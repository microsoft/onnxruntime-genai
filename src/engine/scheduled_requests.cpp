// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include "engine.h"
#include "../search.h"
#include <exception>

namespace Generators {

namespace {

// Collapses Request::GenerateNextTokens' dispatch into the single (k, p, temperature) triple that
// each branch ends up handing to the sampler on CUDA, where SelectTop() is SampleTopKTopP(1, 0, 1),
// SampleTopK(k, t) is SampleTopKTopP(k, 1, t) and SampleTopP(p, t) is SampleTopKTopP(-1, p, t).
// Returns nothing for options the per-request path rejects, so that it keeps raising the error.
std::optional<BatchedSamplingParams> ResolveSampleArgs(const Config::Search& search) {
  if (!search.do_sample || search.top_k == 1 || search.temperature == 0)
    return BatchedSamplingParams{1, 0.0f, 1.0f};

  if (search.num_beams != 1 || search.top_p < 0.0f || search.top_p > 1.0f || search.top_k < 0)
    return std::nullopt;

  if (search.top_p > 0.0f && search.top_p < 1.0f && search.top_k > 1)
    return BatchedSamplingParams{search.top_k, search.top_p, search.temperature};
  if (search.top_k > 1)
    return BatchedSamplingParams{search.top_k, 1.0f, search.temperature};
  return BatchedSamplingParams{-1, search.top_p, search.temperature};
}

// Greedy token for every row, computed on the device so that the full vocabulary never crosses the
// bus. Returns nothing when the rows are not one contiguous block (the decoder only guarantees that
// when it converts the batch in a single launch) or when the device has no top-1 kernel, leaving the
// caller on its host fallback.
std::vector<int32_t> TryDeviceArgmaxPerRow(DeviceInterface& device,
                                           std::vector<DeviceSpan<float>>& rows) {
  if (rows.empty())
    return {};

  const size_t vocab_size = rows[0].size();
  const float* base = rows[0].Span().data();
  for (size_t i = 1; i < rows.size(); ++i) {
    if (rows[i].size() != vocab_size || rows[i].Span().data() != base + i * vocab_size)
      return {};
  }

  std::vector<int32_t> tokens(rows.size());
  if (!device.ArgMax(base, Ort::TypeToTensorType<float>, static_cast<int>(rows.size()),
                     static_cast<int>(vocab_size), tokens.data()))
    return {};
  return tokens;
}

}  // namespace

ScheduledRequests::ScheduledRequests(std::vector<std::shared_ptr<Request>> requests,
                                     std::shared_ptr<Model> model,
                                     BatchedSampler* batched_sampler,
                                     BatchedSamplingPlan* sampling_plan)
    : requests_{std::move(requests)}, model_{std::move(model)}, batched_sampler_{batched_sampler}, sampling_plan_{sampling_plan} {
  // Fixes what each request contributes to this step before anything reads UnprocessedTokens().
  for (auto& request : requests_) {
    request->ScheduleTokens();
  }
}

ScheduledRequests::ScheduledRequests(const StepPlan& plan,
                                     std::shared_ptr<Model> model,
                                     BatchedSampler* batched_sampler,
                                     BatchedSamplingPlan* sampling_plan)
    : model_{std::move(model)}, batched_sampler_{batched_sampler}, sampling_plan_{sampling_plan} {
  requests_.reserve(plan.requests.size());
  draft_token_counts_.reserve(plan.requests.size());
  std::vector<const void*> request_ids;
  request_ids.reserve(plan.requests.size());
  for (const auto& entry : plan.requests) {
    if (!entry.request || entry.request_id != entry.request.get() ||
        std::find(request_ids.begin(), request_ids.end(), entry.request_id) !=
            request_ids.end()) {
      throw std::runtime_error("The dynamic step plan contains an invalid request.");
    }
    if (!IsExecutable(entry.request->status_)) {
      throw std::runtime_error("The dynamic step plan contains a request that is not executable.");
    }
    if (entry.draft_token_count > entry.request->PendingDraftTokenCount()) {
      throw std::runtime_error("The dynamic step plan verifies drafts the request never proposed.");
    }
    // A verify step sends the request's last committed token plus every draft, so the drafts the
    // transaction is about to stage count towards what this step is allowed to schedule.
    const int64_t remaining =
        entry.request->CurrentSequenceLength() -
        entry.request->ProcessedSequenceLength() +
        static_cast<int64_t>(entry.draft_token_count);
    if (remaining <= 0 || entry.unprocessed_token_count == 0 ||
        entry.unprocessed_token_count > static_cast<size_t>(remaining)) {
      throw std::runtime_error(
          "The dynamic step token count must be positive and no greater than the remaining tokens.");
    }
    request_ids.push_back(entry.request_id);
  }
  // Complete every potentially allocating output-bookkeeping operation before binding the plan or
  // executing the model. A partial prefill cannot sample, while a chunk-complete single-sequence
  // request can append its accepted drafts plus one generated token.
  for (const auto& entry : plan.requests) {
    const auto remaining =
        static_cast<size_t>(entry.request->CurrentSequenceLength() -
                            entry.request->ProcessedSequenceLength()) +
        entry.draft_token_count;
    if (entry.unprocessed_token_count == remaining) {
      entry.request->PrepareForStep(kMaxGeneratedTokenIndicesPerStep +
                                    entry.draft_token_count);
    }
  }
  for (const auto& entry : plan.requests) {
    entry.request->BindScheduledTokenCount(
        entry.unprocessed_token_count);
    requests_.push_back(entry.request);
    draft_token_counts_.push_back(entry.draft_token_count);
  }
}

ExecutionContext& ScheduledRequests::CreateExecutionContext() {
  execution_context_ = std::make_unique<ExecutionContext>();
  return *execution_context_;
}

std::shared_ptr<GeneratorParams> ScheduledRequests::Params() {
  if (!params_) {
    params_ = std::make_shared<GeneratorParams>(*model_);
  }
  return params_;
}

void ScheduledRequests::AddDecoderState(std::unique_ptr<DecoderIO> decoder_state) {
  decoder_state_ = std::move(decoder_state);
}

void ScheduledRequests::GenerateNextTokens() {
  if (!decoder_state_) {
    throw std::runtime_error("Cannot generate next tokens without the decoder state.");
  }

  try {
    auto logits = ProcessLogits();

    if (TryGenerateNextTokensBatched(logits))
      return;

    // Every request owns an independent single-sequence search, so token selection runs once per
    // request. Completing each one inline would block the host on the device once per request and
    // serialize the whole batch; launching all of them first means only the first completion below
    // actually waits for the device.
    for (size_t request_idx = 0; request_idx < requests_.size(); ++request_idx) {
      if (IsExecuting(requests_[request_idx]->status_) &&
          requests_[request_idx]->IsChunkComplete()) {
        requests_[request_idx]->GenerateNextTokens(logits[request_idx]);
      }
    }

    for (size_t request_idx = 0; request_idx < requests_.size(); ++request_idx) {
      if (IsExecuting(requests_[request_idx]->status_) &&
          requests_[request_idx]->IsChunkComplete()) {
        requests_[request_idx]->CompleteGeneration();
      }
    }

    for (const auto& request : requests_) {
      if (IsExecuting(request->status_) &&
          !request->IsChunkComplete())
        request->AdvanceChunk();
    }
  } catch (...) {
    const auto error = std::current_exception();
    try {
      model_->p_device_scoring_->Synchronize();
    } catch (...) {
    }
    std::rethrow_exception(error);
  }
}

std::vector<DeviceSpan<float>> ScheduledRequests::ProcessLogits() {
  if (!decoder_state_) {
    throw std::runtime_error("Cannot process logits without the decoder state.");
  }

  // A verify step gets one row per draft on top of the row that predicts the request's next token.
  size_t expected_rows = requests_.size();
  for (size_t draft_count : draft_token_counts_) {
    expected_rows += draft_count;
  }

  std::vector<DeviceSpan<float>> logits = decoder_state_->ProcessLogits();
  if (logits.size() != expected_rows) {
    throw std::runtime_error("Logits size does not match the number of requests.");
  }

  return logits;
}

Tensor* ScheduledRequests::HiddenStates() const {
  return decoder_state_ ? decoder_state_->HiddenStates() : nullptr;
}

std::vector<DeviceSpan<float>> ScheduledRequests::SelectSampledRows(
    std::vector<DeviceSpan<float>>& verify_rows) {
  std::vector<DeviceSpan<float>> sampled_rows;
  sampled_rows.reserve(requests_.size());

  // Verification only needs each draft row's argmax. Reading a whole vocabulary row back to the host
  // costs about a megabyte and a stream synchronization per draft, which on a large-vocabulary model
  // is comparable to the step itself, so ask the device for the token ids in one launch instead.
  const bool step_has_drafts = std::any_of(draft_token_counts_.begin(), draft_token_counts_.end(),
                                           [](size_t count) { return count != 0; });
  const std::vector<int32_t> row_argmax =
      step_has_drafts ? TryDeviceArgmaxPerRow(*model_->p_device_inputs_, verify_rows)
                      : std::vector<int32_t>{};

  size_t row = 0;
  for (size_t i = 0; i < requests_.size(); ++i) {
    const size_t draft_count =
        i < draft_token_counts_.size() ? draft_token_counts_[i] : 0;
    if (draft_count == 0) {
      sampled_rows.push_back(verify_rows[row++]);
      continue;
    }

    // Row j predicts the token at committed_length + j, which is exactly where draft j sits. Accept
    // the longest prefix of the proposal the target model would have produced on its own; the row
    // after it holds the token that replaces the first rejected draft, or the bonus token.
    const auto drafts = requests_[i]->StagedDraftTokens();
    size_t accepted_count = 0;
    while (accepted_count < draft_count) {
      // A stop token ends the turn, so it has to be produced by the sampler, which applies the
      // search's end-of-sequence handling. Appending it as a draft would emit it and carry on.
      if (requests_[i]->IsStopToken(drafts[accepted_count])) {
        break;
      }
      int32_t predicted;
      if (!row_argmax.empty()) {
        predicted = row_argmax[row + accepted_count];
      } else {
        auto row_values = verify_rows[row + accepted_count].CopyDeviceToCpu();
        predicted = static_cast<int32_t>(
            std::max_element(row_values.begin(), row_values.end()) - row_values.begin());
      }
      if (predicted != drafts[accepted_count]) {
        break;
      }
      ++accepted_count;
    }
    requests_[i]->RewindDraftsForTransaction(accepted_count);
    sampled_rows.push_back(verify_rows[row + accepted_count]);
    row += draft_count + 1;
  }
  return sampled_rows;
}

// Samples all active requests through the scheduler-owned sampler. It owns the reusable workspace
// and groups rows by resolved sampling parameters, while each Request owns its persistent RNG state.
bool ScheduledRequests::TryGenerateNextTokensBatched(std::vector<DeviceSpan<float>>& logits) {
  if (!PrepareBatchedSamplingPlan(false))
    return false;

  for (size_t request_idx = 0; request_idx < requests_.size(); ++request_idx) {
    if (IsExecuting(requests_[request_idx]->status_) &&
        requests_[request_idx]->IsChunkComplete())
      sampling_plan_->logits.push_back(logits[request_idx]);
  }

  if (sampling_plan_->requests.empty())
    return true;

  for (size_t request_idx = 0; request_idx < sampling_plan_->requests.size(); ++request_idx) {
    sampling_plan_->requests[request_idx]->PrepareGeneration(sampling_plan_->logits[request_idx]);
  }

  auto next_tokens = batched_sampler_->Sample(sampling_plan_->logits, sampling_plan_->params,
                                              sampling_plan_->states,
                                              model_->config_->model.vocab_size);

  for (size_t request_idx = 0; request_idx < sampling_plan_->requests.size(); ++request_idx) {
    if (!sampling_plan_->requests[request_idx]->BindNextTokensSlot(next_tokens.subspan(request_idx, 1)))
      throw std::runtime_error("The scoring device supports batched sampling but the request search does not.");
    sampling_plan_->requests[request_idx]->OnNextTokensSampled();
  }

  next_tokens.CopyDeviceToCpu();

  for (auto* request : sampling_plan_->requests) {
    request->CompleteGeneration();
  }
  for (const auto& request : requests_) {
    if (IsExecuting(request->status_) &&
        !request->IsChunkComplete())
      request->AdvanceChunk();
  }

  return true;
}

bool ScheduledRequests::PrepareBatchedSamplingPlan(
    bool require_transaction_support) {
  if (!batched_sampler_ || !sampling_plan_ ||
      (require_transaction_support && !batched_sampler_->SupportsTransactions())) {
    return false;
  }

  sampling_plan_->Clear();
  for (const auto& request : requests_) {
    // Dynamic transactions keep newly admitted and continued requests Queued until commit, while
    // the static scheduler moves every executable row to Active before constructing the batch.
    const bool status_is_executable =
        require_transaction_support ? IsExecutable(request->status_)
                                    : IsExecuting(request->status_);
    if (!status_is_executable ||
        !request->IsChunkComplete())
      continue;

    const auto args = ResolveSampleArgs(request->SearchOptions());
    if (!args || !request->SupportsBatchedSampling()) {
      sampling_plan_->Clear();
      return false;
    }
    sampling_plan_->requests.push_back(request.get());
    sampling_plan_->params.push_back(*args);
    sampling_plan_->states.push_back(&request->SamplingState(*batched_sampler_));
  }
  return !sampling_plan_->requests.empty();
}

void ScheduledRequests::BeginTransaction() {
  if (transaction_checkpoint_count_ != 0 || sampler_checkpoint_active_)
    throw std::logic_error("Scheduled request transaction is already active.");

  transaction_uses_batched_sampler_ = PrepareBatchedSamplingPlan(true);
  try {
    for (const auto& request : requests_) {
      if (transaction_uses_batched_sampler_)
        request->SaveStateForExternalSamplingTransaction();
      else
        request->SaveStateForTransaction();
      ++transaction_checkpoint_count_;
    }
    // Drafts join the sequence only after every checkpoint exists, so an abort rewinds them for
    // free through the same restore path as a sampled token.
    for (size_t i = 0; i < draft_token_counts_.size(); ++i) {
      requests_[i]->AppendDraftsForTransaction(draft_token_counts_[i]);
    }
    if (transaction_uses_batched_sampler_) {
      batched_sampler_->SaveStateForTransaction(sampling_plan_->states);
      sampler_checkpoint_active_ = true;
    }
  } catch (...) {
    const auto error = std::current_exception();
    try {
      RestoreStateForTransaction();
    } catch (...) {
    }
    std::rethrow_exception(error);
  }
}

void ScheduledRequests::GenerateNextTokensForTransaction(
    const StepPlan& plan,
    std::vector<RequestStepResult>& results) {
  if (plan.requests.size() != requests_.size() ||
      transaction_checkpoint_count_ != requests_.size()) {
    throw std::logic_error("Scheduled request transaction does not match the step plan.");
  }

  auto verify_rows = ProcessLogits();
  auto logits = SelectSampledRows(verify_rows);
  results.assign(requests_.size(), RequestStepResult{});
  if (transaction_uses_batched_sampler_) {
    sampling_plan_->logits.clear();
    size_t sampling_index = 0;
    for (size_t i = 0; i < requests_.size(); ++i) {
      if (!requests_[i]->IsChunkComplete())
        continue;
      if (sampling_index >= sampling_plan_->requests.size() ||
          sampling_plan_->requests[sampling_index] != requests_[i].get()) {
        throw std::logic_error("Batched sampling plan does not match the scheduled requests.");
      }
      sampling_plan_->logits.push_back(logits[i]);
      requests_[i]->PrepareGenerationForTransaction(logits[i]);
      ++sampling_index;
    }
    if (sampling_index != sampling_plan_->requests.size())
      throw std::logic_error("Batched sampling plan does not match the scheduled requests.");

    auto next_tokens = batched_sampler_->Sample(
        sampling_plan_->logits, sampling_plan_->params,
        sampling_plan_->states, model_->config_->model.vocab_size);
    for (size_t i = 0; i < sampling_plan_->requests.size(); ++i) {
      if (!sampling_plan_->requests[i]->BindNextTokensSlot(next_tokens.subspan(i, 1)))
        throw std::runtime_error("The request search rejected the batched sampler output.");
      sampling_plan_->requests[i]->OnNextTokensSampled();
    }
    next_tokens.CopyDeviceToCpu();
    sampling_index = 0;
    for (size_t i = 0; i < requests_.size(); ++i) {
      if (!requests_[i]->IsChunkComplete())
        continue;
      results[i] = requests_[i]->StageGenerationForTransaction();
      ++sampling_index;
    }
    return;
  }

  for (size_t i = 0; i < requests_.size(); ++i) {
    if (requests_[i]->IsChunkComplete())
      results[i] = requests_[i]->ApplyLogitsForTransaction(logits[i]);
  }
}

void ScheduledRequests::RestoreStateForTransaction() {
  std::exception_ptr error;
  try {
    model_->p_device_scoring_->Synchronize();
  } catch (...) {
    error = std::current_exception();
  }

  if (sampler_checkpoint_active_) {
    try {
      batched_sampler_->RestoreStateForTransaction();
    } catch (...) {
      error = std::current_exception();
    }
    sampler_checkpoint_active_ = false;
  }

  std::vector<Request*> pending_restore_completion;
  pending_restore_completion.reserve(transaction_checkpoint_count_);
  while (transaction_checkpoint_count_ > 0) {
    auto* request = requests_[--transaction_checkpoint_count_].get();
    try {
      request->QueueStateRestoreForTransaction();
      pending_restore_completion.push_back(request);
    } catch (...) {
      if (!error)
        error = std::current_exception();
    }
  }
  transaction_uses_batched_sampler_ = false;

  try {
    model_->p_device_scoring_->Synchronize();
  } catch (...) {
    if (!error)
      error = std::current_exception();
  }
  for (auto* request : pending_restore_completion) {
    try {
      request->CompleteStateRestoreForTransaction();
    } catch (...) {
      if (!error)
        error = std::current_exception();
    }
  }
  if (error)
    std::rethrow_exception(error);
}

void ScheduledRequests::CommitStateForTransaction() {
  if (transaction_checkpoint_count_ != requests_.size())
    throw std::logic_error("Scheduled request transaction is not active.");

  for (const auto& request : requests_) {
    request->CommitStateForTransaction();
  }
  transaction_checkpoint_count_ = 0;
  if (sampler_checkpoint_active_) {
    batched_sampler_->CommitStateForTransaction();
    sampler_checkpoint_active_ = false;
  }
  transaction_uses_batched_sampler_ = false;
}

}  // namespace Generators
