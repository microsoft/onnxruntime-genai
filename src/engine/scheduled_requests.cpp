// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include "engine.h"
#include "request_index.h"
#include "../constrained_logits_processor.h"
#include "../search.h"
#include <cstdint>
#include <exception>
#include <limits>

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
  RequestIndex request_ids{plan.requests.size()};
  for (const auto& entry : plan.requests) {
    if (!entry.request || entry.request_id != entry.request.get() ||
        !request_ids.Insert(entry.request_id, request_ids.Size())) {
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
          "The dynamic step token count (" +
          std::to_string(entry.unprocessed_token_count) +
          ") must be positive and no greater than the remaining token count (" +
          std::to_string(remaining) + ").");
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

Tensor* ScheduledRequests::HiddenStates() const {
  return decoder_state_ ? decoder_state_->HiddenStates() : nullptr;
}

void ScheduledRequests::GenerateNextTokens(std::vector<RequestStepResult>& results) {
  if (!decoder_state_) {
    throw std::runtime_error("Cannot generate next tokens without the decoder state.");
  }

  try {
    auto logits = ProcessLogits();
    const bool guidance_applied = TryApplyBatchedGuidanceMasks(logits);
    results.assign(requests_.size(), RequestStepResult{});

    if (TryGenerateNextTokensBatched(logits, guidance_applied, &results))
      return;

    // Every request owns an independent single-sequence search, so token selection runs once per
    // request. Completing each one inline would block the host on the device once per request and
    // serialize the whole batch; launching all of them first means only the first completion below
    // actually waits for the device.
    for (size_t request_idx = 0; request_idx < requests_.size(); ++request_idx) {
      if (IsExecuting(requests_[request_idx]->status_) &&
          requests_[request_idx]->IsChunkComplete()) {
        requests_[request_idx]->GenerateNextTokens(logits[request_idx], guidance_applied);
      }
    }

    for (size_t request_idx = 0; request_idx < requests_.size(); ++request_idx) {
      if (IsExecuting(requests_[request_idx]->status_) &&
          requests_[request_idx]->IsChunkComplete()) {
        results[request_idx] = requests_[request_idx]->CompleteGeneration();
      }
    }

    ScheduleGuidanceMasks();
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
    throw std::runtime_error(
        "Logits row count does not match the packed step: expected " +
        std::to_string(expected_rows) + ", got " + std::to_string(logits.size()) + ".");
  }

  return logits;
}

std::vector<DeviceSpan<float>> ScheduledRequests::SelectSampledRows(
    std::vector<DeviceSpan<float>>& verify_rows) {
  if (std::none_of(draft_token_counts_.begin(), draft_token_counts_.end(),
                   [](size_t draft_count) { return draft_count != 0; })) {
    return std::move(verify_rows);
  }
  if (!sampling_plan_) {
    throw std::logic_error("Draft verification requires scheduler-owned argmax storage.");
  }
  auto& predicted_tokens = sampling_plan_->verification_tokens;
  predicted_tokens.resize(verify_rows.size());

  bool rows_are_contiguous = !verify_rows.empty();
  const size_t vocab_size = static_cast<size_t>(model_->config_->model.vocab_size);
  const float* first_row = rows_are_contiguous
                               ? verify_rows.front().Span().data()
                               : nullptr;
  for (size_t row = 0; row < verify_rows.size() && rows_are_contiguous; ++row) {
    rows_are_contiguous =
        verify_rows[row].size() == vocab_size &&
        verify_rows[row].Span().data() == first_row + row * vocab_size;
  }

  const bool used_device_argmax =
      rows_are_contiguous &&
      model_->p_device_scoring_->ArgMax(
          first_row, Ort::TypeToTensorType<float>,
          static_cast<int>(verify_rows.size()),
          model_->config_->model.vocab_size, predicted_tokens.data());
  if (!used_device_argmax) {
    const bool rows_share_buffer =
        !verify_rows.empty() &&
        std::all_of(verify_rows.begin() + 1, verify_rows.end(),
                    [&verify_rows](const DeviceSpan<float>& row) {
                      return row.SameBufferAs(verify_rows.front());
                    });
    if (rows_share_buffer) {
      verify_rows.front().CopyDeviceToCpu();
    }
    for (size_t row = 0; row < verify_rows.size(); ++row) {
      auto row_values = rows_share_buffer
                            ? verify_rows[row].CpuSpan()
                            : verify_rows[row].CopyDeviceToCpu();
      predicted_tokens[row] = static_cast<int32_t>(
          std::max_element(row_values.begin(), row_values.end()) -
          row_values.begin());
    }
  }

  std::vector<DeviceSpan<float>> sampled_rows;
  sampled_rows.reserve(requests_.size());
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
      if (predicted_tokens[row + accepted_count] != drafts[accepted_count]) {
        break;
      }
      ++accepted_count;
    }
    requests_[i]->CommitAcceptedDraftsForTransaction(accepted_count);
    sampled_rows.push_back(verify_rows[row + accepted_count]);
    row += draft_count + 1;
  }
  return sampled_rows;
}

// Samples all active requests through the scheduler-owned sampler. It owns the reusable workspace
// and groups rows by resolved sampling parameters, while each Request owns its persistent RNG state.
bool ScheduledRequests::TryGenerateNextTokensBatched(
    std::vector<DeviceSpan<float>>& logits,
    bool guidance_applied,
    std::vector<RequestStepResult>* results) {
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
    sampling_plan_->requests[request_idx]->PrepareGeneration(
        sampling_plan_->logits[request_idx], guidance_applied);
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

  for (size_t sampling_index = 0;
       sampling_index < sampling_plan_->requests.size();
       ++sampling_index) {
    const auto result =
        sampling_plan_->requests[sampling_index]->CompleteGeneration();
    if (results) {
      (*results)[sampling_plan_->result_indices[sampling_index]] = result;
    }
  }
  ScheduleGuidanceMasks();
  for (const auto& request : requests_) {
    if (IsExecuting(request->status_) &&
        !request->IsChunkComplete())
      request->AdvanceChunk();
  }

  return true;
}

void ScheduledRequests::ScheduleGuidanceMasks() noexcept {
  try {
    std::vector<ConstrainedLogitsProcessor*> processors;
    processors.reserve(requests_.size());
    for (const auto& request : requests_) {
      if (request->guidance_logits_processor_ &&
          !request->IsTurnComplete()) {
        processors.push_back(request->guidance_logits_processor_.get());
      }
    }
    ScheduleGuidanceMaskComputation(processors);
  } catch (const std::logic_error& error) {
    if (g_log.enabled) {
      Log("error") << "Guidance mask precomputation invariant violated: "
                   << error.what() << std::endl;
    }
  } catch (const std::exception& error) {
    if (g_log.enabled && g_log.warning) {
      Log("warning") << "Guidance mask precomputation was deferred: "
                     << error.what() << std::endl;
    }
  } catch (...) {
    if (g_log.enabled && g_log.warning) {
      Log("warning",
          "Guidance mask precomputation was deferred after a non-standard exception.");
    }
  }
}

BatchedGuidanceMaskStatus CollectBatchedGuidanceMasks(
    std::span<const std::shared_ptr<Request>> requests,
    size_t words_per_row,
    std::vector<uint32_t>& masks) {
  masks.assign(requests.size() * words_per_row,
               std::numeric_limits<uint32_t>::max());
  bool has_guidance = false;
  for (size_t row = 0; row < requests.size(); ++row) {
    const auto& request = requests[row];
    if (!request->HasGuidance() || !request->IsChunkComplete()) {
      continue;
    }
    if (request->ScheduledTokenCount() == 0) {
      return BatchedGuidanceMaskStatus::FallbackRequired;
    }
    const auto mask = request->GetReadyGuidanceMask();
    if (mask.size() != words_per_row) {
      return BatchedGuidanceMaskStatus::FallbackRequired;
    }
    std::copy(mask.begin(), mask.end(),
              masks.begin() +
                  static_cast<std::ptrdiff_t>(row * words_per_row));
    has_guidance = true;
  }
  return has_guidance ? BatchedGuidanceMaskStatus::Ready
                      : BatchedGuidanceMaskStatus::NoEligibleGuidance;
}

bool ScheduledRequests::TryApplyBatchedGuidanceMasks(std::vector<DeviceSpan<float>>& logits) {
  if (!sampling_plan_ || logits.empty()) {
    return false;
  }
  const auto device_type = model_->p_device_scoring_->GetType();
  if (device_type != DeviceType::CUDA && device_type != DeviceType::NvTensorRtRtx) {
    return false;
  }
  if (std::none_of(
          requests_.begin(), requests_.end(),
          [](const auto& request) { return request->HasGuidance(); })) {
    return false;
  }

  const size_t vocab_size = static_cast<size_t>(model_->config_->model.vocab_size);
  const size_t words_per_row = (vocab_size + 31) / 32;
  float* const first_row = logits.front().Span().data();
  for (size_t i = 0; i < logits.size(); ++i) {
    if (logits[i].size() != vocab_size ||
        logits[i].Span().data() != first_row + i * vocab_size) {
      return false;
    }
  }

  if (CollectBatchedGuidanceMasks(
          requests_, words_per_row,
          sampling_plan_->guidance_masks) !=
      BatchedGuidanceMaskStatus::Ready) {
    return false;
  }

  if (sampling_plan_->guidance_device_masks.size() !=
      sampling_plan_->guidance_masks.size()) {
    sampling_plan_->guidance_device_masks =
        model_->p_device_scoring_->Allocate<uint32_t>(sampling_plan_->guidance_masks.size());
  }
  copy(std::span<const uint32_t>{sampling_plan_->guidance_masks},
       sampling_plan_->guidance_device_masks.CpuSpan());
  sampling_plan_->guidance_device_masks.CopyCpuToDevice();
  model_->p_device_scoring_->LaunchAddLogitsMask(
      first_row, static_cast<int>(logits.size()), static_cast<int>(vocab_size),
      sampling_plan_->guidance_device_masks.Span().data());
  return true;
}

bool ScheduledRequests::PrepareBatchedSamplingPlan(
    bool require_transaction_support) {
  if (!batched_sampler_ || !sampling_plan_ ||
      (require_transaction_support && !batched_sampler_->SupportsTransactions())) {
    return false;
  }

  sampling_plan_->Clear();
  for (size_t request_index = 0;
       request_index < requests_.size();
       ++request_index) {
    const auto& request = requests_[request_index];
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
    sampling_plan_->result_indices.push_back(request_index);
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
  const bool guidance_applied = TryApplyBatchedGuidanceMasks(logits);
  results.assign(requests_.size(), RequestStepResult{});
  if (transaction_uses_batched_sampler_) {
    sampling_plan_->logits.clear();
    const size_t original_sampling_count = sampling_plan_->requests.size();
    size_t source_sampling_index = 0;
    size_t active_sampling_count = 0;
    for (size_t i = 0; i < requests_.size(); ++i) {
      if (!requests_[i]->IsChunkComplete())
        continue;
      if (source_sampling_index >= original_sampling_count ||
          sampling_plan_->requests[source_sampling_index] != requests_[i].get()) {
        throw std::logic_error("Batched sampling plan does not match the scheduled requests.");
      }
      if (requests_[i]->DraftVerificationCompletedGeneration()) {
        results[i] =
            requests_[i]->StageDraftCompletionForTransaction();
      } else {
        if (active_sampling_count != source_sampling_index) {
          sampling_plan_->requests[active_sampling_count] =
              sampling_plan_->requests[source_sampling_index];
          sampling_plan_->params[active_sampling_count] =
              sampling_plan_->params[source_sampling_index];
          sampling_plan_->states[active_sampling_count] =
              sampling_plan_->states[source_sampling_index];
        }
        sampling_plan_->logits.push_back(logits[i]);
        requests_[i]->PrepareGenerationForTransaction(logits[i], guidance_applied);
        ++active_sampling_count;
      }
      ++source_sampling_index;
    }
    if (source_sampling_index != original_sampling_count)
      throw std::logic_error("Batched sampling plan does not match the scheduled requests.");
    sampling_plan_->requests.resize(active_sampling_count);
    sampling_plan_->params.resize(active_sampling_count);
    sampling_plan_->states.resize(active_sampling_count);

    if (active_sampling_count == 0)
      return;

    auto next_tokens = batched_sampler_->Sample(
        sampling_plan_->logits, sampling_plan_->params,
        sampling_plan_->states, model_->config_->model.vocab_size);
    for (size_t i = 0; i < sampling_plan_->requests.size(); ++i) {
      if (!sampling_plan_->requests[i]->BindNextTokensSlot(next_tokens.subspan(i, 1)))
        throw std::runtime_error("The request search rejected the batched sampler output.");
      sampling_plan_->requests[i]->OnNextTokensSampled();
    }
    next_tokens.CopyDeviceToCpu();
    for (size_t i = 0; i < requests_.size(); ++i) {
      if (!requests_[i]->IsChunkComplete() ||
          requests_[i]->DraftVerificationCompletedGeneration())
        continue;
      results[i] = requests_[i]->StageGenerationForTransaction(plan.requests[i]);
    }
    return;
  }

  for (size_t i = 0; i < requests_.size(); ++i) {
    if (requests_[i]->DraftVerificationCompletedGeneration()) {
      results[i] =
          requests_[i]->StageDraftCompletionForTransaction();
    } else if (requests_[i]->IsChunkComplete()) {
      results[i] = requests_[i]->ApplyLogitsForTransaction(logits[i], guidance_applied);
    }
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
