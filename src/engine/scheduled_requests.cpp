// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include "engine.h"
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
    const int64_t remaining =
        entry.request->CurrentSequenceLength() -
        entry.request->ProcessedSequenceLength();
    if (remaining <= 0 || entry.unprocessed_token_count == 0 ||
        entry.unprocessed_token_count > static_cast<size_t>(remaining)) {
      throw std::runtime_error(
          "The dynamic step token count must be positive and no greater than the remaining tokens.");
    }
    request_ids.push_back(entry.request_id);
  }
  // Complete every potentially allocating output-bookkeeping operation before binding the plan or
  // executing the model. A partial prefill cannot sample, while a chunk-complete single-sequence
  // request can append at most one generated index.
  for (const auto& entry : plan.requests) {
    const auto remaining =
        static_cast<size_t>(entry.request->CurrentSequenceLength() -
                            entry.request->ProcessedSequenceLength());
    if (entry.unprocessed_token_count == remaining) {
      entry.request->PrepareForStep(kMaxGeneratedTokenIndicesPerStep);
    }
  }
  for (const auto& entry : plan.requests) {
    entry.request->BindScheduledTokenCount(
        entry.unprocessed_token_count);
    requests_.push_back(entry.request);
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
    const bool guidance_applied = TryApplyBatchedGuidanceMasks(logits);

    if (TryGenerateNextTokensBatched(logits, guidance_applied))
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
        requests_[request_idx]->CompleteGeneration();
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

  std::vector<DeviceSpan<float>> logits = decoder_state_->ProcessLogits();
  if (logits.size() != requests_.size()) {
    throw std::runtime_error("Logits size does not match the number of requests.");
  }

  return logits;
}

// Samples all active requests through the scheduler-owned sampler. It owns the reusable workspace
// and groups rows by resolved sampling parameters, while each Request owns its persistent RNG state.
bool ScheduledRequests::TryGenerateNextTokensBatched(std::vector<DeviceSpan<float>>& logits,
                                                     bool guidance_applied) {
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
    sampling_plan_->requests[request_idx]->PrepareGeneration(sampling_plan_->logits[request_idx],
                                                             guidance_applied);
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

// Builds one contiguous mask row per scheduled request. Unguided and partial-prefill rows remain
// pass-through; eligible guided decode rows receive their ready grammar mask. Keeping this
// selection independent of the CUDA transfer makes row routing directly testable.
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

  auto logits = ProcessLogits();
  const bool guidance_applied = TryApplyBatchedGuidanceMasks(logits);
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
      requests_[i]->PrepareGenerationForTransaction(logits[i], guidance_applied);
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
      results[i] = requests_[i]->StageGenerationForTransaction(plan.requests[i]);
      ++sampling_index;
    }
    return;
  }

  for (size_t i = 0; i < requests_.size(); ++i) {
    if (requests_[i]->IsChunkComplete())
      results[i] = requests_[i]->ApplyLogitsForTransaction(logits[i], guidance_applied);
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
