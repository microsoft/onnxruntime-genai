// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "request.h"

#include "engine.h"
#include "request_index.h"
#include "../constrained_logits_processor.h"
#include "../decoding/speculative_sampling.h"
#include "../search.h"
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>

namespace Generators {

namespace {

// Collapses Request::SelectNextToken's dispatch into the single (k, p, temperature) triple that
// each branch ends up handing to the sampler on CUDA, where SelectTop() is SampleTopKTopP(1, 0, 1),
// SampleTopK(k, t) is SampleTopKTopP(k, 1, t) and SampleTopP(p, t) is SampleTopKTopP(-1, p, t).
BatchedSamplingParams ResolveSampleArgs(const EffectiveTurnPolicy& policy) {
  if (policy.IsGreedy())
    return BatchedSamplingParams{1, 0.0f, 1.0f};

  if (policy.top_p > 0.0f && policy.top_p < 1.0f && policy.top_k > 1)
    return BatchedSamplingParams{policy.top_k, policy.top_p, policy.temperature};
  if (policy.top_k > 1)
    return BatchedSamplingParams{policy.top_k, 1.0f, policy.temperature};
  return BatchedSamplingParams{-1, policy.top_p, policy.temperature};
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

struct TopKScores {
  int k{};
  std::vector<int32_t> tokens;
  std::vector<float> scores;
};

TopKScores TryDeviceTopKScoresPerRow(DeviceInterface& device,
                                     std::vector<DeviceSpan<float>>& rows,
                                     int k) {
  if (rows.empty() || k <= 1)
    return {};

  const size_t vocab_size = rows[0].size();
  const float* base = rows[0].Span().data();
  for (size_t i = 1; i < rows.size(); ++i) {
    if (rows[i].size() != vocab_size || rows[i].Span().data() != base + i * vocab_size)
      return {};
  }

  TopKScores result;
  result.k = std::min(k, static_cast<int>(vocab_size));
  const size_t value_count = rows.size() * static_cast<size_t>(result.k);
  result.tokens.resize(value_count);
  result.scores.resize(value_count);
  if (!device.TopKScores(base, Ort::TypeToTensorType<float>, static_cast<int>(rows.size()),
                         static_cast<int>(vocab_size), result.k,
                         result.tokens.data(), result.scores.data())) {
    return {};
  }
  return result;
}

TargetTokenSelection BuildTargetSelection(
    size_t row, DeviceSpan<float> logits, const EffectiveTurnPolicy& policy,
    const TopKScores& topk, SampledCategorical& scratch) {
  TargetTokenSelection selection;
  if (topk.k == 0) {
    const auto cpu_logits = logits.CopyDeviceToCpu();
    ComputeSampledCategorical(cpu_logits, policy.top_k, policy.top_p,
                              policy.temperature, scratch);
    selection.indices = scratch.indices;
    selection.probs = scratch.probs;
    return selection;
  }

  const int k = std::min(policy.top_k, topk.k);
  const size_t offset = row * static_cast<size_t>(topk.k);
  const float max_score = topk.scores[offset];
  const float inverse_temperature = 1.0f / policy.temperature;
  std::vector<float> probabilities(static_cast<size_t>(k));
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    probabilities[static_cast<size_t>(i)] =
        std::exp((topk.scores[offset + static_cast<size_t>(i)] - max_score) *
                 inverse_temperature);
    sum += probabilities[static_cast<size_t>(i)];
  }
  for (float& probability : probabilities)
    probability /= sum;

  int keep = k;
  if (policy.top_p > 0.0f && policy.top_p < 1.0f) {
    float cumulative = 0.0f;
    for (int i = 0; i < k; ++i) {
      cumulative += probabilities[static_cast<size_t>(i)];
      if (cumulative >= policy.top_p) {
        keep = i + 1;
        break;
      }
    }
  }
  float kept_sum = 0.0f;
  for (int i = 0; i < keep; ++i)
    kept_sum += probabilities[static_cast<size_t>(i)];
  selection.indices.assign(
      topk.tokens.begin() + static_cast<ptrdiff_t>(offset),
      topk.tokens.begin() + static_cast<ptrdiff_t>(offset + static_cast<size_t>(keep)));
  selection.probs.assign(probabilities.begin(), probabilities.begin() + keep);
  for (float& probability : selection.probs)
    probability /= kept_sum;
  return selection;
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

Tensor* ScheduledRequests::AuxHiddenStates() const {
  return decoder_state_ ? decoder_state_->AuxHiddenStates() : nullptr;
}

std::vector<DeviceSpan<float>> ScheduledRequests::SelectSampledRows(
    std::vector<DeviceSpan<float>>& verify_rows,
    std::vector<std::vector<int32_t>>& selected_tokens,
    std::vector<size_t>& confirmed_draft_counts,
    std::vector<std::vector<std::mt19937>>& rng_checkpoints) {
  std::vector<DeviceSpan<float>> sampled_rows;
  sampled_rows.reserve(requests_.size());
  selected_tokens.resize(requests_.size());
  confirmed_draft_counts.assign(requests_.size(), 0);
  // Left completely empty (no allocation at all) unless the pre-scan below finds at least one
  // random-sampled drafted request with an active stop controller; only then is it sized once, to
  // requests_.size() so per-request indexing below stays simple. A step with no drafts, no random
  // sampling, or no stop-enabled request among them never allocates or grows this at all.
  rng_checkpoints.clear();

  // Verification only needs each draft row's argmax. Reading a whole vocabulary row back to the host
  // costs about a megabyte and a stream synchronization per draft, which on a large-vocabulary model
  // is comparable to the step itself, so ask the device for the token ids in one launch instead.
  const bool step_has_drafts = std::any_of(draft_token_counts_.begin(), draft_token_counts_.end(),
                                           [](size_t count) { return count != 0; });
  std::vector<int32_t> row_argmax =
      step_has_drafts ? TryDeviceArgmaxPerRow(*model_->p_device_inputs_, verify_rows)
                      : std::vector<int32_t>{};
  if (step_has_drafts && row_argmax.empty()) {
    row_argmax.resize(verify_rows.size());
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
      const auto row_values = rows_share_buffer
                                  ? verify_rows[row].CpuSpan()
                                  : verify_rows[row].CopyDeviceToCpu();
      row_argmax[row] = static_cast<int32_t>(
          std::max_element(row_values.begin(), row_values.end()) -
          row_values.begin());
    }
  }
  int max_sampling_top_k = 0;
  bool any_checkpoint_needed = false;
  for (size_t i = 0; i < requests_.size(); ++i) {
    if (draft_token_counts_[i] != 0 && !requests_[i]->TurnPolicy().IsGreedy()) {
      max_sampling_top_k = std::max(max_sampling_top_k, requests_[i]->TurnPolicy().top_k);
      any_checkpoint_needed = any_checkpoint_needed || requests_[i]->stop_controller_ != nullptr;
    }
  }
  if (any_checkpoint_needed) {
    rng_checkpoints.resize(requests_.size());
  }
  const TopKScores topk = TryDeviceTopKScoresPerRow(
      *model_->p_device_inputs_, verify_rows, max_sampling_top_k);
  SampledCategorical sampling_scratch;
  size_t row = 0;
  for (size_t i = 0; i < requests_.size(); ++i) {
    const size_t draft_count =
        i < draft_token_counts_.size() ? draft_token_counts_[i] : 0;
    if (draft_count == 0) {
      sampled_rows.push_back(verify_rows[row++]);
      continue;
    }

    if (!requests_[i]->TurnPolicy().IsGreedy()) {
      // Draft acceptance is sequential: each row is drawn only if the previous draw matched its
      // draft, which the batched device sampler cannot express. These draws therefore come from
      // the Request's own host stream (checkpointed with the transaction) rather than its device
      // BatchedSamplerState. The turn seed consequently reproduces a given decode path, not
      // output across decode paths, because whether drafts are admitted depends on batch
      // composition. See "Seeded sampling" in docs/paged_attention_engine.md.
      //
      // A stop-enabled request additionally checkpoints rng_ right after every draw here (a plain
      // copy of the fixed-size std::mt19937 state, intentionally allocated/copied). The outer
      // rng_checkpoints vector itself is left completely empty (see above) unless at least one
      // random-sampled drafted request in this whole step needs it, and each such request's own
      // inner vector is reserved once up front for this call's exact upper bound, draft_count + 1,
      // so growing it below never reallocates or re-copies already-captured checkpoints. If a
      // later stop match truncates this round before its natural end, the caller restores rng_ to
      // the checkpoint captured after the retained match, undoing every draw made for a now-
      // discarded, never-visible token so those candidates do not advance the host verification
      // RNG beyond the retained prefix. A
      // request with no active stop controller never checkpoints (no reservation, no copies, no
      // vector growth) and a step with no such request at all never even allocates the outer
      // vector, so this adds no cost to the existing no-stop/greedy/non-drafted paths.
      const bool checkpoint_rng = requests_[i]->stop_controller_ != nullptr;
      if (checkpoint_rng) {
        rng_checkpoints[i].reserve(draft_count + 1);
      }
      const auto drafts = requests_[i]->StagedDraftTokens();
      const size_t token_budget = requests_[i]->RemainingTurnTokenBudget();
      requests_[i]->RewindDraftsForTransaction(0);
      size_t accepted_count = 0;
      while (accepted_count < draft_count &&
             selected_tokens[i].size() < token_budget) {
        const auto selection = BuildTargetSelection(
            row + accepted_count, verify_rows[row + accepted_count],
            requests_[i]->TurnPolicy(), topk, sampling_scratch);
        const int32_t token = SampleTargetToken(selection, requests_[i]->rng_);
        selected_tokens[i].push_back(token);
        if (checkpoint_rng) {
          rng_checkpoints[i].push_back(requests_[i]->rng_);
        }
        if (token != drafts[accepted_count] || requests_[i]->IsStopToken(token))
          break;
        ++accepted_count;
      }
      if (accepted_count == draft_count &&
          selected_tokens[i].size() < token_budget) {
        const auto selection = BuildTargetSelection(
            row + draft_count, verify_rows[row + draft_count],
            requests_[i]->TurnPolicy(), topk, sampling_scratch);
        selected_tokens[i].push_back(
            SampleTargetToken(selection, requests_[i]->rng_));
        if (checkpoint_rng) {
          rng_checkpoints[i].push_back(requests_[i]->rng_);
        }
      }
      confirmed_draft_counts[i] = accepted_count;
      sampled_rows.push_back({});
      row += draft_count + 1;
      continue;
    }

    // Row j predicts the token at committed_length + j, which is exactly where draft j sits. Accept
    // the longest prefix of the proposal the target model would have produced on its own; the row
    // after it holds the token that replaces the first rejected draft, or the bonus token.
    const auto drafts = requests_[i]->StagedDraftTokens();
    size_t accepted_count = 0;
    while (accepted_count < draft_count) {
      if (row_argmax[row + accepted_count] != drafts[accepted_count]) {
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

float* PackedLogitsRowBase(std::vector<DeviceSpan<float>>& logits, size_t vocab_size) {
  if (logits.empty() || vocab_size == 0) {
    return nullptr;
  }
  // Span() dereferences the row's backing buffer, and an empty row has none, so every row must be
  // proven to be a full vocabulary row before any pointer is taken.
  for (const auto& row : logits) {
    if (row.size() != vocab_size) {
      return nullptr;
    }
  }
  float* const first_row = logits.front().Span().data();
  for (size_t i = 0; i < logits.size(); ++i) {
    if (logits[i].Span().data() != first_row + i * vocab_size) {
      return nullptr;
    }
  }
  return first_row;
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
  float* const first_row = PackedLogitsRowBase(logits, vocab_size);
  if (!first_row) {
    return false;
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
    if (require_transaction_support && draft_token_counts_[request_index] != 0 &&
        !request->TurnPolicy().IsGreedy()) {
      continue;
    }

    if (!request->SupportsBatchedSampling()) {
      sampling_plan_->Clear();
      return false;
    }
    sampling_plan_->requests.push_back(request.get());
    sampling_plan_->result_indices.push_back(request_index);
    sampling_plan_->params.push_back(ResolveSampleArgs(request->TurnPolicy()));
    sampling_plan_->states.push_back(&request->SamplingState(*batched_sampler_));
  }
  return !sampling_plan_->requests.empty();
}

void ScheduledRequests::BeginTransaction() {
  if (transaction_checkpoint_count_ != 0 || sampler_checkpoint_active_)
    throw std::logic_error("Scheduled request transaction is already active.");

  transaction_uses_batched_sampler_ = PrepareBatchedSamplingPlan(true);
  checkpointed_sampler_states_.clear();
  try {
    for (const auto& request : requests_) {
      const bool uses_batched_sampler =
          transaction_uses_batched_sampler_ &&
          std::find(sampling_plan_->requests.begin(), sampling_plan_->requests.end(),
                    request.get()) != sampling_plan_->requests.end();
      if (uses_batched_sampler)
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
    if (batched_sampler_ && batched_sampler_->SupportsTransactions()) {
      if (transaction_uses_batched_sampler_) {
        checkpointed_sampler_states_ = sampling_plan_->states;
      }
      // A pending turn reseed overwrites a device stream in place, so its state has to be
      // checkpointed even when this step does not sample that request through the batched sampler
      // (a sampled draft verification draws on the host, and a plan can be abandoned entirely).
      // Without this the reseed would be an unrecoverable mutation inside the transaction.
      for (const auto& request : requests_) {
        if (!request->HasPendingTurnSeed() || !request->IsChunkComplete())
          continue;
        auto* state = request->ExistingSamplingState(*batched_sampler_);
        if (!state)
          continue;
        if (std::find(checkpointed_sampler_states_.begin(),
                      checkpointed_sampler_states_.end(),
                      state) == checkpointed_sampler_states_.end()) {
          checkpointed_sampler_states_.push_back(state);
        }
      }
      if (!checkpointed_sampler_states_.empty()) {
        batched_sampler_->SaveStateForTransaction(checkpointed_sampler_states_);
        sampler_checkpoint_active_ = true;
      }
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

  // Strictly after every Request and sampler checkpoint taken in BeginTransaction() and strictly
  // before the first host or device RNG consumer below, so a rolled back step restores both streams
  // and leaves the reseed pending for the retry. A device state is reseeded only when this step
  // checkpointed it; BeginTransaction() checkpoints every state a pending reseed targets.
  for (const auto& request : requests_) {
    if (!request->HasPendingTurnSeed() || !request->IsChunkComplete()) {
      continue;
    }
    const bool device_state_checkpointed =
        sampler_checkpoint_active_ && batched_sampler_ &&
        std::find(checkpointed_sampler_states_.begin(),
                  checkpointed_sampler_states_.end(),
                  request->ExistingSamplingState(*batched_sampler_)) !=
            checkpointed_sampler_states_.end();
    request->ApplyPendingSeedForTransaction(batched_sampler_,
                                            device_state_checkpointed);
  }

  auto verify_rows = ProcessLogits();
  std::vector<std::vector<int32_t>> selected_tokens;
  std::vector<size_t> confirmed_draft_counts;
  std::vector<std::vector<std::mt19937>> rng_checkpoints;
  auto logits = SelectSampledRows(verify_rows, selected_tokens, confirmed_draft_counts,
                                  rng_checkpoints);
  const bool guidance_applied = TryApplyBatchedGuidanceMasks(logits);
  results.assign(requests_.size(), RequestStepResult{});
  std::vector<bool> sampled_by_batched_sampler(requests_.size(), false);
  if (transaction_uses_batched_sampler_) {
    sampling_plan_->logits.clear();
    const size_t original_sampling_count = sampling_plan_->requests.size();
    size_t source_sampling_index = 0;
    size_t active_sampling_count = 0;
    for (; source_sampling_index < original_sampling_count;
         ++source_sampling_index) {
      const size_t i = sampling_plan_->result_indices[source_sampling_index];
      if (i >= requests_.size() ||
          sampling_plan_->requests[source_sampling_index] != requests_[i].get() ||
          !requests_[i]->IsChunkComplete()) {
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
          sampling_plan_->result_indices[active_sampling_count] = i;
        }
        sampling_plan_->logits.push_back(logits[i]);
        requests_[i]->PrepareGenerationForTransaction(logits[i], guidance_applied);
        sampled_by_batched_sampler[i] = true;
        ++active_sampling_count;
      }
    }
    sampling_plan_->requests.resize(active_sampling_count);
    sampling_plan_->result_indices.resize(active_sampling_count);
    sampling_plan_->params.resize(active_sampling_count);
    sampling_plan_->states.resize(active_sampling_count);

    if (active_sampling_count != 0) {
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
        if (sampled_by_batched_sampler[i]) {
          results[i] = requests_[i]->StageGenerationForTransaction(plan.requests[i]);
        }
      }
    }
  }

  size_t sampled_request_count = 0;
  size_t max_selected_tokens = 0;
  for (const auto& tokens : selected_tokens) {
    if (!tokens.empty()) {
      ++sampled_request_count;
      max_selected_tokens = std::max(max_selected_tokens, tokens.size());
    }
  }
  if (sampled_request_count != 0) {
    bool can_batch_commits = true;
    for (size_t i = 0; i < requests_.size(); ++i) {
      if (!selected_tokens[i].empty() && !requests_[i]->SupportsBatchedSampling()) {
        can_batch_commits = false;
        break;
      }
    }
    DeviceSpan<int32_t> committed_tokens;
    if (can_batch_commits)
      committed_tokens = model_->p_device_->Allocate<int32_t>(sampled_request_count);

    const auto commit_sampled_stage = [&](size_t request_index, size_t stage) {
      auto& request = requests_[request_index];
      // The plan's original sequence_length_before, plus accepted_draft_count_ (prior stages
      // only -- never this one), always equals CurrentSequenceLength() just before this stage's
      // own append. Every earlier stage advanced Search and accepted_draft_count_ by exactly one,
      // so the unmodified plan works for every stage.
      const auto& stage_plan = plan.requests[request_index];
      const auto stage_result = request->StageGenerationForTransaction(stage_plan);
      const bool is_natural_last_stage =
          stage + 1 == selected_tokens[request_index].size();
      const bool stopped =
          stage_result.finish_reason == GenerationFinishReason::StopString;
      if (stopped || is_natural_last_stage) {
        if (stopped && !is_natural_last_stage &&
            (request_index >= rng_checkpoints.size() ||
             stage >= rng_checkpoints[request_index].size())) {
          throw std::logic_error(
              "A stop-truncated sampled round is missing its RNG checkpoint.");
        }
        auto final_result = stage_result;
        // A target-confirmed final draft contributes to both the accepted and evaluated counts.
        // A final stage within the proposed range but beyond the confirmed prefix is evaluated
        // but not accepted. A trailing bonus token was never proposed and contributes to neither.
        if (stage < confirmed_draft_counts[request_index]) {
          request->PromoteFinalStageAsAcceptedDraft(final_result);
        } else if (stage < stage_plan.draft_token_count) {
          request->MarkFinalStageAsEvaluatedNonAcceptedDraft();
        }
        results[request_index] = final_result;
        if (stopped && !is_natural_last_stage) {
          // Later selected tokens are never committed. Restore the RNG checkpoint immediately
          // after the retained token so discarded samples do not affect future decoding.
          selected_tokens[request_index].resize(stage + 1);
          request->rng_ = rng_checkpoints[request_index][stage];
        }
      } else if (!stage_result.token_appended || stage_result.done) {
        throw std::logic_error("An accepted sampled draft unexpectedly ended the request.");
      } else {
        request->AppendAcceptedSampledToken(stage_result.token);
      }
    };

    for (size_t stage = 0; stage < max_selected_tokens; ++stage) {
      std::vector<size_t> active;
      active.reserve(sampled_request_count);
      for (size_t i = 0; i < selected_tokens.size(); ++i) {
        if (stage < selected_tokens[i].size())
          active.push_back(i);
      }
      // Every request that had selected_tokens this round has already finalized (a stop match or
      // its own natural end) by the time none remain active for a later stage: skip the batched
      // buffer round-trip entirely rather than transferring and synchronizing an empty span.
      if (active.empty()) {
        break;
      }

      if (can_batch_commits) {
        auto stage_tokens = committed_tokens.subspan(0, active.size());
        auto cpu_tokens = stage_tokens.CpuSpan();
        for (size_t row = 0; row < active.size(); ++row)
          cpu_tokens[row] = selected_tokens[active[row]][stage];
        stage_tokens.CopyCpuToDevice();
        for (size_t row = 0; row < active.size(); ++row) {
          auto& request = requests_[active[row]];
          if (!request->BindNextTokensSlot(stage_tokens.subspan(row, 1)))
            throw std::logic_error("Sampled draft commit lost batched-search support.");
          request->OnNextTokensSampled();
        }
        stage_tokens.CopyDeviceToCpu();
        for (size_t row = 0; row < active.size(); ++row) {
          commit_sampled_stage(active[row], stage);
        }
      } else {
        for (size_t request_index : active) {
          requests_[request_index]->search_->CommitToken(selected_tokens[request_index][stage]);
          commit_sampled_stage(request_index, stage);
        }
      }
    }
  }

  for (size_t i = 0; i < requests_.size(); ++i) {
    if (requests_[i]->DraftVerificationCompletedGeneration()) {
      results[i] =
          requests_[i]->StageDraftCompletionForTransaction();
    } else if (requests_[i]->IsChunkComplete() && selected_tokens[i].empty() &&
               !sampled_by_batched_sampler[i]) {
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
  checkpointed_sampler_states_.clear();

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
  checkpointed_sampler_states_.clear();
  transaction_uses_batched_sampler_ = false;
}

}  // namespace Generators
