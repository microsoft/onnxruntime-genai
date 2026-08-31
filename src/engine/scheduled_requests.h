// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "execution_context.h"
#include "request.h"

namespace Generators {

struct DecoderIO;

enum class BatchedGuidanceMaskStatus {
  NoEligibleGuidance,
  Ready,
  FallbackRequired,
};

BatchedGuidanceMaskStatus CollectBatchedGuidanceMasks(
    std::span<const std::shared_ptr<Request>> requests,
    size_t words_per_row,
    std::vector<uint32_t>& masks);

struct BatchedSamplingPlan {
  void Reserve(size_t capacity, size_t verification_capacity) {
    requests.reserve(capacity);
    logits.reserve(capacity);
    params.reserve(capacity);
    states.reserve(capacity);
    verification_tokens.reserve(verification_capacity);
  }

  void Clear() {
    requests.clear();
    logits.clear();
    params.clear();
    states.clear();
  }

  std::vector<Request*> requests;
  std::vector<DeviceSpan<float>> logits;
  std::vector<BatchedSamplingParams> params;
  std::vector<BatchedSamplerState*> states;
  std::vector<uint32_t> guidance_masks;
  DeviceSpan<uint32_t> guidance_device_masks;
  std::vector<int32_t> verification_tokens;
};

struct ScheduledRequests {
  ScheduledRequests(std::vector<std::shared_ptr<Request>> requests,
                    std::shared_ptr<Model> model,
                    BatchedSampler* batched_sampler,
                    BatchedSamplingPlan* sampling_plan);

  ScheduledRequests(const StepPlan& plan,
                    std::shared_ptr<Model> model,
                    BatchedSampler* batched_sampler,
                    BatchedSamplingPlan* sampling_plan);

  ExecutionContext& CreateExecutionContext();

  std::shared_ptr<GeneratorParams> Params();

  auto begin() const {
    return requests_.begin();
  }

  auto end() const {
    return requests_.end();
  }

  size_t size() const {
    return requests_.size();
  }

  explicit operator bool() const { return !requests_.empty(); };

  auto operator[](size_t idx) const {
    assert(idx < requests_.size());
    return requests_[idx];
  }

  const std::vector<std::shared_ptr<Request>>& Requests() const { return requests_; }

  void AddDecoderState(std::unique_ptr<DecoderIO> decoder_state);

  std::vector<DeviceSpan<float>> ProcessLogits();

  void GenerateNextTokens();
  void ScheduleGuidanceMasks() noexcept;
  void BeginTransaction();
  void GenerateNextTokensForTransaction(
      const StepPlan& plan,
      std::vector<RequestStepResult>& results);
  void RestoreStateForTransaction();
  void CommitStateForTransaction();

 private:
  bool PrepareBatchedSamplingPlan(bool require_transaction_support);
  bool TryGenerateNextTokensBatched(std::vector<DeviceSpan<float>>& logits, bool guidance_applied);
  bool TryApplyBatchedGuidanceMasks(std::vector<DeviceSpan<float>>& logits);
  // Verifies each drafted request's proposal against the target model's own rows, rewinds the
  // rejected tail, and returns the one row per request that the sampler must select from.
  std::vector<DeviceSpan<float>> SelectSampledRows(
      std::vector<DeviceSpan<float>>& verify_rows);

  std::vector<std::shared_ptr<Request>> requests_;
  // Drafts the transaction stages onto each request's sequence, in scheduled row order. Empty
  // outside a dynamic step plan.
  std::vector<size_t> draft_token_counts_;
  std::shared_ptr<Model> model_;
  std::unique_ptr<DecoderIO> decoder_state_;
  std::unique_ptr<ExecutionContext> execution_context_;
  std::shared_ptr<GeneratorParams> params_;
  BatchedSampler* batched_sampler_{};
  BatchedSamplingPlan* sampling_plan_{};
  size_t transaction_checkpoint_count_{};
  bool transaction_uses_batched_sampler_{};
  bool sampler_checkpoint_active_{};
};

}  // namespace Generators
