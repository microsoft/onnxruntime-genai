// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"

namespace Generators {

struct DecoderIO;

struct BatchedSamplingPlan {
  void Reserve(size_t capacity) {
    requests.reserve(capacity);
    logits.reserve(capacity);
    params.reserve(capacity);
    states.reserve(capacity);
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
};

struct ScheduledRequests {
  ScheduledRequests(std::vector<std::shared_ptr<Request>> requests,
                    std::shared_ptr<Model> model,
                    BatchedSampler* batched_sampler,
                    BatchedSamplingPlan* sampling_plan);

  std::unique_ptr<OrtRunOptions> RunOptions();

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

  void AddDecoderState(std::unique_ptr<DecoderIO> decoder_state);

  void GenerateNextTokens();

 private:
  bool TryGenerateNextTokensBatched(std::vector<DeviceSpan<float>>& logits);

  std::vector<std::shared_ptr<Request>> requests_;
  std::shared_ptr<Model> model_;
  std::unique_ptr<DecoderIO> decoder_state_;
  std::shared_ptr<GeneratorParams> params_;
  BatchedSampler* batched_sampler_{};
  BatchedSamplingPlan* sampling_plan_{};
};

}  // namespace Generators
