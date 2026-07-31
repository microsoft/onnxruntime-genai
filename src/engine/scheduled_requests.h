// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"

namespace Generators {

struct DecoderIO;

struct ScheduledRequests {
  ScheduledRequests(std::vector<std::shared_ptr<Request>> requests,
                    std::shared_ptr<Model> model);

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

  // Scratch buffer owned by the engine, big enough for one token per scheduled request. When
  // supplied, token selection for the whole batch can be sampled in a single call.
  void SetSharedNextTokens(DeviceSpan<int32_t> next_tokens) { shared_next_tokens_ = next_tokens; }

  void GenerateNextTokens();

 private:
  bool TryGenerateNextTokensBatched(std::vector<DeviceSpan<float>>& logits);

  std::vector<std::shared_ptr<Request>> requests_;
  std::shared_ptr<Model> model_;
  std::unique_ptr<DecoderIO> decoder_state_;
  std::shared_ptr<GeneratorParams> params_;
  DeviceSpan<int32_t> shared_next_tokens_;
};

}  // namespace Generators
