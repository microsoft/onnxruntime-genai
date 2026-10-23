// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <deque>

#include "decoding/decoding_strategy.h"
#include "decoding/speculative_stats.h"
#include "smartptrs.h"

namespace Generators {

struct Generator;
struct Qwen38GraphState;

// Drives the Qwen3.8-Flash CUDA-graph engine (model.type "qwen38_flash_graph"). One GenerateNextToken call commits exactly
// one token to the search; an MTP(1) cycle that accepts its draft yields two tokens, the second is served from `queue_`
// by the next call without touching the GPU. The cycle after the current one is already running on the GPU (look-ahead)
// while the caller consumes the tokens.
struct Qwen38GraphStrategy : DecodingStrategy {
  explicit Qwen38GraphStrategy(Generator& generator);

  void Step(Generator& generator) override;
  SpeculativeStats GetStats() const override;
  void Reset() override;
  void PrepareForAppend(Generator& generator) override;
  void PrepareForSetLogits(Generator& generator) override;
  bool TryGetExternalLogits(Generator& generator, DeviceSpan<float>& logits) override;

 private:
  bool IsEos(const Generator& generator, int64_t token) const;

  Qwen38GraphState& state_;
  std::deque<int32_t> queue_;
  SpeculativeStats stats_;
  DeviceSpan<float> external_logits_;
  bool started_{};
};

}  // namespace Generators
