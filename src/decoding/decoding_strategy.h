// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <memory>

namespace Generators {

struct Generator;
struct SpeculativeStats;
template <typename T>
struct DeviceSpan;

// Base interface for per-token generation dispatch. Chosen once at Generator
// construction based on the model type.
struct DecodingStrategy {
  virtual ~DecodingStrategy() = default;

  // Drives one user-visible "generate next token" step, committing exactly one
  // token to the search sequence per call. Speculative variants compute several
  // tokens per round internally but emit them one-per-call.
  virtual void Step(Generator& generator) = 0;

  // Default: no stats. Speculative strategies override.
  virtual SpeculativeStats GetStats() const;

  // Drop any per-round buffered state so a rewind/restart resumes cleanly.
  // Speculative strategy needs override to clear its pending-token buffer.
  virtual void Reset() {}

  // Called at the start of a mid-stream AppendTokens (continuous decoding) so a strategy can
  // reconcile any deferred/buffered per-round state with the committed sequence before the append
  // runs. Default: nothing to do. Speculative overrides to realign its two inner KV caches.
  virtual void PrepareForAppend(Generator& generator) { (void)generator; }

  // Allows a strategy to expose logits without changing its buffered state. Returns true when
  // logits were supplied; false lets Generator use the regular computed-logits path.
  virtual bool TryGetExternalLogits(Generator& generator, DeviceSpan<float>& logits) {
    (void)generator;
    (void)logits;
    return false;
  }

  // Called before externally replacing logits. Strategies may discard speculative lookahead and
  // realign model state with the committed sequence before the replacement is installed.
  virtual void PrepareForSetLogits(Generator& generator) { (void)generator; }
};

// Picks the right strategy after state_ and search_ are set up.
std::unique_ptr<DecodingStrategy> MakeDecodingStrategy(Generator& generator);

}  // namespace Generators
