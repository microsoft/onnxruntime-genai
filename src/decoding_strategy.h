// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <memory>

namespace Generators {

struct Generator;
struct SpeculativeStats;

// Base interface for per-token generation dispatch. Chosen once at Generator
// construction based on the model type.
struct DecodingStrategy {
  virtual ~DecodingStrategy() = default;

  // Drives one user-visible "generate next token" step. For standard decoding
  // this samples one token; for speculative variants it may commit multiple
  // tokens to the search sequence in a single call.
  virtual void Step(Generator& generator) = 0;

  // Default: no stats. Speculative strategies override.
  virtual SpeculativeStats GetStats() const;
};

// Picks the right strategy after state_ and search_ are set up.
std::unique_ptr<DecodingStrategy> MakeDecodingStrategy(Generator& generator);

}  // namespace Generators
