// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <random>
#include "decoding_strategy.h"

namespace Generators {

struct Generator;

// Shared single-token path used by standard decoding and speculative strategies when no proposal
// is available. sampling_rng keeps speculative fallback sampling on its RNG stream.
void RunStandardDecodingStep(Generator& g, std::mt19937* sampling_rng = nullptr);

// StandardDecodingStrategy
// The classic single-token GenerateNextToken body (greedy / top-k / top-p).
// Commits exactly one token per Step.
struct StandardDecodingStrategy final : DecodingStrategy {
  void Step(Generator& g) override;
};

}  // namespace Generators