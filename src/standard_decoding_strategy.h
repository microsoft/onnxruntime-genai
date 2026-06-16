// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "decoding_strategy.h"

namespace Generators {

struct Generator;

// StandardDecodingStrategy
// The classic single-token GenerateNextToken body (greedy / top-k / top-p).
// Commits exactly one token per Step.
struct StandardDecodingStrategy final : DecodingStrategy {
  void Step(Generator& g) override;
};

}  // namespace Generators