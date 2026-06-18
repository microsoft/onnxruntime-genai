// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "speculative_decoding_strategy.h"

namespace Generators {

struct SpeculativeDecodingState;

// BaseSpeculativeStrategy
// Draft decoder proposes, the target decoder verifies. Greedy mode uses argmax-match; sampling mode
// samples draft from its distribution q and accepts with u < min(1, p_t/p_d).
// All counters and the propose -> verify -> commit -> re-anchor skeleton live in the base.
struct BaseSpeculativeStrategy final : SpeculativeDecodingStrategy {
  explicit BaseSpeculativeStrategy(Generator& g);

 protected:
  Proposal Propose(Generator& g, int K, int seed_length,
                   const SamplingConfig& sampling) override;
  void Advance(Generator& g,
               const Proposal& proposal,
               int n_direct,
               int32_t final_token,
               int seed_length) override;

 private:
  SpeculativeDecodingState& spec_state_;
};

}  // namespace Generators
