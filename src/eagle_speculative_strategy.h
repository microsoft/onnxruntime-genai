// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "speculative_decoding_strategy.h"
#include "models/eagle.h"

namespace Generators {

// Greedy batch-1 EAGLE-3 strategy using AngelSlim's fixed 60/7/10 tree.
struct EagleSpeculativeStrategy final : SpeculativeDecodingStrategy {
   explicit EagleSpeculativeStrategy(Generator& generator);

  private:
   Proposal Propose(Generator& g, int K, int seed_length) override;
   void Advance(Generator& g, const Proposal& proposal, int n_direct,
                int32_t final_token, int seed_length) override;
   void ResetProposer() override;
   void ReconcileProposer(Generator& g, int floor,
                        std::span<const int32_t> committed,
                        int committed_length, bool record_stats) override;
   void FinalizeGuidanceProposer(
       Generator& g, int seed_length, int proposal_length,
       std::span<const int32_t> committed) override;

   void RunRound(Generator& g) override;
   RoundState::Phase FinalizeRound(Generator& g) override;

  int32_t ArgMax(std::span<const float> logits) const;
   int32_t PrepareDraft(Generator& g, bool& folded_root);

  EagleState& state_;
  EagleTree tree_;
  std::vector<size_t> selected_tree_indices_;
};

}  // namespace Generators
