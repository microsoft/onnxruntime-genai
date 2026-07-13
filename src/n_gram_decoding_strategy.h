// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "n_gram_lookup.h"
#include "speculative_decoding_strategy.h"

namespace Generators {

struct NGramDecodingStrategy final : SpeculativeDecodingStrategy {
  explicit NGramDecodingStrategy(Generator& g);

 protected:
  Proposal Propose(Generator& g, int K, int seed_length) override;
  void Advance(Generator& g,
               const Proposal& proposal,
               int n_direct,
               int32_t final_token,
               int seed_length) override;
  void ReconcileProposer(Generator& g,
                         int floor,
                         std::span<const int32_t> committed,
                         int committed_length,
                         bool record_stats) override;
  void FinalizeGuidanceProposer(Generator& g,
                                int seed_length,
                                int proposal_length,
                                std::span<const int32_t> committed) override;
  void ResetProposer() override;

 private:
  void Sync(Generator& g);

  NGramLookup lookup_;
};

}  // namespace Generators
