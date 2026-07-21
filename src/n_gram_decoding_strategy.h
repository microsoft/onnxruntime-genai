// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <stdexcept>

#include "n_gram_lookup.h"
#include "speculative_decoding_strategy.h"

namespace Generators {

struct GeneratorParams;
struct Model;

struct NGramDecodingCapabilities {
  int batch_size{1};
  int num_beams{1};
  int num_return_sequences{1};
  bool uses_guidance{};
  bool uses_draft_model{};
  bool is_plain_decoder_only_text{true};
  bool uses_sliding_kv_cache{};
  bool uses_hybrid_state{};
  bool uses_model_managed_state{};
  bool has_pruned_logits{};
};

inline void ValidateNGramDecodingCapabilities(
    const NGramDecodingCapabilities& capabilities) {
  if (capabilities.uses_draft_model)
    throw std::runtime_error(
        "N-gram decoding cannot be combined with draft-model speculative decoding.");
  if (capabilities.batch_size != 1)
    throw std::runtime_error("N-gram decoding requires batch_size=1 in this release.");
  if (capabilities.num_beams != 1)
    throw std::runtime_error("N-gram decoding does not support beam search in this release.");
  if (capabilities.num_return_sequences != 1)
    throw std::runtime_error(
        "N-gram decoding requires num_return_sequences=1 in this release.");
  if (capabilities.uses_guidance)
    throw std::runtime_error("N-gram decoding does not support guidance in this release.");
  if (!capabilities.is_plain_decoder_only_text)
    throw std::runtime_error(
        "N-gram decoding requires a plain decoder-only text model in this release.");
  if (capabilities.uses_sliding_kv_cache || capabilities.uses_hybrid_state ||
      capabilities.uses_model_managed_state)
    throw std::runtime_error(
        "N-gram decoding requires a rewindable KV cache; sliding-window, hybrid, and "
        "model-managed state are not supported in this release.");
  if (capabilities.has_pruned_logits)
    throw std::runtime_error(
        "N-gram decoding requires target logits for every verified token; pruned last-token-only "
        "logits are not supported in this release.");
}

void ValidateNGramDecoding(const Model& model, const GeneratorParams& params);

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
