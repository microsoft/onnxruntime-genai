// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <stdexcept>

#include "config.h"
#include "n_gram_lookup.h"
#include "speculative_decoding_strategy.h"

namespace Generators {

struct GeneratorParams;
struct Model;

struct NGramDecodingCapabilities {
  bool uses_guidance{};
  bool uses_draft_model{};
  bool is_plain_decoder_only_text{true};
  bool uses_sliding_kv_cache{};
  bool uses_hybrid_state{};
  bool uses_model_managed_state{};
  bool has_pruned_logits{};
};

inline void ValidateNGramDecodingCapabilities(
    const Config::Search& search,
    const NGramDecodingCapabilities& capabilities) {
  if (capabilities.uses_draft_model)
    throw std::runtime_error(
        "N-gram decoding cannot be combined with draft-model speculative decoding.");
  if (search.batch_size != 1)
    throw std::runtime_error("N-gram decoding requires batch_size=1 in this release.");
  if (search.num_beams != 1)
    throw std::runtime_error("N-gram decoding does not support beam search in this release.");
  if (search.num_return_sequences != 1)
    throw std::runtime_error(
        "N-gram decoding requires num_return_sequences=1 in this release.");
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
  void PopulateProposerStats(SpeculativeStats& stats) const override;

 private:
  void Sync(Generator& g);
  void RecordHistorySync(size_t token_count, float elapsed_ms);
  void RecordLookup(std::span<const int32_t> candidates,
                    size_t lookup_tokens_proposed,
                    float elapsed_ms);

  NGramLookup lookup_;
  bool chained_lookup_;
  size_t lookup_hits_{};
  size_t lookup_misses_{};
  size_t lookup_tokens_proposed_{};
  size_t chained_tokens_proposed_{};
  size_t grammar_candidate_rejections_{};
  size_t history_syncs_{};
  size_t history_tokens_synced_{};
  float total_history_sync_ms_{};
  float total_lookup_ms_{};
};

}  // namespace Generators
