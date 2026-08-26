// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <cstddef>

namespace Generators {

inline constexpr size_t kSpeculativeAcceptanceLengthBins = 8;

// Separates draft work from output delivery and exposes the speedup formula terms.
// Target-dependent formula fields are zero without a baseline or when guidance is active.
struct SpeculativeStats {
  size_t rounds{};
  size_t completed_rounds{};
  size_t interrupted_rounds{};
  size_t active_rounds{};
  size_t draft_tokens_proposed{};
  size_t draft_tokens_evaluated{};
  size_t draft_tokens_accepted{};
  size_t correction_tokens{};
  size_t bonus_tokens{};
  size_t tokens_queued{};
  size_t tokens_emitted{};
  size_t tokens_discarded{};
  size_t tokens_buffered{};
  size_t draft_forward_passes{};
  size_t target_forward_passes{};
  size_t effective_k{};
  size_t adaptive_k_increases{};
  size_t adaptive_k_decreases{};
  size_t adaptive_k_observations{};
  size_t adaptive_k_probes{};
  size_t cooldown_entries{};
  size_t cooldown_steps{};
  size_t cooldown_remaining{};
  size_t standard_fallback_steps{};
  size_t full_accept_rounds{};
  size_t partial_accept_rounds{};
  size_t zero_accept_rounds{};
  size_t target_verify_forward_passes{};
  size_t target_reanchor_forward_passes{};
  size_t target_reconciliation_forward_passes{};
  size_t ngram_lookup_hits{};
  size_t ngram_lookup_misses{};
  size_t ngram_lookup_tokens_proposed{};
  size_t ngram_chained_tokens_proposed{};
  size_t ngram_grammar_candidate_rejections{};
  size_t ngram_history_syncs{};
  size_t ngram_history_tokens_synced{};
  size_t fixed_state_prefill_direct_rows{};
  size_t fixed_state_prefill_gathered_rows{};
  size_t fixed_state_decode_direct_rows{};
  size_t fixed_state_decode_gathered_rows{};
  size_t fixed_state_speculative_direct_rows{};
  size_t fixed_state_speculative_gathered_rows{};
  size_t fixed_state_full_state_bytes_avoided{};
  size_t fixed_state_replay_descriptor_count{};
  size_t fixed_state_replayed_transition_count{};
  size_t fixed_state_noncontiguous_slot_fallbacks{};
  size_t fixed_state_mixed_active_bank_fallbacks{};
  std::array<size_t, kSpeculativeAcceptanceLengthBins> acceptance_length_histogram{};
  size_t formula_supported{};
  float total_draft_ms{};
  float total_target_ms{};
  float total_reconciliation_ms{};
  float total_target_verify_ms{};
  float total_target_reanchor_ms{};
  float total_ngram_history_sync_ms{};
  float total_ngram_lookup_ms{};
  float avg_draft_ms_per_token{};
  float acceptance_rate{};
  float avg_draft_tokens_per_round{};
  float mean_emitted_tokens_per_round{};
  float expected_tokens_per_round{};
  float avg_target_ms_per_round{};
  float target_baseline_ms_per_token{};
  float target_overhead_ratio{};
  float estimated_speedup{};
  float observed_speedup{};
  float adaptive_k_throughput{};
};

}  // namespace Generators
