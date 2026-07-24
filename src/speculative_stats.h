// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstddef>

namespace Generators {

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
  size_t formula_supported{};
  float total_draft_ms{};
  float total_target_ms{};
  float total_reconciliation_ms{};
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
