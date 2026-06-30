// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstddef>

namespace Generators {

// Instrumentation snapshot returned by Generator::GetSpeculativeStats().
// All fields are zero for non-speculative models.
//
// Definitions. A "round" is one propose -> verify -> accept/correct/bonus cycle. Each round
// proposes K draft tokens, accepts the first n_direct of them, and then commits exactly one more
// token: a correction token (on the first rejection) or a bonus token (when all K are accepted).
// So every round commits n_direct + 1 tokens and increments either correction_tokens or
// bonus_tokens (correction_tokens + bonus_tokens == rounds).
//
// Derived fields (computed in SpeculativeDecodingStrategy::GetStats; zero when undefined):
//   acceptance_rate        = draft_tokens_accepted / draft_tokens_proposed
//   avg_draft_ms_per_token = total_propose_ms / draft_tokens_proposed
//   avg_target_ms_per_token= total_target_ms  / draft_tokens_proposed
//   mean_accepted_tokens   = committed / rounds,
//                            where committed = draft_tokens_accepted + correction_tokens + bonus_tokens
//   effective_speedup      = committed * t_target / (total_propose_ms + total_target_ms + total_reanchor_ms),
//                            where t_target = total_reanchor_ms / reanchor_runs (a single-token target
//                            forward); 0 until a non-fold re-anchor has measured a target step.
struct SpeculativeStats {
  size_t rounds{};
  size_t draft_tokens_proposed{};
  size_t draft_tokens_accepted{};
  size_t correction_tokens{};
  size_t bonus_tokens{};
  float avg_draft_ms_per_token{};
  float avg_target_ms_per_token{};
  float acceptance_rate{};
  float mean_accepted_tokens{};
  float effective_speedup{};
};

}  // namespace Generators
