// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "recompute_preemption_policy.h"

#include <algorithm>

namespace Generators {

std::vector<size_t> RecomputeVictimOrder(
    std::span<const RecomputePreemptionCandidate> candidates) {
  std::vector<size_t> order;
  order.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].eligible)
      order.push_back(i);
  }

  std::sort(order.begin(), order.end(),
            [&candidates](size_t left, size_t right) {
              if (candidates[left].preemption_count !=
                  candidates[right].preemption_count) {
                return candidates[left].preemption_count <
                       candidates[right].preemption_count;
              }
              return left > right;
            });
  return order;
}

RecomputePreemptionDecision SelectRecomputeVictims(
    std::span<const RecomputePreemptionCandidate> candidates,
    size_t required_blocks,
    const RecomputePreemptionSettings& settings) {
  RecomputePreemptionDecision decision;
  if (!settings.enabled || required_blocks == 0 ||
      settings.max_victims_per_step == 0) {
    return decision;
  }

  const auto order = RecomputeVictimOrder(candidates);
  for (const size_t index : order) {
    const auto& candidate = candidates[index];
    if (settings.max_preemptions_per_request != 0 &&
        candidate.preemption_count >= settings.max_preemptions_per_request) {
      continue;
    }
    // Every admission has to buy the request a minimum amount of committed decoding, otherwise two
    // requests competing for the same blocks would trade residency every step and recompute far
    // more than they produce.
    if (candidate.decode_steps_since_admission <
        settings.min_decode_steps_before_preemption) {
      continue;
    }
    // A candidate that owns nothing frees nothing; suspending it would restart its prefill for no
    // capacity gain.
    if (candidate.committed_blocks == 0) {
      continue;
    }

    decision.victims.push_back(index);
    decision.reclaimed_blocks += candidate.committed_blocks;
    if (decision.reclaimed_blocks >= required_blocks) {
      return decision;
    }
    if (decision.victims.size() == settings.max_victims_per_step) {
      break;
    }
  }

  // Suspending this set would not unblock the waiting request, so keep every resident intact.
  decision.victims.clear();
  decision.reclaimed_blocks = 0;
  return decision;
}

}  // namespace Generators
