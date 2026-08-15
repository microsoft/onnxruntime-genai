// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <vector>

#include "../span.h"

/**
 * @file recompute_preemption_policy.h
 * @brief Pure victim-selection arithmetic for recompute preemption on the dynamic engine path.
 *
 * Recompute preemption suspends a resident request, returns its committed paged-cache blocks to the
 * pool, and rebuilds that request's KV later through the ordinary chunked-prefill path. The policy
 * here decides *whether* preempting helps and *which* residents to suspend; it deliberately takes
 * plain counters rather than a Request or a cache so it can be unit-tested without a model, a
 * device, or a block pool. The scheduler wraps it with request-derived and cache-derived values.
 */

namespace Generators {

/**
 * @brief Bounded knobs controlling how aggressively the dynamic scheduler may preempt residents.
 *
 * Preemption is off by default. Enabling it changes which requests hold KV blocks across steps, so
 * an engine keeps its previous wait-only behavior unless the model configuration asks for it.
 */
struct RecomputePreemptionSettings {
  bool enabled{};
  // Upper bound on victims suspended within one planning pass, which bounds the recompute work a
  // single step can create.
  size_t max_victims_per_step{1};
  // Upper bound on how many times one request may be suspended. Zero leaves it unbounded; a
  // positive value protects a repeatedly chosen victim from unbounded recompute amplification.
  size_t max_preemptions_per_request{};
  // Committed decode steps a resident must complete after being admitted before it can be
  // suspended again. This is the anti-churn quantum: without it two requests competing for the same
  // blocks can trade residency every step, each recomputing its whole sequence to produce one
  // token. Raising it damps that trade at the cost of admission latency for waiting requests.
  size_t min_decode_steps_before_preemption{8};
};

/**
 * @brief One resident request the scheduler could suspend, described by the counters the policy
 *        needs.
 *
 * `committed_blocks` is the number of physical blocks the request would return to the pool. It is
 * summed over every block table the cache holds for the request, so a cache that splits a request
 * across more than one table still reports a single reclaimable total.
 */
struct RecomputePreemptionCandidate {
  const void* request_id{};
  size_t committed_blocks{};
  size_t preemption_count{};
  // Decode steps this request has committed since it was last admitted. Only a committed
  // transaction advances it, so a resident never becomes a victim on the strength of work that has
  // not been committed.
  size_t decode_steps_since_admission{};
  // False when suspending the request would waste in-flight work or could not make progress: a
  // request that is still (re)building its KV owns blocks for a prefill it has not finished, so
  // discarding it would trade one stalled request for another without freeing durable capacity.
  bool eligible{};
};

struct RecomputePreemptionDecision {
  std::vector<size_t> victims;  // Indices into the candidate span, in the order they are suspended.
  size_t reclaimed_blocks{};

  bool Empty() const { return victims.empty(); }
};

/**
 * @brief Deterministic order in which eligible candidates are considered for suspension.
 *
 * The order is:
 *   1. Fewest previous preemptions first, so no request is suspended twice while a peer that has
 *      never been suspended is still resident. This bounds unfairness and is what keeps a single
 *      victim from being starved by repeated recomputation.
 *   2. Latest resident position first, preserving the accumulated work of the requests that have
 *      been resident longest.
 *
 * Candidate indices are unique, so the order is total and the result is reproducible.
 */
std::vector<size_t> RecomputeVictimOrder(
    std::span<const RecomputePreemptionCandidate> candidates);

/**
 * @brief Chooses the victims whose combined committed blocks satisfy `required_blocks`.
 *
 * Returns an empty decision -- meaning "do not preempt" -- when preemption is disabled, when
 * nothing is blocked, or when the eligible candidates within `max_victims_per_step` cannot cover
 * the demand. Preempting without covering the demand would discard committed KV without unblocking
 * anything, which is exactly the churn this policy exists to avoid.
 */
RecomputePreemptionDecision SelectRecomputeVictims(
    std::span<const RecomputePreemptionCandidate> candidates,
    size_t required_blocks,
    const RecomputePreemptionSettings& settings);

}  // namespace Generators
