// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "request_status.h"

/**
 * @file engine_invariants.h
 * @brief Immutable snapshots of the continuous-batching Engine's Request and paged-cache state,
 *        plus a central invariant validator that operates purely on those snapshots.
 *
 * These value types are read-only records: they copy out the state needed to reason about
 * correctness and never expose mutable pointers into the live Request or cache. Defining the
 * invariant formulas here once keeps them in a single place instead of being restated wherever the
 * state is inspected; the same snapshots and checks back runtime and telemetry consumers as the
 * Engine grows, and the Engine's correctness tests.
 *
 * This header deliberately depends only on lightweight standard types and RequestStatus so the
 * validator remains model- and device-independent and can be exercised without a real model.
 */

namespace Generators {

// Immutable view of a single Request's progress counters. Produced by Request::Snapshot().
struct RequestStateSnapshot {
  // Stable opaque identity of the Request (its address), used only to cross-reference this snapshot
  // with the paged-cache snapshot. Never dereferenced by the validator.
  const void* request_id{nullptr};
  RequestStatus status{RequestStatus::Unassigned};
  int64_t current_sequence_length{};    // Total tokens the Request's search currently holds.
  int64_t processed_sequence_length{};  // Tokens the model has already processed into the cache.
  int64_t seen_sequence_length{};       // Tokens already streamed to (seen by) the application.
  int64_t unprocessed_token_count{};    // Tokens produced/appended but not yet processed.
  bool is_prefill{};
};

// Immutable view of one Request's block ownership within the paged cache.
struct RequestBlockSnapshot {
  const void* request_id{nullptr};
  std::vector<size_t> block_ids;  // Physical block ids owned by this Request, in block-table order.
  size_t used_slots{};            // Slots already written (committed) across the Request's blocks.
  size_t empty_slots{};           // Slots reserved for the Request but not yet written.
};

// Immutable view of the paged cache's block accounting. Produced by PagedKeyValueCache::Snapshot().
struct PagedCacheSnapshot {
  size_t block_size{};
  size_t total_blocks{};
  size_t free_blocks{};          // Blocks currently owned by no Request.
  size_t block_table_columns{};  // Padded block-table width the model sees, or 0 when unused.
  std::vector<RequestBlockSnapshot> requests;

  // Blocks currently owned by some Request (sum of per-Request block counts).
  size_t AllocatedBlocks() const;
};

// A single invariant violation, carrying a human-readable description of what was inconsistent.
struct InvariantViolation {
  std::string message;
};

// Pure invariant checks over the paged-cache snapshot. Returns every violation found; an empty
// result means the snapshot satisfies all cache invariants.
std::vector<InvariantViolation> ValidateCacheInvariants(const PagedCacheSnapshot& cache);

// Pure invariant checks over a single Request snapshot.
std::vector<InvariantViolation> ValidateRequestInvariants(const RequestStateSnapshot& request);

// Cross-cutting checks that need both a Request's progress and its cache ownership, in addition to
// the per-snapshot checks above.
std::vector<InvariantViolation> ValidateInvariants(const PagedCacheSnapshot& cache,
                                                   const std::vector<RequestStateSnapshot>& requests);

// Throwing convenience wrapper. Runs ValidateInvariants and throws std::runtime_error listing every
// violation if the snapshots are inconsistent, so callers can fail fast with the full list.
void ThrowIfInvariantsViolated(const PagedCacheSnapshot& cache,
                               const std::vector<RequestStateSnapshot>& requests);

}  // namespace Generators
