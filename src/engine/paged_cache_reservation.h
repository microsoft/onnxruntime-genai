// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "../span.h"
#include "block.h"

namespace Generators {

struct PagedCacheBlockTable {
  const void* request_id{};
  size_t committed_slots{};
  std::vector<std::shared_ptr<Block>> blocks;
  std::vector<std::shared_ptr<Block>> window_blocks;
};

struct PagedCacheReservationRequest {
  const void* request_id{};
  size_t target_slots{};
  bool newly_admitted{};
  // Slots the reservation has to own blocks for, which is the whole sequence rather than this
  // step's target when a prefill is chunked. Clamped up to target_slots when left unset.
  size_t reserved_slots{};
};

struct PagedCacheReservationDelta {
  const void* request_id{};
  size_t committed_slots{};
  size_t target_slots{};
  size_t tail_slots_to_consume{};
  size_t reserved_block_offset{};
  size_t reserved_block_count{};
  size_t reserved_window_block_offset{};
  size_t reserved_window_block_count{};
  bool newly_admitted{};
};

void RemovePagedCacheBlockTable(BlockPool& block_pool,
                                BlockPool* window_block_pool,
                                std::vector<PagedCacheBlockTable>& committed_tables,
                                const void* request_id);

enum class PagedCacheReservationState {
  Reserved,
  Committed,
  Released,
};

class PagedCacheReservation {
 public:
  // Requests omitted from the reservation keep their committed tables unchanged.
  PagedCacheReservation(BlockPool& block_pool,
                        std::vector<PagedCacheBlockTable>& committed_tables,
                        std::span<const PagedCacheReservationRequest> requests,
                        BlockPool* window_block_pool = nullptr,
                        size_t window_ring_blocks = 0);
  PagedCacheReservation(PagedCacheReservation&& other) noexcept;
  PagedCacheReservation& operator=(PagedCacheReservation&&) = delete;
  PagedCacheReservation(const PagedCacheReservation&) = delete;
  PagedCacheReservation& operator=(const PagedCacheReservation&) = delete;
  ~PagedCacheReservation();

  PagedCacheReservationState State() const { return state_; }
  size_t ReservedBlockCount() const { return reserved_blocks_.size(); }
  const std::vector<std::shared_ptr<Block>>& ReservedBlocks() const {
    return reserved_blocks_;
  }
  const std::vector<std::shared_ptr<Block>>& ReservedWindowBlocks() const {
    return reserved_window_blocks_;
  }
  size_t RequiredBlockTableColumns() const;
  const std::vector<PagedCacheReservationDelta>& Deltas() const { return deltas_; }

  void FillBlockTable(std::span<const void* const> request_ids,
                      size_t columns,
                      std::span<int32_t> output) const;
  void FillWindowBlockTable(std::span<const void* const> request_ids,
                            size_t columns,
                            std::span<int32_t> output) const;
  // Commits only the first `kept_slots` of the `step_slots` this request's step planned, leaving
  // the rejected tail slots reserved-but-unwritten inside the blocks the request already owns. The
  // speculative counterpart of FixedStateReservation::CommitPrefix; both have to be called with the
  // same accepted prefix so the two states commit at one token boundary.
  void CommitPrefix(const void* request_id, size_t step_slots, size_t kept_slots);
  // Validates every precondition this reservation's own CommitValidated relies on, without
  // mutating any state. For a reservation that still satisfies the publish contract it is the only
  // part of committing that can throw; CommitValidated additionally throws if the reservation is no
  // longer Reserved (double publish), so an orchestrator must still guard CommitValidated too.
  //
  // Composite use: reservations produced by the same cache share one committed_tables_ vector
  // (PagedKeyValueCache::Reserve hands the same vector to every reservation). ValidateCommit only
  // checks this reservation's own share of that vector's headroom, so an orchestrator that validates
  // several reservations up front and then publishes them all must additionally guarantee, before
  // the publish phase, that (a) the reservations cover disjoint request ids and (b) committed_tables_
  // has capacity for the sum of their newly-admitted tables. Otherwise a later CommitValidated could
  // reallocate committed_tables_ and throw mid-publish.
  void ValidateCommit() const;
  // Publishes a reservation that ValidateCommit has already accepted. For a single reservation it
  // moves preallocated blocks and tables into the committed cache and performs no fallible allocation
  // or device work; its only guard is a state-only, allocation-free check that the reservation is
  // still Reserved (it throws rather than publish twice, which would be undefined behavior). Across a
  // composite publish it stays allocation-free only while the orchestrator honors the ValidateCommit
  // preconditions above. See the implementation for why it is intentionally not marked noexcept.
  void CommitValidated();
  // Convenience wrapper preserving the original single-call contract: ValidateCommit then
  // CommitValidated.
  void Commit();
  void Release();

 private:
  const PagedCacheBlockTable* FindCommittedTable(const void* request_id) const;
  const PagedCacheReservationDelta& FindDelta(const void* request_id) const;
  void AdvanceCommittedSlots(PagedCacheBlockTable& table, size_t target_slots);

  BlockPool* block_pool_{};
  BlockPool* window_block_pool_{};
  size_t window_ring_blocks_{};
  std::vector<PagedCacheBlockTable>* committed_tables_{};
  std::vector<std::shared_ptr<Block>> reserved_blocks_;
  std::vector<std::shared_ptr<Block>> reserved_window_blocks_;
  std::vector<PagedCacheReservationDelta> deltas_;
  std::vector<PagedCacheBlockTable> new_tables_;
  PagedCacheReservationState state_{PagedCacheReservationState::Released};
};

}  // namespace Generators
