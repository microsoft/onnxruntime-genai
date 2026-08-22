// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "../span.h"
#include "block.h"

namespace Generators {

struct PagedKeyValueCache;

class PagedCacheBlockTable {
 public:
  PagedCacheBlockTable() = default;
  explicit PagedCacheBlockTable(
      const void* request_id,
      size_t committed_slots,
      std::vector<std::shared_ptr<Block>> blocks = {},
      std::vector<std::shared_ptr<Block>> window_blocks = {})
      : request_id_{request_id},
        committed_slots_{committed_slots},
        blocks_{std::move(blocks)},
        window_blocks_{std::move(window_blocks)} {}
  PagedCacheBlockTable(const PagedCacheBlockTable&) = default;
  PagedCacheBlockTable(PagedCacheBlockTable&&) noexcept = default;
  PagedCacheBlockTable& operator=(const PagedCacheBlockTable& other);
  PagedCacheBlockTable& operator=(PagedCacheBlockTable&& other) noexcept;

  const void* RequestId() const { return request_id_; }
  size_t CommittedSlots() const { return committed_slots_; }
  const std::vector<std::shared_ptr<Block>>& Blocks() const { return blocks_; }
  const std::vector<std::shared_ptr<Block>>& WindowBlocks() const { return window_blocks_; }
  uint64_t MutationGeneration() const { return mutation_generation_; }

 private:
  friend class PagedCacheReservation;
  friend struct PagedKeyValueCache;

  const void* request_id_{};
  size_t committed_slots_{};
  std::vector<std::shared_ptr<Block>> blocks_;
  std::vector<std::shared_ptr<Block>> window_blocks_;
  uint64_t mutation_generation_{};
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
  uint64_t table_generation{};
  size_t target_slots{};
  size_t reserved_block_offset{};
  size_t reserved_block_count{};
  size_t reserved_window_block_offset{};
  size_t reserved_window_block_count{};
  size_t advance_block_offset{};
  size_t advance_block_count{};
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
  // part of committing expected to fail under the Engine's serialized transaction contract.
  void ValidateCommit() const;
  // Publishes a reservation that ValidateCommit has already accepted. For a single reservation it
  // moves preallocated blocks and tables into the committed cache and performs no fallible
  // allocation or device work. Before mutation it preflights every delta again using only
  // allocation-free state, ownership, generation, boundary, and occupancy checks. See the
  // implementation for why it is intentionally not marked noexcept.
  void CommitValidated();
  // Convenience wrapper preserving the original single-call contract: ValidateCommit then
  // CommitValidated.
  void Commit();
  void Release();

 private:
  struct ResidentTableSnapshot {
    const void* request_id{};
    size_t committed_slots{};
    uint64_t mutation_generation{};
    const std::shared_ptr<Block>* blocks_data{};
    size_t block_count{};
    const std::shared_ptr<Block>* window_blocks_data{};
    size_t window_block_count{};
  };

  const PagedCacheBlockTable* FindCommittedTable(const void* request_id) const;
  const PagedCacheReservationDelta& FindDelta(const void* request_id) const;
  void ValidateResidentTablesUnchanged() const;
  void ValidateAdvancePreconditions(const PagedCacheReservationDelta& delta,
                                    const PagedCacheBlockTable& table) const;
  void AdvanceCommittedSlots(PagedCacheBlockTable& table, size_t target_slots);

  BlockPool* block_pool_{};
  BlockPool* window_block_pool_{};
  size_t window_ring_blocks_{};
  std::vector<PagedCacheBlockTable>* committed_tables_{};
  std::vector<std::shared_ptr<Block>> reserved_blocks_;
  std::vector<std::shared_ptr<Block>> reserved_window_blocks_;
  std::vector<PagedCacheReservationDelta> deltas_;
  std::vector<PagedCacheBlockTable> new_tables_;
  std::vector<const Block*> advance_blocks_;
  std::vector<ResidentTableSnapshot> resident_table_snapshots_;
  uint64_t block_pool_generation_{};
  uint64_t window_block_pool_generation_{};
  PagedCacheReservationState state_{PagedCacheReservationState::Released};
};

}  // namespace Generators
