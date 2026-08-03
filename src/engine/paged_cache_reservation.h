// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "block.h"

namespace Generators {

struct PagedCacheBlockTable {
  const void* request_id{};
  size_t committed_slots{};
  std::vector<std::shared_ptr<Block>> blocks;
};

struct PagedCacheReservationRequest {
  const void* request_id{};
  size_t target_slots{};
  bool newly_admitted{};
};

struct PagedCacheReservationDelta {
  const void* request_id{};
  size_t committed_slots{};
  size_t target_slots{};
  size_t tail_slots_to_consume{};
  size_t reserved_block_offset{};
  size_t reserved_block_count{};
  bool newly_admitted{};
};

enum class PagedCacheReservationState {
  Reserved,
  Committed,
  Released,
};

class PagedCacheReservation {
 public:
  PagedCacheReservation(BlockPool& block_pool,
                        std::vector<PagedCacheBlockTable>& committed_tables,
                        std::span<const PagedCacheReservationRequest> requests);
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
  size_t RequiredBlockTableColumns() const;
  const std::vector<PagedCacheReservationDelta>& Deltas() const { return deltas_; }

  void FillBlockTable(std::span<const void* const> request_ids,
                      size_t columns,
                      std::span<int32_t> output) const;
  void Commit();
  void Release();

 private:
  const PagedCacheBlockTable* FindCommittedTable(const void* request_id) const;
  const PagedCacheReservationDelta& FindDelta(const void* request_id) const;
  void AdvanceCommittedSlots(PagedCacheBlockTable& table, size_t target_slots);

  BlockPool* block_pool_{};
  std::vector<PagedCacheBlockTable>* committed_tables_{};
  std::vector<std::shared_ptr<Block>> reserved_blocks_;
  std::vector<PagedCacheReservationDelta> deltas_;
  std::vector<PagedCacheBlockTable> new_tables_;
  PagedCacheReservationState state_{PagedCacheReservationState::Released};
};

}  // namespace Generators
