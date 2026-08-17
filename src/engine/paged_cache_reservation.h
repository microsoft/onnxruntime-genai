// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "../span.h"
#include "block.h"
#include "prefix_cache.h"

namespace Generators {

struct PagedCacheBlockTable {
  const void* request_id{};
  size_t committed_slots{};
  std::vector<std::shared_ptr<Block>> blocks;
  // Leading blocks that already carry a content identity, so a later step only has to seal the
  // blocks that filled up since. Adopted prefix blocks arrive already sealed.
  size_t sealed_blocks{};
  // Identity the next block to seal chains from. Null while nothing is sealed.
  std::shared_ptr<const BlockIdentity> sealed_identity;
};

// Reclaims capacity a reservation needs from owners that can give it back on demand, which is what
// lets retained prefix blocks count as available without ever starving a live request.
struct BlockReclaimer {
  virtual size_t Reclaim(size_t blocks_needed) = 0;
  virtual ~BlockReclaimer() = default;
};

struct PagedCacheReservationRequest {
  const void* request_id{};
  size_t target_slots{};
  bool newly_admitted{};
  // Slots the reservation has to own blocks for, which is the whole sequence rather than this
  // step's target when a prefill is chunked. Clamped up to target_slots when left unset.
  size_t reserved_slots{};
  // Resident blocks a newly admitted request adopts for the leading tokens of its prompt. The
  // reservation takes a reference on each so a rolled back step cannot leak one, and seeds the
  // request's block table with them on commit. Null for every other request.
  const PrefixCacheMatch* prefix_match{};
};

struct PagedCacheReservationDelta {
  const void* request_id{};
  size_t committed_slots{};
  size_t target_slots{};
  size_t tail_slots_to_consume{};
  size_t reserved_block_offset{};
  size_t reserved_block_count{};
  size_t adopted_block_offset{};
  size_t adopted_block_count{};
  bool newly_admitted{};
};

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
                        BlockReclaimer* reclaimer = nullptr);
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
  size_t AdoptedBlockCount() const { return adopted_blocks_.size(); }
  const std::vector<std::shared_ptr<Block>>& AdoptedBlocks() const {
    return adopted_blocks_;
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
  std::vector<std::shared_ptr<Block>> adopted_blocks_;
  std::vector<PagedCacheReservationDelta> deltas_;
  std::vector<PagedCacheBlockTable> new_tables_;
  PagedCacheReservationState state_{PagedCacheReservationState::Released};
};

}  // namespace Generators
