// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../span.h"

namespace Generators {

class PagedCacheBlockTable;
class PagedCacheReservation;
struct PagedKeyValueCache;
struct BlockPool;

/*
 * Block represents a contiguous set of slots in the paged key-value cache.
 * Each block has a fixed capacity (number of slots it can hold) and tracks
 * the number of currently used slots.
 */
struct Block {
  Block(size_t id, size_t slots, size_t block_size);
  Block(const Block&) = default;
  Block(Block&&) = default;
  Block& operator=(const Block&) = delete;
  Block& operator=(Block&&) = delete;

  size_t Id() const;

  size_t Size() const;

  bool IsFull() const;

  size_t Capacity() const;

  size_t EmptySlots() const;

  std::vector<size_t> SlotIds() const;

 private:
  friend class PagedCacheBlockTable;
  friend class PagedCacheReservation;
  friend struct PagedKeyValueCache;
  friend struct BlockPool;
  void AddSlot();
  void AddSlots(size_t slots);

  size_t id_;
  size_t size_;
  size_t capacity_;
};

/*
 * BlockPool manages a pool of blocks for the paged key-value cache.
 * It allows allocation and deallocation of blocks, and keeps track
 * of the total capacity and currently available blocks.
 */
struct BlockPool {
  BlockPool(size_t block_size, size_t num_blocks);

  size_t AvailableBlocks() const;

  size_t Size() const;

  size_t Capacity() const;

  size_t BlockSize() const;

  bool Owns(const std::shared_ptr<Block>& block) const;
  uint64_t MutationGeneration() const { return mutation_generation_; }

  // Allocates enough blocks to hold `num_slots` and marks those slots used.
  std::vector<std::shared_ptr<Block>> AllocateBlocks(size_t num_slots);

  // Allocates enough blocks to hold `num_slots` but leaves every slot empty. Use this to
  // reserve capacity for tokens that have not been processed yet; the caller marks the slots
  // used via Block::AddSlot() as the tokens are actually written to the cache.
  std::vector<std::shared_ptr<Block>> ReserveBlocks(size_t num_slots);

  void Free(const std::vector<std::shared_ptr<Block>>& blocks);
  void ValidateFree(std::span<const std::shared_ptr<Block>> blocks) const;
  // Allocation-free publication for an unchanged span accepted by ValidateFree(). Defensive
  // guards make direct misuse a no-op rather than an out-of-bounds access.
  void FreeValidated(std::span<const std::shared_ptr<Block>> blocks) noexcept;
  bool CanFreeValidated(std::span<const std::shared_ptr<Block>> blocks) const noexcept;

  size_t BlocksNeeded(size_t num_slots);

 private:
  friend class PagedCacheReservation;
  friend struct PagedKeyValueCache;

  std::vector<std::shared_ptr<Block>> AllocateBlocks(size_t num_slots, bool mark_slots_used);
  void RollbackReservedBlocks(const std::vector<std::shared_ptr<Block>>& blocks) noexcept;
  void RecordOccupancyMutation() noexcept { ++mutation_generation_; }

  const size_t block_size_;
  const size_t capacity_;
  std::vector<std::shared_ptr<Block>> blocks_;
  // Preallocated scratch for allocation-free duplicate detection in validated publication.
  mutable std::vector<uint64_t> validation_marks_;
  mutable uint64_t validation_epoch_{};
  uint64_t mutation_generation_{};
};

}  // namespace Generators
