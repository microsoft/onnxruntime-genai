// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "../span.h"

namespace Generators {

class PagedCacheBlockTable;
class PagedCacheReservation;
class PrefixCache;
struct BlockCopier;
struct PagedKeyValueCache;
struct BlockPool;
struct Block;

struct BlockReferenceObserver {
  virtual ~BlockReferenceObserver() = default;
  virtual void OnBlockBecameReferenced(Block& block, void* cookie) noexcept = 0;
  virtual void OnBlockBecameReclaimable(Block& block, void* cookie) noexcept = 0;
};

struct BlockIdentity {
  uint64_t hash{};
  std::shared_ptr<const BlockIdentity> parent;
  std::vector<int32_t> tokens;
};

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

  size_t RefCount() const { return ref_count_; }
  bool IsShared() const { return ref_count_ > 1; }
  bool IsShareable() const { return IsFull() && HasIdentity(); }
  bool HasIdentity() const { return identity_ != nullptr; }
  const BlockIdentity& Identity() const;
  const std::shared_ptr<const BlockIdentity>& IdentityPtr() const { return identity_; }
  void SetIdentity(std::shared_ptr<const BlockIdentity> identity);
  void ClearIdentity();

 private:
  friend class PagedCacheBlockTable;
  friend class PagedCacheReservation;
  friend class PrefixCache;
  friend bool MakeTailBlockExclusive(PagedCacheBlockTable&, size_t, BlockPool&, BlockCopier&);
  friend struct PagedKeyValueCache;
  friend struct BlockPool;
  void AddSlot();
  void AddSlots(size_t slots);
  void AddRef();
  size_t ReleaseRef();
  void SetReferenceObserverCookie(void* cookie) noexcept { reference_observer_cookie_ = cookie; }
  void ClearReferenceObserverCookie() noexcept { reference_observer_cookie_ = nullptr; }

  size_t id_;
  size_t size_;
  size_t capacity_;
  size_t ref_count_{1};
  std::shared_ptr<const BlockIdentity> identity_;
  void* reference_observer_cookie_{};
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

  void AddRef(std::span<const std::shared_ptr<Block>> blocks);
  void AddRef(const std::shared_ptr<Block>& block);
  void Release(const std::shared_ptr<Block>& block);
  void SetReferenceObserver(BlockReferenceObserver* observer);

  void Free(const std::vector<std::shared_ptr<Block>>& blocks);
  void ValidateFree(std::span<const std::shared_ptr<Block>> blocks) const;
  // Allocation-free publication for an unchanged span accepted by ValidateFree(). A guard failure
  // is an impossible publication invariant violation and terminates rather than orphaning blocks.
  void FreeValidated(std::span<const std::shared_ptr<Block>> blocks) noexcept;
  bool CanFreeValidated(std::span<const std::shared_ptr<Block>> blocks) const noexcept;

  size_t BlocksNeeded(size_t num_slots);
  std::vector<std::shared_ptr<Block>> OwnedBlocks() const;

 private:
  friend class PagedCacheReservation;
  friend struct PagedKeyValueCache;
  friend bool MakeTailBlockExclusive(PagedCacheBlockTable&, size_t, BlockPool&, BlockCopier&);

  std::vector<std::shared_ptr<Block>> AllocateBlocks(size_t num_slots, bool mark_slots_used);
  void RollbackReservedBlocks(const std::vector<std::shared_ptr<Block>>& blocks) noexcept;
  void RecordOccupancyMutation() noexcept { ++mutation_generation_; }
  std::vector<std::pair<size_t, size_t>> ValidateOwnership(
      std::span<const std::shared_ptr<Block>> blocks,
      const char* operation,
      bool require_references) const;

  const size_t block_size_;
  const size_t capacity_;
  std::vector<std::shared_ptr<Block>> blocks_;
  // Preallocated scratch for allocation-free duplicate detection in validated publication.
  // Validation mutates this scratch even through const methods and therefore relies on the
  // Engine's external serialization; it is not safe for concurrent inspection.
  mutable std::vector<uint64_t> validation_marks_;
  mutable std::vector<size_t> validation_counts_;
  mutable uint64_t validation_epoch_{};
  uint64_t mutation_generation_{};
  BlockReferenceObserver* reference_observer_{};
};

}  // namespace Generators
