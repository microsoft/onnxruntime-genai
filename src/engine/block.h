// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace Generators {

/*
 * BlockIdentity is the content address of a filled block.
 *
 * `hash` chains the identity of the block that precedes this one into the hash of this block's
 * tokens, so two blocks only carry the same identity when the entire token sequence from the start
 * of the prompt through the end of the block matches. A hash is never treated as proof of a match:
 * `tokens` is compared before a block is handed out, and `parent` is the exact identity this block
 * was computed behind rather than a hash of it, so no combination of collisions can splice a block
 * onto a prefix it was not computed behind.
 */
struct BlockIdentity {
  uint64_t hash{};
  std::shared_ptr<const BlockIdentity> parent;  // Null when this block starts the sequence.
  std::vector<int32_t> tokens;
};

/*
 * Block represents a contiguous set of slots in the paged key-value cache.
 * Each block has a fixed capacity (number of slots it can hold) and tracks
 * the number of currently used slots.
 *
 * A block is reference counted. Several owners -- the committed block tables of different requests,
 * an in-flight reservation, and the prefix cache -- can hold the same physical block, and the block
 * only returns to the pool once the last of them releases it.
 */
struct Block {
  Block(size_t id, size_t slots, size_t block_size);

  size_t Id() const;

  size_t Size() const;

  bool IsFull() const;

  size_t Capacity() const;

  size_t EmptySlots() const;

  void AddSlot();

  void AddSlots(size_t slots);

  std::vector<size_t> SlotIds() const;

  // Owners currently holding this block. Maintained by BlockPool.
  size_t RefCount() const { return ref_count_; }

  bool IsShared() const { return ref_count_ > 1; }

  // A block is only safe to share once it is full: a full block is never written again, so no owner
  // can diverge from another. A partially filled tail block stays private to its request.
  bool IsShareable() const { return IsFull() && HasIdentity(); }

  bool HasIdentity() const { return identity_ != nullptr; }

  // Throws when the block carries no identity.
  const BlockIdentity& Identity() const;

  // The identity object itself, which is what a chained lookup compares against so that a hash
  // collision can never splice a block onto a prefix it was not computed behind. Null when unset.
  const std::shared_ptr<const BlockIdentity>& IdentityPtr() const { return identity_; }

  void SetIdentity(std::shared_ptr<const BlockIdentity> identity);

  void ClearIdentity();

 private:
  friend struct BlockPool;

  void AddRef();
  // Returns the reference count remaining after the release.
  size_t ReleaseRef();

  size_t id_;
  size_t size_;
  size_t capacity_;
  size_t ref_count_{1};
  std::shared_ptr<const BlockIdentity> identity_;
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

  // Allocates enough blocks to hold `num_slots` and marks those slots used.
  std::vector<std::shared_ptr<Block>> AllocateBlocks(size_t num_slots);

  // Allocates enough blocks to hold `num_slots` but leaves every slot empty. Use this to
  // reserve capacity for tokens that have not been processed yet; the caller marks the slots
  // used via Block::AddSlot() as the tokens are actually written to the cache.
  std::vector<std::shared_ptr<Block>> ReserveBlocks(size_t num_slots);

  // True when this pool currently owns `block`, which is the precondition for AddRef and Release.
  bool Owns(const std::shared_ptr<Block>& block) const;

  // Adds one reference for every listed block so that an additional owner can hold them. Every
  // block must currently be owned by this pool; the batch is validated before any reference moves.
  void AddRef(const std::vector<std::shared_ptr<Block>>& blocks);

  // Single-block reference operations. They allocate nothing, so they are safe to call from cleanup
  // and teardown paths where a failed allocation would otherwise become a hard failure.
  void AddRef(const std::shared_ptr<Block>& block);
  void Release(const std::shared_ptr<Block>& block);

  // Releases one reference for every listed block, returning a block to the pool once its last
  // owner releases it. A block may appear more than once when the caller holds several references
  // to it, which happens when one reservation admits two requests that adopt the same prefix.
  void Free(const std::vector<std::shared_ptr<Block>>& blocks);

  size_t BlocksNeeded(size_t num_slots);

  // Blocks this pool currently owns, in ascending id order. Used to build cache snapshots.
  std::vector<std::shared_ptr<Block>> OwnedBlocks() const;

 private:
  std::vector<std::shared_ptr<Block>> AllocateBlocks(size_t num_slots, bool mark_slots_used);

  // Validates that every listed block is owned by this pool and, when releasing, that its reference
  // count can absorb the listed multiplicity. Returns the (block id, occurrences) pairs.
  std::vector<std::pair<size_t, size_t>> ValidateOwnership(
      const std::vector<std::shared_ptr<Block>>& blocks,
      const char* operation,
      bool require_references) const;

  const size_t block_size_;
  const size_t capacity_;
  std::vector<std::shared_ptr<Block>> blocks_{capacity_};
};

}  // namespace Generators
