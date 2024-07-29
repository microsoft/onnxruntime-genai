// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <vector>

namespace Generators {

struct Block {
  Block(size_t id, size_t num_tokens, std::shared_ptr<Block> previous_block, size_t block_size);

  Block(const Block&) = delete;
  Block& operator=(const Block&) = delete;

  size_t Id() const;
  size_t Size() const;

  bool IsFull() const;
  size_t NumOccupiedSlots() const;
  size_t NumEmptySlots() const;

  void AddSlot();

  std::shared_ptr<Block> PreviosBlock() const;

  std::vector<size_t> SlotIds() const;

  size_t RefCount() const;
  void IncrementRefCount();
  void DecrementRefCount();

 private:
  size_t id_;
  size_t num_occupied_slots_;
  size_t ref_count_;
  std::shared_ptr<Block> previous_block_;
  size_t size_;
};

struct BlockAllocator {
  BlockAllocator(size_t block_size, size_t num_blocks);

  BlockAllocator(const BlockAllocator&) = delete;
  BlockAllocator& operator=(const BlockAllocator&) = delete;

  size_t NumFreeBlocks() const;
  size_t NumAllocatedBlocks() const;
  size_t NumBlocks() const;

  std::shared_ptr<Block> AllocateBlock(size_t num_slots, std::shared_ptr<Block> previous_block);
  std::vector<std::shared_ptr<Block>> AllocateBlocks(size_t num_slots);
  void Free(const std::vector<std::shared_ptr<Block>>& blocks);
  std::vector<std::shared_ptr<Block>> Fork(const std::vector<std::shared_ptr<Block>>& blocks);

  size_t NumBlocksNeeded(size_t num_slots);

 private:
  const size_t block_size_;
  const size_t num_blocks_;
  std::vector<std::shared_ptr<Block>> blocks_{num_blocks_, nullptr};
};

}  // namespace Generators