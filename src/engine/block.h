// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

namespace Generators {

struct Block {
  Block(size_t id, size_t slots, size_t block_size);

  size_t Id() const;

  size_t Size() const;

  bool IsFull() const;

  size_t Capacity() const;

  size_t EmptySlots() const;

  void AddSlot();

  std::vector<size_t> SlotIds() const;

 private:
  size_t id_;
  size_t size_;
  size_t capacity_;
};

struct BlockPool {
  BlockPool(size_t block_size, size_t num_blocks);

  size_t AvailableBlocks() const;

  size_t Size() const;

  size_t Capacity() const;

  std::vector<std::shared_ptr<Block>> AllocateBlocks(size_t num_slots);

  void Free(const std::vector<std::shared_ptr<Block>>& blocks);

  size_t BlocksNeeded(size_t num_slots);

 private:
  const size_t block_size_;
  const size_t capacity_;
  std::vector<std::shared_ptr<Block>> blocks_{capacity_};
};

}  // namespace Generators
