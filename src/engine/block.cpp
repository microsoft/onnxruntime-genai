// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "block.h"

#include <numeric>
#include <algorithm>

namespace Generators {

Block::Block(size_t id, size_t slots, size_t block_size)
    : id_(id), size_(slots), capacity_(block_size) {}

size_t Block::Id() const {
  return id_;
}

size_t Block::Size() const {
  return size_;
}

bool Block::IsFull() const {
  return Size() == Capacity();
}

size_t Block::EmptySlots() const {
  return Capacity() - Size();
}

size_t Block::Capacity() const {
  return capacity_;
}

void Block::AddSlot() {
  if (IsFull()) {
    throw std::runtime_error("Cannot add a slot. The block is full.");
  }

  size_++;
}

std::vector<size_t> Block::SlotIds() const {
  std::vector<size_t> slot_ids(Size(), 0);
  std::iota(slot_ids.begin(), slot_ids.end(), Id() * Capacity());
  return slot_ids;
}

BlockPool::BlockPool(size_t block_size, size_t num_blocks)
    : block_size_(block_size), capacity_(num_blocks) {}

std::vector<std::shared_ptr<Block>> BlockPool::AllocateBlocks(size_t num_slots, bool mark_slots_used) {
  const auto allocate_block = [this](size_t slots) {
    for (size_t i = 0; i < Capacity(); ++i) {
      if (blocks_[i] == nullptr) {
        blocks_[i] = std::make_shared<Block>(i, slots, block_size_);
        return blocks_[i];
      }
    }
    return std::shared_ptr<Block>();
  };

  if (BlocksNeeded(num_slots) > AvailableBlocks()) {
    throw std::runtime_error("Requested number of blocks " + std::to_string(BlocksNeeded(num_slots)) +
                             " for number of slots " + std::to_string(num_slots) +
                             " exceeds available blocks " + std::to_string(AvailableBlocks()) + ".");
  }

  std::vector<std::shared_ptr<Block>> allocated_blocks;
  for (size_t i = 0; i < num_slots; i += block_size_) {
    auto block = allocate_block(mark_slots_used ? std::min(block_size_, num_slots - i) : 0);
    if (!block) {
      throw std::runtime_error("Failed to allocate a block.");
    }
    allocated_blocks.push_back(block);
  }
  return allocated_blocks;
}

std::vector<std::shared_ptr<Block>> BlockPool::AllocateBlocks(size_t num_slots) {
  return AllocateBlocks(num_slots, /*mark_slots_used=*/true);
}

std::vector<std::shared_ptr<Block>> BlockPool::ReserveBlocks(size_t num_slots) {
  return AllocateBlocks(num_slots, /*mark_slots_used=*/false);
}

void BlockPool::Free(const std::vector<std::shared_ptr<Block>>& blocks) {
  // Validate every block before mutating any pool state so that an invalid request (a null block,
  // an out-of-range id, a block this pool does not currently own, or the same block listed twice)
  // is rejected without partially freeing the batch.
  std::vector<bool> seen(Capacity(), false);
  for (const auto& block : blocks) {
    if (!block) {
      throw std::runtime_error("Cannot free a null block.");
    }

    const size_t id = block->Id();
    if (id >= Capacity()) {
      throw std::runtime_error("Cannot free block with out-of-range id " + std::to_string(id) +
                               " for a pool with capacity " + std::to_string(Capacity()) + ".");
    }

    if (blocks_[id] != block) {
      throw std::runtime_error("Cannot free block with id " + std::to_string(id) +
                               " that is not currently allocated by this pool.");
    }

    if (seen[id]) {
      throw std::runtime_error("Cannot free block with id " + std::to_string(id) +
                               " more than once in the same call.");
    }
    seen[id] = true;
  }

  for (const auto& block : blocks) {
    blocks_[block->Id()].reset();
  }
}

size_t BlockPool::AvailableBlocks() const {
  return std::count_if(blocks_.begin(), blocks_.end(), [](const std::shared_ptr<Block>& block) { return block == nullptr; });
}

size_t BlockPool::Size() const {
  return Capacity() - AvailableBlocks();
}

size_t BlockPool::Capacity() const {
  return capacity_;
}

size_t BlockPool::BlockSize() const {
  return block_size_;
}

size_t BlockPool::BlocksNeeded(size_t num_slots) {
  return (num_slots + block_size_ - 1) / block_size_;
}

}  // namespace Generators
