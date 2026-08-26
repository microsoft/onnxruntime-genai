// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
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
  AddSlots(1);
}

void Block::AddSlots(size_t slots) {
  if (slots > EmptySlots()) {
    throw std::runtime_error("Cannot add slots beyond the block capacity.");
  }

  size_ += slots;
}

std::vector<size_t> Block::SlotIds() const {
  std::vector<size_t> slot_ids(Size(), 0);
  std::iota(slot_ids.begin(), slot_ids.end(), Id() * Capacity());
  return slot_ids;
}

BlockPool::BlockPool(size_t block_size, size_t num_blocks)
    : block_size_(block_size), capacity_(num_blocks) {}

std::vector<std::shared_ptr<Block>> BlockPool::AllocateBlocks(size_t num_slots, bool mark_slots_used) {
  const size_t blocks_needed = BlocksNeeded(num_slots);
  if (blocks_needed > AvailableBlocks()) {
    throw std::runtime_error("Requested number of blocks " + std::to_string(blocks_needed) +
                             " for number of slots " + std::to_string(num_slots) +
                             " exceeds available blocks " + std::to_string(AvailableBlocks()) + ".");
  }

  std::vector<std::shared_ptr<Block>> allocated_blocks;
  allocated_blocks.reserve(blocks_needed);
  for (size_t id = 0; id < Capacity() && allocated_blocks.size() < blocks_needed; ++id) {
    if (blocks_[id] == nullptr) {
      const size_t allocated_slots = allocated_blocks.size() * block_size_;
      const size_t slots =
          mark_slots_used ? std::min(block_size_, num_slots - allocated_slots) : 0;
      allocated_blocks.push_back(std::make_shared<Block>(id, slots, block_size_));
    }
  }

  // Publish only after every allocation succeeds. Until this loop, an exception leaves the pool
  // unchanged and the local handles clean themselves up.
  for (const auto& block : allocated_blocks) {
    blocks_[block->Id()] = block;
  }
  if (!allocated_blocks.empty()) {
    ++mutation_generation_;
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
  // is rejected without partially freeing the batch. Work stays proportional to the batch size
  // rather than the pool capacity.
  std::vector<size_t> ids;
  ids.reserve(blocks.size());
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

    ids.push_back(id);
  }

  std::sort(ids.begin(), ids.end());
  const auto duplicate = std::adjacent_find(ids.begin(), ids.end());
  if (duplicate != ids.end()) {
    throw std::runtime_error("Cannot free block with id " + std::to_string(*duplicate) +
                             " more than once in the same call.");
  }

  for (const auto& block : blocks) {
    blocks_[block->Id()].reset();
  }
  if (!blocks.empty()) {
    ++mutation_generation_;
  }
}

void BlockPool::RollbackReservedBlocks(
    const std::vector<std::shared_ptr<Block>>& blocks) noexcept {
  for (const auto& block : blocks) {
    blocks_[block->Id()].reset();
  }
  if (!blocks.empty()) {
    ++mutation_generation_;
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

bool BlockPool::Owns(const std::shared_ptr<Block>& block) const {
  return block && block->Id() < Capacity() && blocks_[block->Id()] == block;
}

size_t BlockPool::BlocksNeeded(size_t num_slots) {
  return (num_slots + block_size_ - 1) / block_size_;
}

}  // namespace Generators
