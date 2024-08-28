// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "block.h"

#include <numeric>
#include <algorithm>

namespace Generators {

Block::Block(size_t id, size_t num_tokens,
             std::shared_ptr<Block> previous_block, size_t block_size)
    : id_(id),
      num_occupied_slots_(num_tokens),
      ref_count_(0),
      previous_block_(previous_block),
      size_(block_size) {}

size_t Block::Id() const { return id_; }

size_t Block::Size() const { return size_; }

bool Block::IsFull() const { return NumOccupiedSlots() == Size(); }

size_t Block::NumEmptySlots() const { return Size() - NumOccupiedSlots(); }

size_t Block::NumOccupiedSlots() const { return num_occupied_slots_; }

void Block::AddSlot() {
  if (num_occupied_slots_ == Size()) {
    throw std::runtime_error("Cannot add a slot. Block is full.");
  }
  num_occupied_slots_++;
}

std::shared_ptr<Block> Block::PreviosBlock() const { return previous_block_; }

std::vector<size_t> Block::SlotIds() const {
  std::vector<size_t> slot_ids(NumOccupiedSlots(), 0);
  std::iota(slot_ids.begin(), slot_ids.end(), Id() * Size());
  return slot_ids;
}

size_t Block::RefCount() const { return ref_count_; }

void Block::IncrementRefCount() { ref_count_++; }

void Block::DecrementRefCount() { ref_count_--; }

BlockAllocator::BlockAllocator(size_t block_size, size_t num_blocks)
    : block_size_(block_size),
      num_blocks_(num_blocks),
      blocks_(num_blocks_, nullptr) {
  for (size_t i = 0; i < num_blocks_; i++) {
    free_blocks_[i] = std::make_shared<Block>(i, 0, nullptr, block_size_);
  }
}

std::shared_ptr<Block> BlockAllocator::AllocateBlock(
    size_t num_slots, std::shared_ptr<Block> previous_block) {
  if (num_slots > block_size_) {
    throw std::runtime_error(
        "Cannot allocate a block with more slots than the block size.");
  }

  for (size_t i = 0; i < num_blocks_; ++i) {
    if (blocks_[i] == nullptr) {
      blocks_[i] =
          std::make_shared<Block>(i, num_slots, previous_block, block_size_);
      return blocks_[i];
    }
  }

  throw std::runtime_error("No free blocks available.");
}
std::shared_ptr<Block> BlockAllocator::Allocate() {
  if (free_blocks_.empty()) {
    throw std::runtime_error("Out of memory! No free blocks available.");
  }
  auto block = free_blocks_.back();
  free_blocks_.pop_back();
  block->IncrementRefCount();
  return block;
}

void BlockAllocator::Free(const std::shared_ptr<Block>& block) {
  if (block->RefCount() == 0) {
    throw std::runtime_error("Block has already been freed.");
  }
  block->DecrementRefCount();
  if (block->RefCount() == 0) {
    free_blocks_.push_back(block);
  }
}

std::vector<std::shared_ptr<Block>> BlockAllocator::AllocateBlocks(
    size_t num_slots) {
  std::vector<std::shared_ptr<Block>> allocated_blocks;
  std::shared_ptr<Block> previous_block = nullptr;
  for (size_t i = 0; i < num_slots; i += block_size_) {
    previous_block =
        AllocateBlock(std::min(block_size_, num_slots - i), previous_block);
    allocated_blocks.push_back(previous_block);
  }
  return allocated_blocks;
}

void BlockAllocator::Free(const std::vector<std::shared_ptr<Block>>& blocks) {
  for (const auto& block : blocks) {
    block->DecrementRefCount();
    if (block->RefCount() == 0) {
      blocks_[block->Id()] = nullptr;
    }
  }
}



std::vector<std::shared_ptr<Block>> BlockAllocator::Fork(
    const std::vector<std::shared_ptr<Block>>& blocks) {
  for (const auto& block : blocks) {
    block->IncrementRefCount();
  }
  return blocks;
}

size_t BlockAllocator::NumFreeBlocks() const {
  return std::count_if(
      blocks_.begin(), blocks_.end(),
      [](const std::shared_ptr<Block>& block) { return block == nullptr; });
}

size_t BlockAllocator::GetNumFreeBlocks() const { return free_blocks_.size(); }

size_t BlockAllocator::NumAllocatedBlocks() const {
  return num_blocks_ - NumFreeBlocks();
}

size_t BlockAllocator::NumBlocks() const { return num_blocks_; }

size_t BlockAllocator::NumBlocksNeeded(size_t num_slots) {
  return (num_slots + block_size_ - 1) / block_size_;
}

}  // namespace Generators