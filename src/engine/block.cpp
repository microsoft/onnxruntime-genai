// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "block.h"

#include <numeric>
#include <algorithm>
#include <string>
#include <utility>

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

const BlockIdentity& Block::Identity() const {
  if (!identity_) {
    throw std::runtime_error("Block " + std::to_string(id_) + " carries no content identity.");
  }
  return *identity_;
}

void Block::SetIdentity(std::shared_ptr<const BlockIdentity> identity) {
  if (!identity) {
    throw std::runtime_error("Cannot give a block a null content identity.");
  }
  if (!IsFull()) {
    throw std::runtime_error("Only a full block can carry a content identity.");
  }
  if (identity->tokens.size() != Capacity()) {
    throw std::runtime_error("Block content identity does not cover every slot in the block.");
  }
  identity_ = std::move(identity);
}

void Block::ClearIdentity() {
  identity_.reset();
}

void Block::AddRef() {
  ++ref_count_;
}

size_t Block::ReleaseRef() {
  if (ref_count_ == 0) {
    throw std::runtime_error("Cannot release a block that has no remaining references.");
  }
  return --ref_count_;
}

BlockPool::BlockPool(size_t block_size, size_t num_blocks)
    : block_size_(block_size), capacity_(num_blocks) {}

std::vector<std::shared_ptr<Block>> BlockPool::AllocateBlocks(size_t num_slots, bool mark_slots_used) {
  if (BlocksNeeded(num_slots) > AvailableBlocks()) {
    throw std::runtime_error("Requested number of blocks " + std::to_string(BlocksNeeded(num_slots)) +
                             " for number of slots " + std::to_string(num_slots) +
                             " exceeds available blocks " + std::to_string(AvailableBlocks()) + ".");
  }

  const auto allocate_block = [this](size_t slots) {
    for (size_t i = 0; i < Capacity(); ++i) {
      if (blocks_[i] == nullptr) {
        blocks_[i] = std::make_shared<Block>(i, slots, block_size_);
        return blocks_[i];
      }
    }
    return std::shared_ptr<Block>();
  };

  std::vector<std::shared_ptr<Block>> allocated_blocks;
  allocated_blocks.reserve(BlocksNeeded(num_slots));
  try {
    for (size_t i = 0; i < num_slots; i += block_size_) {
      auto block = allocate_block(mark_slots_used ? std::min(block_size_, num_slots - i) : 0);
      if (!block) {
        throw std::runtime_error("Failed to allocate a block.");
      }
      allocated_blocks.push_back(block);
    }
  } catch (...) {
    // Allocation is all-or-nothing: a block installed before the failure has no owner to release
    // it, so it would sit in the pool forever.
    for (const auto& block : allocated_blocks) {
      blocks_[block->Id()].reset();
    }
    throw;
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
  // an out-of-range id, a block this pool does not currently own, or more releases than the block
  // has references) is rejected without partially freeing the batch. Work stays proportional to the
  // batch size rather than the pool capacity.
  const auto occurrences = ValidateOwnership(blocks, "free", /*require_references=*/true);

  for (const auto& [id, count] : occurrences) {
    size_t remaining = blocks_[id]->RefCount();
    for (size_t i = 0; i < count; ++i) {
      remaining = blocks_[id]->ReleaseRef();
    }
    if (remaining == 0) {
      blocks_[id].reset();
    }
  }
}

void BlockPool::AddRef(const std::vector<std::shared_ptr<Block>>& blocks) {
  // Several owners can take a reference to the same block in one call: one reservation admitting a
  // batch of requests that all adopt the same prefix does exactly that.
  const auto occurrences = ValidateOwnership(blocks, "add a reference to",
                                             /*require_references=*/false);

  for (const auto& [id, count] : occurrences) {
    for (size_t i = 0; i < count; ++i) {
      blocks_[id]->AddRef();
    }
  }
}

bool BlockPool::Owns(const std::shared_ptr<Block>& block) const {
  return block && block->Id() < Capacity() && blocks_[block->Id()] == block;
}

void BlockPool::AddRef(const std::shared_ptr<Block>& block) {
  if (!Owns(block)) {
    throw std::runtime_error("Cannot add a reference to a block this pool does not own.");
  }
  block->AddRef();
}

void BlockPool::Release(const std::shared_ptr<Block>& block) {
  if (!Owns(block)) {
    throw std::runtime_error("Cannot release a block this pool does not own.");
  }
  if (block->ReleaseRef() == 0) {
    blocks_[block->Id()].reset();
  }
}

std::vector<std::pair<size_t, size_t>> BlockPool::ValidateOwnership(
    const std::vector<std::shared_ptr<Block>>& blocks,
    const char* operation,
    bool require_references) const {
  std::vector<std::pair<size_t, size_t>> occurrences;
  occurrences.reserve(blocks.size());
  for (const auto& block : blocks) {
    if (!block) {
      throw std::runtime_error(std::string{"Cannot "} + operation + " a null block.");
    }

    const size_t id = block->Id();
    if (id >= Capacity()) {
      throw std::runtime_error(std::string{"Cannot "} + operation + " block with out-of-range id " +
                               std::to_string(id) + " for a pool with capacity " +
                               std::to_string(Capacity()) + ".");
    }

    if (blocks_[id] != block) {
      throw std::runtime_error(std::string{"Cannot "} + operation + " block with id " +
                               std::to_string(id) + " that is not currently allocated by this pool.");
    }

    const auto entry = std::find_if(occurrences.begin(), occurrences.end(),
                                    [id](const std::pair<size_t, size_t>& value) {
                                      return value.first == id;
                                    });
    if (entry == occurrences.end()) {
      occurrences.emplace_back(id, size_t{1});
    } else {
      ++entry->second;
    }
  }

  for (const auto& [id, count] : occurrences) {
    if (require_references && blocks_[id]->RefCount() < count) {
      throw std::runtime_error(std::string{"Cannot "} + operation + " block with id " +
                               std::to_string(id) + " " + std::to_string(count) +
                               " times when it only holds " + std::to_string(blocks_[id]->RefCount()) +
                               " reference(s).");
    }
  }

  return occurrences;
}

std::vector<std::shared_ptr<Block>> BlockPool::OwnedBlocks() const {
  std::vector<std::shared_ptr<Block>> owned;
  for (const auto& block : blocks_) {
    if (block) {
      owned.push_back(block);
    }
  }
  return owned;
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
