// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "block.h"

#include <algorithm>
#include <exception>
#include <numeric>
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
    : block_size_(block_size),
      capacity_(num_blocks),
      blocks_(num_blocks),
      validation_marks_(num_blocks),
      validation_counts_(num_blocks) {}

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
  ValidateFree(blocks);
  FreeValidated(blocks);
}

void BlockPool::ValidateFree(
    std::span<const std::shared_ptr<Block>> blocks) const {
  ValidateOwnership(blocks, "free", /*require_references=*/true);
}

void BlockPool::AddRef(std::span<const std::shared_ptr<Block>> blocks) {
  const auto occurrences =
      ValidateOwnership(blocks, "add a reference to", /*require_references=*/false);
  for (const auto [id, count] : occurrences) {
    for (size_t i = 0; i < count; ++i) {
      auto& block = *blocks_[id];
      const bool was_reclaimable =
          block.RefCount() == 1 && block.reference_observer_cookie_;
      block.AddRef();
      if (was_reclaimable && reference_observer_) {
        reference_observer_->OnBlockBecameReferenced(
            block, block.reference_observer_cookie_);
      }
    }
  }
}

void BlockPool::AddRef(const std::shared_ptr<Block>& block) {
  if (!Owns(block)) {
    throw std::runtime_error("Cannot add a reference to a block this pool does not own.");
  }
  const bool was_reclaimable =
      block->RefCount() == 1 && block->reference_observer_cookie_;
  block->AddRef();
  if (was_reclaimable && reference_observer_) {
    reference_observer_->OnBlockBecameReferenced(
        *block, block->reference_observer_cookie_);
  }
}

void BlockPool::Release(const std::shared_ptr<Block>& block) {
  if (!Owns(block)) {
    throw std::runtime_error("Cannot release a block this pool does not own.");
  }
  const size_t remaining_references = block->ReleaseRef();
  if (remaining_references == 1 && block->reference_observer_cookie_ &&
      reference_observer_) {
    reference_observer_->OnBlockBecameReclaimable(
        *block, block->reference_observer_cookie_);
  } else if (remaining_references == 0) {
    block->ClearIdentity();
    blocks_[block->Id()].reset();
    ++mutation_generation_;
  }
}

void BlockPool::SetReferenceObserver(BlockReferenceObserver* observer) {
  if (reference_observer_ && observer && reference_observer_ != observer) {
    throw std::runtime_error("A block pool supports only one reference observer.");
  }
  reference_observer_ = observer;
}

void BlockPool::SetReferenceObserverCookie(
    const std::shared_ptr<Block>& block, void* cookie) {
  if (!Owns(block) || !cookie) {
    throw std::invalid_argument(
        "A reference observer cookie requires an owned block and non-null cookie.");
  }
  block->SetReferenceObserverCookie(cookie);
}

void BlockPool::ClearReferenceObserverCookie(
    const std::shared_ptr<Block>& block) noexcept {
  if (!Owns(block)) {
    std::terminate();
  }
  block->ClearReferenceObserverCookie();
}

std::vector<std::pair<size_t, size_t>> BlockPool::ValidateOwnership(
    std::span<const std::shared_ptr<Block>> blocks,
    const char* operation,
    bool require_references) const {
  std::vector<size_t> ids;
  ids.reserve(blocks.size());
  for (const auto& block : blocks) {
    if (!block) {
      throw std::runtime_error(std::string{"Cannot "} + operation + " a null block.");
    }

    const size_t id = block->Id();
    if (id >= Capacity()) {
      throw std::runtime_error(std::string{"Cannot "} + operation +
                               " block with out-of-range id " + std::to_string(id) +
                               " for a pool with capacity " + std::to_string(Capacity()) + ".");
    }

    if (blocks_[id] != block) {
      throw std::runtime_error(std::string{"Cannot "} + operation +
                               " block with id " + std::to_string(id) +
                               " that is not currently allocated by this pool.");
    }

    ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());

  std::vector<std::pair<size_t, size_t>> occurrences;
  occurrences.reserve(ids.size());
  for (const size_t id : ids) {
    if (occurrences.empty() || occurrences.back().first != id) {
      occurrences.emplace_back(id, 1);
    } else {
      ++occurrences.back().second;
    }
  }
  for (const auto [id, count] : occurrences) {
    if (require_references && blocks_[id]->RefCount() < count) {
      throw std::runtime_error(std::string{"Cannot "} + operation +
                               " block with id " + std::to_string(id) + " " +
                               std::to_string(count) + " times when it only holds " +
                               std::to_string(blocks_[id]->RefCount()) + " reference(s).");
    }
  }
  return occurrences;
}

void BlockPool::FreeValidated(
    std::span<const std::shared_ptr<Block>> blocks) noexcept {
  if (!CanFreeValidated(blocks)) {
    std::terminate();
  }
  for (const auto& block : blocks) {
    const size_t remaining_references = block->ReleaseRef();
    if (remaining_references == 1 && block->reference_observer_cookie_ &&
        reference_observer_) {
      reference_observer_->OnBlockBecameReclaimable(
          *block, block->reference_observer_cookie_);
    } else if (remaining_references == 0) {
      block->ClearIdentity();
      blocks_[block->Id()].reset();
    }
  }
  if (!blocks.empty()) {
    ++mutation_generation_;
  }
}

bool BlockPool::CanFreeValidated(
    std::span<const std::shared_ptr<Block>> blocks) const noexcept {
  ++validation_epoch_;
  if (validation_epoch_ == 0) {
    std::fill(validation_marks_.begin(), validation_marks_.end(), 0);
    ++validation_epoch_;
  }
  for (const auto& block : blocks) {
    if (!block || block->Id() >= blocks_.size() ||
        blocks_[block->Id()] != block) {
      return false;
    }
    const size_t id = block->Id();
    if (validation_marks_[id] != validation_epoch_) {
      validation_marks_[id] = validation_epoch_;
      validation_counts_[id] = 0;
    }
    if (++validation_counts_[id] > block->RefCount()) {
      return false;
    }
  }
  return true;
}

void BlockPool::RollbackReservedBlocks(
    const std::vector<std::shared_ptr<Block>>& blocks) noexcept {
  bool released = false;
  for (const auto& block : blocks) {
    if (block && block->Id() < blocks_.size() &&
        blocks_[block->Id()] == block) {
      blocks_[block->Id()].reset();
      released = true;
    }
  }
  if (released) {
    ++mutation_generation_;
  }
}

std::vector<std::shared_ptr<Block>> BlockPool::OwnedBlocks() const {
  std::vector<std::shared_ptr<Block>> owned;
  owned.reserve(Size());
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

bool BlockPool::Owns(const std::shared_ptr<Block>& block) const {
  return block && block->Id() < Capacity() && blocks_[block->Id()] == block;
}

size_t BlockPool::BlocksNeeded(size_t num_slots) {
  return num_slots / block_size_ + (num_slots % block_size_ != 0);
}

}  // namespace Generators
