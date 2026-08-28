// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "paged_cache_reservation.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "window_ring.h"

namespace Generators {

namespace {

PagedCacheBlockTable* FindTable(std::vector<PagedCacheBlockTable>& tables, const void* request_id) {
  const auto it = std::find_if(tables.begin(), tables.end(),
                               [request_id](const PagedCacheBlockTable& table) {
                                 return table.RequestId() == request_id;
                               });
  return it == tables.end() ? nullptr : &*it;
}

void ValidateEmptyReservedBlocks(
    std::span<const std::shared_ptr<Block>> blocks,
    size_t block_size,
    std::string_view description) {
  for (const auto& block : blocks) {
    if (!block || block->Capacity() != block_size || block->Size() != 0) {
      throw std::logic_error(
          "Paged cache " + std::string{description} + " blocks are invalid.");
    }
  }
}

size_t CheckedBlockSlots(
    size_t block_count,
    size_t block_size,
    std::string_view description) {
  if (block_size != 0 &&
      block_count > std::numeric_limits<size_t>::max() / block_size) {
    throw std::overflow_error(
        "Paged cache " + std::string{description} + " block capacity overflow.");
  }
  return block_count * block_size;
}

size_t CheckedAdd(
    size_t left,
    size_t right,
    std::string_view description) {
  if (right > std::numeric_limits<size_t>::max() - left) {
    throw std::overflow_error(
        "Paged cache " + std::string{description} + " overflow.");
  }
  return left + right;
}

}  // namespace

PagedCacheBlockTable& PagedCacheBlockTable::operator=(
    const PagedCacheBlockTable& other) {
  if (this != &other) {
    PagedCacheBlockTable copy{other};
    *this = std::move(copy);
  }
  return *this;
}

PagedCacheBlockTable& PagedCacheBlockTable::operator=(
    PagedCacheBlockTable&& other) noexcept {
  if (this != &other) {
    const uint64_t next_generation = mutation_generation_ + 1;
    request_id_ = other.request_id_;
    committed_slots_ = other.committed_slots_;
    blocks_ = std::move(other.blocks_);
    window_blocks_ = std::move(other.window_blocks_);
    mutation_generation_ = next_generation;
  }
  return *this;
}

void RemovePagedCacheBlockTable(
    BlockPool& block_pool,
    BlockPool* window_block_pool,
    std::vector<PagedCacheBlockTable>& committed_tables,
    const void* request_id) {
  ValidateRemovePagedCacheBlockTable(
      block_pool, window_block_pool, committed_tables, request_id);
  RemoveValidatedPagedCacheBlockTable(
      block_pool, window_block_pool, committed_tables, request_id);
}

void ValidateRemovePagedCacheBlockTable(
    const BlockPool& block_pool,
    const BlockPool* window_block_pool,
    const std::vector<PagedCacheBlockTable>& committed_tables,
    const void* request_id) {
  const auto table = std::find_if(
      committed_tables.begin(), committed_tables.end(),
      [request_id](const PagedCacheBlockTable& candidate) {
        return candidate.RequestId() == request_id;
      });
  if (table == committed_tables.end()) {
    return;
  }

  block_pool.ValidateFree(table->Blocks());
  if (window_block_pool) {
    window_block_pool->ValidateFree(table->WindowBlocks());
  }
}

void RemoveValidatedPagedCacheBlockTable(
    BlockPool& block_pool,
    BlockPool* window_block_pool,
    std::vector<PagedCacheBlockTable>& committed_tables,
    const void* request_id) noexcept {
  const auto table = std::find_if(
      committed_tables.begin(), committed_tables.end(),
      [request_id](const PagedCacheBlockTable& candidate) {
        return candidate.RequestId() == request_id;
      });
  if (table == committed_tables.end()) {
    return;
  }

  if (!block_pool.CanFreeValidated(table->Blocks()) ||
      (window_block_pool &&
       !window_block_pool->CanFreeValidated(table->WindowBlocks()))) {
    std::terminate();
  }
  block_pool.FreeValidated(table->Blocks());
  if (window_block_pool) {
    window_block_pool->FreeValidated(table->WindowBlocks());
  }
  committed_tables.erase(table);
}

PagedCacheReservation::PagedCacheReservation(
    BlockPool& block_pool,
    std::vector<PagedCacheBlockTable>& committed_tables,
    std::span<const PagedCacheReservationRequest> requests,
    BlockPool* window_block_pool,
    size_t window_ring_blocks,
    RequestIndex* table_index)
    : block_pool_{&block_pool},
      window_block_pool_{window_block_pool},
      window_ring_blocks_{window_ring_blocks},
      committed_tables_{&committed_tables},
      table_index_{table_index},
      resident_table_index_{committed_tables.size()},
      delta_index_{requests.size()},
      delta_visit_generations_(requests.size()) {
  if ((window_block_pool_ == nullptr) != (window_ring_blocks_ == 0)) {
    throw std::runtime_error("Paged cache window reservation configuration is inconsistent.");
  }
  deltas_.reserve(requests.size());
  new_tables_.reserve(requests.size());
  committed_tables.reserve(CheckedAdd(
      committed_tables.size(), requests.size(), "table capacity"));
  if (table_index_ &&
      table_index_->Size() != committed_tables.size()) {
    throw std::logic_error(
        "Paged cache request index does not match committed tables.");
  }
  for (size_t index = 0; index < committed_tables.size(); ++index) {
    const auto request_id = committed_tables[index].request_id_;
    if (!request_id ||
        !resident_table_index_.Insert(request_id, index)) {
      throw std::logic_error(
          "Paged cache committed tables contain an invalid or duplicate request.");
    }
    if (table_index_) {
      const auto indexed = table_index_->Find(request_id);
      if (!indexed || *indexed != index) {
        throw std::logic_error(
            "Paged cache request index does not match committed tables.");
      }
    }
  }

  size_t reserved_block_count = 0;
  size_t reserved_window_block_count = 0;
  size_t advance_block_count = 0;
  std::unordered_set<const Block*> touched_committed_blocks;
  std::unordered_set<const Block*> resident_window_blocks;
  for (const auto& request : requests) {
    if (!request.request_id ||
        !delta_index_.Insert(request.request_id, deltas_.size())) {
      throw std::runtime_error("Paged cache reservation contains an invalid or duplicate request.");
    }

    auto* committed_table = FindCommittedTable(request.request_id);
    if (request.newly_admitted == (committed_table != nullptr)) {
      throw std::runtime_error("Paged cache reservation request membership does not match the committed cache.");
    }

    const size_t committed_slots = committed_table ? committed_table->committed_slots_ : 0;
    const size_t committed_blocks = committed_table ? committed_table->blocks_.size() : 0;
    const uint64_t table_generation =
        committed_table ? committed_table->mutation_generation_ : 0;
    if (request.target_slots < committed_slots) {
      throw std::runtime_error("Paged cache reservation cannot reduce committed slots.");
    }
    if (committed_table) {
      if (window_block_pool_) {
        if (committed_table->window_blocks_.size() != window_ring_blocks_) {
          throw std::logic_error(
              "Paged cache resident window ring has an invalid size.");
        }
        for (const auto& block : committed_table->window_blocks_) {
          if (!window_block_pool_->Owns(block) ||
              block->Capacity() != window_block_pool_->BlockSize()) {
            throw std::logic_error(
                "Paged cache resident window ring contains an invalid block.");
          }
          if (!resident_window_blocks.insert(block.get()).second) {
            throw std::logic_error(
                "Paged cache resident window rings cannot share a physical block.");
          }
        }
      } else if (!committed_table->window_blocks_.empty()) {
        throw std::logic_error(
            "Paged cache without a window pool cannot own window blocks.");
      }
    }

    const size_t committed_capacity = CheckedBlockSlots(
        committed_blocks, block_pool.BlockSize(), "committed");
    if (committed_slots > committed_capacity) {
      throw std::logic_error(
          "Paged cache committed boundary exceeds its block capacity.");
    }
    if (committed_table && request.target_slots > committed_slots) {
      const size_t block_size = block_pool.BlockSize();
      const size_t first_block = committed_slots / block_size;
      const size_t end_block = std::min(
          committed_blocks, block_pool.BlocksNeeded(request.target_slots));
      for (size_t block_index = first_block; block_index < end_block; ++block_index) {
        const auto& block = committed_table->blocks_[block_index];
        const size_t expected_size =
            block_index == first_block ? committed_slots % block_size : 0;
        if (!block_pool.Owns(block) || block->Capacity() != block_size ||
            block->Size() != expected_size) {
          throw std::logic_error(
              "Paged cache block occupancy is inconsistent with its committed boundary.");
        }
        if (!touched_committed_blocks.insert(block.get()).second) {
          throw std::logic_error(
              "Paged cache growth cannot share a physical block.");
        }
      }
    }
    // Blocks are taken for the whole sequence so that a chunked prefill cannot lose the rest of its
    // capacity to another request between steps, but only target_slots is committed below.
    const size_t reserved_slots = std::max(request.reserved_slots, request.target_slots);
    const size_t additional_slots =
        reserved_slots > committed_capacity ? reserved_slots - committed_capacity : 0;
    const size_t new_blocks = block_pool.BlocksNeeded(additional_slots);
    const size_t first_advance_block = committed_slots / block_pool.BlockSize();
    const size_t past_last_advance_block =
        request.target_slots / block_pool.BlockSize() +
        (request.target_slots % block_pool.BlockSize() != 0);
    const size_t request_advance_blocks =
        past_last_advance_block - first_advance_block;

    deltas_.push_back(PagedCacheReservationDelta{
        request.request_id,
        committed_slots,
        table_generation,
        request.target_slots,
        reserved_block_count,
        new_blocks,
        reserved_window_block_count,
        request.newly_admitted ? window_ring_blocks_ : 0,
        advance_block_count,
        request_advance_blocks,
        request.newly_admitted,
    });
    reserved_block_count = CheckedAdd(
        reserved_block_count, new_blocks, "reserved block count");
    advance_block_count = CheckedAdd(
        advance_block_count, request_advance_blocks,
        "advance block count");
    if (request.newly_admitted) {
      reserved_window_block_count = CheckedAdd(
          reserved_window_block_count, window_ring_blocks_,
          "reserved window block count");
    }

    if (committed_table) {
      committed_table->blocks_.reserve(CheckedAdd(
          committed_blocks, new_blocks, "committed block capacity"));
    } else {
      PagedCacheBlockTable table;
      table.request_id_ = request.request_id;
      table.blocks_.reserve(new_blocks);
      table.window_blocks_.reserve(window_ring_blocks_);
      new_tables_.push_back(std::move(table));
    }
  }

  resident_table_snapshots_.reserve(committed_tables.size());
  if (reserved_block_count > block_pool.AvailableBlocks()) {
    throw std::runtime_error("Not enough free blocks for the complete paged cache reservation.");
  }
  if (window_block_pool_ && reserved_window_block_count > window_block_pool_->AvailableBlocks()) {
    throw std::runtime_error("Not enough free window blocks for the complete paged cache reservation.");
  }

  advance_blocks_.reserve(advance_block_count);
  reserved_blocks_ = block_pool.ReserveBlocks(CheckedBlockSlots(
      reserved_block_count, block_pool.BlockSize(), "reserved"));
  try {
    if (window_block_pool_) {
      reserved_window_blocks_ = window_block_pool_->ReserveBlocks(CheckedBlockSlots(
          reserved_window_block_count, window_block_pool_->BlockSize(),
          "reserved window"));
    }
  } catch (...) {
    block_pool.RollbackReservedBlocks(reserved_blocks_);
    reserved_blocks_.clear();
    throw;
  }
  block_pool_generation_ = block_pool_->MutationGeneration();
  window_block_pool_generation_ =
      window_block_pool_ ? window_block_pool_->MutationGeneration() : 0;
  for (const auto& delta : deltas_) {
    const auto* table = FindCommittedTable(delta.request_id);
    const size_t committed_blocks = table ? table->blocks_.size() : 0;
    const size_t first_block = delta.committed_slots / block_pool_->BlockSize();
    for (size_t i = 0; i < delta.advance_block_count; ++i) {
      const size_t block_index = first_block + i;
      advance_blocks_.push_back(
          block_index < committed_blocks
              ? table->blocks_[block_index].get()
              : reserved_blocks_[delta.reserved_block_offset +
                                 block_index - committed_blocks]
                    .get());
    }
  }
  for (const auto& table : committed_tables) {
    resident_table_snapshots_.push_back(ResidentTableSnapshot{
        table.request_id_,
        table.committed_slots_,
        table.mutation_generation_,
        table.blocks_.data(),
        table.blocks_.size(),
        table.window_blocks_.data(),
        table.window_blocks_.size(),
    });
  }
  state_ = PagedCacheReservationState::Reserved;
}

PagedCacheReservation::PagedCacheReservation(PagedCacheReservation&& other) noexcept
    : block_pool_{std::exchange(other.block_pool_, nullptr)},
      window_block_pool_{std::exchange(other.window_block_pool_, nullptr)},
      window_ring_blocks_{std::exchange(other.window_ring_blocks_, 0)},
      committed_tables_{std::exchange(other.committed_tables_, nullptr)},
      table_index_{std::exchange(other.table_index_, nullptr)},
      resident_table_index_{std::move(other.resident_table_index_)},
      reserved_blocks_{std::move(other.reserved_blocks_)},
      reserved_window_blocks_{std::move(other.reserved_window_blocks_)},
      deltas_{std::move(other.deltas_)},
      delta_index_{std::move(other.delta_index_)},
      delta_visit_generations_{
          std::move(other.delta_visit_generations_)},
      next_delta_visit_generation_{
          std::exchange(other.next_delta_visit_generation_, 1)},
      new_tables_{std::move(other.new_tables_)},
      advance_blocks_{std::move(other.advance_blocks_)},
      resident_table_snapshots_{std::move(other.resident_table_snapshots_)},
      block_pool_generation_{std::exchange(other.block_pool_generation_, 0)},
      window_block_pool_generation_{std::exchange(other.window_block_pool_generation_, 0)},
      state_{std::exchange(other.state_, PagedCacheReservationState::Released)} {}

PagedCacheReservation::~PagedCacheReservation() noexcept {
  if (state_ == PagedCacheReservationState::Reserved) {
    block_pool_->RollbackReservedBlocks(reserved_blocks_);
    if (window_block_pool_) {
      window_block_pool_->RollbackReservedBlocks(
          reserved_window_blocks_);
    }
  }
}

size_t PagedCacheReservation::RequiredBlockTableColumns() const {
  size_t columns = 0;
  for (const auto& delta : deltas_) {
    const auto* table = FindCommittedTable(delta.request_id);
    const size_t committed_blocks = table ? table->blocks_.size() : 0;
    columns = std::max(columns, committed_blocks + delta.reserved_block_count);
  }
  return columns;
}

void PagedCacheReservation::FillBlockTable(std::span<const void* const> request_ids,
                                           size_t columns,
                                           std::span<int32_t> output) const {
  if (state_ != PagedCacheReservationState::Reserved) {
    throw std::logic_error("Paged cache block table is only available while the reservation is active.");
  }
  if (request_ids.size() != deltas_.size() ||
      columns < RequiredBlockTableColumns() ||
      output.size() != request_ids.size() * columns) {
    throw std::runtime_error("Paged cache block table output has an invalid shape.");
  }

  std::fill(output.begin(), output.end(), int32_t{-1});
  const uint64_t visit_generation = BeginDeltaResolution();
  for (size_t row = 0; row < request_ids.size(); ++row) {
    const auto& delta =
        ResolveDelta(request_ids[row], visit_generation);
    const auto* table = FindCommittedTable(request_ids[row]);
    size_t column = 0;
    if (table) {
      for (const auto& block : table->blocks_) {
        output[row * columns + column++] = static_cast<int32_t>(block->Id());
      }
    }
    for (size_t i = 0; i < delta.reserved_block_count; ++i) {
      output[row * columns + column++] =
          static_cast<int32_t>(reserved_blocks_[delta.reserved_block_offset + i]->Id());
    }
  }
}

void PagedCacheReservation::FillWindowBlockTable(
    std::span<const void* const> request_ids,
    size_t columns,
    std::span<int32_t> output) const {
  if (state_ != PagedCacheReservationState::Reserved || !window_block_pool_) {
    throw std::logic_error("Paged cache window block table is unavailable.");
  }
  if (request_ids.size() != deltas_.size() ||
      output.size() != request_ids.size() * columns) {
    throw std::runtime_error("Paged cache window block table output has an invalid shape.");
  }

  const uint64_t visit_generation = BeginDeltaResolution();
  for (size_t row = 0; row < request_ids.size(); ++row) {
    const auto& delta =
        ResolveDelta(request_ids[row], visit_generation);
    const auto* table = FindCommittedTable(request_ids[row]);
    for (size_t column = 0; column < columns; ++column) {
      const size_t ring_column = WindowRingColumn(column, window_ring_blocks_);
      const auto& block = table
                              ? table->window_blocks_.at(ring_column)
                              : reserved_window_blocks_.at(
                                    delta.reserved_window_block_offset + ring_column);
      output[row * columns + column] = static_cast<int32_t>(block->Id());
    }
  }
}

void PagedCacheReservation::Commit() {
  ValidateCommit();
  CommitValidated();
}

void PagedCacheReservation::ValidateCommit() const {
  if (state_ != PagedCacheReservationState::Reserved) {
    throw std::logic_error("Paged cache reservation can only be committed once.");
  }
  if (!block_pool_ || !committed_tables_) {
    throw std::logic_error("Paged cache reservation has no owning cache.");
  }
  if ((window_block_pool_ == nullptr) != (window_ring_blocks_ == 0)) {
    throw std::logic_error("Paged cache window reservation configuration is inconsistent.");
  }
  if (block_pool_->MutationGeneration() != block_pool_generation_ ||
      (window_block_pool_ &&
       window_block_pool_->MutationGeneration() != window_block_pool_generation_)) {
    throw std::logic_error(
        "Paged cache block pool ownership changed after reservation.");
  }
  ValidateResidentTablesUnchanged();

  // Re-derive everything CommitValidated depends on straight from the committed cache so that a
  // change to ownership, token boundaries, delta layout, or preallocated capacity between reserve
  // and commit is rejected here rather than during publication.
  size_t new_table_count = 0;
  size_t assigned_reserved_blocks = 0;
  size_t assigned_reserved_window_blocks = 0;
  for (size_t index = 0; index < committed_tables_->size(); ++index) {
    const auto& table = (*committed_tables_)[index];
    const auto indexed = resident_table_index_.Find(table.request_id_);
    if (!table.request_id_ || !indexed || *indexed != index) {
      throw std::logic_error(
          "Paged cache committed tables contain an invalid or duplicate request.");
    }
  }
  for (size_t delta_index = 0; delta_index < deltas_.size();
       ++delta_index) {
    const auto& delta = deltas_[delta_index];
    const auto indexed = delta_index_.Find(delta.request_id);
    if (!delta.request_id || !indexed || *indexed != delta_index) {
      throw std::logic_error(
          "Paged cache reservation contains an invalid request delta.");
    }

    const auto* table = FindCommittedTable(delta.request_id);
    if (delta.newly_admitted == (table != nullptr)) {
      throw std::logic_error("Paged cache ownership changed after reservation.");
    }
    if (table &&
        (table->committed_slots_ != delta.committed_slots ||
         table->mutation_generation_ != delta.table_generation)) {
      throw std::logic_error(
          "Paged cache committed state changed after reservation.");
    }

    // The delta must consume exactly its own contiguous slice of the reserved block and window
    // block pools, in order, and never grow below the already-committed boundary.
    if (delta.target_slots < delta.committed_slots ||
        delta.reserved_block_offset != assigned_reserved_blocks ||
        assigned_reserved_blocks > reserved_blocks_.size() ||
        delta.reserved_block_count >
            reserved_blocks_.size() - assigned_reserved_blocks ||
        delta.reserved_window_block_offset != assigned_reserved_window_blocks ||
        assigned_reserved_window_blocks > reserved_window_blocks_.size() ||
        delta.reserved_window_block_count >
            reserved_window_blocks_.size() - assigned_reserved_window_blocks) {
      throw std::logic_error("Paged cache reservation delta is inconsistent.");
    }

    ValidateEmptyReservedBlocks(
        std::span<const std::shared_ptr<Block>>{reserved_blocks_}.subspan(
            assigned_reserved_blocks, delta.reserved_block_count),
        block_pool_->BlockSize(), "reserved");
    for (const auto& block :
         std::span<const std::shared_ptr<Block>>{reserved_blocks_}.subspan(
             assigned_reserved_blocks, delta.reserved_block_count)) {
      if (!block_pool_->Owns(block)) {
        throw std::logic_error(
            "Paged cache reserved blocks are not owned by the block pool.");
      }
    }
    if (window_block_pool_) {
      const auto reserved_window_blocks =
          std::span<const std::shared_ptr<Block>>{reserved_window_blocks_}.subspan(
              assigned_reserved_window_blocks, delta.reserved_window_block_count);
      ValidateEmptyReservedBlocks(
          reserved_window_blocks,
          window_block_pool_->BlockSize(), "reserved window");
      for (const auto& block : reserved_window_blocks) {
        if (!window_block_pool_->Owns(block)) {
          throw std::logic_error(
              "Paged cache reserved window blocks are not owned by the window block pool.");
        }
      }
    }

    const size_t committed_blocks = table ? table->blocks_.size() : 0;
    const size_t total_blocks = CheckedAdd(
        committed_blocks, delta.reserved_block_count,
        "target block capacity");
    if (delta.target_slots >
        CheckedBlockSlots(
            total_blocks, block_pool_->BlockSize(), "target")) {
      throw std::logic_error(
          "Paged cache reservation cannot reach its target token boundary.");
    }
    // Existing tables must already have room for the appended blocks so CommitValidated's insert
    // cannot reallocate. New tables preallocated their capacity at reservation time.
    if (table && table->blocks_.capacity() < total_blocks) {
      throw std::logic_error(
          "Paged cache reservation did not preallocate commit capacity.");
    }
    const auto& target_table =
        table ? *table : new_tables_.at(new_table_count);
    ValidateAdvancePreconditions(delta, target_table);

    assigned_reserved_blocks += delta.reserved_block_count;
    assigned_reserved_window_blocks += delta.reserved_window_block_count;
    new_table_count += delta.newly_admitted ? 1 : 0;
  }

  if (assigned_reserved_blocks != reserved_blocks_.size() ||
      assigned_reserved_window_blocks != reserved_window_blocks_.size() ||
      new_table_count != new_tables_.size() ||
      (table_index_ &&
       new_table_count > table_index_->Capacity() - table_index_->Size()) ||
      new_table_count >
          committed_tables_->capacity() - committed_tables_->size()) {
    throw std::logic_error(
        "Paged cache reservation commit resources are inconsistent.");
  }
}

// Publication path for a reservation that ValidateCommit has already accepted. It only moves
// shared_ptr block handles and preallocated tables into the committed cache; ValidateCommit
// guarantees the reserved-block/table spans indexed below, the occupancy of every block the current
// growth touches, and both insert capacities, so none of the block/table moves reallocate or touch
// the device. The one caveat is committed_tables_->push_back, whose headroom was reserved by the
// constructor and rechecked by ValidateCommit. It is deliberately not marked noexcept: the
// std::vector insert/push_back and .at() calls it relies on are not themselves noexcept-declared, so
// marking it noexcept would turn any unexpected precondition violation into std::terminate instead
// of a catchable failure the Engine can surface as unhealthy. Keeping it potentially-throwing is the
// safest honest signature while still contributing no new fallible work of its own.
void PagedCacheReservation::CommitValidated() {
  if (state_ != PagedCacheReservationState::Reserved) {
    throw std::logic_error("Paged cache reservation can only be committed once.");
  }

  // Preflight every ownership/generation check before mutating the first table. A later mismatch
  // must never surface after an earlier request has already published.
  if (block_pool_->MutationGeneration() != block_pool_generation_ ||
      (window_block_pool_ &&
       window_block_pool_->MutationGeneration() != window_block_pool_generation_)) {
    throw std::logic_error(
        "Paged cache block pool ownership changed after validation.");
  }
  ValidateResidentTablesUnchanged();
  size_t preflight_new_table_index = 0;
  size_t preflight_new_table_count = 0;
  for (const auto& delta : deltas_) {
    const auto* table = FindCommittedTable(delta.request_id);
    if (delta.newly_admitted) {
      if (table) {
        throw std::logic_error(
            "Paged cache ownership changed after validation.");
      }
    } else if (!table ||
               table->mutation_generation_ != delta.table_generation ||
               table->committed_slots_ != delta.committed_slots) {
      throw std::logic_error(
          "Paged cache committed state changed after validation.");
    }
    const auto& target_table =
        table ? *table : new_tables_.at(preflight_new_table_index++);
    if (table &&
        delta.reserved_block_count >
            table->blocks_.capacity() - table->blocks_.size()) {
      throw std::logic_error(
          "Paged cache reservation lost preallocated commit capacity after validation.");
    }
    preflight_new_table_count += delta.newly_admitted ? 1 : 0;
    ValidateAdvancePreconditions(delta, target_table);
  }
  if (preflight_new_table_count >
      committed_tables_->capacity() - committed_tables_->size()) {
    throw std::logic_error(
        "Paged cache reservation lost table capacity after validation.");
  }

  size_t new_table_index = 0;
  bool occupancy_changed = false;
  for (const auto& delta : deltas_) {
    PagedCacheBlockTable* table = FindCommittedTable(delta.request_id);
    if (delta.newly_admitted) {
      table = &new_tables_.at(new_table_index++);
    }

    const auto first = reserved_blocks_.begin() + static_cast<ptrdiff_t>(delta.reserved_block_offset);
    table->blocks_.insert(table->blocks_.end(), first, first + static_cast<ptrdiff_t>(delta.reserved_block_count));
    const auto window_first = reserved_window_blocks_.begin() +
                              static_cast<ptrdiff_t>(delta.reserved_window_block_offset);
    table->window_blocks_.insert(
        table->window_blocks_.end(), window_first,
        window_first + static_cast<ptrdiff_t>(delta.reserved_window_block_count));
    AdvanceCommittedSlots(*table, delta.target_slots);
    occupancy_changed |= delta.target_slots != delta.committed_slots;
    ++table->mutation_generation_;

    if (delta.newly_admitted) {
      committed_tables_->push_back(std::move(*table));
      if (table_index_ &&
          !table_index_->Insert(
              delta.request_id, committed_tables_->size() - 1)) {
        std::terminate();
      }
    }
  }
  if (occupancy_changed) {
    block_pool_->RecordOccupancyMutation();
  }

  reserved_blocks_.clear();
  reserved_window_blocks_.clear();
  new_tables_.clear();
  advance_blocks_.clear();
  state_ = PagedCacheReservationState::Committed;
}

void PagedCacheReservation::Release() {
  if (state_ == PagedCacheReservationState::Released) {
    return;
  }
  if (state_ == PagedCacheReservationState::Committed) {
    throw std::logic_error("Cannot release a committed paged cache reservation.");
  }

  block_pool_->ValidateFree(reserved_blocks_);
  if (window_block_pool_) {
    window_block_pool_->ValidateFree(reserved_window_blocks_);
  }
  block_pool_->FreeValidated(reserved_blocks_);
  if (window_block_pool_) {
    window_block_pool_->FreeValidated(reserved_window_blocks_);
  }
  reserved_blocks_.clear();
  reserved_window_blocks_.clear();
  new_tables_.clear();
  advance_blocks_.clear();
  state_ = PagedCacheReservationState::Released;
}

PagedCacheBlockTable* PagedCacheReservation::FindCommittedTable(
    const void* request_id) {
  return const_cast<PagedCacheBlockTable*>(
      std::as_const(*this).FindCommittedTable(request_id));
}

const PagedCacheBlockTable* PagedCacheReservation::FindCommittedTable(
    const void* request_id) const {
  const auto index = resident_table_index_.Find(request_id);
  if (!index) {
    return nullptr;
  }
  if (*index >= committed_tables_->size() ||
      (*committed_tables_)[*index].request_id_ != request_id) {
    throw std::logic_error(
        "Paged cache reservation index does not match committed tables.");
  }
  return &(*committed_tables_)[*index];
}

uint64_t PagedCacheReservation::BeginDeltaResolution() const {
  if (next_delta_visit_generation_ ==
      std::numeric_limits<uint64_t>::max()) {
    std::fill(delta_visit_generations_.begin(),
              delta_visit_generations_.end(), 0);
    next_delta_visit_generation_ = 1;
  }
  return next_delta_visit_generation_++;
}

const PagedCacheReservationDelta& PagedCacheReservation::ResolveDelta(
    const void* request_id, uint64_t visit_generation) const {
  const auto index = delta_index_.Find(request_id);
  if (!index || *index >= deltas_.size() ||
      delta_visit_generations_[*index] == visit_generation ||
      deltas_[*index].request_id != request_id) {
    throw std::runtime_error(
        "Paged cache block table contains an invalid or duplicate request.");
  }
  delta_visit_generations_[*index] = visit_generation;
  return deltas_[*index];
}

void PagedCacheReservation::ValidateResidentTablesUnchanged() const {
  if (committed_tables_->size() != resident_table_snapshots_.size()) {
    throw std::logic_error(
        "Paged cache committed table membership changed after reservation.");
  }
  for (size_t index = 0; index < resident_table_snapshots_.size(); ++index) {
    const auto& table = (*committed_tables_)[index];
    const auto& snapshot = resident_table_snapshots_[index];
    if (table.request_id_ != snapshot.request_id ||
        table.committed_slots_ != snapshot.committed_slots ||
        table.mutation_generation_ != snapshot.mutation_generation ||
        table.blocks_.data() != snapshot.blocks_data ||
        table.blocks_.size() != snapshot.block_count ||
        table.window_blocks_.data() != snapshot.window_blocks_data ||
        table.window_blocks_.size() != snapshot.window_block_count) {
      throw std::logic_error(
          "Paged cache committed table mapping changed after reservation.");
    }
  }
}

void PagedCacheReservation::ValidateAdvancePreconditions(
    const PagedCacheReservationDelta& delta,
    const PagedCacheBlockTable& table) const {
  const size_t block_size = block_pool_->BlockSize();
  size_t remaining = delta.target_slots - delta.committed_slots;
  size_t block_index = delta.committed_slots / block_size;
  const size_t first_block_index = block_index;
  while (remaining > 0) {
    const std::shared_ptr<Block>* block{};
    if (block_index < table.blocks_.size()) {
      block = &table.blocks_[block_index];
    } else {
      const size_t reserved_index = block_index - table.blocks_.size();
      if (reserved_index >= delta.reserved_block_count) {
        throw std::logic_error(
            "Paged cache reservation cannot reach its target token boundary.");
      }
      block = &reserved_blocks_.at(delta.reserved_block_offset + reserved_index);
    }

    const size_t expected_size =
        block_index == first_block_index ? delta.committed_slots % block_size : 0;
    if (!*block || (*block)->Capacity() != block_size ||
        (*block)->Size() != expected_size ||
        block_index - first_block_index >= delta.advance_block_count ||
        block->get() !=
            advance_blocks_.at(delta.advance_block_offset +
                               block_index - first_block_index)) {
      throw std::logic_error(
          "Paged cache block mapping or occupancy changed after reservation.");
    }
    const size_t slots = std::min(remaining, block_size - expected_size);
    remaining -= slots;
    ++block_index;
  }
}

void PagedCacheReservation::AdvanceCommittedSlots(PagedCacheBlockTable& table, size_t target_slots) {
  size_t remaining = target_slots - table.committed_slots_;
  size_t block_index = table.committed_slots_ / block_pool_->BlockSize();
  while (remaining > 0) {
    auto& block = table.blocks_.at(block_index++);
    const size_t slots = std::min(remaining, block->EmptySlots());
    block->AddSlots(slots);
    remaining -= slots;
  }
  table.committed_slots_ = target_slots;
}

}  // namespace Generators
