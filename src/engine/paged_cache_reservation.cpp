// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "paged_cache_reservation.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "window_ring.h"

namespace Generators {

namespace {

PagedCacheBlockTable* FindTable(std::vector<PagedCacheBlockTable>& tables, const void* request_id) {
  const auto it = std::find_if(tables.begin(), tables.end(),
                               [request_id](const PagedCacheBlockTable& table) {
                                 return table.request_id == request_id;
                               });
  return it == tables.end() ? nullptr : &*it;
}

}  // namespace

void RemovePagedCacheBlockTable(
    BlockPool& block_pool,
    BlockPool* window_block_pool,
    std::vector<PagedCacheBlockTable>& committed_tables,
    const void* request_id) {
  const auto table = std::find_if(
      committed_tables.begin(), committed_tables.end(),
      [request_id](const PagedCacheBlockTable& candidate) {
        return candidate.request_id == request_id;
      });
  if (table == committed_tables.end()) {
    return;
  }

  block_pool.Free(table->blocks);
  if (window_block_pool) {
    window_block_pool->Free(table->window_blocks);
  }
  committed_tables.erase(table);
}

PagedCacheReservation::PagedCacheReservation(
    BlockPool& block_pool,
    std::vector<PagedCacheBlockTable>& committed_tables,
    std::span<const PagedCacheReservationRequest> requests,
    BlockPool* window_block_pool,
    size_t window_ring_blocks)
    : block_pool_{&block_pool},
      window_block_pool_{window_block_pool},
      window_ring_blocks_{window_ring_blocks},
      committed_tables_{&committed_tables} {
  if ((window_block_pool_ == nullptr) != (window_ring_blocks_ == 0)) {
    throw std::runtime_error("Paged cache window reservation configuration is inconsistent.");
  }
  deltas_.reserve(requests.size());
  new_tables_.reserve(requests.size());
  committed_tables.reserve(committed_tables.size() + requests.size());

  size_t reserved_block_count = 0;
  size_t reserved_window_block_count = 0;
  for (const auto& request : requests) {
    const bool duplicate_request =
        std::any_of(deltas_.begin(), deltas_.end(),
                    [&request](const PagedCacheReservationDelta& delta) {
                      return delta.request_id == request.request_id;
                    });
    if (!request.request_id || duplicate_request) {
      throw std::runtime_error("Paged cache reservation contains an invalid or duplicate request.");
    }

    auto* committed_table = FindTable(committed_tables, request.request_id);
    if (request.newly_admitted == (committed_table != nullptr)) {
      throw std::runtime_error("Paged cache reservation request membership does not match the committed cache.");
    }

    const size_t committed_slots = committed_table ? committed_table->committed_slots : 0;
    const size_t committed_blocks = committed_table ? committed_table->blocks.size() : 0;
    if (request.target_slots < committed_slots) {
      throw std::runtime_error("Paged cache reservation cannot reduce committed slots.");
    }

    const size_t committed_capacity = committed_blocks * block_pool.BlockSize();
    // Blocks are taken for the whole sequence so that a chunked prefill cannot lose the rest of its
    // capacity to another request between steps, but only target_slots is committed below.
    const size_t reserved_slots = std::max(request.reserved_slots, request.target_slots);
    const size_t additional_slots =
        reserved_slots > committed_capacity ? reserved_slots - committed_capacity : 0;
    const size_t new_blocks = block_pool.BlocksNeeded(additional_slots);
    const size_t tail_capacity = committed_capacity - committed_slots;
    const size_t growth = request.target_slots - committed_slots;

    deltas_.push_back(PagedCacheReservationDelta{
        request.request_id,
        committed_slots,
        request.target_slots,
        std::min(growth, tail_capacity),
        reserved_block_count,
        new_blocks,
        reserved_window_block_count,
        request.newly_admitted ? window_ring_blocks_ : 0,
        request.newly_admitted,
    });
    reserved_block_count += new_blocks;
    if (request.newly_admitted) {
      reserved_window_block_count += window_ring_blocks_;
    }

    if (committed_table) {
      committed_table->blocks.reserve(committed_blocks + new_blocks);
    } else {
      PagedCacheBlockTable table;
      table.request_id = request.request_id;
      table.blocks.reserve(new_blocks);
      table.window_blocks.reserve(window_ring_blocks_);
      new_tables_.push_back(std::move(table));
    }
  }

  if (reserved_block_count > block_pool.AvailableBlocks()) {
    throw std::runtime_error("Not enough free blocks for the complete paged cache reservation.");
  }
  if (window_block_pool_ && reserved_window_block_count > window_block_pool_->AvailableBlocks()) {
    throw std::runtime_error("Not enough free window blocks for the complete paged cache reservation.");
  }

  reserved_blocks_ = block_pool.ReserveBlocks(reserved_block_count * block_pool.BlockSize());
  if (window_block_pool_) {
    reserved_window_blocks_ = window_block_pool_->ReserveBlocks(
        reserved_window_block_count * window_block_pool_->BlockSize());
  }
  state_ = PagedCacheReservationState::Reserved;
}

PagedCacheReservation::PagedCacheReservation(PagedCacheReservation&& other) noexcept
    : block_pool_{std::exchange(other.block_pool_, nullptr)},
      window_block_pool_{std::exchange(other.window_block_pool_, nullptr)},
      window_ring_blocks_{std::exchange(other.window_ring_blocks_, 0)},
      committed_tables_{std::exchange(other.committed_tables_, nullptr)},
      reserved_blocks_{std::move(other.reserved_blocks_)},
      reserved_window_blocks_{std::move(other.reserved_window_blocks_)},
      deltas_{std::move(other.deltas_)},
      new_tables_{std::move(other.new_tables_)},
      state_{std::exchange(other.state_, PagedCacheReservationState::Released)} {}

PagedCacheReservation::~PagedCacheReservation() {
  if (state_ == PagedCacheReservationState::Reserved) {
    Release();
  }
}

size_t PagedCacheReservation::RequiredBlockTableColumns() const {
  size_t columns = 0;
  for (const auto& delta : deltas_) {
    const auto* table = FindCommittedTable(delta.request_id);
    const size_t committed_blocks = table ? table->blocks.size() : 0;
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
  if (request_ids.size() > deltas_.size() ||
      columns < RequiredBlockTableColumns() ||
      output.size() != request_ids.size() * columns) {
    throw std::runtime_error("Paged cache block table output has an invalid shape.");
  }

  std::fill(output.begin(), output.end(), int32_t{-1});
  for (size_t row = 0; row < request_ids.size(); ++row) {
    if (std::find(request_ids.begin(), request_ids.begin() + static_cast<ptrdiff_t>(row), request_ids[row]) !=
        request_ids.begin() + static_cast<ptrdiff_t>(row)) {
      throw std::runtime_error("Paged cache block table contains a duplicate request.");
    }
    const auto& delta = FindDelta(request_ids[row]);
    const auto* table = FindCommittedTable(request_ids[row]);
    size_t column = 0;
    if (table) {
      for (const auto& block : table->blocks) {
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
  if (request_ids.size() > deltas_.size() ||
      output.size() != request_ids.size() * columns) {
    throw std::runtime_error("Paged cache window block table output has an invalid shape.");
  }

  for (size_t row = 0; row < request_ids.size(); ++row) {
    const auto& delta = FindDelta(request_ids[row]);
    const auto* table = FindCommittedTable(request_ids[row]);
    for (size_t column = 0; column < columns; ++column) {
      const size_t ring_column = WindowRingColumn(column, window_ring_blocks_);
      const auto& block = table
                              ? table->window_blocks.at(ring_column)
                              : reserved_window_blocks_.at(
                                    delta.reserved_window_block_offset + ring_column);
      output[row * columns + column] = static_cast<int32_t>(block->Id());
    }
  }
}

void PagedCacheReservation::CommitPrefix(const void* request_id,
                                         size_t step_slots,
                                         size_t kept_slots) {
  if (state_ != PagedCacheReservationState::Reserved) {
    throw std::logic_error("Paged cache reservation is no longer accepting prefix commits.");
  }
  const auto delta = std::find_if(deltas_.begin(), deltas_.end(),
                                  [request_id](const PagedCacheReservationDelta& candidate) {
                                    return candidate.request_id == request_id;
                                  });
  if (delta == deltas_.end()) {
    throw std::runtime_error("Request is not part of the paged cache reservation.");
  }
  if (kept_slots == 0 || kept_slots > step_slots ||
      step_slots != delta->target_slots - delta->committed_slots) {
    throw std::runtime_error(
        "Paged cache prefix commit must keep between one and step_slots of the slots this step "
        "planned.");
  }

  delta->target_slots = delta->committed_slots + kept_slots;
  // The rejected slots stay inside blocks this request already owns, so only the boundary moves.
  const auto* table = FindCommittedTable(request_id);
  const size_t committed_capacity =
      (table ? table->blocks.size() : 0) * block_pool_->BlockSize();
  const size_t tail_capacity = committed_capacity - delta->committed_slots;
  delta->tail_slots_to_consume = std::min(kept_slots, tail_capacity);
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

  // Re-derive everything CommitValidated depends on straight from the committed cache so that a
  // change to ownership, token boundaries, delta layout, or preallocated capacity between reserve
  // and commit is rejected here rather than during publication.
  size_t new_table_count = 0;
  size_t assigned_reserved_blocks = 0;
  size_t assigned_reserved_window_blocks = 0;
  std::vector<const void*> request_ids;
  request_ids.reserve(deltas_.size());
  for (const auto& delta : deltas_) {
    if (!delta.request_id ||
        std::find(request_ids.begin(), request_ids.end(), delta.request_id) !=
            request_ids.end()) {
      throw std::logic_error(
          "Paged cache reservation contains an invalid request delta.");
    }
    request_ids.push_back(delta.request_id);

    const auto* table = FindCommittedTable(delta.request_id);
    if (delta.newly_admitted == (table != nullptr)) {
      throw std::logic_error("Paged cache ownership changed after reservation.");
    }
    if (table && table->committed_slots != delta.committed_slots) {
      throw std::logic_error(
          "Paged cache token boundary changed after reservation.");
    }

    // The delta must consume exactly its own contiguous slice of the reserved block and window
    // block pools, in order, and never grow below the already-committed boundary.
    if (delta.target_slots < delta.committed_slots ||
        delta.reserved_block_offset != assigned_reserved_blocks ||
        delta.reserved_block_count >
            reserved_blocks_.size() - assigned_reserved_blocks ||
        delta.reserved_window_block_offset != assigned_reserved_window_blocks ||
        delta.reserved_window_block_count >
            reserved_window_blocks_.size() - assigned_reserved_window_blocks) {
      throw std::logic_error("Paged cache reservation delta is inconsistent.");
    }

    const size_t committed_blocks = table ? table->blocks.size() : 0;
    const size_t total_blocks = committed_blocks + delta.reserved_block_count;
    if (delta.target_slots > total_blocks * block_pool_->BlockSize()) {
      throw std::logic_error(
          "Paged cache reservation cannot reach its target token boundary.");
    }
    // Existing tables must already have room for the appended blocks so CommitValidated's insert
    // cannot reallocate. New tables preallocated their capacity at reservation time.
    if (table && table->blocks.capacity() < total_blocks) {
      throw std::logic_error(
          "Paged cache reservation did not preallocate commit capacity.");
    }

    assigned_reserved_blocks += delta.reserved_block_count;
    assigned_reserved_window_blocks += delta.reserved_window_block_count;
    new_table_count += delta.newly_admitted ? 1 : 0;
  }

  // The committed_tables_ headroom checked here is this reservation's own share only. A composite
  // orchestrator that publishes sibling reservations against the same vector must ensure aggregate
  // headroom itself (see the ValidateCommit contract in the header).
  if (assigned_reserved_blocks != reserved_blocks_.size() ||
      assigned_reserved_window_blocks != reserved_window_blocks_.size() ||
      new_table_count != new_tables_.size() ||
      committed_tables_->capacity() <
          committed_tables_->size() + new_table_count) {
    throw std::logic_error(
        "Paged cache reservation commit resources are inconsistent.");
  }
}

// Publication path for a reservation that ValidateCommit has already accepted. It only moves
// shared_ptr block handles and preallocated tables into the committed cache; ValidateCommit
// guarantees the reserved-block/table spans indexed below and both insert capacities, so none of
// the block/table moves reallocate or touch the device. (The AdvanceCommittedSlots slot walk relies
// on the same UsedSlots == committed_slots invariant the whole reservation model maintains, which
// ValidateCommit re-derives via the token-boundary and target-reachability checks.) The one caveat
// is committed_tables_->push_back: it cannot reallocate for a
// single validated reservation (the constructor reserved this reservation's headroom), but when an
// orchestrator publishes several reservations that share one committed_tables_ vector it stays
// allocation-free only if that orchestrator guaranteed the aggregate headroom documented on
// ValidateCommit. It is deliberately not marked noexcept: the std::vector insert/push_back and .at()
// calls it relies on are not themselves noexcept-declared, so marking it noexcept would turn any
// unexpected precondition violation into std::terminate instead of a catchable failure the Engine
// can surface as unhealthy. Keeping it potentially-throwing is the safest honest signature while
// still contributing no new fallible work of its own on the single-reservation path.
void PagedCacheReservation::CommitValidated() {
  // The one precondition re-checked here: publishing twice would walk cleared reserved-block spans
  // with stale delta offsets, which is undefined behavior rather than a catchable error. This is a
  // state-only comparison performed before any mutation, so it introduces no allocation or
  // device work and never leaves a half-published reservation. It intentionally does NOT re-run the
  // ownership, boundary, or capacity checks -- those belong to ValidateCommit and a composite
  // transaction runs them for every reservation up front.
  if (state_ != PagedCacheReservationState::Reserved) {
    throw std::logic_error("Paged cache reservation can only be committed once.");
  }

  size_t new_table_index = 0;
  for (const auto& delta : deltas_) {
    PagedCacheBlockTable* table = FindTable(*committed_tables_, delta.request_id);
    if (!table) {
      table = &new_tables_.at(new_table_index++);
    }

    const auto first = reserved_blocks_.begin() + static_cast<ptrdiff_t>(delta.reserved_block_offset);
    table->blocks.insert(table->blocks.end(), first, first + static_cast<ptrdiff_t>(delta.reserved_block_count));
    const auto window_first = reserved_window_blocks_.begin() +
                              static_cast<ptrdiff_t>(delta.reserved_window_block_offset);
    table->window_blocks.insert(
        table->window_blocks.end(), window_first,
        window_first + static_cast<ptrdiff_t>(delta.reserved_window_block_count));
    AdvanceCommittedSlots(*table, delta.target_slots);

    if (delta.newly_admitted) {
      committed_tables_->push_back(std::move(*table));
    }
  }

  reserved_blocks_.clear();
  reserved_window_blocks_.clear();
  new_tables_.clear();
  state_ = PagedCacheReservationState::Committed;
}

void PagedCacheReservation::Release() {
  if (state_ == PagedCacheReservationState::Released) {
    return;
  }
  if (state_ == PagedCacheReservationState::Committed) {
    throw std::logic_error("Cannot release a committed paged cache reservation.");
  }

  block_pool_->Free(reserved_blocks_);
  if (window_block_pool_) {
    window_block_pool_->Free(reserved_window_blocks_);
  }
  reserved_blocks_.clear();
  reserved_window_blocks_.clear();
  new_tables_.clear();
  state_ = PagedCacheReservationState::Released;
}

const PagedCacheBlockTable* PagedCacheReservation::FindCommittedTable(const void* request_id) const {
  const auto* table = FindTable(*committed_tables_, request_id);
  return table;
}

const PagedCacheReservationDelta& PagedCacheReservation::FindDelta(const void* request_id) const {
  const auto it = std::find_if(deltas_.begin(), deltas_.end(),
                               [request_id](const PagedCacheReservationDelta& delta) {
                                 return delta.request_id == request_id;
                               });
  if (it == deltas_.end()) {
    throw std::runtime_error("Request is not part of the paged cache reservation.");
  }
  return *it;
}

void PagedCacheReservation::AdvanceCommittedSlots(PagedCacheBlockTable& table, size_t target_slots) {
  size_t remaining = target_slots - table.committed_slots;
  size_t block_index = table.committed_slots / block_pool_->BlockSize();
  while (remaining > 0) {
    auto& block = table.blocks.at(block_index++);
    const size_t slots = std::min(remaining, block->EmptySlots());
    block->AddSlots(slots);
    remaining -= slots;
  }
  table.committed_slots = target_slots;
}

}  // namespace Generators
