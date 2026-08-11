// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "paged_cache_reservation.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

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

PagedCacheReservation::PagedCacheReservation(
    BlockPool& block_pool,
    std::vector<PagedCacheBlockTable>& committed_tables,
    std::span<const PagedCacheReservationRequest> requests)
    : block_pool_{&block_pool},
      committed_tables_{&committed_tables} {
  deltas_.reserve(requests.size());
  new_tables_.reserve(requests.size());
  committed_tables.reserve(committed_tables.size() + requests.size());

  size_t reserved_block_count = 0;
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
        request.newly_admitted,
    });
    reserved_block_count += new_blocks;

    if (committed_table) {
      committed_table->blocks.reserve(committed_blocks + new_blocks);
    } else {
      PagedCacheBlockTable table;
      table.request_id = request.request_id;
      table.blocks.reserve(new_blocks);
      new_tables_.push_back(std::move(table));
    }
  }

  if (reserved_block_count > block_pool.AvailableBlocks()) {
    throw std::runtime_error("Not enough free blocks for the complete paged cache reservation.");
  }

  reserved_blocks_ = block_pool.ReserveBlocks(reserved_block_count * block_pool.BlockSize());
  state_ = PagedCacheReservationState::Reserved;
}

PagedCacheReservation::PagedCacheReservation(PagedCacheReservation&& other) noexcept
    : block_pool_{std::exchange(other.block_pool_, nullptr)},
      committed_tables_{std::exchange(other.committed_tables_, nullptr)},
      reserved_blocks_{std::move(other.reserved_blocks_)},
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
  if (request_ids.size() != deltas_.size() ||
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

void PagedCacheReservation::Commit() {
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
    AdvanceCommittedSlots(*table, delta.target_slots);

    if (delta.newly_admitted) {
      committed_tables_->push_back(std::move(*table));
    }
  }

  reserved_blocks_.clear();
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
  reserved_blocks_.clear();
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
