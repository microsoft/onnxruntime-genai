// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine_invariants.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace Generators {

namespace {

std::string PtrId(const void* p) {
  std::ostringstream oss;
  oss << p;
  return oss.str();
}

}  // namespace

size_t PagedCacheSnapshot::AllocatedBlocks() const {
  size_t allocated = 0;
  for (const auto& request : requests) {
    allocated += request.block_ids.size();
  }
  return allocated;
}

std::vector<InvariantViolation> ValidateCacheInvariants(const PagedCacheSnapshot& cache) {
  std::vector<InvariantViolation> violations;
  const auto add = [&violations](std::string message) {
    violations.push_back(InvariantViolation{std::move(message)});
  };

  // Total accounting: every block is free, transaction-reserved, or committed to one Request.
  const size_t allocated = cache.AllocatedBlocks();
  const size_t transaction_reserved = cache.TransactionReservedBlocks();
  if (cache.free_blocks > cache.total_blocks) {
    add("free_blocks (" + std::to_string(cache.free_blocks) + ") exceeds total_blocks (" +
        std::to_string(cache.total_blocks) + ").");
  }
  if (cache.free_blocks + transaction_reserved + allocated != cache.total_blocks) {
    add("free (" + std::to_string(cache.free_blocks) + ") + transaction_reserved (" +
        std::to_string(transaction_reserved) + ") + allocated (" +
        std::to_string(allocated) + ") != total_blocks (" +
        std::to_string(cache.total_blocks) + ").");
  }

  // Single ownership: a physical block id appears in at most one Request's table.
  std::unordered_map<size_t, const void*> owner_of_block;
  std::unordered_set<const void*> seen_requests;
  size_t max_blocks_per_request = 0;

  for (const auto& request : cache.requests) {
    max_blocks_per_request = std::max(max_blocks_per_request, request.block_ids.size());

    // Each Request appears in the block-table listing at most once. A repeated row would let the
    // same Request re-list a physical block without tripping the single-ownership check below.
    if (!seen_requests.insert(request.request_id).second) {
      add("Request " + PtrId(request.request_id) + " appears in more than one block table.");
    }

    std::unordered_set<size_t> seen_within_request;
    for (const size_t block_id : request.block_ids) {
      if (block_id >= cache.total_blocks) {
        add("Request " + PtrId(request.request_id) + " owns out-of-range block id " +
            std::to_string(block_id) + " (total_blocks " + std::to_string(cache.total_blocks) + ").");
      }

      if (!seen_within_request.insert(block_id).second) {
        add("Request " + PtrId(request.request_id) + " lists block id " + std::to_string(block_id) +
            " more than once.");
      }

      const auto [it, inserted] = owner_of_block.emplace(block_id, request.request_id);
      if (!inserted && it->second != request.request_id) {
        add("Block id " + std::to_string(block_id) + " is owned by more than one Request (" +
            PtrId(it->second) + " and " + PtrId(request.request_id) + ").");
      }
    }

    // Slot accounting: used + empty must fill exactly the owned blocks' capacity.
    if (cache.block_size != 0) {
      const size_t capacity = request.block_ids.size() * cache.block_size;
      if (request.used_slots + request.empty_slots != capacity) {
        add("Request " + PtrId(request.request_id) + " used (" + std::to_string(request.used_slots) +
            ") + empty (" + std::to_string(request.empty_slots) + ") slots != owned capacity (" +
            std::to_string(capacity) + ").");
      }
      if (request.used_slots > capacity) {
        add("Request " + PtrId(request.request_id) + " used slots (" +
            std::to_string(request.used_slots) + ") exceed owned capacity (" +
            std::to_string(capacity) + ").");
      }
    }
  }

  std::unordered_set<size_t> reserved_blocks;
  for (const size_t block_id : cache.transaction_reserved_block_ids) {
    if (block_id >= cache.total_blocks) {
      add("Transaction reserves out-of-range block id " + std::to_string(block_id) + ".");
    }
    if (!reserved_blocks.insert(block_id).second) {
      add("Transaction reserves block id " + std::to_string(block_id) + " more than once.");
    }
    if (owner_of_block.find(block_id) != owner_of_block.end()) {
      add("Transaction-reserved block id " + std::to_string(block_id) +
          " is also committed to a Request.");
    }
  }

  std::unordered_set<const void*> reservation_requests;
  std::unordered_set<size_t> blocks_assigned_to_delta;
  std::unordered_set<size_t> window_blocks_assigned_to_delta;
  for (const auto& reservation : cache.reservations) {
    if (!reservation_requests.insert(reservation.request_id).second) {
      add("Request " + PtrId(reservation.request_id) +
          " appears in more than one transaction reservation.");
    }
    if (reservation.target_slots < reservation.committed_slots) {
      add("Request " + PtrId(reservation.request_id) +
          " transaction target precedes its committed slot boundary.");
    }
    if (reservation.tail_slots_to_consume >
        reservation.target_slots - reservation.committed_slots) {
      add("Request " + PtrId(reservation.request_id) +
          " transaction tail-slot growth exceeds total growth.");
    }
    const auto committed_owner = std::find_if(
        cache.requests.begin(), cache.requests.end(),
        [&reservation](const RequestBlockSnapshot& request) {
          return request.request_id == reservation.request_id;
        });
    if (reservation.newly_admitted == (committed_owner != cache.requests.end())) {
      add("Request " + PtrId(reservation.request_id) +
          " transaction membership disagrees with committed cache ownership.");
    }
    if (committed_owner != cache.requests.end() &&
        reservation.committed_slots != committed_owner->used_slots) {
      add("Request " + PtrId(reservation.request_id) +
          " transaction committed slots disagree with committed cache usage.");
    }
    for (const size_t block_id : reservation.reserved_block_ids) {
      if (reserved_blocks.find(block_id) == reserved_blocks.end()) {
        add("Request " + PtrId(reservation.request_id) +
            " references block id " + std::to_string(block_id) +
            " that is not transaction-reserved.");
      }
      if (!blocks_assigned_to_delta.insert(block_id).second) {
        add("Transaction-reserved block id " + std::to_string(block_id) +
            " is assigned to more than one Request delta.");
      }
    }
    for (const size_t block_id : reservation.reserved_window_block_ids) {
      if (!window_blocks_assigned_to_delta.insert(block_id).second) {
        add("Transaction-reserved window block id " + std::to_string(block_id) +
            " is assigned to more than one Request delta.");
      }
    }
  }
  if (blocks_assigned_to_delta != reserved_blocks) {
    add("Not every transaction-reserved block belongs to exactly one Request delta.");
  }

  const auto& window = cache.window_blocks;
  size_t allocated_window_blocks = 0;
  std::unordered_map<size_t, const void*> window_owner_of_block;
  std::unordered_set<const void*> window_requests;
  for (const auto& request : window.requests) {
    allocated_window_blocks += request.block_ids.size();
    if (!window_requests.insert(request.request_id).second) {
      add("Request " + PtrId(request.request_id) +
          " appears in more than one window block table.");
    }
    if (request.block_ids.size() != window.blocks_per_request) {
      add("Request " + PtrId(request.request_id) + " owns " +
          std::to_string(request.block_ids.size()) + " window blocks instead of " +
          std::to_string(window.blocks_per_request) + ".");
    }
    for (const size_t block_id : request.block_ids) {
      if (block_id >= window.total_blocks) {
        add("Request " + PtrId(request.request_id) + " owns out-of-range window block id " +
            std::to_string(block_id) + " (total_blocks " +
            std::to_string(window.total_blocks) + ").");
      }
      const auto [it, inserted] = window_owner_of_block.emplace(block_id, request.request_id);
      if (!inserted) {
        add("Window block id " + std::to_string(block_id) +
            " is listed more than once.");
      }
    }
  }

  std::unordered_set<size_t> reserved_window_blocks;
  for (const size_t block_id : window.transaction_reserved_block_ids) {
    if (block_id >= window.total_blocks) {
      add("Transaction reserves out-of-range window block id " + std::to_string(block_id) + ".");
    }
    if (!reserved_window_blocks.insert(block_id).second) {
      add("Transaction reserves window block id " + std::to_string(block_id) + " more than once.");
    }
    if (window_owner_of_block.find(block_id) != window_owner_of_block.end()) {
      add("Transaction-reserved window block id " + std::to_string(block_id) +
          " is also committed to a Request.");
    }
  }
  if (window_blocks_assigned_to_delta != reserved_window_blocks) {
    add("Not every transaction-reserved window block belongs to exactly one Request delta.");
  }
  if (window.total_blocks == 0 && !window_blocks_assigned_to_delta.empty()) {
    add("A non-windowed cache has Request-attributed window reservations.");
  }
  if (window.total_blocks != 0) {
    for (const auto& reservation : cache.reservations) {
      const size_t expected_window_blocks =
          reservation.newly_admitted ? window.blocks_per_request : 0;
      if (reservation.reserved_window_block_ids.size() !=
          expected_window_blocks) {
        add("Request " + PtrId(reservation.request_id) + " owns " +
            std::to_string(reservation.reserved_window_block_ids.size()) +
            " transaction-reserved window blocks instead of " +
            std::to_string(expected_window_blocks) + ".");
      }
    }
  }
  if (window.free_blocks > window.total_blocks) {
    add("window free_blocks (" + std::to_string(window.free_blocks) +
        ") exceeds total_blocks (" + std::to_string(window.total_blocks) + ").");
  }
  if (window.free_blocks + allocated_window_blocks + reserved_window_blocks.size() !=
      window.total_blocks) {
    add("window free (" + std::to_string(window.free_blocks) + ") + transaction_reserved (" +
        std::to_string(reserved_window_blocks.size()) + ") + allocated (" +
        std::to_string(allocated_window_blocks) + ") != total_blocks (" +
        std::to_string(window.total_blocks) + ").");
  }

  // The padded block-table width must be able to hold the widest Request's table.
  if (cache.block_table_columns != 0 && cache.block_table_columns < max_blocks_per_request) {
    add("block_table_columns (" + std::to_string(cache.block_table_columns) +
        ") is narrower than the widest Request table (" + std::to_string(max_blocks_per_request) + ").");
  }

  return violations;
}

std::vector<InvariantViolation> ValidateRequestInvariants(const RequestStateSnapshot& request) {
  std::vector<InvariantViolation> violations;
  const auto add = [&violations](std::string message) {
    violations.push_back(InvariantViolation{std::move(message)});
  };

  const std::string id = PtrId(request.request_id);

  if (request.current_sequence_length < 0 || request.processed_sequence_length < 0 ||
      request.seen_sequence_length < 0) {
    add("Request " + id + " has a negative sequence-length counter.");
  }

  if (request.processed_sequence_length > request.current_sequence_length) {
    add("Request " + id + " processed length (" + std::to_string(request.processed_sequence_length) +
        ") exceeds current length (" + std::to_string(request.current_sequence_length) + ").");
  }

  if (request.seen_sequence_length > request.current_sequence_length) {
    add("Request " + id + " seen length (" + std::to_string(request.seen_sequence_length) +
        ") exceeds current length (" + std::to_string(request.current_sequence_length) + ").");
  }

  return violations;
}

std::vector<InvariantViolation> ValidateInvariants(const PagedCacheSnapshot& cache,
                                                   const std::vector<RequestStateSnapshot>& requests) {
  std::vector<InvariantViolation> violations = ValidateCacheInvariants(cache);

  for (const auto& request : requests) {
    auto request_violations = ValidateRequestInvariants(request);
    violations.insert(violations.end(), request_violations.begin(), request_violations.end());
  }

  // Block tables exist only for known Requests: every block-owning id must be a Request we were
  // handed. (The reverse does not hold: an Assigned-but-unallocated Request owns no blocks yet.)
  std::set<const void*> known_requests;
  for (const auto& request : requests) {
    known_requests.insert(request.request_id);
  }
  for (const auto& owner : cache.requests) {
    if (known_requests.find(owner.request_id) == known_requests.end()) {
      violations.push_back(InvariantViolation{
          "Cache holds a block table for unknown Request " + PtrId(owner.request_id) + "."});
    }
  }
  for (const auto& owner : cache.window_blocks.requests) {
    if (known_requests.find(owner.request_id) == known_requests.end()) {
      violations.push_back(InvariantViolation{
          "Window cache holds a block table for unknown Request " +
          PtrId(owner.request_id) + "."});
    }
  }
  for (const auto& reservation : cache.reservations) {
    if (known_requests.find(reservation.request_id) == known_requests.end()) {
      violations.push_back(InvariantViolation{
          "Cache holds a transaction reservation for unknown Request " +
          PtrId(reservation.request_id) + "."});
    }
  }

  std::unordered_map<const void*, size_t> processed_by_request;
  for (const auto& request : requests) {
    if (request.processed_sequence_length >= 0) {
      processed_by_request.emplace(
          request.request_id,
          static_cast<size_t>(request.processed_sequence_length));
    }
  }
  for (const auto& owner : cache.requests) {
    const auto processed = processed_by_request.find(owner.request_id);
    if (processed != processed_by_request.end() &&
        owner.used_slots != processed->second) {
      violations.push_back(InvariantViolation{
          "Request " + PtrId(owner.request_id) + " committed cache usage (" +
          std::to_string(owner.used_slots) + ") differs from processed sequence length (" +
          std::to_string(processed->second) + ")."});
    }
  }

  if (cache.window_blocks.total_blocks != 0) {
    std::set<const void*> full_owners;
    std::set<const void*> window_owners;
    for (const auto& owner : cache.requests) {
      full_owners.insert(owner.request_id);
    }
    for (const auto& owner : cache.window_blocks.requests) {
      window_owners.insert(owner.request_id);
    }
    if (full_owners != window_owners) {
      violations.push_back(InvariantViolation{
          "Full-cache and window-cache owner sets disagree."});
    }
  }

  return violations;
}

void ThrowIfInvariantsViolated(const PagedCacheSnapshot& cache,
                               const std::vector<RequestStateSnapshot>& requests) {
  const auto violations = ValidateInvariants(cache, requests);
  if (violations.empty()) {
    return;
  }

  std::ostringstream oss;
  oss << "Engine invariant validation failed with " << violations.size() << " violation(s):";
  for (const auto& violation : violations) {
    oss << "\n  - " << violation.message;
  }
  throw std::runtime_error(oss.str());
}

}  // namespace Generators
