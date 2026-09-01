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
  for (const auto& reservation : cache.reservations) {
    if (!reservation_requests.insert(reservation.request_id).second) {
      add("Request " + PtrId(reservation.request_id) +
          " appears in more than one transaction reservation.");
    }
    if (reservation.target_slots < reservation.committed_slots) {
      add("Request " + PtrId(reservation.request_id) +
          " transaction target precedes its committed slot boundary.");
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

  if (request.current_sequence_length < 0 || request.processed_sequence_length < 0) {
    add("Request " + id + " has a negative sequence-length counter.");
  }

  if (request.processed_sequence_length > request.current_sequence_length) {
    add("Request " + id + " processed length (" + std::to_string(request.processed_sequence_length) +
        ") exceeds current length (" + std::to_string(request.current_sequence_length) + ").");
  }

  if (request.has_current_turn && request.current_turn_id == 0) {
    add("Request " + id + " uses reserved turn id zero.");
  }

  if (IsExecutable(request.status)) {
    if (!request.has_current_turn) {
      add("Executable Request " + id + " has no assigned turn id.");
    }
    if (request.finish_reason != GenerationFinishReason::None) {
      add("Executable Request " + id + " has a terminal generation finish reason.");
    }
  }

  if (IsTurnComplete(request.status)) {
    if (!request.has_current_turn) {
      add("Turn-complete Request " + id + " has no assigned turn id.");
    }
    if (request.finish_reason == GenerationFinishReason::None) {
      add("Turn-complete Request " + id + " has no generation finish reason.");
    }
    // Cancellation is observed between Engine::Run calls and deliberately retains accepted input.
    // Fatal failure can likewise stop before all accepted input is processed.
    if (request.is_prefill &&
        request.finish_reason != GenerationFinishReason::Canceled &&
        request.finish_reason != GenerationFinishReason::Failed) {
      add("Turn-complete Request " + id +
          " still has unprocessed input without cancellation or failure.");
    }
  }

  return violations;
}

std::vector<InvariantViolation> ValidateFixedStateInvariants(
    const FixedStatePoolSnapshot& fixed) {
  std::vector<InvariantViolation> violations;
  const auto add = [&violations](std::string message) {
    violations.push_back(InvariantViolation{std::move(message)});
  };

  if (!fixed.healthy) {
    add("Fixed state pool is unhealthy.");
  }

  // Total accounting: every slot is free, reserved by the in-flight transaction, or committed.
  if (fixed.free_slots + fixed.reserved_slots + fixed.committed_slots != fixed.capacity) {
    add("Fixed state free (" + std::to_string(fixed.free_slots) + ") + reserved (" +
        std::to_string(fixed.reserved_slots) + ") + committed (" +
        std::to_string(fixed.committed_slots) + ") != capacity (" +
        std::to_string(fixed.capacity) + ").");
  }
  if (fixed.slots.size() != fixed.capacity) {
    add("Fixed state slot snapshot size (" + std::to_string(fixed.slots.size()) +
        ") != capacity (" + std::to_string(fixed.capacity) + ").");
  }

  size_t free_slots = 0;
  size_t reserved_slots = 0;
  size_t committed_slots = 0;
  std::unordered_set<const void*> owners;
  std::unordered_set<size_t> slot_ids;
  for (const auto& slot : fixed.slots) {
    if (slot.slot >= fixed.capacity || !slot_ids.insert(slot.slot).second) {
      add("Fixed state slot " + std::to_string(slot.slot) + " is out of range or duplicated.");
    }
    switch (slot.ownership) {
      case FixedStateSlotOwnership::Free:
        ++free_slots;
        // A free slot must retain no request identity or committed progress.
        if (slot.request_id || slot.state_generation != 0 || slot.committed_tokens != 0) {
          add("Free fixed state slot " + std::to_string(slot.slot) + " retains request state.");
        }
        break;
      case FixedStateSlotOwnership::Reserved:
        ++reserved_slots;
        break;
      case FixedStateSlotOwnership::Committed:
        ++committed_slots;
        break;
    }
    // Single ownership: an owned slot names a request, and no request owns two slots.
    if (slot.ownership != FixedStateSlotOwnership::Free) {
      if (!slot.request_id) {
        add("Owned fixed state slot " + std::to_string(slot.slot) + " has no request identity.");
      } else if (!owners.insert(slot.request_id).second) {
        add("Request " + PtrId(slot.request_id) + " owns more than one fixed state slot.");
      }
    }
  }
  if (free_slots != fixed.free_slots || reserved_slots != fixed.reserved_slots ||
      committed_slots != fixed.committed_slots) {
    add("Fixed state ownership totals do not match the slot listing.");
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

  return violations;
}

std::vector<InvariantViolation> ValidateCompositeStateInvariants(
    const PagedCacheSnapshot& cache,
    const FixedStatePoolSnapshot& fixed,
    const std::vector<RequestStateSnapshot>& requests) {
  auto violations = ValidateInvariants(cache, requests);
  auto fixed_violations = ValidateFixedStateInvariants(fixed);
  violations.insert(violations.end(), fixed_violations.begin(), fixed_violations.end());
  const auto add = [&violations](std::string message) {
    violations.push_back(InvariantViolation{std::move(message)});
  };

  // Committed paged and fixed ownership are published together in one transaction, so their
  // committed sets must match exactly and each request must sit at one token boundary in both. This
  // holds at every observation point: nothing is published until commit, so an in-flight
  // reservation's new admissions appear only as paged reservation deltas and reserved fixed slots,
  // never as committed ownership on either side.
  std::unordered_map<const void*, const RequestBlockSnapshot*> paged_owners;
  for (const auto& request : cache.requests) {
    paged_owners.emplace(request.request_id, &request);
  }
  std::unordered_map<const void*, const FixedStateSlotSnapshot*> fixed_owners;
  for (const auto& slot : fixed.slots) {
    if (slot.ownership == FixedStateSlotOwnership::Committed) {
      fixed_owners.emplace(slot.request_id, &slot);
    }
  }
  std::unordered_map<const void*, const RequestStateSnapshot*> request_states;
  for (const auto& request : requests) {
    request_states.emplace(request.request_id, &request);
  }

  for (const auto& paged : cache.requests) {
    const void* request_id = paged.request_id;
    const auto fixed_it = fixed_owners.find(request_id);
    if (fixed_it == fixed_owners.end()) {
      add("Paged cache Request " + PtrId(request_id) + " has no committed fixed state slot.");
      continue;
    }
    if (fixed_it->second->committed_tokens != paged.used_slots) {
      add("Request " + PtrId(request_id) +
          " has different paged and fixed committed token boundaries.");
    }
    const auto request_it = request_states.find(request_id);
    if (request_it != request_states.end() &&
        (request_it->second->processed_sequence_length < 0 ||
         static_cast<uint64_t>(
             request_it->second->processed_sequence_length) !=
             paged.used_slots)) {
      add("Request " + PtrId(request_id) +
          " has different request and decoder-state committed token boundaries.");
    }
  }
  for (const auto& slot : fixed.slots) {
    if (slot.ownership == FixedStateSlotOwnership::Committed &&
        paged_owners.find(slot.request_id) == paged_owners.end()) {
      add("Fixed state Request " + PtrId(slot.request_id) +
          " has no committed paged cache ownership.");
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

void ThrowIfCompositeStateInvariantsViolated(
    const PagedCacheSnapshot& cache,
    const FixedStatePoolSnapshot& fixed,
    const std::vector<RequestStateSnapshot>& requests) {
  const auto violations = ValidateCompositeStateInvariants(cache, fixed, requests);
  if (violations.empty()) {
    return;
  }

  std::ostringstream oss;
  oss << "Composite Engine invariant validation failed with " << violations.size()
      << " violation(s):";
  for (const auto& violation : violations) {
    oss << "\n  - " << violation.message;
  }
  throw std::runtime_error(oss.str());
}

}  // namespace Generators
