// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "fixed_slot_scheduler.h"

#include <algorithm>
#include <stdexcept>

namespace Generators {

// ------------------------------------------------------------------------------------------------
// Cache manager
// ------------------------------------------------------------------------------------------------
FixedSlotCacheManager::FixedSlotCacheManager(std::shared_ptr<Model> model, std::shared_ptr<FixedSlotSession> session)
    : CacheManager{std::move(model)},
      session_{std::move(session)},
      max_slots_{session_ ? session_->MaxBatchSize() : 1},
      slots_(max_slots_, nullptr) {
  if (!session_) {
    throw std::invalid_argument("FixedSlotCacheManager requires a session.");
  }
}

bool FixedSlotCacheManager::CanAllocate(const std::vector<std::shared_ptr<Request>>& requests) const {
  if (requests.empty() || requests.size() > max_slots_) {
    return false;
  }
  // A slot is free, or holds a row whose turn is over and can be replaced.
  size_t available = 0;
  for (const auto& resident : slots_) {
    if (!resident || IsTurnComplete(resident->status_) || IsClosed(resident->status_)) {
      ++available;
    }
  }
  return requests.size() <= available;
}

void FixedSlotCacheManager::Allocate(const std::vector<std::shared_ptr<Request>>& requests) {
  if (!CanAllocate(requests)) {
    throw std::runtime_error(max_slots_ == 1 ? "The fixed-slot backend serves one Request at a time."
                                             : "The fixed-slot backend cannot allocate the requested batch.");
  }
  for (const auto& req : requests) {
    if (IsResident(req)) {
      continue;
    }
    size_t target_slot = max_slots_;
    for (size_t s = 0; s < max_slots_; ++s) {
      if (!slots_[s] || IsClosed(slots_[s]->status_)) {
        target_slot = s;
        break;
      }
    }
    if (target_slot == max_slots_) {
      for (size_t s = 0; s < max_slots_; ++s) {
        if (IsTurnComplete(slots_[s]->status_)) {
          target_slot = s;
          break;
        }
      }
    }
    if (target_slot == max_slots_) {
      throw std::runtime_error("The fixed-slot backend has no free slot.");
    }
    if (slots_[target_slot]) {
      session_->ReleaseSlotForTeardown(target_slot);
    }
    slots_[target_slot] = req;
  }
}

void FixedSlotCacheManager::Deallocate(std::vector<std::shared_ptr<Request>>& requests) {
  if (requests.empty()) {
    throw std::runtime_error("Cannot dynamically deallocate statically batched requests.");
  }
  for (const auto& req : requests) {
    if (!IsResident(req)) {
      throw std::runtime_error("Cannot dynamically deallocate statically batched requests.");
    }
  }
  for (const auto& req : requests) {
    const size_t s = SlotOf(req);
    session_->ReleaseSlot(s);
    slots_[s].reset();
  }
}

void FixedSlotCacheManager::ValidateRewind(const std::shared_ptr<Request>&) const {
  // Each resident row owns its dedicated slot, so releasing it for a replay never affects a peer.
}

void FixedSlotCacheManager::ReleaseForRewind(const std::shared_ptr<Request>& request) {
  if (!IsResident(request)) {
    return;
  }
  const size_t s = SlotOf(request);
  session_->ReleaseSlot(s);
  slots_[s].reset();
}

void FixedSlotCacheManager::DetachRequestForTeardown(const std::shared_ptr<Request>& request) noexcept {
  if (!IsResident(request)) {
    return;
  }
  const size_t s = SlotOf(request);
  session_->ReleaseSlotForTeardown(s);
  slots_[s].reset();
}

std::vector<std::shared_ptr<Request>> FixedSlotCacheManager::AllocatedRequests() const {
  std::vector<std::shared_ptr<Request>> allocated;
  allocated.reserve(max_slots_);
  for (const auto& req : slots_) {
    if (req) {
      allocated.push_back(req);
    }
  }
  return allocated;
}

bool FixedSlotCacheManager::IsResident(const std::shared_ptr<Request>& request) const {
  return request && std::find(slots_.begin(), slots_.end(), request) != slots_.end();
}

size_t FixedSlotCacheManager::ResidentRequestCount() const {
  return static_cast<size_t>(std::count_if(slots_.begin(), slots_.end(), [](const auto& req) { return req != nullptr; }));
}

size_t FixedSlotCacheManager::SlotOf(const std::shared_ptr<Request>& request) const {
  const auto it = std::find(slots_.begin(), slots_.end(), request);
  if (it == slots_.end()) throw std::logic_error("Request has no fixed-slot residency lease.");
  return static_cast<size_t>(it - slots_.begin());
}

// ------------------------------------------------------------------------------------------------
// Scheduler (multi-slot devices only: a device with one slot uses the Engine's static scheduler)
// ------------------------------------------------------------------------------------------------

}  // namespace Generators
