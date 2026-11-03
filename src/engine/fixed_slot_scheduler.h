// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <utility>

#include "cache_manager.h"
#include "scheduler.h"

namespace Generators {

// Fixed storage lifecycle only. Implementations retain graph bindings, buffers, and phase constraints.
struct FixedSlotSession {
  virtual ~FixedSlotSession() = default;
  virtual size_t MaxRequestLength() const = 0;
  virtual size_t MaxBatchSize() const = 0;
  // Quiesce all work referencing a slot and inspect pending status before returning it for reuse.
  virtual void ReleaseSlot(size_t slot) = 0;
  virtual void ReleaseSlotForTeardown(size_t slot) noexcept = 0;
};

// Static-cache contract for 1..MaxBatchSize() slots backed by a FixedSlotSession.
struct FixedSlotCacheManager final : CacheManager {
  FixedSlotCacheManager(std::shared_ptr<Model> model, std::shared_ptr<FixedSlotSession> session);

  bool CanAllocate(const std::vector<std::shared_ptr<Request>>& requests) const override;
  void Allocate(const std::vector<std::shared_ptr<Request>>& requests) override;
  void Step() override {}
  void Deallocate(std::vector<std::shared_ptr<Request>>& requests) override;
  void ValidateRewind(const std::shared_ptr<Request>& request) const override;
  void ReleaseForRewind(const std::shared_ptr<Request>& request) override;
  void DetachRequestForTeardown(const std::shared_ptr<Request>& request) noexcept override;
  bool SupportsDynamicBatching() const override { return false; }
  bool SupportsContinuousMultiSlotDecoding() const override { return max_slots_ > 1; }
  size_t MaxBatchSize() const override { return max_slots_; }
  size_t MaxStaticBatchSize() const override { return max_slots_; }
  size_t MaxStaticRequestLength() const override { return session_->MaxRequestLength(); }
  std::vector<std::shared_ptr<Request>> AllocatedRequests() const override;
  bool IsResident(const std::shared_ptr<Request>& request) const override;
  size_t ResidentRequestCount() const override;
  size_t SlotOf(const std::shared_ptr<Request>& request) const;

 private:
  std::shared_ptr<FixedSlotSession> session_;
  size_t max_slots_{1};
  std::vector<std::shared_ptr<Request>> slots_;
};

struct FixedSlotScheduler final : Scheduler {
  FixedSlotScheduler(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager)
      : Scheduler{model}, model_{std::move(model)}, cache_manager_{std::move(cache_manager)} {}

  bool SupportsIndependentSlotAdmission() const override {
    return cache_manager_->SupportsContinuousMultiSlotDecoding();
  }

  void AddRequest(std::shared_ptr<Request> request) override {
    if (request->PrefillChunkSize().value_or(0) != 0) {
      throw std::runtime_error(
          "search.chunk_size requires dynamic batching; the static batch scheduler cannot chunk a "
          "prefill.");
    }
    if (std::find(requests_pool_.begin(), requests_pool_.end(), request) != requests_pool_.end()) {
      return;
    }
    requests_pool_.reserve(requests_pool_.size() + 1);
    auto sampling_state = CreateSamplingState(*request);
    request->CommitSamplingState(std::move(sampling_state));
    requests_pool_.push_back(std::move(request));
  }

  void RemoveRequest(std::shared_ptr<Request> request) override {
    if (cache_manager_->IsResident(request)) {
      std::vector<std::shared_ptr<Request>> to_deallocate{request};
      cache_manager_->Deallocate(to_deallocate);
    }
    Forget(request);
  }

  void DetachRequestForTeardown(const std::shared_ptr<Request>& request) noexcept override {
    cache_manager_->DetachRequestForTeardown(request);
    Forget(request);
  }

  ScheduledRequests Schedule() override {
    for (const auto& req : cache_manager_->AllocatedRequests()) {
      if (IsClosed(req->status_)) {
        cache_manager_->DetachRequestForTeardown(req);
        Forget(req);
      } else if (IsQueued(req->status_)) {
        req->Schedule();
      }
    }

    std::vector<std::shared_ptr<Request>> requests_to_schedule;
    for (const auto& request : requests_pool_) {
      if (IsQueued(request->status_) && !cache_manager_->IsResident(request)) {
        requests_to_schedule.push_back(request);
      }
    }

    for (const auto& request : requests_to_schedule) {
      if (!cache_manager_->CanAllocate({request})) {
        break;
      }
      if (cache_manager_->ResidentRequestCount() >= cache_manager_->MaxBatchSize()) {
        for (const auto& resident : cache_manager_->AllocatedRequests()) {
          if (IsTurnComplete(resident->status_) || IsClosed(resident->status_)) {
            cache_manager_->DetachRequestForTeardown(resident);
            if (!StaysEnrolledWhenDisplaced(*resident)) Forget(resident);
            break;
          }
        }
      }
      cache_manager_->Allocate({request});
      request->Schedule();
    }

    ScheduledRequests scheduled_requests(cache_manager_->AllocatedRequests(), model_, GetBatchedSampler(),
                                         GetBatchedSamplingPlan());
    return scheduled_requests;
  }

  NontransactionalSchedulingResult ScheduleStep() override {
    auto scheduled = Schedule();
    if (!scheduled) {
      return {{HasPendingRequests() ? StepOutcomeKind::CapacityDeferred : StepOutcomeKind::NoWork, 0, nullptr},
              std::nullopt};
    }
    return {{StepOutcomeKind::Committed, 0, nullptr}, std::move(scheduled)};
  }

  bool HasPendingRequests() const override {
    for (const auto& request : requests_pool_) {
      if (IsExecutable(request->status_)) {
        return true;
      }
    }
    return false;
  }

 private:
  void Forget(const std::shared_ptr<Request>& request) {
    requests_pool_.erase(std::remove(requests_pool_.begin(), requests_pool_.end(), request), requests_pool_.end());
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<CacheManager> cache_manager_;
  std::vector<std::shared_ptr<Request>> requests_pool_;
};

}  // namespace Generators
