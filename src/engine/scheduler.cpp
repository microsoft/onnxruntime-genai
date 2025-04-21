// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"

namespace Generators {

Scheduler::Scheduler(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager)
    : model_{model}, cache_manager_{cache_manager} {}

void Scheduler::AddRequest(std::shared_ptr<Request> request) {
  requests_pool_.push_back(request);
}

void Scheduler::RemoveRequest(std::shared_ptr<Request> request) {
  requests_pool_.erase(std::remove(requests_pool_.begin(), requests_pool_.end(), request), requests_pool_.end());
}

ScheduledRequests Scheduler::Schedule() {
  std::vector<std::shared_ptr<Request>> requests_to_schedule;
  std::vector<std::shared_ptr<Request>> scheduled_requests;
  for (auto& request : requests_pool_) {
    if (request->status_ == RequestStatus::Assigned) {
      requests_to_schedule.push_back(request);
    } else if (request->status_ == RequestStatus::InProgress) {
      scheduled_requests.push_back(request);
    }
  }

  if (cache_manager_->SupportsDynamicBatching()) {
    for (auto& request : requests_to_schedule) {
      if (cache_manager_->CanAllocate({request})) {
        cache_manager_->Allocate({request});
        request->Schedule();
        scheduled_requests.push_back(request);
      }
    }
  } else if (scheduled_requests.empty()) {
    constexpr size_t static_batch_size = 4;
    for (size_t batch_size = std::min(static_batch_size, requests_to_schedule.size());
         batch_size != 0; batch_size /= 2) {
      std::vector<std::shared_ptr<Request>> batch_requests(requests_to_schedule.begin(),
                                                           requests_to_schedule.begin() + batch_size);
      if (cache_manager_->CanAllocate(batch_requests)) {
        cache_manager_->Allocate(batch_requests);
        for (auto& request : batch_requests) {
          request->Schedule();
          scheduled_requests.push_back(request);
        }
        requests_to_schedule.erase(requests_to_schedule.begin(), requests_to_schedule.begin() + batch_size);
        break;
      }
    }
  }
  return ScheduledRequests(scheduled_requests, model_);
}

bool Scheduler::HasPendingRequests() const {
  for (auto& request : requests_pool_) {
    if (request->status_ != RequestStatus::Completed) {
      return true;
    }
  }
  return false;
}

}  // namespace Generators
