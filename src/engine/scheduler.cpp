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
  // For statically batched requests, memory is managed as a single block for the entire batch,
  // so individual requests cannot be deallocated until the whole batch is completed.
  // Therefore, deallocation is only performed for dynamically batched requests below.
  // we simply mark the request to be removed and it will be deallocated when the
  // entire batch is completed.
  to_be_removed_requests_.insert(request);
}

ScheduledRequests Scheduler::Schedule() {
  std::vector<std::shared_ptr<Request>> requests_to_schedule;
  for (auto& request : requests_pool_) {
    if (request->status_ == RequestStatus::Assigned) {
      requests_to_schedule.push_back(request);
    }
  }

  constexpr size_t static_batch_size = 4;
  for (size_t batch_size = std::min(static_batch_size, requests_to_schedule.size());
       batch_size != 0; batch_size /= 2) {
    std::vector<std::shared_ptr<Request>> batch_requests(requests_to_schedule.begin(),
                                                         requests_to_schedule.begin() + batch_size);
    if (cache_manager_->CanAllocate(batch_requests)) {
      // Before allocating, we need to ensure that the existing requests in the cache manager
      // are complete and that if they were previously removed from the engine, they are no longer
      // in the requests pool.
      for (auto& request : cache_manager_->AllocatedRequests()) {
        if (request->status_ != RequestStatus::Completed && to_be_removed_requests_.count(request)) {
          throw std::runtime_error("Encountered a request that was removed from the engine but was not completed.");
        }
        requests_pool_.erase(std::remove(requests_pool_.begin(), requests_pool_.end(), request), requests_pool_.end());
      }

      cache_manager_->Allocate(batch_requests);
      for (auto& request : batch_requests) {
        request->Schedule();
      }
      requests_to_schedule.erase(requests_to_schedule.begin(), requests_to_schedule.begin() + batch_size);
      break;
    }
  }

  ScheduledRequests scheduled_requests(cache_manager_->AllocatedRequests(), model_);

  if (!scheduled_requests) {
    throw std::runtime_error("Unable to schedule requests: no requests available or all requests are completed.");
  }

  return scheduled_requests;
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
