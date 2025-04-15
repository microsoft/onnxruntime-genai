// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"

namespace Generators {

Scheduler::Scheduler(std::shared_ptr<Model> model)
    : model_{model} {}

void Scheduler::AddRequest(std::shared_ptr<Request> request) {
  requests_pool_.insert(request);
}

void Scheduler::RemoveRequest(std::shared_ptr<Request> request) {
  if (auto request_it = requests_pool_.find(request); request_it != requests_pool_.end()) {
    requests_pool_.erase(request_it);
  }
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
  std::sort(requests_to_schedule.begin(), requests_to_schedule.end(),
            [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
              return a->assigned_time_ < b->assigned_time_;
            });
  for (auto& request : requests_to_schedule) {
    request->Schedule();
    scheduled_requests.push_back(request);
  }
  return ScheduledRequests(scheduled_requests);
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
