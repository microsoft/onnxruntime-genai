// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"
#include "scheduled_requests.h"

namespace Generators {

struct Scheduler {
  Scheduler(std::shared_ptr<Model> model);

  void AddRequest(std::shared_ptr<Request> request);

  void RemoveRequest(std::shared_ptr<Request> request);

  ScheduledRequests Schedule();

  bool HasPendingRequests() const;

 private:
  std::unordered_set<std::shared_ptr<Request>> requests_pool_;
  std::shared_ptr<Model> model_;
};

}  // namespace Generators
