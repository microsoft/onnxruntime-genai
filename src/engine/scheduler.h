// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"
#include "scheduled_requests.h"
#include "cache_manager.h"

/**
 * @file scheduler.h
 * @brief Defines the Scheduler class, which is responsible for managing
 *        the scheduling of requests for model execution.
 */

namespace Generators {

struct Scheduler {
  Scheduler(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager);

  void AddRequest(std::shared_ptr<Request> request);

  void RemoveRequest(std::shared_ptr<Request> request);

  ScheduledRequests Schedule();

  bool HasPendingRequests() const;

 private:
  std::shared_ptr<Model> model_;
  std::shared_ptr<CacheManager> cache_manager_;
  std::vector<std::shared_ptr<Request>> requests_pool_;
  std::set<std::shared_ptr<Request>> to_be_removed_requests_;
};

}  // namespace Generators
