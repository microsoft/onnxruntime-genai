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
  /**
   * @brief Constructs a Scheduler instance with the specified model and cache manager.
   * @param model A shared pointer to the Model object to be used by the Scheduler.
   * @param cache_manager A shared pointer to the CacheManager for managing cache states.
   */
  Scheduler(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager);

  /**
   * @brief Adds a request to the Scheduler for processing.
   * @param request A shared pointer to the Request object to be added.
   *
   * This function adds the request to the internal pool of requests and marks it
   * as pending for scheduling.
   */
  void AddRequest(std::shared_ptr<Request> request);

  /**
   * @brief Removes a request from the Scheduler.
   * @param request A shared pointer to the Request object to be removed.
   *
   * This function marks the request for removal and cleans up any associated resources.
   */
  void RemoveRequest(std::shared_ptr<Request> request);

  /**
   * @brief Steps through the Scheduler to process requests.
   * @return An instance of ScheduledRequests struct.
   *
   * This function processes the requests in the pool, scheduling them for execution
   * and returning any requests that have been scheduled.
   */
  ScheduledRequests Schedule();

  /**
   * @brief Checks if the Scheduler has any pending requests.
   * @return True if there are pending requests, false otherwise.
   *
   * This function checks the internal pool of requests to determine if there are
   * any requests that have not yet been processed.
   */
  bool HasPendingRequests() const;

 private:
  std::shared_ptr<Model> model_;
  std::shared_ptr<CacheManager> cache_manager_;
  std::vector<std::shared_ptr<Request>> requests_pool_;
  std::set<std::shared_ptr<Request>> to_be_removed_requests_;
};

}  // namespace Generators
