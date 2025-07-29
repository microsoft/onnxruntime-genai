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
  Scheduler() = default;

  static std::unique_ptr<Scheduler> Create(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager);

  /**
   * @brief Adds a request to the Scheduler for processing.
   * @param request A shared pointer to the Request object to be added.
   *
   * This function adds the request to the internal pool of requests and marks it
   * as pending for scheduling.
   */
  virtual void AddRequest(std::shared_ptr<Request> request) = 0;

  /**
   * @brief Removes a request from the Scheduler.
   * @param request A shared pointer to the Request object to be removed.
   *
   * This function marks the request for removal and cleans up any associated resources.
   */
  virtual void RemoveRequest(std::shared_ptr<Request> request) = 0;

  /**
   * @brief Steps through the Scheduler to process requests.
   * @return An instance of ScheduledRequests struct.
   *
   * This function processes the requests in the pool, scheduling them for execution
   * and returning any requests that have been scheduled.
   */
  virtual ScheduledRequests Schedule() = 0;

  /**
   * @brief Checks if the Scheduler has any pending requests.
   * @return True if there are pending requests, false otherwise.
   *
   * This function checks the internal pool of requests to determine if there are
   * any requests that have not yet been processed.
   */
  virtual bool HasPendingRequests() const = 0;

  virtual ~Scheduler() = default;
};

struct StaticBatchScheduler : Scheduler {
  StaticBatchScheduler(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager);

  void AddRequest(std::shared_ptr<Request> request) override;

  void RemoveRequest(std::shared_ptr<Request> request) override;

  ScheduledRequests Schedule() override;

  bool HasPendingRequests() const override;

 private:
  std::shared_ptr<Model> model_;
  std::shared_ptr<CacheManager> cache_manager_;
  std::vector<std::shared_ptr<Request>> requests_pool_;
  std::set<std::shared_ptr<Request>> to_be_removed_requests_;
};

struct DynamicBatchScheduler : Scheduler {
  DynamicBatchScheduler(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager);

  void AddRequest(std::shared_ptr<Request> request) override;

  void RemoveRequest(std::shared_ptr<Request> request) override;

  ScheduledRequests Schedule() override;

  bool HasPendingRequests() const override;

 private:
  std::shared_ptr<Model> model_;
  std::shared_ptr<CacheManager> cache_manager_;
  std::vector<std::shared_ptr<Request>> requests_pool_;
};

std::unique_ptr<Scheduler> CreateScheduler(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager);

}  // namespace Generators
