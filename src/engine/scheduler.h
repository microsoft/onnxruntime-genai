// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"
#include "scheduled_requests.h"
#include "cache_manager.h"
#include "recompute_preemption_policy.h"
#include "step_plan.h"

/**
 * @file scheduler.h
 * @brief Defines the Scheduler class, which is responsible for managing
 *        the scheduling of requests for model execution.
 */

namespace Generators {

// Counters describing how often the dynamic scheduler had to reclaim resident cache capacity to
// admit waiting work, and what that cost. They are pure observations: nothing in the scheduler
// reads them back to make a decision.
struct SchedulerPreemptionMetrics {
  uint64_t block_starved_passes{};  // Passes where free blocks alone held a request back.
  uint64_t preemption_passes{};     // Passes that suspended at least one resident.
  uint64_t preemptions{};           // Residents suspended.
  uint64_t declined_preemptions{};  // Starved passes no eligible victim set could relieve.
  uint64_t reclaimed_blocks{};      // Blocks returned to the pool by those suspensions.
  uint64_t recomputed_tokens{};     // Committed key-value tokens discarded and queued for rebuild.
};

struct Scheduler {
  /**
   * @brief Constructs a Scheduler instance with the specified model and cache manager.
   * @param model A shared pointer to the Model object to be used by the Scheduler.
   * @param cache_manager A shared pointer to the CacheManager for managing cache states.
   */
  explicit Scheduler(std::shared_ptr<Model> model);

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

  virtual StepPlanningResult PlanStep(
      StepPlan&, const StepPlanningLimits& = {}) {
    throw std::logic_error("Scheduler does not support transactional step planning.");
  }

  ScheduledRequests CreateScheduledRequests(const StepPlan& plan);

  /**
   * @brief Checks if the Scheduler has any pending requests.
   * @return True if there are pending requests, false otherwise.
   *
   * This function checks the internal pool of requests to determine if there are
   * any requests that have not yet been processed.
   */
  virtual bool HasPendingRequests() const = 0;

  virtual ~Scheduler() = default;

 protected:
  BatchedSampler* GetBatchedSampler() const { return batched_sampler_.get(); }
  BatchedSamplingPlan* GetBatchedSamplingPlan() { return &batched_sampling_plan_; }

 private:
  std::shared_ptr<Model> model_;
  std::unique_ptr<BatchedSampler> batched_sampler_;
  BatchedSamplingPlan batched_sampling_plan_;
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

  StepPlanningResult PlanStep(
      StepPlan& plan, const StepPlanningLimits& limits = {}) override;

  bool HasPendingRequests() const override;

  const SchedulerPreemptionMetrics& PreemptionMetrics() const {
    return preemption_metrics_;
  }

 private:
  void ReapCompletedRequests();

  // One complete planning attempt over the current residents and waiting requests. Requests listed
  // in suspended_this_step_ are skipped so a request suspended by this engine step cannot take back
  // the capacity the step just reclaimed for someone else.
  StepPlanningResult PlanStepOnce(StepPlan& plan, const StepPlanningLimits& limits);

  // Suspends residents until the blocked request's shortfall is covered. Returns the number of
  // residents suspended, which is zero when no eligible victim set could unblock it.
  size_t PreemptForBlockShortfall(const BlockCapacityShortfall& shortfall);

  static size_t CountNewAdmissions(const StepPlan& plan);

  std::shared_ptr<Model> model_;
  std::shared_ptr<CacheManager> cache_manager_;
  std::vector<std::shared_ptr<Request>> requests_pool_;
  RecomputePreemptionSettings preemption_settings_;
  SchedulerPreemptionMetrics preemption_metrics_;
  std::vector<const void*> suspended_this_step_;
};

std::unique_ptr<Scheduler> CreateScheduler(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager);

}  // namespace Generators
