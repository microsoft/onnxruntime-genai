// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"
#include "model_executor.h"
#include "scheduler.h"

/**
 * @file engine.h
 * @brief Defines the Engine class, which serves as the core component for managing
 *        model executions and request handling. It provides a way to continuously
 *        add, remove, and process requests by dynamically scheduling them.
 */

namespace Generators {

enum class EngineHealth {
  Healthy,
  Unhealthy,
};

/**
 * @struct EngineDependencies
 * @brief Bundle of the collaborators the Engine drives.
 *
 * The Engine constructor assembles these from the model via CreateDependencies and forwards them
 * to the dependency-injecting constructor, so there is a single construction flow. The injecting
 * constructor accepts a pre-assembled bundle, which also lets a caller supply alternative
 * collaborators. The cache manager, scheduler, and model executor are all abstract interfaces, so
 * an alternative implementation of any of them can be supplied in place of the production one.
 * This is a composition seam, not a setter: the dependencies are supplied once at construction and
 * never replaced mid-run.
 */
struct EngineDependencies {
  std::shared_ptr<CacheManager> cache_manager;
  std::unique_ptr<Scheduler> scheduler;
  std::unique_ptr<ModelExecutor> model_executor;
};

struct EngineTransactionMetrics {
  uint64_t committed_steps{};
  uint64_t capacity_deferrals{};
  uint64_t reservation_failures{};
  uint64_t rollbacks{};
  uint64_t retryable_aborts{};
  uint64_t post_processing_aborts{};
  uint64_t fatal_execution_failures{};
};

/**
 * @class Engine
 * @brief The Engine class is responsible for managing requests, executing models,
 *        and coordinating scheduling and caching mechanisms.
 *
 * The Engine class is designed to handle multiple requests concurrently, allowing
 * for efficient execution of models by dynamically batching requests.
 * It is the entry point for adding and processing requests.
 */
struct Engine : std::enable_shared_from_this<Engine>,
                LeakChecked<Engine>,
                ExternalRefCounted<Engine> {
  /**
   * @brief Constructs an Engine instance with the specified model.
   * @param model A shared pointer to the Model object to be used by the Engine
   *              and its components.
   *
   * Assembles the Engine's scheduler, cache manager, and model executor for the model via
   * CreateDependencies and delegates to the dependency-injecting constructor. Behavior is identical
   * to constructing those collaborators inline.
   */
  Engine(std::shared_ptr<Model> model);

  /**
   * @brief Dependency-injecting constructor.
   * @param model A shared pointer to the Model object to be used by the Engine.
   * @param dependencies A pre-assembled bundle of the Engine's collaborators.
   *
   * Takes ownership of the supplied collaborators as-is. The model-only constructor delegates here
   * with the default bundle; callers may also supply an alternative bundle.
   */
  Engine(std::shared_ptr<Model> model, EngineDependencies dependencies);

  /**
   * @brief Assembles the Engine's collaborators (cache manager, scheduler, model executor) for a
   *        model, in the order they depend on one another.
   */
  static EngineDependencies CreateDependencies(std::shared_ptr<Model> model);

  /**
   * @brief Adds a request to the Engine for processing.
   * @param request A shared pointer to the Request object to be added.
   */
  void AddRequest(std::shared_ptr<Request> request);

  /**
   * @brief Removes a previously added request from the Engine.
   * @param request A shared pointer to the Request object to be removed.
   */
  void RemoveRequest(std::shared_ptr<Request> request);

  /**
   * @brief Advances the state of a subset of the Requests the Engine is currently
   *        serving.
   *
   * This function schedules the execution of the model for the subset of requests
   * that are ready to be processed (as determined by the scheduling strategy).
   * Once these requests are scheduled, the Engine offloads the execution to the
   * model executor and updates the requests' states with the newly generated
   * tokens.
   */
  std::shared_ptr<Request> Step();

  /**
   * @brief Checks if there are any pending requests in the Engine.
   * @return True if there are pending requests; otherwise, false.
   */
  bool HasPendingRequests() const;

  /**
   * @brief Transaction counters for this Engine's dynamic steps.
   */
  const EngineTransactionMetrics& TransactionMetrics() const { return transaction_metrics_; }

  /**
   * @brief Prefix-cache counters, or null when the cache does not content-address its blocks.
   */
  const PrefixCacheMetrics* PrefixCacheStats() const { return cache_manager_->PrefixMetrics(); }

 private:
  std::shared_ptr<Request> DrainReadyRequest();
  std::shared_ptr<Request> StepDynamic();
  std::shared_ptr<Request> StepStatic();
  [[noreturn]] void MarkUnhealthyAndThrow(StepOutcomeKind outcome,
                                          StepTransactionId transaction_id,
                                          const void* request_id,
                                          std::string message,
                                          std::exception_ptr error);

  std::shared_ptr<Model> model_;                   // The model used by the Engine.
  std::shared_ptr<CacheManager> cache_manager_;    // The cache manager for handling cached data.
  std::unique_ptr<Scheduler> scheduler_;           // The scheduler responsible for managing execution order.
  std::unique_ptr<ModelExecutor> model_executor_;  // The executor responsible for running the model.
  EngineHealth health_{EngineHealth::Healthy};
  std::exception_ptr fatal_error_;
  StepTransactionId next_transaction_id_{1};
  EngineTransactionMetrics transaction_metrics_;
  StepPlan step_plan_;
  std::vector<RequestStepResult> step_results_;
  std::vector<std::shared_ptr<Request>> ready_requests_;
  std::vector<std::shared_ptr<Request>> staged_ready_requests_;
  size_t ready_request_index_{};
};

}  // namespace Generators
