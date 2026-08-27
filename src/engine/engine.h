// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"
#include "model_executor.h"
#include "scheduler.h"

#include <thread>

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

enum class EngineErrorCode : uint32_t {
  None = 0,
  CapacityDeferred = 1,
  ExecutionCapacityExceeded = 2,
  RetryableExecution = 3,
  RequestUnserviceable = 4,
  EngineContractFailure = 5,
  EngineExecutionFailure = 6,
};

enum EngineEventFlag : uint32_t {
  EngineEventFlagNone = 0,
  EngineEventFlagToken = 1u << 0,
  EngineEventFlagTurnFinished = 1u << 1,
  EngineEventFlagCapacityBlocked = 1u << 2,
  EngineEventFlagFailed = 1u << 3,
  EngineEventFlagRetryable = 1u << 4,
};

struct TurnUsage {
  uint64_t prompt_tokens{};
  uint64_t generated_tokens{};
  uint64_t cached_prompt_tokens{};
};

struct EngineEvent {
  std::shared_ptr<Request> request;
  uint64_t turn_id{};
  uint32_t flags{};
  int32_t token{};
  GenerationFinishReason finish_reason{GenerationFinishReason::None};
  EngineErrorCode error_code{EngineErrorCode::None};
  TurnUsage usage{};
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
  ~Engine();

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

  std::shared_ptr<Request> CreateRequest(const GeneratorParams& params,
                                         size_t max_total_tokens);
  std::shared_ptr<Request> CreateRequest(const GeneratorParams& params) {
    return CreateRequest(
        params, static_cast<size_t>(params.search.max_length));
  }

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
  EngineEvent Run();

  /**
   * @brief Checks if there are any pending requests in the Engine.
   *
   * Reclaims Requests whose final public handle was released before checking retained events and
   * schedulable work.
   * @return True if there are pending requests; otherwise, false.
   */
  bool HasPendingRequests();

 private:
  void ValidateOwnerThread() const;
  uint64_t BeginTurn(const std::shared_ptr<Request>& request,
                     std::span<const int32_t> tokens,
                     std::optional<size_t> max_generated_tokens);
  void CloseRequest(const std::shared_ptr<Request>& request);
  bool CancelRequest(const std::shared_ptr<Request>& request, uint64_t turn_id);
  void ReclaimAbandonedRequests();
  EngineEvent DrainPendingEvent();
  EngineEvent RunDynamic();
  EngineEvent RunStatic();
  EngineEvent EventFromStep(const std::shared_ptr<Request>& request,
                            const RequestStepResult& result) const;
  EngineEvent EventFromStepError(const EngineStepError& error);
  std::shared_ptr<Request> FindTrackedRequest(const void* request_id) const;
  EngineEvent FailUnserviceableRequest(const void* request_id);
  void ValidateRequestCanContinue(
      const std::shared_ptr<Request>& request,
      bool allow_nonresident = false) const;
  [[noreturn]] void HandleContinuationRestoreFailure(
      const std::shared_ptr<Request>& request,
      std::exception_ptr append_error,
      std::exception_ptr restore_error);
  [[noreturn]] void MarkUnhealthyAndThrow(StepOutcomeKind outcome,
                                          StepTransactionId transaction_id,
                                          const void* request_id,
                                          std::string message,
                                          std::exception_ptr error);

  std::shared_ptr<Model> model_;                   // The model used by the Engine.
  std::shared_ptr<CacheManager> cache_manager_;    // The cache manager for handling cached data.
  std::unique_ptr<Scheduler> scheduler_;           // The scheduler responsible for managing execution order.
  std::unique_ptr<ModelExecutor> model_executor_;  // The executor responsible for running the model.
  const std::thread::id owner_thread_{std::this_thread::get_id()};
  EngineHealth health_{EngineHealth::Healthy};
  std::exception_ptr fatal_error_;
  StepTransactionId next_transaction_id_{1};
  EngineTransactionMetrics transaction_metrics_;
  StepPlan step_plan_;
  std::vector<RequestStepResult> step_results_;
  std::vector<std::weak_ptr<Request>> tracked_requests_;
  std::vector<EngineEvent> pending_events_;
  std::vector<EngineEvent> staged_events_;
  size_t pending_event_index_{};

  friend struct Request;
};

}  // namespace Generators
