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

struct Engine;

namespace detail {
uint64_t BeginRequestTurn(
    Engine& engine,
    const std::shared_ptr<Request>& request,
    std::span<const int32_t> tokens,
    std::optional<size_t> max_generated_tokens);
void CloseRequest(
    Engine& engine,
    const std::shared_ptr<Request>& request);
bool CancelRequest(
    Engine& engine,
    const std::shared_ptr<Request>& request,
    uint64_t turn_id);
}  // namespace detail

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
   * @brief Drains retained events or advances one model transaction.
   *
   * A non-empty output span first reclaims abandoned Requests. If events are retained, it copies
   * only those events. Otherwise it executes at most one static step or dynamic transaction, copies
   * the available event prefix, and retains overflow. An empty span validates the owner thread and
   * Engine health and performs no other work.
   *
   * @param events Caller-owned storage for internal Engine events.
   * @return The number of populated records in the output prefix.
   */
  size_t Run(std::span<EngineEvent> events);

  /** @brief Throws unless called from the Engine owner thread. */
  void ValidateOwnerThread() const;

  /**
   * @brief Checks if there are any pending requests in the Engine.
   *
   * Reclaims Requests whose final public handle was released before checking retained events and
   * schedulable work.
   * @return True if there are pending requests; otherwise, false.
   */
  bool HasPendingRequests();

  /**
   * @brief Speculative draft tokens a request may attach to one decode step.
   * @return Zero when this Engine cannot verify and roll back draft tokens.
   */
  size_t MaxDraftTokensPerStep() const;

 private:
  uint64_t BeginTurn(const std::shared_ptr<Request>& request,
                     std::span<const int32_t> tokens,
                     std::optional<size_t> max_generated_tokens);
  void CloseRequest(const std::shared_ptr<Request>& request);
  bool CancelRequest(const std::shared_ptr<Request>& request, uint64_t turn_id);

  void DetachRequestForTeardown(
      const std::shared_ptr<Request>& request) noexcept;
  void ReclaimAbandonedRequests();
  void CompleteNonresidentClosedRequests();
  size_t DrainPendingEvents(std::span<EngineEvent> events);
  void RetainEvent(EngineEvent event);
  void RunDynamic();
  void RunStatic();
  void AppendEventsFromStep(const std::shared_ptr<Request>& request,
                            const RequestStepResult& result);
  EngineEvent EventFromStepError(
      const EngineStepError& error,
      std::exception_ptr caught_error) noexcept;
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
  std::vector<size_t> staged_event_order_;
  std::vector<std::shared_ptr<Request>> tracked_requests_;
  std::vector<EngineEvent> pending_events_;
  std::vector<EngineEvent> staged_events_;
  size_t pending_event_index_{};
  const std::shared_ptr<std::atomic<bool>> abandonment_pending_{
      std::make_shared<std::atomic<bool>>(false)};

  friend uint64_t detail::BeginRequestTurn(
      Engine& engine,
      const std::shared_ptr<Request>& request,
      std::span<const int32_t> tokens,
      std::optional<size_t> max_generated_tokens);
  friend void detail::CloseRequest(
      Engine& engine,
      const std::shared_ptr<Request>& request);
  friend bool detail::CancelRequest(
      Engine& engine,
      const std::shared_ptr<Request>& request,
      uint64_t turn_id);
};

}  // namespace Generators
