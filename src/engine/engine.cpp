// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"

namespace Generators {

namespace {

std::string AddExceptionCause(std::string message, std::exception_ptr error) {
  try {
    std::rethrow_exception(error);
  } catch (const std::exception& cause) {
    message += " Cause: ";
    message += cause.what();
  } catch (...) {
    message += " Cause: non-standard exception.";
  }
  return message;
}

}  // namespace

Engine::Engine(std::shared_ptr<Model> model)
    : Engine(model, CreateDependencies(model)) {}

Engine::Engine(std::shared_ptr<Model> model, EngineDependencies dependencies)
    : model_{std::move(model)},
      cache_manager_{std::move(dependencies.cache_manager)},
      scheduler_{std::move(dependencies.scheduler)},
      model_executor_{std::move(dependencies.model_executor)} {
  // Fail fast on a missing collaborator rather than crashing later on first use.
  if (!cache_manager_) {
    throw std::runtime_error("Engine requires a non-null cache manager.");
  }
  if (!scheduler_) {
    throw std::runtime_error("Engine requires a non-null scheduler.");
  }
  if (!model_executor_) {
    throw std::runtime_error("Engine requires a non-null model executor.");
  }

  const size_t max_batch_size = cache_manager_->MaxBatchSize();
  step_plan_.requests.reserve(max_batch_size);
  step_results_.reserve(max_batch_size);
  ready_requests_.reserve(max_batch_size);
  staged_ready_requests_.reserve(max_batch_size);
}

EngineDependencies Engine::CreateDependencies(std::shared_ptr<Model> model) {
  std::shared_ptr<CacheManager> cache_manager = CacheManager::Create(model);
  auto scheduler = Scheduler::Create(model, cache_manager);
  auto model_executor = ModelExecutor::Create(model, cache_manager);
  return EngineDependencies{std::move(cache_manager), std::move(scheduler), std::move(model_executor)};
}

void Engine::AddRequest(std::shared_ptr<Request> request) {
  if (cache_manager_->SupportsDynamicBatching()) {
    request->ValidateEngineCompatibility();
  }
  request->Assign(shared_from_this());
  scheduler_->AddRequest(request);
}

void Engine::RemoveRequest(std::shared_ptr<Request> request) {
  scheduler_->RemoveRequest(request);
}

std::shared_ptr<Request> Engine::Step() {
  if (auto request = DrainReadyRequest()) {
    return request;
  }
  if (health_ == EngineHealth::Unhealthy) {
    std::rethrow_exception(fatal_error_);
  }
  return cache_manager_->SupportsDynamicBatching() ? StepDynamic() : StepStatic();
}

std::shared_ptr<Request> Engine::StepStatic() {
  while (scheduler_->HasPendingRequests()) {
    auto scheduled_requests = scheduler_->Schedule();
    model_executor_->Decode(scheduled_requests);
    scheduled_requests.GenerateNextTokens();

    for (auto& request : scheduled_requests) {
      if (request->HasUnseenTokens() || request->IsDone()) {
        ready_requests_.push_back(request);
      }
    }
    if (auto request = DrainReadyRequest()) {
      return request;
    }
  }
  return nullptr;
}

std::shared_ptr<Request> Engine::StepDynamic() {
  while (scheduler_->HasPendingRequests()) {
    // A dynamic step is a transaction with six phases:
    // plan -> reserve cache -> checkpoint request state -> execute -> stage sampled tokens -> commit.
    // Nothing becomes externally visible until the final commit succeeds.
    step_plan_.transaction_id = next_transaction_id_++;
    auto planning_result = scheduler_->PlanStep(step_plan_);
    if (planning_result.capacity_deferred) {
      ++transaction_metrics_.capacity_deferrals;
    }
    if (!planning_result.executable) {
      const auto outcome = planning_result.outcome;
      if (outcome.kind == StepOutcomeKind::NoWork &&
          !scheduler_->HasPendingRequests()) {
        return nullptr;
      }
      if (outcome.kind == StepOutcomeKind::UnserviceableRequest) {
        throw EngineStepError{
            outcome,
            "Request cannot be serviced by the configured paged cache.",
        };
      }
      if (outcome.kind == StepOutcomeKind::CapacityDeferred) {
        throw EngineStepError{
            outcome,
            "Paged cache capacity deferred all pending requests.",
        };
      }
      MarkUnhealthyAndThrow(
          StepOutcomeKind::ExecutionContractFailure,
          step_plan_.transaction_id,
          outcome.request_id,
          "Dynamic scheduler returned no executable work while requests remain pending.",
          std::make_exception_ptr(std::logic_error{
              "Invalid dynamic scheduler planning outcome."}));
    }

    std::unique_ptr<CacheStepReservation> reservation;
    try {
      // Reserve every block needed by the complete plan up front. The reservation can be used to
      // build model inputs, but it does not alter committed block tables until Commit().
      reservation = cache_manager_->ReserveStep(step_plan_);
    } catch (...) {
      ++transaction_metrics_.reservation_failures;
      MarkUnhealthyAndThrow(
          StepOutcomeKind::ExecutionContractFailure,
          step_plan_.transaction_id,
          nullptr,
          "Failed to reserve the planned paged cache transaction.",
          std::current_exception());
    }

    auto scheduled_requests =
        scheduler_->CreateScheduledRequests(step_plan_);
    ExecutionContext context{&step_plan_};
    context.cache_reservation = reservation->PagedReservation();

    bool request_transaction_active = false;
    const auto rollback_transaction = [&]() {
      // Request/search state and paged-cache state are checkpointed separately. Both must be
      // restored so a retry observes exactly the state that existed before this Step() call.
      std::exception_ptr rollback_error;
      if (request_transaction_active) {
        try {
          scheduled_requests.RestoreStateForTransaction();
        } catch (...) {
          rollback_error = std::current_exception();
        }
        request_transaction_active = false;
      }
      try {
        reservation->Release();
      } catch (...) {
        if (!rollback_error)
          rollback_error = std::current_exception();
      }
      ++transaction_metrics_.rollbacks;
      if (rollback_error) {
        MarkUnhealthyAndThrow(
            StepOutcomeKind::FatalExecutionFailure,
            step_plan_.transaction_id,
            nullptr,
            "Transaction rollback failed and the Engine is no longer healthy.",
            rollback_error);
      }
    };

    try {
      for (const auto& entry : step_plan_.requests) {
        entry.request->ValidateEngineCompatibility();
      }
    } catch (...) {
      const auto validation_error = std::current_exception();
      rollback_transaction();
      ++transaction_metrics_.post_processing_aborts;
      ++transaction_metrics_.retryable_aborts;
      throw EngineStepError{
          {StepOutcomeKind::RetryableBatchAbort,
           step_plan_.transaction_id,
           nullptr},
          AddExceptionCause(
              "Request validation failed; the batch was rolled back.",
              validation_error),
      };
    }

    try {
      // Sampling mutates each request's Search state. Checkpoint it before the model run so failures
      // in execution or post-processing can discard the whole batch rather than partially advancing
      // whichever requests happened to finish first.
      scheduled_requests.BeginTransaction();
      request_transaction_active = true;
      model_executor_->Decode(scheduled_requests, context);
    } catch (const ModelExecutionError& error) {
      const auto execution_error = std::current_exception();
      rollback_transaction();
      if (error.FailureKind() == ExecutionFailureKind::RetryableAbort ||
          error.FailureKind() == ExecutionFailureKind::CapacityExceeded) {
        ++transaction_metrics_.retryable_aborts;
        throw EngineStepError{
            {error.FailureKind() == ExecutionFailureKind::CapacityExceeded
                 ? StepOutcomeKind::ExecutionCapacityExceeded
                 : StepOutcomeKind::RetryableBatchAbort,
             step_plan_.transaction_id,
             nullptr},
            error.what(),
        };
      }
      MarkUnhealthyAndThrow(
          StepOutcomeKind::FatalExecutionFailure,
          step_plan_.transaction_id,
          nullptr,
          error.what(),
          execution_error);
    } catch (...) {
      const auto execution_error = std::current_exception();
      rollback_transaction();
      MarkUnhealthyAndThrow(
          StepOutcomeKind::FatalExecutionFailure,
          step_plan_.transaction_id,
          nullptr,
          "Model execution failed and the Engine is no longer healthy.",
          execution_error);
    }

    try {
      // Turn the final logits row for each packed request into a staged next-token result. Request
      // counters, host token mirrors, and completion status still remain unchanged in this phase.
      scheduled_requests.GenerateNextTokensForTransaction(
          step_plan_, step_results_);
      staged_ready_requests_.clear();
      for (size_t i = 0; i < step_plan_.requests.size(); ++i) {
        auto& request = step_plan_.requests[i].request;
        if (step_results_[i].token_appended || step_results_[i].done) {
          staged_ready_requests_.push_back(request);
        }
      }
    } catch (...) {
      const auto post_processing_error = std::current_exception();
      rollback_transaction();
      ++transaction_metrics_.post_processing_aborts;
      ++transaction_metrics_.retryable_aborts;
      throw EngineStepError{
          {StepOutcomeKind::RetryableBatchAbort,
           step_plan_.transaction_id,
           nullptr},
          AddExceptionCause(
              "Request post-processing failed; the batch was rolled back.",
              post_processing_error),
      };
    }

    try {
      // Commit order is deliberate: make staged search state durable, publish cache growth, then
      // advance the lightweight Request bookkeeping that readers observe.
      scheduled_requests.CommitStateForTransaction();
      request_transaction_active = false;
      reservation->Commit();
      for (size_t i = 0; i < step_plan_.requests.size(); ++i) {
        step_plan_.requests[i].request->CommitStep(
            step_plan_.requests[i], step_results_[i]);
      }
    } catch (...) {
      MarkUnhealthyAndThrow(
          StepOutcomeKind::ExecutionContractFailure,
          step_plan_.transaction_id,
          nullptr,
          "Transaction commit failed and the Engine is no longer healthy.",
          std::current_exception());
    }

    // Step() returns one ready request at a time. Keep the rest queued so draining this committed
    // batch does not trigger another model execution.
    ready_requests_.swap(staged_ready_requests_);
    ready_request_index_ = 0;
    ++transaction_metrics_.committed_steps;
    if (auto request = DrainReadyRequest()) {
      return request;
    }
  }
  return nullptr;
}

std::shared_ptr<Request> Engine::DrainReadyRequest() {
  if (ready_request_index_ == ready_requests_.size()) {
    ready_requests_.clear();
    ready_request_index_ = 0;
    return nullptr;
  }
  return ready_requests_[ready_request_index_++];
}

[[noreturn]] void Engine::MarkUnhealthyAndThrow(
    StepOutcomeKind outcome,
    StepTransactionId transaction_id,
    const void* request_id,
    std::string message,
    std::exception_ptr error) {
  health_ = EngineHealth::Unhealthy;
  if (outcome == StepOutcomeKind::FatalExecutionFailure ||
      outcome == StepOutcomeKind::ExecutionContractFailure) {
    ++transaction_metrics_.fatal_execution_failures;
  }
  fatal_error_ = std::make_exception_ptr(EngineStepError{
      {outcome, transaction_id, request_id},
      AddExceptionCause(std::move(message), error),
  });
  std::rethrow_exception(fatal_error_);
}

bool Engine::HasPendingRequests() const {
  return ready_request_index_ < ready_requests_.size() ||
         scheduler_->HasPendingRequests();
}

}  // namespace Generators
