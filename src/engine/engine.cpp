// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"

namespace Generators {

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
    step_plan_.transaction_id = next_transaction_id_++;
    auto planning_result = scheduler_->PlanStep(step_plan_);
    if (planning_result.capacity_deferred) {
      ++transaction_metrics_.capacity_deferrals;
    }
    if (!planning_result.executable) {
      const auto outcome = planning_result.terminal_outcome;
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

    StepTransaction transaction{step_plan_};
    std::unique_ptr<CacheStepReservation> reservation;
    try {
      reservation = cache_manager_->ReserveStep(step_plan_);
      transaction.MarkReserved();
    } catch (...) {
      ++transaction_metrics_.reservation_failures;
      transaction.RollBack();
      MarkUnhealthyAndThrow(
          StepOutcomeKind::ExecutionContractFailure,
          step_plan_.transaction_id,
          nullptr,
          "Failed to reserve the planned paged cache transaction.",
          std::current_exception());
    }

    std::vector<std::shared_ptr<Request>> requests;
    requests.reserve(step_plan_.requests.size());
    for (const auto& entry : step_plan_.requests) {
      requests.push_back(entry.request);
    }
    ScheduledRequests scheduled_requests{std::move(requests), model_};
    ExecutionContext context{step_plan_.transaction_id, &step_plan_};
    context.cache_reservation = reservation->PagedReservation();
    context.block_table_columns = step_plan_.proposed_block_table_columns;
    context.graph_capture_eligible = step_plan_.graph_capture_eligible;

    const auto rollback_reservation = [&]() {
      try {
        reservation->Release();
        transaction.RollBack();
        ++transaction_metrics_.rollbacks;
      } catch (...) {
        MarkUnhealthyAndThrow(
            StepOutcomeKind::FatalExecutionFailure,
            step_plan_.transaction_id,
            nullptr,
            "Paged cache rollback failed and the Engine is no longer healthy.",
            std::current_exception());
      }
    };

    try {
      transaction.MarkExecuting();
      model_executor_->Decode(scheduled_requests, context);
      transaction.MarkExecuted();
    } catch (const ModelExecutionError& error) {
      const auto execution_error = std::current_exception();
      rollback_reservation();
      if (error.FailureKind() == ExecutionFailureKind::RetryableAbort) {
        ++transaction_metrics_.retryable_aborts;
        throw EngineStepError{
            {StepOutcomeKind::RetryableBatchAbort,
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
      rollback_reservation();
      MarkUnhealthyAndThrow(
          StepOutcomeKind::FatalExecutionFailure,
          step_plan_.transaction_id,
          nullptr,
          "Model execution failed and the Engine is no longer healthy.",
          execution_error);
    }

    size_t checkpoint_count = 0;
    try {
      for (const auto& entry : step_plan_.requests) {
        entry.request->SaveStateForTransaction();
        ++checkpoint_count;
      }
    } catch (...) {
      try {
        while (checkpoint_count > 0) {
          step_plan_.requests[--checkpoint_count].request->RestoreStateForTransaction();
        }
        rollback_reservation();
      } catch (...) {
        MarkUnhealthyAndThrow(
            StepOutcomeKind::FatalExecutionFailure,
            step_plan_.transaction_id,
            nullptr,
            "Search checkpoint cleanup failed and the Engine is no longer healthy.",
            std::current_exception());
      }
      MarkUnhealthyAndThrow(
          StepOutcomeKind::FatalExecutionFailure,
          step_plan_.transaction_id,
          nullptr,
          "Search checkpoint creation failed and the Engine is no longer healthy.",
          std::current_exception());
    }

    try {
      const auto logits = scheduled_requests.ProcessLogits();
      step_results_.clear();
      staged_ready_requests_.clear();
      for (size_t i = 0; i < step_plan_.requests.size(); ++i) {
        auto& request = step_plan_.requests[i].request;
        step_results_.push_back(
            request->ApplyLogitsForTransaction(logits[i]));
        if (request->HasUnseenTokens() || step_results_.back().done) {
          staged_ready_requests_.push_back(request);
        }
      }
    } catch (...) {
      const auto post_processing_error = std::current_exception();
      try {
        for (const auto& entry : step_plan_.requests) {
          entry.request->RestoreStateForTransaction();
        }
        rollback_reservation();
      } catch (...) {
        MarkUnhealthyAndThrow(
            StepOutcomeKind::FatalExecutionFailure,
            step_plan_.transaction_id,
            nullptr,
            "Transaction rollback failed and the Engine is no longer healthy.",
            std::current_exception());
      }
      ++transaction_metrics_.post_processing_aborts;
      ++transaction_metrics_.retryable_aborts;
      std::string message =
          "Request post-processing failed; the batch was rolled back.";
      try {
        std::rethrow_exception(post_processing_error);
      } catch (const std::exception& error) {
        message += " Cause: ";
        message += error.what();
      } catch (...) {
        message += " Cause: non-standard exception.";
      }
      throw EngineStepError{
          {StepOutcomeKind::RetryableBatchAbort,
           step_plan_.transaction_id,
           nullptr},
          std::move(message),
      };
    }

    try {
      for (const auto& entry : step_plan_.requests) {
        entry.request->CommitStateForTransaction();
      }
      reservation->Commit();
      scheduler_->CommitStepPlan(step_plan_);
      for (size_t i = 0; i < step_plan_.requests.size(); ++i) {
        step_plan_.requests[i].request->CommitStep(
            step_plan_.requests[i], step_results_[i]);
      }
    } catch (...) {
      transaction.RollBack();
      MarkUnhealthyAndThrow(
          StepOutcomeKind::ExecutionContractFailure,
          step_plan_.transaction_id,
          nullptr,
          "Transaction commit failed and the Engine is no longer healthy.",
          std::current_exception());
    }

    ready_requests_.swap(staged_ready_requests_);
    ready_request_index_ = 0;
    transaction.Commit();
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
  fatal_cause_ = std::move(error);
  fatal_error_ = std::make_exception_ptr(EngineStepError{
      {outcome, transaction_id, request_id},
      std::move(message),
  });
  std::rethrow_exception(fatal_error_);
}

bool Engine::HasPendingRequests() const {
  return ready_request_index_ < ready_requests_.size() ||
         scheduler_->HasPendingRequests();
}

}  // namespace Generators
