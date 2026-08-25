// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"

#include "../search.h"

namespace Generators {

namespace {

std::shared_ptr<GeneratorParams> CloneRequestParams(
    const GeneratorParams& source,
    const Model& model) {
  auto copy = std::make_shared<GeneratorParams>(model);
  copy->search = source.search;
  copy->speculative = source.speculative;
  copy->max_batch_size = source.max_batch_size;
  copy->use_graph_capture = source.use_graph_capture;
  copy->max_graph_capture_length = source.max_graph_capture_length;
  copy->use_multi_profile = source.use_multi_profile;
  copy->p_device = source.p_device;
  copy->guidance_type = source.guidance_type;
  copy->guidance_data = source.guidance_data;
  copy->guidance_ff_tokens_enabled = source.guidance_ff_tokens_enabled;
  return copy;
}

DeviceSpan<int32_t> AllocateOnDevice(
    GeneratorParams& params,
    std::span<const int32_t> input_ids) {
  auto device_tokens = params.p_device->Allocate<int32_t>(input_ids.size());
  auto cpu_tokens = device_tokens.CpuSpan();
  std::copy(input_ids.begin(), input_ids.end(), cpu_tokens.begin());
  device_tokens.CopyCpuToDevice();
  return device_tokens;
}

void ValidateAppendLength(const GeneratorParams& params,
                          size_t current_sequence_length,
                          size_t token_count) {
  const size_t max_length = static_cast<size_t>(params.search.max_length);
  if (current_sequence_length >= max_length ||
      token_count >= max_length - current_sequence_length) {
    throw std::runtime_error(
        "Input tokens must leave room for at least one generated token before max_length (" +
        std::to_string(params.search.max_length) + ").");
  }
}

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

Engine::~Engine() {
  while (!tracked_requests_.empty()) {
    auto request = tracked_requests_.back().lock();
    tracked_requests_.pop_back();
    if (!request) {
      continue;
    }
    if (IsClosed(request->status_)) {
      continue;
    }
    try {
      CloseRequest(request);
    } catch (...) {
      request->CompleteClose();
    }
  }
}

EngineDependencies Engine::CreateDependencies(std::shared_ptr<Model> model) {
  std::shared_ptr<CacheManager> cache_manager = CacheManager::Create(model);
  auto scheduler = Scheduler::Create(model, cache_manager);
  auto model_executor = ModelExecutor::Create(model, cache_manager);
  return EngineDependencies{std::move(cache_manager), std::move(scheduler), std::move(model_executor)};
}

std::shared_ptr<Request> Engine::CreateRequest(const GeneratorParams& params) {
  ReclaimAbandonedRequests();
  if (health_ == EngineHealth::Unhealthy) {
    std::rethrow_exception(fatal_error_);
  }
  if (params.model_.get() != model_.get()) {
    throw std::runtime_error(
        "Engine request parameters must belong to the Engine's model.");
  }

  auto request = std::make_shared<Request>(
      CloneRequestParams(params, *model_));
  if (cache_manager_->SupportsDynamicBatching()) {
    request->ValidateEngineCompatibility();
  }

  tracked_requests_.push_back(request);
  request->engine_ = shared_from_this();
  return request;
}

void Engine::BeginTurn(const std::shared_ptr<Request>& request,
                       std::span<const int32_t> tokens,
                       std::optional<size_t> max_generated_tokens) {
  ReclaimAbandonedRequests();
  if (health_ == EngineHealth::Unhealthy) {
    std::rethrow_exception(fatal_error_);
  }
  if (!request || request->engine_.lock().get() != this) {
    throw std::runtime_error(
        "Cannot begin a turn for a request that does not belong to this engine.");
  }
  if (max_generated_tokens && *max_generated_tokens == 0) {
    throw std::runtime_error(
        "max_generated_tokens must be greater than zero.");
  }
  if (tokens.empty()) {
    throw std::runtime_error(
        "Expected at least one input token for generation. Received 0.");
  }

  const bool first_turn = request->status_ == RequestStatus::Unassigned;
  if (!first_turn && !IsTurnComplete(request->status_)) {
    if (IsClosed(request->status_)) {
      throw std::runtime_error("Cannot begin a turn for a closed request.");
    }
    throw std::runtime_error(
        "BeginTurn is only valid for a new request or after the current turn is complete.");
  }

  if (first_turn) {
    if (cache_manager_->SupportsDynamicBatching()) {
      request->ValidateEngineCompatibility();
    }
  } else {
    const DeviceType cache_device =
        request->params_->model_->p_device_kvcache_->GetType();
    if (!SupportsContinuousDecoding(cache_device)) {
      throw std::runtime_error(
          "Continuous decoding is not supported on the selected KV-cache device type (" +
          to_string(cache_device) + ").");
    }
    ValidateRequestCanContinue(request);
  }

  const size_t sequence_length =
      first_turn ? 0 : static_cast<size_t>(request->CurrentSequenceLength());
  ValidateAppendLength(*request->params_, sequence_length, tokens.size());
  if (request->tokens_host_.capacity() <
      request->tokens_host_.size() + tokens.size()) {
    throw std::logic_error(
        "The request host token mirror does not have reserved turn capacity.");
  }

  auto device_tokens = AllocateOnDevice(*request->params_, tokens);
  const RequestStatus status_before = request->status_;
  const size_t host_size_before = request->tokens_host_.size();
  const int64_t prompt_length_before = request->prompt_sequence_length_;
  const int64_t seen_length_before = request->seen_sequence_length_;
  const int64_t processed_length_before =
      request->processed_sequence_length_;
  const auto turn_max_generated_tokens_before =
      request->turn_max_generated_tokens_;
  const size_t turn_generated_tokens_before =
      request->turn_generated_tokens_;
  bool added_to_scheduler = false;

  request->search_->SaveStateForTransaction();
  try {
    request->search_->AppendTokens(device_tokens);
    request->tokens_host_.insert(
        request->tokens_host_.end(), tokens.begin(), tokens.end());
    request->prompt_sequence_length_ = request->CurrentSequenceLength();
    if (first_turn) {
      request->processed_sequence_length_ = 0;
      request->seen_sequence_length_ = request->CurrentSequenceLength();
      request->status_ = RequestStatus::Assigned;
      scheduler_->AddRequest(request);
      added_to_scheduler = true;
    } else {
      request->status_ = RequestStatus::Assigned;
    }
    request->search_->CommitStateForTransaction();
    // Reset turn-scoped publication state only after every validating, allocating, appending, and
    // scheduler operation has succeeded. The caller-owned options storage is represented only by
    // this copied value.
    request->turn_max_generated_tokens_ = max_generated_tokens;
    request->turn_generated_tokens_ = 0;
  } catch (...) {
    const auto append_error = std::current_exception();
    try {
      if (added_to_scheduler) {
        scheduler_->RemoveRequest(request);
      }
      request->status_ = status_before;
      request->tokens_host_.resize(host_size_before);
      request->prompt_sequence_length_ = prompt_length_before;
      request->seen_sequence_length_ = seen_length_before;
      request->processed_sequence_length_ = processed_length_before;
      request->turn_max_generated_tokens_ =
          turn_max_generated_tokens_before;
      request->turn_generated_tokens_ =
          turn_generated_tokens_before;
      request->search_->RestoreStateForTransaction();
    } catch (...) {
      HandleContinuationRestoreFailure(
          request, append_error, std::current_exception());
    }
    std::rethrow_exception(append_error);
  }
}

void Engine::CloseRequest(const std::shared_ptr<Request>& request) {
  if (request && IsClosed(request->status_)) {
    return;
  }
  if (!request || request->engine_.lock().get() != this) {
    throw std::runtime_error("Cannot close a request that does not belong to this engine.");
  }

  scheduler_->RemoveRequest(request);

  ready_requests_.erase(
      ready_requests_.begin(),
      ready_requests_.begin() + static_cast<ptrdiff_t>(ready_request_index_));
  ready_request_index_ = 0;
  ready_requests_.erase(
      std::remove(ready_requests_.begin(), ready_requests_.end(), request),
      ready_requests_.end());
  staged_ready_requests_.erase(
      std::remove(staged_ready_requests_.begin(), staged_ready_requests_.end(), request),
      staged_ready_requests_.end());
  request->CompleteClose();
  tracked_requests_.erase(
      std::remove_if(
          tracked_requests_.begin(), tracked_requests_.end(),
          [&request](const std::weak_ptr<Request>& tracked) {
            const auto owned = tracked.lock();
            return !owned || owned == request;
          }),
      tracked_requests_.end());
}

void Engine::ReclaimAbandonedRequests() {
  // ExternalRelease only publishes an atomic abandonment marker. The host's owner-thread boundary
  // can safely perform the normal removal sequence: scheduler/cache release, ready-notification
  // purge, and terminal close.
  std::vector<std::shared_ptr<Request>> abandoned_requests;
  abandoned_requests.reserve(tracked_requests_.size());
  tracked_requests_.erase(
      std::remove_if(
          tracked_requests_.begin(), tracked_requests_.end(),
          [&abandoned_requests, this](const std::weak_ptr<Request>& tracked) {
            const auto request = tracked.lock();
            if (!request) {
              return true;
            }
            if (!IsClosed(request->status_) &&
                request->engine_.lock().get() == this &&
                request->IsExternallyAbandoned()) {
              abandoned_requests.push_back(request);
            }
            return false;
          }),
      tracked_requests_.end());

  for (const auto& request : abandoned_requests) {
    // Recheck defensively in case an external owner was reacquired before this serialized boundary.
    if (request->IsExternallyAbandoned()) {
      CloseRequest(request);
    }
  }
}

void Engine::ValidateRequestCanContinue(const std::shared_ptr<Request>& request) const {
  if (health_ == EngineHealth::Unhealthy) {
    std::rethrow_exception(fatal_error_);
  }
  if (request->engine_.lock().get() != this) {
    throw std::runtime_error("Cannot continue a request that does not belong to this engine.");
  }

  if (!cache_manager_->IsResident(request)) {
    throw std::runtime_error("Cannot continue a request whose model state is no longer resident.");
  }

  if (std::find(ready_requests_.begin() + static_cast<ptrdiff_t>(ready_request_index_),
                ready_requests_.end(), request) != ready_requests_.end()) {
    throw std::runtime_error(
        "Cannot continue a request while its ready notification is pending; "
        "call Engine::Run() to drain the ready notification before continuing.");
  }

  if (!cache_manager_->SupportsDynamicBatching() &&
      cache_manager_->ResidentRequestCount() > 1) {
    throw std::runtime_error(
        "Continuous decoding is only supported when a static engine batch contains one request.");
  }
}

[[noreturn]] void Engine::HandleContinuationRestoreFailure(
    const std::shared_ptr<Request>& request,
    std::exception_ptr append_error,
    std::exception_ptr restore_error) {
  std::string message = AddExceptionCause(
      "Continuation append failed and its Search state could not be restored.",
      append_error);
  try {
    CloseRequest(request);
  } catch (...) {
    message = AddExceptionCause(
        std::move(message) + " Closing the poisoned request also failed.",
        std::current_exception());
    request->CompleteClose();
  }
  MarkUnhealthyAndThrow(
      StepOutcomeKind::FatalExecutionFailure,
      /*transaction_id=*/0,
      request.get(),
      std::move(message),
      restore_error);
}

std::shared_ptr<Request> Engine::Run() {
  ReclaimAbandonedRequests();
  if (auto request = DrainReadyRequest()) {
    return request;
  }
  if (health_ == EngineHealth::Unhealthy) {
    std::rethrow_exception(fatal_error_);
  }
  return cache_manager_->SupportsDynamicBatching() ? RunDynamic() : RunStatic();
}

std::shared_ptr<Request> Engine::RunStatic() {
  while (scheduler_->HasPendingRequests()) {
    auto scheduled_requests = scheduler_->Schedule();
    std::vector<RequestStatus> statuses_before_step;
    statuses_before_step.reserve(scheduled_requests.size());
    for (const auto& request : scheduled_requests) {
      statuses_before_step.push_back(request->status_);
    }

    model_executor_->Decode(scheduled_requests);
    scheduled_requests.GenerateNextTokens();

    for (size_t i = 0; i < scheduled_requests.size(); ++i) {
      auto request = scheduled_requests[i];
      const bool turn_completed_this_step =
          !IsTurnComplete(statuses_before_step[i]) &&
          IsTurnComplete(request->status_);
      if (!IsClosed(request->status_) &&
          (request->HasUnseenTokens() || turn_completed_this_step)) {
        ready_requests_.push_back(request);
      }
    }
    if (auto request = DrainReadyRequest()) {
      return request;
    }
  }
  return nullptr;
}

std::shared_ptr<Request> Engine::RunDynamic() {
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

    auto scheduled_requests = [&]() -> ScheduledRequests {
      try {
        return scheduler_->CreateScheduledRequests(step_plan_);
      } catch (...) {
        const auto construction_error = std::current_exception();
        try {
          reservation->Release();
        } catch (...) {
          ++transaction_metrics_.rollbacks;
          MarkUnhealthyAndThrow(
              StepOutcomeKind::FatalExecutionFailure,
              step_plan_.transaction_id,
              nullptr,
              "Failed to release cache state after scheduled-request construction failed.",
              std::current_exception());
        }
        ++transaction_metrics_.rollbacks;
        MarkUnhealthyAndThrow(
            StepOutcomeKind::ExecutionContractFailure,
            step_plan_.transaction_id,
            nullptr,
            "Failed to construct the scheduled request transaction.",
            construction_error);
      }
    }();
    ExecutionContext context{&step_plan_};
    context.cache_reservation = reservation->PagedReservation();

    bool request_transaction_active = false;
    const auto rollback_transaction = [&]() {
      // Request/search state and paged-cache state are checkpointed separately. Both must be
      // restored so a retry observes exactly the state that existed before this Run() call.
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

    // Run() returns one ready request at a time. Keep the rest queued so draining this committed
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
