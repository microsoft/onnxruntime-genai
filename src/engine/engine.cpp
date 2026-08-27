// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"

#include <limits>

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

void ValidateAppendLength(size_t max_total_tokens,
                          size_t current_sequence_length,
                          size_t token_count) {
  if (current_sequence_length >= max_total_tokens ||
      token_count >= max_total_tokens - current_sequence_length) {
    throw std::runtime_error(
        "Input tokens must leave room for at least one generated token before max_total_tokens (" +
        std::to_string(max_total_tokens) + ").");
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
  pending_events_.reserve(max_batch_size);
  staged_events_.reserve(max_batch_size);
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

void Engine::ValidateOwnerThread() const {
  if (std::this_thread::get_id() != owner_thread_) {
    throw std::runtime_error(
        "Engine operations must be called from the Engine owner thread.");
  }
}

std::shared_ptr<Request> Engine::CreateRequest(const GeneratorParams& params,
                                               size_t max_total_tokens) {
  ValidateOwnerThread();
  ReclaimAbandonedRequests();
  if (health_ == EngineHealth::Unhealthy) {
    std::rethrow_exception(fatal_error_);
  }
  if (params.model_.get() != model_.get()) {
    throw std::runtime_error(
        "Engine request parameters must belong to the Engine's model.");
  }
  if (max_total_tokens == 0 ||
      max_total_tokens > static_cast<size_t>(params.search.max_length)) {
    throw std::runtime_error(
        "max_total_tokens must be greater than zero and no greater than max_length.");
  }

  auto request = std::make_shared<Request>(
      CloneRequestParams(params, *model_), max_total_tokens);
  request->ValidateEngineCompatibility();

  tracked_requests_.push_back(request);
  request->engine_ = shared_from_this();
  return request;
}

uint64_t Engine::BeginTurn(const std::shared_ptr<Request>& request,
                           std::span<const int32_t> tokens,
                           std::optional<size_t> max_generated_tokens) {
  ValidateOwnerThread();
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
  if (request->turn_id_exhausted_) {
    throw std::overflow_error(
        "The request cannot admit another turn because its uint64 turn id space is exhausted.");
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
    const bool restartable_canceled_turn =
        request->finish_reason_ == GenerationFinishReason::Canceled &&
        request->processed_sequence_length_ == 0 &&
        !cache_manager_->IsResident(request);
    ValidateRequestCanContinue(request, restartable_canceled_turn);
    if (!restartable_canceled_turn) {
      const DeviceType cache_device =
          request->params_->model_->p_device_kvcache_->GetType();
      if (!SupportsContinuousDecoding(cache_device)) {
        throw std::runtime_error(
            "Continuous decoding is not supported on the selected KV-cache device type (" +
            to_string(cache_device) + ").");
      }
    }
  }

  const size_t sequence_length =
      first_turn ? 0 : static_cast<size_t>(request->CurrentSequenceLength());
  ValidateAppendLength(
      request->max_total_tokens_, sequence_length, tokens.size());
  if (request->tokens_host_.capacity() <
      request->tokens_host_.size() + tokens.size()) {
    throw std::logic_error(
        "The request host token mirror does not have reserved turn capacity.");
  }

  auto device_tokens = AllocateOnDevice(*request->params_, tokens);
  const RequestStatus status_before = request->status_;
  const size_t host_size_before = request->tokens_host_.size();
  const int64_t prompt_length_before = request->prompt_sequence_length_;
  const int64_t processed_length_before =
      request->processed_sequence_length_;
  const auto turn_max_generated_tokens_before =
      request->turn_max_generated_tokens_;
  const size_t turn_generated_tokens_before =
      request->turn_generated_tokens_;
  const size_t turn_prompt_tokens_before = request->turn_prompt_tokens_;
  const uint64_t current_turn_id_before = request->current_turn_id_;
  const uint64_t next_turn_id_before = request->next_turn_id_;
  const bool has_current_turn_before = request->has_current_turn_;
  const bool turn_id_exhausted_before = request->turn_id_exhausted_;
  bool added_to_scheduler = false;

  request->search_->SaveStateForTransaction();
  try {
    request->search_->AppendTokens(device_tokens);
    request->tokens_host_.insert(
        request->tokens_host_.end(), tokens.begin(), tokens.end());
    request->prompt_sequence_length_ = request->CurrentSequenceLength();
    if (first_turn) {
      request->processed_sequence_length_ = 0;
      request->status_ = RequestStatus::Assigned;
      scheduler_->AddRequest(request);
      added_to_scheduler = true;
    } else {
      request->status_ = RequestStatus::Assigned;
    }
    request->search_->CommitStateForTransaction();
    // Install the new turn's generation budget only after turn admission succeeds. This keeps
    // BeginTurn transactional and copies the caller-owned options before they may go out of scope.
    request->turn_max_generated_tokens_ = max_generated_tokens;
    request->turn_prompt_tokens_ = tokens.size();
    request->turn_generated_tokens_ = 0;
    request->current_turn_id_ = request->next_turn_id_;
    request->has_current_turn_ = true;
    if (request->next_turn_id_ == std::numeric_limits<uint64_t>::max()) {
      request->turn_id_exhausted_ = true;
    } else {
      ++request->next_turn_id_;
    }
    request->finish_reason_ = GenerationFinishReason::None;
    return request->current_turn_id_;
  } catch (...) {
    const auto append_error = std::current_exception();
    try {
      if (added_to_scheduler) {
        scheduler_->RemoveRequest(request);
      }
      request->status_ = status_before;
      request->tokens_host_.resize(host_size_before);
      request->prompt_sequence_length_ = prompt_length_before;
      request->processed_sequence_length_ = processed_length_before;
      request->turn_max_generated_tokens_ =
          turn_max_generated_tokens_before;
      request->turn_generated_tokens_ =
          turn_generated_tokens_before;
      request->turn_prompt_tokens_ = turn_prompt_tokens_before;
      request->current_turn_id_ = current_turn_id_before;
      request->next_turn_id_ = next_turn_id_before;
      request->has_current_turn_ = has_current_turn_before;
      request->turn_id_exhausted_ = turn_id_exhausted_before;
      request->search_->RestoreStateForTransaction();
    } catch (...) {
      HandleContinuationRestoreFailure(
          request, append_error, std::current_exception());
    }
    std::rethrow_exception(append_error);
  }
}

bool Engine::CancelRequest(const std::shared_ptr<Request>& request, uint64_t turn_id) {
  ValidateOwnerThread();
  if (!request || request->engine_.lock().get() != this) {
    throw std::runtime_error(
        "Cannot cancel a request that does not belong to this engine.");
  }
  if (IsClosed(request->status_)) {
    throw std::runtime_error("Cannot cancel a closed request.");
  }
  if (!request->has_current_turn_) {
    throw std::runtime_error("Cannot cancel a request before a turn has begun.");
  }
  if (turn_id != request->current_turn_id_ ||
      IsTurnComplete(request->status_)) {
    return false;
  }
  if (!IsExecutable(request->status_)) {
    return false;
  }

  pending_events_.erase(
      pending_events_.begin(),
      pending_events_.begin() + static_cast<ptrdiff_t>(pending_event_index_));
  pending_event_index_ = 0;
  auto existing = std::find_if(
      pending_events_.begin(), pending_events_.end(),
      [&request, turn_id](const EngineEvent& event) {
        return event.request == request && event.turn_id == turn_id;
      });
  const bool has_existing_event = existing != pending_events_.end();
  if (!has_existing_event) {
    pending_events_.reserve(pending_events_.size() + 1);
  }

  request->status_ = RequestStatus::TurnComplete;
  request->finish_reason_ = GenerationFinishReason::Canceled;
  EngineEvent terminal;
  terminal.request = request;
  terminal.turn_id = turn_id;
  terminal.flags = EngineEventFlagTurnFinished;
  terminal.finish_reason = GenerationFinishReason::Canceled;
  terminal.usage = {
      request->turn_prompt_tokens_,
      request->turn_generated_tokens_,
      0};
  if (has_existing_event) {
    existing->flags |= terminal.flags;
    existing->finish_reason = terminal.finish_reason;
    existing->usage = terminal.usage;
  } else {
    pending_events_.push_back(std::move(terminal));
  }
  return true;
}

void Engine::CloseRequest(const std::shared_ptr<Request>& request) {
  ValidateOwnerThread();
  if (request && IsClosed(request->status_)) {
    return;
  }
  if (!request || request->engine_.lock().get() != this) {
    throw std::runtime_error("Cannot close a request that does not belong to this engine.");
  }

  scheduler_->RemoveRequest(request);

  pending_events_.erase(
      pending_events_.begin(),
      pending_events_.begin() + static_cast<ptrdiff_t>(pending_event_index_));
  pending_event_index_ = 0;
  pending_events_.erase(
      std::remove_if(
          pending_events_.begin(), pending_events_.end(),
          [&request](const EngineEvent& event) { return event.request == request; }),
      pending_events_.end());
  staged_events_.erase(
      std::remove_if(
          staged_events_.begin(), staged_events_.end(),
          [&request](const EngineEvent& event) { return event.request == request; }),
      staged_events_.end());
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
  // can safely perform the normal removal sequence: logical scheduler removal, ready-notification
  // purge, and terminal close. Dynamic cache ownership is released immediately; a resident static
  // batch row can remain physically retained until its shared batch is recycled.
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

void Engine::ValidateRequestCanContinue(
    const std::shared_ptr<Request>& request,
    bool allow_nonresident) const {
  if (health_ == EngineHealth::Unhealthy) {
    std::rethrow_exception(fatal_error_);
  }
  if (request->engine_.lock().get() != this) {
    throw std::runtime_error("Cannot continue a request that does not belong to this engine.");
  }

  if (!allow_nonresident && !cache_manager_->IsResident(request)) {
    throw std::runtime_error("Cannot continue a request whose model state is no longer resident.");
  }

  if (std::find_if(
          pending_events_.begin() + static_cast<ptrdiff_t>(pending_event_index_),
          pending_events_.end(),
          [&request](const EngineEvent& event) {
            return event.request == request;
          }) != pending_events_.end()) {
    throw std::runtime_error(
        "Cannot continue a request while an Engine event is pending; "
        "call Engine::Run() to drain the event before continuing.");
  }

  if (!allow_nonresident &&
      !cache_manager_->SupportsDynamicBatching() &&
      cache_manager_->ResidentRequestCount() > 1) {
    throw std::runtime_error(
        "Continuous decoding is only supported when a static engine batch contains one request.");
  }
}

[[noreturn]] void Engine::HandleContinuationRestoreFailure(
    const std::shared_ptr<Request>& request,
    std::exception_ptr append_error,
    std::exception_ptr restore_error) {
  request->finish_reason_ = GenerationFinishReason::Failed;
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

EngineEvent Engine::Run() {
  ValidateOwnerThread();
  ReclaimAbandonedRequests();
  if (pending_event_index_ < pending_events_.size()) {
    return DrainPendingEvent();
  }
  if (health_ == EngineHealth::Unhealthy) {
    std::rethrow_exception(fatal_error_);
  }
  try {
    return cache_manager_->SupportsDynamicBatching() ? RunDynamic() : RunStatic();
  } catch (const EngineStepError& error) {
    return EventFromStepError(error);
  }
}

EngineEvent Engine::RunStatic() {
  while (scheduler_->HasPendingRequests()) {
    auto scheduled_requests = scheduler_->Schedule();

    try {
      model_executor_->Decode(scheduled_requests);
      scheduled_requests.GenerateNextTokens(step_results_);
    } catch (...) {
      MarkUnhealthyAndThrow(
          StepOutcomeKind::FatalExecutionFailure,
          /*transaction_id=*/0,
          nullptr,
          "Static-batch execution failed and the Engine is no longer healthy.",
          std::current_exception());
    }

    staged_events_.clear();
    for (size_t i = 0; i < scheduled_requests.size(); ++i) {
      if (!IsClosed(scheduled_requests[i]->status_) &&
          (step_results_[i].token_appended || step_results_[i].done)) {
        staged_events_.push_back(
            EventFromStep(scheduled_requests[i], step_results_[i]));
      }
    }
    if (!staged_events_.empty()) {
      pending_events_.swap(staged_events_);
      pending_event_index_ = 0;
      return DrainPendingEvent();
    }
  }
  return {};
}

EngineEvent Engine::RunDynamic() {
  while (scheduler_->HasPendingRequests()) {
    // A dynamic step is a transaction with six phases:
    // plan -> reserve cache -> checkpoint request state -> execute -> stage sampled tokens -> commit.
    // Nothing becomes externally visible until the final commit succeeds.
    step_plan_.transaction_id = next_transaction_id_++;
    StepPlanningResult planning_result;
    try {
      planning_result = scheduler_->PlanStep(step_plan_);
    } catch (...) {
      MarkUnhealthyAndThrow(
          StepOutcomeKind::ExecutionContractFailure,
          step_plan_.transaction_id,
          nullptr,
          "Dynamic scheduler planning failed and the Engine is no longer healthy.",
          std::current_exception());
    }
    if (planning_result.capacity_deferred) {
      ++transaction_metrics_.capacity_deferrals;
    }
    if (planning_result.unserviceable_request_id) {
      return FailUnserviceableRequest(
          planning_result.unserviceable_request_id);
    }
    if (!planning_result.executable) {
      const auto outcome = planning_result.outcome;
      if (outcome.kind == StepOutcomeKind::NoWork &&
          !scheduler_->HasPendingRequests()) {
        return {};
      }
      if (outcome.kind == StepOutcomeKind::UnserviceableRequest) {
        return FailUnserviceableRequest(outcome.request_id);
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
    } catch (const std::logic_error&) {
      const auto post_processing_error = std::current_exception();
      rollback_transaction();
      ++transaction_metrics_.post_processing_aborts;
      MarkUnhealthyAndThrow(
          StepOutcomeKind::ExecutionContractFailure,
          step_plan_.transaction_id,
          nullptr,
          "Request post-processing violated the transaction contract.",
          post_processing_error);
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

    staged_events_.clear();
    for (size_t i = 0; i < step_plan_.requests.size(); ++i) {
      if (step_results_[i].token_appended || step_results_[i].done) {
        staged_events_.push_back(
            EventFromStep(step_plan_.requests[i].request, step_results_[i]));
      }
    }
    pending_events_.swap(staged_events_);
    pending_event_index_ = 0;
    ++transaction_metrics_.committed_steps;
    if (!pending_events_.empty()) {
      return DrainPendingEvent();
    }
  }
  return {};
}

EngineEvent Engine::DrainPendingEvent() {
  if (pending_event_index_ == pending_events_.size()) {
    pending_events_.clear();
    pending_event_index_ = 0;
    return {};
  }
  return pending_events_[pending_event_index_++];
}

EngineEvent Engine::EventFromStep(
    const std::shared_ptr<Request>& request,
    const RequestStepResult& result) const {
  EngineEvent event;
  event.request = request;
  event.turn_id = request->current_turn_id_;
  if (result.token_appended) {
    event.flags |= EngineEventFlagToken;
    event.token = result.token;
  }
  if (result.done) {
    event.flags |= EngineEventFlagTurnFinished;
    event.finish_reason = result.finish_reason;
    event.usage = {
        request->turn_prompt_tokens_,
        request->turn_generated_tokens_,
        0};
  }
  return event;
}

std::shared_ptr<Request> Engine::FindTrackedRequest(const void* request_id) const {
  for (const auto& tracked : tracked_requests_) {
    if (auto request = tracked.lock();
        request && request.get() == request_id) {
      return request;
    }
  }
  return nullptr;
}

EngineEvent Engine::FailUnserviceableRequest(const void* request_id) {
  auto request = FindTrackedRequest(request_id);
  if (!request || !IsExecutable(request->status_)) {
    MarkUnhealthyAndThrow(
        StepOutcomeKind::ExecutionContractFailure,
        step_plan_.transaction_id,
        request_id,
        "The scheduler identified an unknown or non-executable unserviceable Request.",
        std::make_exception_ptr(std::logic_error{
            "Invalid unserviceable Request identity."}));
  }
  scheduler_->RemoveRequest(request);
  request->status_ = RequestStatus::TurnComplete;
  request->finish_reason_ = GenerationFinishReason::Failed;

  EngineEvent event;
  event.request = request;
  event.turn_id = request->current_turn_id_;
  event.flags = EngineEventFlagTurnFinished | EngineEventFlagFailed;
  event.finish_reason = GenerationFinishReason::Failed;
  event.error_code = EngineErrorCode::RequestUnserviceable;
  event.usage = {
      request->turn_prompt_tokens_,
      request->turn_generated_tokens_,
      0};
  return event;
}

EngineEvent Engine::EventFromStepError(const EngineStepError& error) {
  EngineEvent event;
  switch (error.Outcome().kind) {
    case StepOutcomeKind::CapacityDeferred:
      event.flags = EngineEventFlagCapacityBlocked;
      event.error_code = EngineErrorCode::CapacityDeferred;
      break;
    case StepOutcomeKind::ExecutionCapacityExceeded:
      event.flags = EngineEventFlagCapacityBlocked;
      event.error_code = EngineErrorCode::ExecutionCapacityExceeded;
      break;
    case StepOutcomeKind::RetryableBatchAbort:
      event.flags = EngineEventFlagRetryable;
      event.error_code = EngineErrorCode::RetryableExecution;
      break;
    case StepOutcomeKind::ExecutionContractFailure:
      event.flags = EngineEventFlagFailed;
      event.error_code = EngineErrorCode::EngineContractFailure;
      break;
    case StepOutcomeKind::FatalExecutionFailure:
      event.flags = EngineEventFlagFailed;
      event.error_code = EngineErrorCode::EngineExecutionFailure;
      break;
    case StepOutcomeKind::UnserviceableRequest:
      return FailUnserviceableRequest(error.Outcome().request_id);
    case StepOutcomeKind::NoWork:
    case StepOutcomeKind::Committed:
      throw;
  }
  return event;
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
  for (const auto& tracked : tracked_requests_) {
    if (auto request = tracked.lock();
        request && IsExecutable(request->status_)) {
      request->status_ = RequestStatus::TurnComplete;
      request->finish_reason_ = GenerationFinishReason::Failed;
    }
  }
  fatal_error_ = std::make_exception_ptr(EngineStepError{
      {outcome, transaction_id, request_id},
      AddExceptionCause(std::move(message), error),
  });
  std::rethrow_exception(fatal_error_);
}

bool Engine::HasPendingRequests() {
  ValidateOwnerThread();
  ReclaimAbandonedRequests();
  return pending_event_index_ < pending_events_.size() ||
         scheduler_->HasPendingRequests();
}

}  // namespace Generators
