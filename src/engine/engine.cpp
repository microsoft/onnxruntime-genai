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

  fatal_fallback_error_ = std::make_exception_ptr(EngineStepError{
      {StepOutcomeKind::ExecutionContractFailure, 0, nullptr},
      "The Engine encountered a fatal failure while constructing its diagnostic error.",
  });
  const size_t max_batch_size = cache_manager_->MaxBatchSize();
  step_plan_.requests.reserve(max_batch_size);
  step_results_.reserve(max_batch_size);
  staged_event_order_.reserve(max_batch_size);
  pending_events_.reserve(max_batch_size);
  staged_events_.reserve(max_batch_size);
  fatal_events_.reserve(max_batch_size + 1);
  deferred_static_closes_.reserve(max_batch_size);
}

Engine::~Engine() {
  while (!tracked_requests_.empty()) {
    auto request = std::move(tracked_requests_.back());
    tracked_requests_.pop_back();
    DetachRequestForTeardown(request);
  }
  ReleaseDeferredStaticCloseState();
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
  if (params.search.max_length <= 0) {
    throw std::runtime_error(
        "max_length must be greater than zero; actual value is " +
        std::to_string(params.search.max_length) + ".");
  }
  if (max_total_tokens == 0 ||
      max_total_tokens > static_cast<size_t>(params.search.max_length)) {
    throw std::runtime_error(
        "max_total_tokens (" + std::to_string(max_total_tokens) +
        ") must be greater than zero and no greater than max_length (" +
        std::to_string(params.search.max_length) + ").");
  }

  auto request = std::make_shared<Request>(
      CloneRequestParams(params, *model_), max_total_tokens,
      abandonment_pending_);
  request->ValidateEngineCompatibility();

  // Fatal handling may need one terminal event for every tracked Request plus a request-less
  // fallback. Grow dedicated storage before publishing the Request so terminalization never
  // depends on the capacity of the normal event vectors, which swap after every committed step.
  fatal_events_.reserve(tracked_requests_.size() + 2);
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
        "max_generated_tokens (0) must be greater than zero.");
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
  Request::ValidateAppendLength(
      request->max_total_tokens_, sequence_length, tokens.size());
  if (request->tokens_host_.capacity() <
      request->tokens_host_.size() + tokens.size()) {
    throw std::logic_error(
        "The request host token mirror does not have reserved turn capacity.");
  }

  auto device_tokens = Request::AllocateOnDevice(*request->params_, tokens);
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

  request->SaveStateForTransaction();
  try {
    request->ResetGuidanceForNewTurn();
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
    request->CommitStateForTransaction();
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
      request->RestoreStateForTransaction();
    } catch (...) {
      HandleContinuationRestoreFailure(
          request, std::current_exception());
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
  const bool release_deferred_static_state =
      !deferred_static_closes_.empty() &&
      !cache_manager_->SupportsDynamicBatching() &&
      cache_manager_->IsResident(request) &&
      !StaticBatchNeedsRequest(request);

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
  if (release_deferred_static_state) {
    ReleaseDeferredStaticCloseState();
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
  const bool resident_static_row =
      !cache_manager_->SupportsDynamicBatching() &&
      cache_manager_->IsResident(request);
  const bool retain_static_row_state =
      resident_static_row && StaticBatchNeedsRequest(request);

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
  if (retain_static_row_state) {
    deferred_static_closes_.push_back(request);
    request->CompleteLogicalClose();
  } else {
    request->CompleteClose();
    if (resident_static_row) {
      ReleaseDeferredStaticCloseState();
    }
  }
  tracked_requests_.erase(
      std::remove_if(
          tracked_requests_.begin(), tracked_requests_.end(),
          [&request](const std::shared_ptr<Request>& tracked) {
            return tracked == request;
          }),
      tracked_requests_.end());
}

void Engine::ReleaseDeferredStaticCloseState() noexcept {
  for (const auto& request : deferred_static_closes_) {
    request->CompleteClose();
  }
  deferred_static_closes_.clear();
}

void Engine::DetachRequestForTeardown(
    const std::shared_ptr<Request>& request) noexcept {
  if (!request) {
    return;
  }

  scheduler_->DetachRequestForTeardown(request);
  pending_events_.erase(
      std::remove_if(
          pending_events_.begin(), pending_events_.end(),
          [&request](const EngineEvent& event) {
            return event.request == request;
          }),
      pending_events_.end());
  staged_events_.erase(
      std::remove_if(
          staged_events_.begin(), staged_events_.end(),
          [&request](const EngineEvent& event) {
            return event.request == request;
          }),
      staged_events_.end());
  pending_event_index_ = 0;
  request->CompleteClose();
}

void Engine::ReclaimAbandonedRequests() {
  // ExternalRelease only publishes an atomic abandonment marker. The host's owner-thread boundary
  // can safely perform the normal removal sequence: logical scheduler removal, ready-notification
  // purge, and terminal close. Dynamic cache ownership is released immediately; a resident static
  // batch row can remain physically retained until its shared batch is recycled.
  if (!abandonment_pending_->exchange(false, std::memory_order_acq_rel)) {
    return;
  }

  try {
    while (true) {
      const auto abandoned = std::find_if(
          tracked_requests_.begin(), tracked_requests_.end(),
          [this](const std::shared_ptr<Request>& request) {
            return request &&
                   !IsClosed(request->status_) &&
                   request->engine_.lock().get() == this &&
                   request->IsExternallyAbandoned();
          });
      if (abandoned == tracked_requests_.end()) {
        return;
      }

      auto request = *abandoned;
      // Recheck immediately before removal in case an external owner was reacquired before this
      // serialized boundary.
      if (request->IsExternallyAbandoned()) {
        CloseRequest(request);
      }
    }
  } catch (const std::bad_alloc&) {
    abandonment_pending_->store(true, std::memory_order_release);
    throw;
  } catch (...) {
    MarkUnhealthyAndThrow(
        StepOutcomeKind::ExecutionContractFailure,
        /*transaction_id=*/0,
        nullptr,
        "Abandoned Request cleanup violated Engine ownership invariants.",
        std::current_exception());
  }
}

bool Engine::StaticBatchNeedsRequest(
    const std::shared_ptr<Request>& request) const {
  if (cache_manager_->SupportsDynamicBatching() ||
      !cache_manager_->IsResident(request)) {
    return false;
  }
  const auto residents = cache_manager_->AllocatedRequests();
  return std::any_of(
      residents.begin(), residents.end(),
      [&request](const std::shared_ptr<Request>& resident) {
        return resident != request && IsExecutable(resident->status_);
      });
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
        "Continuous decoding requires exactly one resident request in a "
        "static engine batch; actual resident request count is " +
        std::to_string(cache_manager_->ResidentRequestCount()) + ".");
  }
}

[[noreturn]] void Engine::HandleContinuationRestoreFailure(
    const std::shared_ptr<Request>& request,
    std::exception_ptr restore_error) {
  request->finish_reason_ = GenerationFinishReason::Failed;
  try {
    CloseRequest(request);
  } catch (...) {
    request->CompleteClose();
    MarkUnhealthyAndThrow(
        StepOutcomeKind::FatalExecutionFailure,
        /*transaction_id=*/0,
        request.get(),
        "Continuation append and restore failed, and closing the poisoned Request also failed.",
        std::current_exception());
  }
  MarkUnhealthyAndThrow(
      StepOutcomeKind::FatalExecutionFailure,
      /*transaction_id=*/0,
      request.get(),
      "Continuation append failed and its Search state could not be restored.",
      restore_error);
}

size_t Engine::Run(std::span<EngineEvent> events) {
  ValidateOwnerThread();
  if (events.empty()) {
    if (health_ == EngineHealth::Unhealthy) {
      std::rethrow_exception(fatal_error_);
    }
    return 0;
  }
  try {
    ReclaimAbandonedRequests();
  } catch (const EngineStepError&) {
    if (pending_event_index_ < pending_events_.size()) {
      return DrainPendingEvents(events);
    }
    throw;
  }
  if (pending_event_index_ < pending_events_.size()) {
    return DrainPendingEvents(events);
  }
  if (health_ == EngineHealth::Unhealthy) {
    std::rethrow_exception(fatal_error_);
  }
  try {
    if (cache_manager_->SupportsDynamicBatching()) {
      RunDynamic();
    } else {
      RunStatic();
    }
  } catch (const EngineStepError& error) {
    if (pending_events_.empty()) {
      RetainEvent(EventFromStepError(
          error, std::current_exception()));
    }
  }
  return DrainPendingEvents(events);
}

void Engine::RunStatic() {
  if (scheduler_->HasPendingRequests()) {
    auto scheduled_requests = [&]() -> ScheduledRequests {
      try {
        return scheduler_->Schedule();
      } catch (...) {
        MarkUnhealthyAndThrow(
            StepOutcomeKind::ExecutionContractFailure,
            /*transaction_id=*/0,
            nullptr,
            "Static scheduler failed and the Engine is no longer healthy.",
            std::current_exception());
      }
    }();

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
    pending_events_.swap(staged_events_);
    pending_event_index_ = 0;
    if (std::none_of(
            scheduled_requests.begin(), scheduled_requests.end(),
            [](const std::shared_ptr<Request>& request) {
              return IsExecutable(request->status_);
            })) {
      ReleaseDeferredStaticCloseState();
    }
  }
}

void Engine::RunDynamic() {
  if (scheduler_->HasPendingRequests()) {
    // A dynamic step is a transaction with six phases:
    // plan -> reserve state -> checkpoint request state -> execute -> stage sampled tokens -> commit.
    // Nothing becomes externally visible until the final commit succeeds.
    step_plan_.transaction_id = next_transaction_id_++;
    StepPlanningResult planning_result;
    try {
      // Planning must report ordinary non-executable outcomes through its result. A composite
      // cache consistency failure proves that paged and fixed committed ownership disagree and is
      // fatal. Incidental failures such as allocation errors have not mutated Engine state and may
      // propagate without poisoning the Engine.
      planning_result = scheduler_->PlanStep(step_plan_);
    } catch (const StepPlanningConsistencyError&) {
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
      RetainEvent(FailUnserviceableRequest(
          planning_result.unserviceable_request_id));
      return;
    }
    if (!planning_result.executable) {
      const auto outcome = planning_result.outcome;
      if (outcome.kind == StepOutcomeKind::NoWork &&
          !scheduler_->HasPendingRequests()) {
        return;
      }
      if (outcome.kind == StepOutcomeKind::UnserviceableRequest) {
        RetainEvent(FailUnserviceableRequest(outcome.request_id));
        return;
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
          fatal_fallback_error_);
    }

    std::unique_ptr<CacheStepReservation> reservation;
    try {
      // Reserve every paged block, fixed slot, and fixed staging tensor the complete plan needs up
      // front. The reservation can build model inputs, but it does not alter committed ownership or
      // token boundaries until Commit(). Prove the reservation matches the plan exactly -- required
      // flag, row count, new-slot count, staging bytes, and per-row request identity -- so a plan/reservation
      // divergence fails here (fatal) rather than silently committing mismatched state.
      reservation = cache_manager_->ReserveStep(step_plan_);
      const auto fixed_slots = reservation->FixedStateSlots();
      const bool has_fixed_state = !fixed_slots.empty();
      if (step_plan_.fixed_state.required != has_fixed_state ||
          (has_fixed_state &&
           step_plan_.fixed_state.row_count != step_plan_.requests.size()) ||
          fixed_slots.size() != step_plan_.fixed_state.row_count ||
          reservation->FixedStateNewSlotCount() !=
              step_plan_.fixed_state.new_slot_count ||
          reservation->FixedStateStagingBytes() !=
              step_plan_.fixed_state.staging_bytes) {
        throw std::logic_error(
            "State reservation does not match the planned fixed-state resources.");
      }
      // For a fixed-state plan, every prior condition passing proves
      // fixed_slots.size() == row_count == requests.size(), so indexing requests by a fixed-slot row
      // below is in bounds even when a buggy cache manager over-reports row_count.
      for (size_t row = 0; row < fixed_slots.size(); ++row) {
        if (fixed_slots[row].request_id !=
            step_plan_.requests[row].request_id) {
          throw std::logic_error(
              "Fixed state slots do not match scheduled request row order.");
        }
      }
    } catch (...) {
      ++transaction_metrics_.reservation_failures;
      MarkUnhealthyAndThrow(
          StepOutcomeKind::ExecutionContractFailure,
          step_plan_.transaction_id,
          nullptr,
          "Failed to reserve the planned cache transaction.",
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
    context.fixed_state_slots = reservation->FixedStateSlots();
    context.fixed_state_bindings = reservation->FixedStateBindings();
    context.fixed_state_staging_bytes = reservation->FixedStateStagingBytes();

    bool request_transaction_active = false;
    const auto rollback_transaction = [&]() {
      // Request/search state and composite cache state are checkpointed separately. Both must be
      // restored so a retry observes exactly the state that existed before this Run() call. The
      // reservation's Release() discards fixed provisional slots and staged banks as well as the
      // reserved paged blocks.
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

    staged_event_order_.clear();
    try {
      // Turn the final logits row for each packed request into a staged next-token result. Request
      // counters, host token mirrors, and completion status still remain unchanged in this phase.
      scheduled_requests.GenerateNextTokensForTransaction(
          step_plan_, step_results_);
      // A verify step planned cache slots for every draft. Narrow the reservation to the accepted
      // prefix before anything is staged, so the paged and fixed states commit at one boundary.
      for (size_t i = 0; i < step_plan_.requests.size(); ++i) {
        const auto& entry = step_plan_.requests[i];
        if (entry.draft_token_count == 0) {
          continue;
        }
        reservation->CommitPrefix(
            i, entry.request_id, entry.unprocessed_token_count,
            entry.unprocessed_token_count - entry.draft_token_count +
                entry.request->AcceptedDraftTokenCount());
      }
      for (size_t i = 0; i < step_plan_.requests.size(); ++i) {
        if (step_results_[i].token_appended || step_results_[i].done) {
          staged_event_order_.push_back(i);
        }
      }
      std::sort(
          staged_event_order_.begin(), staged_event_order_.end(),
          [this](size_t left, size_t right) {
            return step_plan_.requests[left].scheduling_order <
                   step_plan_.requests[right].scheduling_order;
          });
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

    // Validate every ownership and capacity precondition and perform all fallible fixed device
    // work into inactive banks without publishing anything.
    try {
      reservation->PrepareCommit();
    } catch (...) {
      const auto preparation_error = std::current_exception();
      rollback_transaction();
      MarkUnhealthyAndThrow(
          StepOutcomeKind::ExecutionContractFailure,
          step_plan_.transaction_id,
          nullptr,
          "Transaction preparation failed and the Engine is no longer healthy.",
          preparation_error);
    }

    try {
      // Everything below crosses the commit boundary and is never retried. Make staged search state
      // durable, publish paged occupancy and the fixed bank flip, then advance the lightweight
      // Request bookkeeping that readers observe. Any failure after this point is fatal because a
      // cooperating component may already have crossed the shared token boundary.
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
    for (size_t i : staged_event_order_) {
      staged_events_.push_back(
          EventFromStep(step_plan_.requests[i].request, step_results_[i]));
    }
    pending_events_.swap(staged_events_);
    pending_event_index_ = 0;
    ++transaction_metrics_.committed_steps;
  }
}

size_t Engine::DrainPendingEvents(std::span<EngineEvent> events) {
  const size_t event_count = std::min(
      events.size(), pending_events_.size() - pending_event_index_);
  std::copy_n(
      pending_events_.begin() + static_cast<ptrdiff_t>(pending_event_index_),
      event_count, events.begin());
  pending_event_index_ += event_count;
  if (pending_event_index_ == pending_events_.size()) {
    pending_events_.clear();
    pending_event_index_ = 0;
  }
  return event_count;
}

void Engine::RetainEvent(EngineEvent event) {
  staged_events_.clear();
  staged_events_.push_back(std::move(event));
  pending_events_.swap(staged_events_);
  pending_event_index_ = 0;
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
    if (tracked && tracked.get() == request_id) {
      return tracked;
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
        fatal_fallback_error_);
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

EngineEvent Engine::EventFromStepError(
    const EngineStepError& error,
    std::exception_ptr caught_error) noexcept {
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
      event.finish_reason = GenerationFinishReason::Failed;
      event.error_code = EngineErrorCode::EngineContractFailure;
      health_ = EngineHealth::Unhealthy;
      if (!fatal_error_) {
        fatal_error_ = caught_error;
      }
      break;
    case StepOutcomeKind::FatalExecutionFailure:
      event.flags = EngineEventFlagFailed;
      event.finish_reason = GenerationFinishReason::Failed;
      event.error_code = EngineErrorCode::EngineExecutionFailure;
      health_ = EngineHealth::Unhealthy;
      if (!fatal_error_) {
        fatal_error_ = caught_error;
      }
      break;
    case StepOutcomeKind::UnserviceableRequest:
    case StepOutcomeKind::NoWork:
    case StepOutcomeKind::Committed:
      // These outcomes must be handled before throwing EngineStepError. Reaching this catch path is
      // itself an Engine contract failure; do not call a helper that can throw while translating it.
      event.flags = EngineEventFlagFailed;
      event.finish_reason = GenerationFinishReason::Failed;
      event.error_code = EngineErrorCode::EngineContractFailure;
      health_ = EngineHealth::Unhealthy;
      if (!fatal_error_) {
        fatal_error_ = caught_error;
      }
      break;
  }
  return event;
}

[[noreturn]] void Engine::MarkUnhealthyAndThrow(
    StepOutcomeKind outcome,
    StepTransactionId transaction_id,
    const void* request_id,
    std::string_view message,
    std::exception_ptr error) {
  if (health_ == EngineHealth::Unhealthy) {
    std::rethrow_exception(fatal_error_);
  }

  std::exception_ptr durable_error = fatal_fallback_error_;
  try {
    durable_error = std::make_exception_ptr(EngineStepError{
        {outcome, transaction_id, request_id},
        AddExceptionCause(std::string{message}, error),
    });
  } catch (...) {
    // The fallback was constructed with the Engine, before any Request could be published.
  }

  fatal_events_.clear();
  for (size_t i = pending_event_index_; i < pending_events_.size(); ++i) {
    const auto& pending = pending_events_[i];
    if (!pending.request || !pending.request->IsExternallyAbandoned()) {
      fatal_events_.push_back(pending);
    }
  }
  const auto error_code =
      outcome == StepOutcomeKind::ExecutionContractFailure
          ? EngineErrorCode::EngineContractFailure
          : EngineErrorCode::EngineExecutionFailure;
  bool affected_turn = false;
  for (const auto& request : tracked_requests_) {
    if (request && IsExecutable(request->status_) &&
        !request->IsExternallyAbandoned()) {
      affected_turn = true;
      EngineEvent event;
      event.request = request;
      event.turn_id = request->current_turn_id_;
      event.flags = EngineEventFlagTurnFinished | EngineEventFlagFailed;
      event.finish_reason = GenerationFinishReason::Failed;
      event.error_code = error_code;
      event.usage = {
          request->turn_prompt_tokens_,
          request->turn_generated_tokens_,
          0};
      const auto existing = std::find_if(
          fatal_events_.begin(), fatal_events_.end(),
          [&request](const EngineEvent& pending) {
            return pending.request == request &&
                   pending.turn_id == request->current_turn_id_;
          });
      if (existing == fatal_events_.end()) {
        fatal_events_.push_back(std::move(event));
      } else {
        existing->flags |= event.flags;
        existing->finish_reason = event.finish_reason;
        existing->error_code = event.error_code;
        existing->usage = event.usage;
      }
    }
  }
  if (!affected_turn) {
    EngineEvent event;
    event.flags = EngineEventFlagFailed;
    event.finish_reason = GenerationFinishReason::Failed;
    event.error_code = error_code;
    fatal_events_.push_back(std::move(event));
  }

  fatal_error_ = durable_error;
  health_ = EngineHealth::Unhealthy;
  if (outcome == StepOutcomeKind::FatalExecutionFailure ||
      outcome == StepOutcomeKind::ExecutionContractFailure) {
    ++transaction_metrics_.fatal_execution_failures;
  }
  for (const auto& request : tracked_requests_) {
    if (request && IsExecutable(request->status_)) {
      request->status_ = RequestStatus::TurnComplete;
      request->finish_reason_ = GenerationFinishReason::Failed;
    }
  }
  ReleaseDeferredStaticCloseState();
  pending_events_.swap(fatal_events_);
  pending_event_index_ = 0;
  std::rethrow_exception(fatal_error_);
}

bool Engine::HasPendingRequests() {
  ValidateOwnerThread();
  try {
    ReclaimAbandonedRequests();
  } catch (const EngineStepError&) {
    if (pending_event_index_ < pending_events_.size()) {
      return true;
    }
    throw;
  }
  return pending_event_index_ < pending_events_.size() ||
         scheduler_->HasPendingRequests();
}

size_t Engine::MaxDraftTokensPerStep() const {
  return cache_manager_->SupportsDynamicBatching() &&
                 model_executor_->SupportsDraftVerification()
             ? cache_manager_->MaxDraftTokensPerStep()
             : 0;
}

}  // namespace Generators
