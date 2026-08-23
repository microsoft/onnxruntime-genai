// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"
#include "../search.h"

namespace Generators {

static_assert(kMaxDraftTokensPerStep < kSpeculativeAcceptanceLengthBins);

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

std::vector<int32_t> GreedyTokens(
    const std::shared_ptr<DecoderOnly_Model>& model,
    std::vector<DeviceSpan<float>>& logits) {
  std::vector<int32_t> tokens(logits.size());
  auto device_tokens = model->p_device_->Allocate<int32_t>(logits.size());
  (void)device_tokens.CpuSpan();
  bool device_argmax = true;
  for (size_t i = 0; i < logits.size(); ++i) {
    device_argmax &= model->p_device_->ArgMaxDevice(
        logits[i].Span().data(), Ort::TypeToTensorType<float>, 1,
        model->config_->model.vocab_size, device_tokens.subspan(i, 1));
  }
  if (device_argmax) {
    device_tokens.CopyDeviceToCpu();
    std::copy(device_tokens.CpuSpan().begin(), device_tokens.CpuSpan().end(),
              tokens.begin());
    return tokens;
  }

  for (size_t i = 0; i < logits.size(); ++i) {
    if (!model->p_device_->ArgMax(
            logits[i].Span().data(), Ort::TypeToTensorType<float>, 1,
            model->config_->model.vocab_size, &tokens[i])) {
      const auto values = logits[i].CopyDeviceToCpu();
      tokens[i] = static_cast<int32_t>(
          std::max_element(values.begin(), values.end()) - values.begin());
    }
  }
  return tokens;
}

}  // namespace

Engine::Engine(std::shared_ptr<Model> model)
    : Engine(model, CreateDependencies(model)) {}

Engine::Engine(std::shared_ptr<Model> model, EngineDependencies dependencies)
    : model_{std::move(model)},
      cache_manager_{std::move(dependencies.cache_manager)},
      scheduler_{std::move(dependencies.scheduler)},
      model_executor_{std::move(dependencies.model_executor)},
      mtp_model_{std::move(dependencies.mtp_model)},
      mtp_cache_manager_{std::move(dependencies.mtp_cache_manager)},
      mtp_model_executor_{std::move(dependencies.mtp_model_executor)} {
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
  mtp_requests_.reserve(max_batch_size);
}

EngineDependencies Engine::CreateDependencies(std::shared_ptr<Model> model) {
  std::shared_ptr<DecoderOnly_Model> mtp_model;
  size_t mtp_bytes_per_block = 0;
  if (!model->config_->model.mtp.filename.empty()) {
    if (!model->config_->engine.dynamic_batching) {
      throw std::runtime_error("An Engine-hosted MTP head requires dynamic batching.");
    }
    mtp_model = std::make_shared<DecoderOnly_Model>(
        CreateMtpDecoderConfig(*model->config_), GetOrtEnv());
    mtp_bytes_per_block = PagedKeyValueCacheBytesPerBlock(mtp_model);
  }

  std::shared_ptr<CacheManager> cache_manager =
      CacheManager::Create(model, mtp_bytes_per_block);
  auto scheduler = Scheduler::Create(model, cache_manager);
  auto model_executor = ModelExecutor::Create(model, cache_manager);

  std::shared_ptr<CacheManager> mtp_cache_manager;
  std::unique_ptr<ModelExecutor> mtp_model_executor;
  if (mtp_model) {
    // Both decoders cover the same resident request set. Fixing the head to the main pool's block
    // count makes the auxiliary bytes included above exact rather than letting it independently
    // consume another gpu_utilization_factor share of currently free memory.
    mtp_model->config_->engine.dynamic_batching->num_blocks =
        cache_manager->Snapshot().total_blocks;
    mtp_cache_manager = CacheManager::Create(mtp_model);
    mtp_model_executor = ModelExecutor::Create(mtp_model, mtp_cache_manager);
  }

  return EngineDependencies{
      std::move(cache_manager), std::move(scheduler), std::move(model_executor),
      std::move(mtp_model), std::move(mtp_cache_manager), std::move(mtp_model_executor)};
}

std::unique_ptr<Engine::MtpStep> Engine::PrepareMtpStep(
    const StepPlan& target_plan,
    const std::vector<RequestStepResult>& target_results,
    ScheduledRequests& target_requests) {
  if (!mtp_model_ || MaxDraftTokensPerStep() == 0) {
    return nullptr;
  }
  if (target_results.size() != target_plan.requests.size()) {
    throw std::logic_error("MTP preparation requires one target result per planned request.");
  }

  Tensor* target_hidden_states = target_requests.HiddenStates();
  if (!target_hidden_states) {
    throw std::logic_error("The main decoder did not expose hidden states for its MTP head.");
  }
  const auto target_hidden_shape = target_hidden_states->GetShape();
  const int64_t hidden_size = model_->config_->model.decoder.hidden_size;
  if (target_hidden_shape !=
      std::vector<int64_t>{static_cast<int64_t>(target_plan.token_count), hidden_size}) {
    throw std::logic_error("The main decoder hidden-state shape does not match its packed step plan.");
  }
  const auto hidden_type = target_hidden_states->GetType();
  const auto& mtp_hidden_name = mtp_model_->config_->model.decoder.inputs.hidden_states;
  if (hidden_type != mtp_model_->session_info_.GetInputDataType(mtp_hidden_name)) {
    throw std::logic_error("The main and MTP hidden-state element types do not match.");
  }

  struct Feed {
    std::shared_ptr<Request> target;
    std::shared_ptr<Request> shadow;
    std::vector<int32_t> tokens;
    size_t target_hidden_row{};
    size_t max_draft_tokens{};
    bool newly_created{};
  };
  std::vector<Feed> feeds;
  feeds.reserve(target_plan.requests.size());
  std::vector<std::shared_ptr<Request>> checkpointed_shadows;
  checkpointed_shadows.reserve(target_plan.requests.size());
  size_t total_rows = 0;
  try {
    for (size_t i = 0; i < target_plan.requests.size(); ++i) {
      const auto& entry = target_plan.requests[i];
      const auto& result = target_results[i];
      const auto& search = entry.request->SearchOptions();
      const bool greedy = !search.do_sample || search.top_k == 1 || search.temperature == 0;
      const int64_t committed_length_after_step =
          entry.request->CurrentSequenceLength() + (result.token_appended ? 1 : 0);
      const size_t max_draft_tokens = std::min({MaxDraftTokensPerStep(),
                                                static_cast<size_t>(entry.request->params_->speculative.max_draft_tokens),
                                                committed_length_after_step + 1 < search.max_length
                                                    ? static_cast<size_t>(search.max_length - committed_length_after_step - 1)
                                                    : size_t{0}});
      if (!result.token_appended || result.done || !greedy ||
          search.repetition_penalty != 1.0f || search.no_repeat_ngram_size > 0 ||
          search.min_length > entry.request->CurrentSequenceLength() ||
          max_draft_tokens == 0) {
        continue;
      }

      Feed feed;
      feed.target = entry.request;
      const size_t accepted = entry.request->AcceptedDraftTokenCount();
      if (accepted > entry.draft_token_count) {
        throw std::logic_error("MTP preparation observed more accepted drafts than the target planned.");
      }
      const auto staged_drafts = entry.request->StagedDraftTokens();
      feed.tokens.insert(feed.tokens.end(), staged_drafts.begin(),
                         staged_drafts.begin() + static_cast<ptrdiff_t>(accepted));
      feed.tokens.push_back(result.token);
      feed.target_hidden_row = entry.packed_token_offset +
                               (entry.draft_token_count == 0
                                    ? entry.unprocessed_token_count - 1
                                    : 0);
      feed.max_draft_tokens = max_draft_tokens;

      const auto existing = mtp_requests_.find(entry.request.get());
      if (existing != mtp_requests_.end()) {
        feed.shadow = existing->second;
        feed.shadow->SaveStateForTransaction();
        checkpointed_shadows.push_back(feed.shadow);
        feed.shadow->AppendTokensForAuxiliaryDecoder(feed.tokens);
      } else {
        auto params = CreateGeneratorParams(*mtp_model_);
        params->search = search;
        feed.shadow = std::make_shared<Request>(std::move(params));
        feed.shadow->AddTokens(feed.tokens);
        feed.shadow->Assign(shared_from_this());
        feed.shadow->Schedule();
        feed.newly_created = true;
      }
      total_rows += feed.tokens.size();
      feeds.push_back(std::move(feed));
    }
  } catch (...) {
    const auto setup_error = std::current_exception();
    for (auto it = checkpointed_shadows.rbegin(); it != checkpointed_shadows.rend(); ++it) {
      try {
        (*it)->RestoreStateForTransaction();
      } catch (...) {
      }
    }
    std::rethrow_exception(setup_error);
  }
  if (feeds.empty()) {
    return nullptr;
  }

  auto step = std::make_unique<MtpStep>();
  step->plan.transaction_id = target_plan.transaction_id;
  step->plan.scheduled_request_limit = feeds.size();
  step->plan.token_count = total_rows;
  step->target_requests.reserve(feeds.size());
  step->newly_created.reserve(feeds.size());
  step->drafts.resize(feeds.size());

  size_t packed_offset = 0;
  for (const auto& feed : feeds) {
    const size_t token_count = feed.tokens.size();
    const size_t processed = static_cast<size_t>(feed.shadow->ProcessedSequenceLength());
    step->plan.requests.push_back(RequestStepPlan{
        feed.shadow,
        feed.shadow.get(),
        feed.shadow->CurrentSequenceLength(),
        token_count,
        0,
        packed_offset,
        packed_offset + token_count - 1,
        processed + token_count,
        static_cast<size_t>(feed.shadow->CurrentSequenceLength()) +
            feed.max_draft_tokens - 1,
        feed.shadow->IsPrefill(),
        feed.newly_created,
    });
    step->target_requests.push_back(feed.target);
    step->newly_created.push_back(feed.newly_created);
    packed_offset += token_count;
  }
  step->plan.graph_capture_eligible = std::all_of(
      step->plan.requests.begin(), step->plan.requests.end(),
      [](const RequestStepPlan& entry) {
        return !entry.is_prefill && entry.unprocessed_token_count == 1;
      });

  try {
    const auto planning = mtp_cache_manager_->PlanStepResources(step->plan);
    if (!planning.executable || step->plan.requests.size() != feeds.size()) {
      throw std::runtime_error("The MTP cache could not reserve every target-committed suffix.");
    }
    step->reservation = mtp_cache_manager_->ReserveStep(step->plan);

    Tensor packed_hidden_states{mtp_model_->p_device_inputs_, hidden_type};
    const std::array<int64_t, 2> packed_hidden_shape{
        static_cast<int64_t>(total_rows), hidden_size};
    packed_hidden_states.CreateTensor(packed_hidden_shape);
    const size_t row_bytes = static_cast<size_t>(hidden_size) * Ort::SizeOf(hidden_type);
    auto source_bytes = target_hidden_states->GetByteSpan();
    auto destination_bytes = packed_hidden_states.GetByteSpan();
    size_t destination_row = 0;
    for (const auto& feed : feeds) {
      const size_t row_count = feed.tokens.size();
      destination_bytes.subspan(destination_row * row_bytes, row_count * row_bytes)
          .CopyFrom(source_bytes.subspan(feed.target_hidden_row * row_bytes,
                                         row_count * row_bytes));
      destination_row += row_count;
    }

    ScheduledRequests mtp_requests{step->plan, mtp_model_, nullptr, nullptr};
    ExecutionContext context{&step->plan};
    context.cache_reservation = step->reservation->PagedReservation();
    context.hidden_states_input = packed_hidden_states.GetOrtTensor();
    mtp_model_executor_->Decode(mtp_requests, context);
    ++speculative_stats_.draft_forward_passes;
    auto logits = mtp_requests.ProcessLogits();
    const auto first_drafts = GreedyTokens(mtp_model_, logits);
    for (size_t i = 0; i < first_drafts.size(); ++i) {
      step->drafts[i].reserve(feeds[i].max_draft_tokens);
      step->drafts[i].push_back(first_drafts[i]);
      step->plan.requests[i].request->CommitAuxiliaryDecoderStep();
    }

    const auto copy_hidden_rows = [&](Tensor& source,
                                      std::span<const size_t> source_rows) {
      auto destination = std::make_unique<Tensor>(mtp_model_->p_device_inputs_, hidden_type);
      const std::array<int64_t, 2> shape{
          static_cast<int64_t>(source_rows.size()), hidden_size};
      destination->CreateTensor(shape);
      const size_t row_bytes = static_cast<size_t>(hidden_size) * Ort::SizeOf(hidden_type);
      auto source_bytes = source.GetByteSpan();
      auto destination_bytes = destination->GetByteSpan();
      for (size_t row = 0; row < source_rows.size(); ++row) {
        destination_bytes.subspan(row * row_bytes, row_bytes)
            .CopyFrom(source_bytes.subspan(source_rows[row] * row_bytes, row_bytes));
      }
      return destination;
    };

    std::vector<size_t> active_feed_indices;
    std::vector<size_t> feedback_rows;
    for (size_t i = 0; i < feeds.size(); ++i) {
      if (feeds[i].max_draft_tokens > 1) {
        active_feed_indices.push_back(i);
        feedback_rows.push_back(step->plan.requests[i].logits_row_index);
      }
    }
    std::unique_ptr<Tensor> feedback_hidden;
    if (!active_feed_indices.empty()) {
      Tensor* head_hidden = mtp_requests.HiddenStates();
      if (!head_hidden ||
          head_hidden->GetShape() !=
              std::vector<int64_t>{static_cast<int64_t>(total_rows), hidden_size}) {
        throw std::runtime_error(
            "Chained MTP drafts require one configured head hidden-state output row per input token.");
      }
      feedback_hidden = copy_hidden_rows(*head_hidden, feedback_rows);
    }

    for (size_t draft_index = 1; !active_feed_indices.empty(); ++draft_index) {
      StepPlan chain_plan;
      chain_plan.transaction_id = target_plan.transaction_id;
      chain_plan.scheduled_request_limit = active_feed_indices.size();
      chain_plan.token_count = active_feed_indices.size();
      chain_plan.proposed_block_table_columns = step->plan.proposed_block_table_columns;
      chain_plan.graph_capture_eligible = true;
      chain_plan.requests.reserve(active_feed_indices.size());
      for (size_t feed_index : active_feed_indices) {
        auto& shadow = feeds[feed_index].shadow;
        const std::array<int32_t, 1> token{step->drafts[feed_index].back()};
        shadow->AppendTokensForAuxiliaryDecoder(token);
        const size_t processed = static_cast<size_t>(shadow->ProcessedSequenceLength());
        chain_plan.requests.push_back(RequestStepPlan{
            shadow,
            shadow.get(),
            shadow->CurrentSequenceLength(),
            1,
            0,
            chain_plan.requests.size(),
            chain_plan.requests.size(),
            processed + 1,
            step->plan.requests[feed_index].whole_sequence_cache_slots,
            false,
            false,
        });
      }

      ScheduledRequests chain_requests{chain_plan, mtp_model_, nullptr, nullptr};
      ExecutionContext chain_context{&chain_plan};
      chain_context.cache_reservation = step->reservation->PagedReservation();
      chain_context.hidden_states_input = feedback_hidden->GetOrtTensor();
      mtp_model_executor_->Decode(chain_requests, chain_context);
      ++speculative_stats_.draft_forward_passes;
      auto chain_logits = chain_requests.ProcessLogits();
      const auto chain_drafts = GreedyTokens(mtp_model_, chain_logits);
      for (size_t row = 0; row < active_feed_indices.size(); ++row) {
        const size_t feed_index = active_feed_indices[row];
        step->drafts[feed_index].push_back(chain_drafts[row]);
        feeds[feed_index].shadow->CommitAuxiliaryDecoderStep();
      }

      std::vector<size_t> next_active_feed_indices;
      std::vector<size_t> next_feedback_rows;
      for (size_t row = 0; row < active_feed_indices.size(); ++row) {
        if (feeds[active_feed_indices[row]].max_draft_tokens > draft_index + 1) {
          next_active_feed_indices.push_back(active_feed_indices[row]);
          next_feedback_rows.push_back(row);
        }
      }
      if (!next_active_feed_indices.empty()) {
        Tensor* head_hidden = chain_requests.HiddenStates();
        if (!head_hidden ||
            head_hidden->GetShape() !=
                std::vector<int64_t>{static_cast<int64_t>(active_feed_indices.size()),
                                     hidden_size}) {
          throw std::runtime_error(
              "Chained MTP hidden-state output does not match the active request batch.");
        }
        feedback_hidden = copy_hidden_rows(*head_hidden, next_feedback_rows);
      }
      active_feed_indices = std::move(next_active_feed_indices);
    }

    for (size_t i = 0; i < feeds.size(); ++i) {
      feeds[i].shadow->RewindAuxiliaryDecoderTo(
          static_cast<size_t>(step->plan.requests[i].target_cache_slots));
    }
    for (size_t i = 0; i < step->plan.requests.size(); ++i) {
      if (step->newly_created[i]) {
        mtp_requests_.emplace(step->target_requests[i].get(),
                              step->plan.requests[i].request);
      }
    }
  } catch (...) {
    RollbackMtpStep(*step);
    throw;
  }
  return step;
}

void Engine::RollbackMtpStep(MtpStep& step) {
  std::exception_ptr rollback_error;
  if (step.reservation) {
    try {
      step.reservation->Release();
    } catch (...) {
      rollback_error = std::current_exception();
    }
    step.reservation.reset();
  }
  for (size_t i = 0; i < step.plan.requests.size(); ++i) {
    if (step.newly_created[i]) {
      mtp_requests_.erase(step.target_requests[i].get());
      continue;
    }
    try {
      step.plan.requests[i].request->RestoreStateForTransaction();
    } catch (...) {
      if (!rollback_error) {
        rollback_error = std::current_exception();
      }
    }
  }
  if (rollback_error) {
    std::rethrow_exception(rollback_error);
  }
}

void Engine::CommitMtpStep(MtpStep& step) {
  for (size_t i = 0; i < step.plan.requests.size(); ++i) {
    if (!step.newly_created[i]) {
      step.plan.requests[i].request->CommitStateForTransaction();
    }
  }
  step.reservation->Commit();
  step.reservation.reset();
  for (size_t i = 0; i < step.plan.requests.size(); ++i) {
    step.plan.requests[i].request->CommitAuxiliaryDecoderStep();
  }
}

void Engine::PublishMtpDrafts(MtpStep& step) {
  for (size_t i = 0; i < step.plan.requests.size(); ++i) {
    step.target_requests[i]->SetDraftTokens(step.drafts[i]);
  }
}

void Engine::RecordSpeculativeCommit(const StepPlan& plan) noexcept {
  for (const auto& entry : plan.requests) {
    if (entry.draft_token_count == 0) {
      continue;
    }

    const size_t accepted = entry.request->AcceptedDraftTokenCount();
    ++speculative_stats_.rounds;
    ++speculative_stats_.completed_rounds;
    speculative_stats_.draft_tokens_proposed += entry.draft_token_count;
    speculative_stats_.draft_tokens_evaluated +=
        std::min(accepted + 1, entry.draft_token_count);
    speculative_stats_.draft_tokens_accepted += accepted;
    ++speculative_stats_.acceptance_length_histogram[accepted];
    if (accepted == 0) {
      ++speculative_stats_.zero_accept_rounds;
    } else if (accepted == entry.draft_token_count) {
      ++speculative_stats_.full_accept_rounds;
    } else {
      ++speculative_stats_.partial_accept_rounds;
    }
  }
}

void Engine::AddRequest(std::shared_ptr<Request> request) {
  ReclaimAbandonedRequests();
  if (cache_manager_->SupportsDynamicBatching()) {
    request->ValidateEngineCompatibility();
  }

  // Track the request before assignment so every successfully submitted request can later be found
  // even when the scheduler and cache hold it through implementation-specific containers. The
  // registry allocation therefore happens before any request lifecycle mutation.
  tracked_requests_.push_back(request);
  try {
    request->Assign(shared_from_this());
    scheduler_->AddRequest(request);
  } catch (...) {
    tracked_requests_.pop_back();
    throw;
  }
}

void Engine::RemoveRequest(std::shared_ptr<Request> request) {
  if (request && IsClosed(request->status_)) {
    return;
  }
  if (!request || request->engine_.lock().get() != this) {
    throw std::runtime_error("Cannot remove a request from an engine it does not belong to.");
  }

  scheduler_->RemoveRequest(request);

  if (const auto mtp_it = mtp_requests_.find(request.get());
      mtp_it != mtp_requests_.end()) {
    std::vector<std::shared_ptr<Request>> mtp_requests{mtp_it->second};
    mtp_cache_manager_->Deallocate(mtp_requests);
    mtp_it->second->CompleteClose();
    mtp_requests_.erase(mtp_it);
  }

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
  // ExternalRelease only publishes an atomic abandonment marker. Engine entry points are externally
  // serialized, so this boundary can safely perform the normal removal sequence: scheduler/cache
  // release, ready-notification purge, and terminal close.
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
      RemoveRequest(request);
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
        "call Engine::Step() to drain the ready notification before continuing.");
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
    RemoveRequest(request);
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

std::shared_ptr<Request> Engine::Step() {
  ReclaimAbandonedRequests();
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
    std::vector<RequestStatus> statuses_before_step;
    statuses_before_step.reserve(scheduled_requests.size());
    for (const auto& request : scheduled_requests) {
      statuses_before_step.push_back(request->status_);
    }

    model_executor_->Decode(scheduled_requests);
    ++speculative_stats_.target_forward_passes;
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

std::shared_ptr<Request> Engine::StepDynamic() {
  while (scheduler_->HasPendingRequests()) {
    // A dynamic step is a transaction with six phases:
    // plan -> reserve state -> checkpoint request state -> execute -> stage sampled tokens -> commit.
    // Nothing becomes externally visible until the final commit succeeds.
    step_plan_.transaction_id = next_transaction_id_++;
    StepPlanningResult planning_result;
    try {
      // PlanStep is expected to report non-executable outcomes through its result, not by throwing.
      // A throw here means a planning-time consistency check failed -- for a composite model,
      // PagedCacheManager::PlanStepResources proves paged and fixed committed ownership agree and
      // throws std::logic_error on any divergence. That is proven state corruption, so route it
      // through the same fatal, structured path as every other execution-contract violation instead
      // of letting a raw std::logic_error escape Step() with the Engine still marked healthy.
      planning_result = scheduler_->PlanStep(step_plan_);
    } catch (...) {
      MarkUnhealthyAndThrow(
          StepOutcomeKind::ExecutionContractFailure,
          step_plan_.transaction_id,
          nullptr,
          "Dynamic step planning failed a cache/state consistency check.",
          std::current_exception());
    }
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
      // Reserve every paged block, fixed slot, and fixed staging tensor the complete plan needs up
      // front. The reservation can build model inputs, but it does not alter committed ownership or
      // token boundaries until Commit(). Prove the reservation matches the plan exactly -- required
      // flag, row count, staging bytes, and per-row request identity -- so a plan/reservation
      // divergence fails here (fatal) rather than silently committing mismatched state.
      reservation = cache_manager_->ReserveStep(step_plan_);
      const auto fixed_slots = reservation->FixedStateSlots();
      const bool has_fixed_state = !fixed_slots.empty();
      if (step_plan_.fixed_state.required != has_fixed_state ||
          (has_fixed_state &&
           step_plan_.fixed_state.row_count != step_plan_.requests.size()) ||
          fixed_slots.size() != step_plan_.fixed_state.row_count ||
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
    std::unique_ptr<MtpStep> mtp_step;
    const auto rollback_transaction = [&]() {
      // Request/search state and composite cache state are checkpointed separately. Both must be
      // restored so a retry observes exactly the state that existed before this Step() call. The
      // reservation's Release() discards fixed provisional slots and staged banks as well as the
      // reserved paged blocks.
      std::exception_ptr rollback_error;
      if (mtp_step) {
        try {
          RollbackMtpStep(*mtp_step);
        } catch (...) {
          rollback_error = std::current_exception();
        }
        mtp_step.reset();
      }
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
      ++speculative_stats_.target_forward_passes;
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
      mtp_step = PrepareMtpStep(step_plan_, step_results_, scheduled_requests);
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
      // Commit order is deliberate. First validate every ownership and capacity precondition of the
      // whole reservation and perform all fallible fixed device work into inactive banks, without
      // publishing anything. A failure here is fatal even though committed state is intact: the
      // fixed inactive banks may be partly written and cannot be proven consistent for a retry.
      reservation->PrepareCommit();
      if (mtp_step) {
        mtp_step->reservation->PrepareCommit();
      }
      // Everything below crosses the commit boundary and is never retried. Make staged search state
      // durable, publish paged occupancy and the fixed bank flip, then advance the lightweight
      // Request bookkeeping that readers observe. Any failure after this point is fatal because a
      // cooperating component may already have crossed the shared token boundary.
      scheduled_requests.CommitStateForTransaction();
      request_transaction_active = false;
      reservation->Commit();
      if (mtp_step) {
        CommitMtpStep(*mtp_step);
      }
      RecordSpeculativeCommit(step_plan_);
      for (size_t i = 0; i < step_plan_.requests.size(); ++i) {
        step_plan_.requests[i].request->CommitStep(
            step_plan_.requests[i], step_results_[i]);
      }
      if (mtp_step) {
        PublishMtpDrafts(*mtp_step);
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

size_t Engine::MaxDraftTokensPerStep() const {
  return cache_manager_->SupportsDynamicBatching()
             ? cache_manager_->MaxDraftTokensPerStep()
             : 0;
}

SpeculativeStats Engine::GetSpeculativeStats() const noexcept {
  auto stats = speculative_stats_;
  if (stats.draft_tokens_evaluated != 0) {
    stats.acceptance_rate = static_cast<float>(stats.draft_tokens_accepted) /
                            static_cast<float>(stats.draft_tokens_evaluated);
  }
  if (stats.rounds != 0) {
    stats.avg_draft_tokens_per_round =
        static_cast<float>(stats.draft_tokens_proposed) /
        static_cast<float>(stats.rounds);
  }
  return stats;
}

}  // namespace Generators
