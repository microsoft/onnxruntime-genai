// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "engine.h"
#include "../search.h"

#include <limits>

namespace Generators {

static_assert(kMaxDraftTokensPerStep < kSpeculativeAcceptanceLengthBins);

namespace {

struct MtpRollbackError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

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

bool TryGreedyTokensToDevice(
    const std::shared_ptr<DecoderOnly_Model>& model,
    std::vector<DeviceSpan<float>>& logits,
    DeviceSpan<int32_t> tokens) {
  if (tokens.size() != logits.size()) {
    return false;
  }

  const size_t vocab_size = static_cast<size_t>(model->config_->model.vocab_size);
  const bool contiguous = !logits.empty() && std::all_of(
                                                 logits.begin(), logits.end(),
                                                 [&](DeviceSpan<float>& row) {
                                                   const size_t index = &row - logits.data();
                                                   return row.size() == vocab_size &&
                                                          row.SameBufferAs(logits.front()) &&
                                                          row.Span().data() ==
                                                              logits.front().Span().data() +
                                                                  index * vocab_size;
                                                 });
  if (contiguous) {
    return model->p_device_->ArgMaxDevice(
        logits.front().Span().data(), Ort::TypeToTensorType<float>,
        static_cast<int>(logits.size()), model->config_->model.vocab_size, tokens);
  }

  for (size_t i = 0; i < logits.size(); ++i) {
    if (!model->p_device_->ArgMaxDevice(
            logits[i].Span().data(), Ort::TypeToTensorType<float>, 1,
            model->config_->model.vocab_size, tokens.subspan(i, 1))) {
      return false;
    }
  }
  return true;
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
  if (max_batch_size > staged_events_.max_size() /
                           kMaxGeneratedTokensPerStep) {
    throw std::overflow_error(
        "Engine event capacity exceeds the supported size.");
  }
  const size_t max_step_events =
      max_batch_size * kMaxGeneratedTokensPerStep;
  step_plan_.requests.reserve(max_batch_size);
  step_results_.reserve(max_batch_size);
  staged_event_order_.reserve(max_batch_size);
  pending_events_.reserve(max_step_events);
  staged_events_.reserve(max_step_events);
  mtp_requests_.reserve(max_batch_size);
}

Engine::~Engine() {
  while (!tracked_requests_.empty()) {
    auto request = std::move(tracked_requests_.back());
    tracked_requests_.pop_back();
    DetachRequestForTeardown(request);
  }
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
    // The head consumes the target decoder's packed hidden states, so the target must emit them.
    model->config_->engine.hidden_states_output_required = true;
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
  const auto restore_checkpointed_shadows = [&]() {
    std::exception_ptr rollback_error;
    for (auto it = checkpointed_shadows.rbegin();
         it != checkpointed_shadows.rend(); ++it) {
      try {
        (*it)->RestoreStateForTransaction();
      } catch (...) {
        if (!rollback_error) {
          rollback_error = std::current_exception();
        }
      }
    }
    if (rollback_error) {
      std::rethrow_exception(rollback_error);
    }
  };
  const auto rollback_setup_and_rethrow =
      [&](std::exception_ptr setup_error) -> void {
    try {
      restore_checkpointed_shadows();
    } catch (...) {
      auto message = AddExceptionCause(
          "MTP shadow setup failed.", setup_error);
      message = AddExceptionCause(
          std::move(message) + " Shadow rollback also failed.",
          std::current_exception());
      throw MtpRollbackError(std::move(message));
    }
    std::rethrow_exception(setup_error);
  };
  size_t total_rows = 0;
  try {
    for (size_t i = 0; i < target_plan.requests.size(); ++i) {
      const auto& entry = target_plan.requests[i];
      const auto& result = target_results[i];
      const auto& search = entry.request->SearchOptions();
      const int64_t committed_length_after_step =
          entry.request->CurrentSequenceLength();
      const size_t sequence_limit =
          std::min(static_cast<size_t>(search.max_length),
                   entry.request->MaxTotalTokens());
      const size_t remaining_turn_tokens =
          entry.request->RemainingTurnTokenBudget();
      const size_t remaining_turn_tokens_after_step =
          result.visible_token_count < remaining_turn_tokens
              ? remaining_turn_tokens - result.visible_token_count
              : 0;
      const size_t max_draft_tokens = std::min({MaxDraftTokensPerStep(),
                                                static_cast<size_t>(entry.request->SpeculativeOptions().max_draft_tokens),
                                                static_cast<size_t>(committed_length_after_step) + 1 < sequence_limit
                                                    ? sequence_limit - static_cast<size_t>(committed_length_after_step) - 1
                                                    : size_t{0},
                                                remaining_turn_tokens_after_step > 1
                                                    ? remaining_turn_tokens_after_step - 1
                                                    : size_t{0}});
      if (!result.token_appended || result.done ||
          entry.request->DraftTokenValidationError() ||
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
        feed.shadow = Request::CreateAuxiliaryDecoderRequest(
            std::move(params), entry.request->MaxTotalTokens(),
            abandonment_pending_, shared_from_this(), feed.tokens);
        feed.newly_created = true;
      }
      total_rows += feed.tokens.size();
      feeds.push_back(std::move(feed));
    }
  } catch (...) {
    rollback_setup_and_rethrow(std::current_exception());
  }
  if (feeds.empty()) {
    return nullptr;
  }

  std::unique_ptr<MtpStep> step;
  try {
    step = std::make_unique<MtpStep>();
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
  } catch (...) {
    rollback_setup_and_rethrow(std::current_exception());
  }

  try {
    const auto planning = mtp_cache_manager_->PlanStepResources(step->plan);
    if (!planning.executable || step->plan.requests.size() != feeds.size()) {
      throw std::runtime_error("The MTP cache could not reserve every target-committed suffix.");
    }
    step->reservation = mtp_cache_manager_->ReserveStep(step->plan);

    const size_t max_draft_tokens = std::max_element(
                                        feeds.begin(), feeds.end(),
                                        [](const Feed& left, const Feed& right) {
                                          return left.max_draft_tokens < right.max_draft_tokens;
                                        })
                                        ->max_draft_tokens;
    bool device_draft_chain =
        max_draft_tokens > 1 && mtp_model_->p_device_->GetType() == DeviceType::CUDA;
    DeviceSpan<int32_t> device_drafts;
    DeviceSpan<int32_t> device_chain_inputs;
    if (device_draft_chain) {
      const size_t draft_capacity = feeds.size() * max_draft_tokens;
      if (mtp_device_drafts_.size() < draft_capacity) {
        mtp_device_drafts_ = mtp_model_->p_device_->Allocate<int32_t>(draft_capacity);
      }
      if (mtp_device_chain_inputs_.size() < feeds.size()) {
        mtp_device_chain_inputs_ = mtp_model_->p_device_->Allocate<int32_t>(feeds.size());
      }
      device_drafts = mtp_device_drafts_.subspan(0, draft_capacity);
      device_chain_inputs = mtp_device_chain_inputs_.subspan(0, feeds.size());
    }
    std::vector<std::vector<size_t>> device_stage_feed_indices;
    device_stage_feed_indices.reserve(max_draft_tokens);

    const auto materialize_device_drafts = [&]() {
      auto draft_ids = device_drafts.CopyDeviceToCpu();
      for (size_t stage = 0; stage < device_stage_feed_indices.size(); ++stage) {
        const auto& feed_indices = device_stage_feed_indices[stage];
        for (size_t row = 0; row < feed_indices.size(); ++row) {
          step->drafts[feed_indices[row]].push_back(
              draft_ids[stage * feeds.size() + row]);
        }
      }
    };

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
    if (device_draft_chain) {
      context.run_options->AddConfigEntry("disable_synchronize_execution_providers", "1");
    }
    mtp_model_executor_->Decode(mtp_requests, context);
    ++speculative_stats_.draft_forward_passes;
    auto logits = mtp_requests.ProcessLogits();
    device_draft_chain =
        device_draft_chain &&
        TryGreedyTokensToDevice(
            mtp_model_, logits, device_drafts.subspan(0, feeds.size()));
    std::vector<int32_t> first_drafts;
    if (device_draft_chain) {
      std::vector<size_t> first_stage_feed_indices;
      first_stage_feed_indices.reserve(feeds.size());
      for (size_t i = 0; i < feeds.size(); ++i) {
        first_stage_feed_indices.push_back(i);
      }
      device_stage_feed_indices.push_back(std::move(first_stage_feed_indices));
    } else {
      first_drafts = GreedyTokens(mtp_model_, logits);
    }
    for (size_t i = 0; i < feeds.size(); ++i) {
      step->drafts[i].reserve(feeds[i].max_draft_tokens);
      if (!device_draft_chain) {
        step->drafts[i].push_back(first_drafts[i]);
      }
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
    std::vector<size_t> previous_stage_rows(feeds.size());
    std::vector<size_t> feedback_rows;
    for (size_t i = 0; i < feeds.size(); ++i) {
      previous_stage_rows[i] = i;
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

    // Keep each stage's decoder I/O alive until the final device-to-host copy completes; device
    // ArgMax can still be reading its logits after Decode returns.
    std::vector<std::unique_ptr<ScheduledRequests>> pending_device_requests;
    pending_device_requests.reserve(max_draft_tokens - 1);
    std::vector<std::unique_ptr<Tensor>> pending_device_inputs;
    pending_device_inputs.reserve(max_draft_tokens - 1);

    for (size_t draft_index = 1; !active_feed_indices.empty(); ++draft_index) {
      StepPlan chain_plan;
      chain_plan.transaction_id = target_plan.transaction_id;
      chain_plan.scheduled_request_limit = active_feed_indices.size();
      chain_plan.token_count = active_feed_indices.size();
      chain_plan.proposed_block_table_columns = step->plan.proposed_block_table_columns;
      chain_plan.graph_capture_eligible = true;
      chain_plan.requests.reserve(active_feed_indices.size());
      DeviceSpan<int32_t> packed_device_inputs;
      if (device_draft_chain) {
        packed_device_inputs = device_chain_inputs.subspan(0, active_feed_indices.size());
      }
      for (size_t row = 0; row < active_feed_indices.size(); ++row) {
        const size_t feed_index = active_feed_indices[row];
        auto& shadow = feeds[feed_index].shadow;
        if (device_draft_chain) {
          auto token = packed_device_inputs.subspan(row, 1);
          token.CopyFrom(device_drafts.subspan(
              (draft_index - 1) * feeds.size() + previous_stage_rows[feed_index], 1));
          shadow->AppendTokensForAuxiliaryDecoder(token);
        } else {
          const std::array<int32_t, 1> token{step->drafts[feed_index].back()};
          shadow->AppendTokensForAuxiliaryDecoder(token);
        }
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

      auto chain_requests = std::make_unique<ScheduledRequests>(
          chain_plan, mtp_model_, nullptr, nullptr);
      ExecutionContext chain_context{&chain_plan};
      chain_context.cache_reservation = step->reservation->PagedReservation();
      chain_context.hidden_states_input = feedback_hidden->GetOrtTensor();
      if (device_draft_chain) {
        chain_context.input_ids = packed_device_inputs;
        chain_context.run_options->AddConfigEntry(
            "disable_synchronize_execution_providers", "1");
      }
      mtp_model_executor_->Decode(*chain_requests, chain_context);
      ++speculative_stats_.draft_forward_passes;
      auto chain_logits = chain_requests->ProcessLogits();
      bool stage_on_device = false;
      if (device_draft_chain) {
        stage_on_device = TryGreedyTokensToDevice(
            mtp_model_, chain_logits,
            device_drafts.subspan(
                draft_index * feeds.size(), active_feed_indices.size()));
      }
      std::vector<int32_t> chain_drafts;
      if (stage_on_device) {
        device_stage_feed_indices.push_back(active_feed_indices);
      } else {
        if (device_draft_chain) {
          materialize_device_drafts();
          pending_device_requests.clear();
          pending_device_inputs.clear();
          device_draft_chain = false;
        }
        chain_drafts = GreedyTokens(mtp_model_, chain_logits);
      }
      for (size_t row = 0; row < active_feed_indices.size(); ++row) {
        const size_t feed_index = active_feed_indices[row];
        if (!device_draft_chain) {
          step->drafts[feed_index].push_back(chain_drafts[row]);
        }
        feeds[feed_index].shadow->CommitAuxiliaryDecoderStep();
        previous_stage_rows[feed_index] = row;
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
        Tensor* head_hidden = chain_requests->HiddenStates();
        if (!head_hidden ||
            head_hidden->GetShape() !=
                std::vector<int64_t>{static_cast<int64_t>(active_feed_indices.size()),
                                     hidden_size}) {
          throw std::runtime_error(
              "Chained MTP hidden-state output does not match the active request batch.");
        }
        auto next_feedback_hidden =
            copy_hidden_rows(*head_hidden, next_feedback_rows);
        if (device_draft_chain) {
          pending_device_inputs.push_back(std::move(feedback_hidden));
        }
        feedback_hidden = std::move(next_feedback_hidden);
      }
      if (device_draft_chain) {
        pending_device_requests.push_back(std::move(chain_requests));
      }
      active_feed_indices = std::move(next_active_feed_indices);
    }

    if (device_draft_chain) {
      materialize_device_drafts();
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
    const auto preparation_error = std::current_exception();
    try {
      RollbackMtpStep(*step);
    } catch (...) {
      auto message = AddExceptionCause(
          "MTP step preparation failed.", preparation_error);
      message = AddExceptionCause(
          std::move(message) + " MTP rollback also failed.",
          std::current_exception());
      throw MtpRollbackError(std::move(message));
    }
    std::rethrow_exception(preparation_error);
  }
  return step;
}

void Engine::RollbackMtpStep(MtpStep& step) {
  std::exception_ptr rollback_error;
  try {
    mtp_model_->p_device_scoring_->Synchronize();
  } catch (...) {
    rollback_error = std::current_exception();
  }
  if (step.reservation) {
    try {
      step.reservation->Release();
    } catch (...) {
      if (!rollback_error) {
        rollback_error = std::current_exception();
      }
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

void Engine::ValidateOwnerThread() const {
  if (std::this_thread::get_id() != owner_thread_) {
    throw std::runtime_error(
        "Engine operations must be called from the Engine owner thread.");
  }
}

std::shared_ptr<Request> Engine::CreateRequest(const GeneratorParams& params,
                                               size_t max_total_tokens) {
  ValidateOwnerThread();
  CompleteNonresidentClosedRequests();
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
  auto engine = shared_from_this();

  // Fatal handling may need one terminal event for every tracked Request. Grow that storage before
  // publishing the Request so failure reporting never allocates after the Engine becomes unhealthy.
  pending_events_.reserve(tracked_requests_.size() + 1);
  tracked_requests_.push_back(request);
  request->AttachToEngine(std::move(engine));
  return request;
}

uint64_t Engine::BeginTurn(const std::shared_ptr<Request>& request,
                           std::span<const int32_t> tokens,
                           std::optional<size_t> max_generated_tokens) {
  ValidateOwnerThread();
  CompleteNonresidentClosedRequests();
  ReclaimAbandonedRequests();
  if (health_ == EngineHealth::Unhealthy) {
    std::rethrow_exception(fatal_error_);
  }
  if (!request || !request->BelongsTo(*this)) {
    throw std::runtime_error(
        "Cannot begin a turn for a request that does not belong to this engine.");
  }
  request->ValidateTurnAdmission(tokens, max_generated_tokens);

  const bool first_turn = request->IsAwaitingFirstTurn();
  if (first_turn) {
    if (cache_manager_->SupportsDynamicBatching()) {
      request->ValidateEngineCompatibility();
    }
  } else {
    const bool restartable_canceled_turn =
        request->IsRestartableCanceledTurn() &&
        !cache_manager_->IsResident(request);
    ValidateRequestCanContinue(request, restartable_canceled_turn);
    if (!restartable_canceled_turn) {
      request->ValidateContinuousDecodingSupport();
    }
  }

  RequestTurnAdmission admission;
  bool added_to_scheduler = false;

  // A continuation appends a prompt the MTP shadow never sees, so keeping the shadow would leave it
  // a concatenation of generated tokens across turns with the intervening prompt missing. Drop it
  // here so the next drafted step rebuilds a suffix-local shadow, matching fresh-turn semantics.
  // Doing it before admission keeps a rolled back turn consistent: the shadow is simply rebuilt.
  CloseMtpRequest(request);

  try {
    request->PrepareTurnAdmission(tokens, admission);
    if (first_turn) {
      scheduler_->AddRequest(request);
      added_to_scheduler = true;
    }
    return request->CommitTurnAdmission(
        max_generated_tokens, admission);
  } catch (...) {
    const auto append_error = std::current_exception();
    try {
      if (added_to_scheduler) {
        scheduler_->RemoveRequest(request);
      }
      request->RollbackTurnAdmission(admission);
    } catch (...) {
      HandleContinuationRestoreFailure(
          request, append_error, std::current_exception());
    }
    std::rethrow_exception(append_error);
  }
}

bool Engine::CancelRequest(const std::shared_ptr<Request>& request, uint64_t turn_id) {
  ValidateOwnerThread();
  if (!request) {
    throw std::runtime_error(
        "Cannot cancel a request that does not belong to this engine.");
  }
  if (!request->CanCancelFromEngine(*this, turn_id)) {
    return false;
  }

  pending_events_.erase(
      pending_events_.begin(),
      pending_events_.begin() + static_cast<ptrdiff_t>(pending_event_index_));
  pending_event_index_ = 0;
  auto existing = std::find_if(
      pending_events_.rbegin(), pending_events_.rend(),
      [&request, turn_id](const EngineEvent& event) {
        return event.request == request && event.turn_id == turn_id;
      });
  const bool has_existing_event = existing != pending_events_.rend();
  if (!has_existing_event) {
    pending_events_.reserve(pending_events_.size() + 1);
  }

  const auto counters =
      request->CompleteCancelFromEngine(*this, turn_id);
  // The canceled turn's suffix is gone, so the shadow that mirrored it must not survive into the
  // next turn.
  CloseMtpRequest(request);
  EngineEvent terminal;
  terminal.request = request;
  terminal.turn_id = turn_id;
  terminal.flags = EngineEventFlagTurnFinished;
  terminal.finish_reason = GenerationFinishReason::Canceled;
  terminal.usage = {
      counters.prompt_tokens,
      counters.generated_tokens,
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
  if (!request || !request->BelongsTo(*this)) {
    throw std::runtime_error("Cannot close a request that does not belong to this engine.");
  }

  scheduler_->RemoveRequest(request);
  const bool retain_static_runtime_state =
      !cache_manager_->SupportsDynamicBatching() &&
      cache_manager_->IsResident(request);

  CloseMtpRequest(request);

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
  request->MarkClosedFromEngine(*this);
  if (retain_static_runtime_state) {
    return;
  }

  request->CompleteCloseFromEngine(*this);
  tracked_requests_.erase(
      std::remove_if(
          tracked_requests_.begin(), tracked_requests_.end(),
          [&request](const std::shared_ptr<Request>& tracked) {
            return tracked == request;
          }),
      tracked_requests_.end());
}

void Engine::CloseMtpRequest(const std::shared_ptr<Request>& request) {
  const auto mtp_it = mtp_requests_.find(request.get());
  if (mtp_it == mtp_requests_.end()) {
    return;
  }
  std::vector<std::shared_ptr<Request>> mtp_requests{mtp_it->second};
  mtp_cache_manager_->Deallocate(mtp_requests);
  mtp_it->second->CompleteCloseFromEngine(*this);
  mtp_requests_.erase(mtp_it);
}

void Engine::DetachRequestForTeardown(
    const std::shared_ptr<Request>& request) noexcept {
  if (!request) {
    return;
  }

  scheduler_->DetachRequestForTeardown(request);
  if (const auto mtp_it = mtp_requests_.find(request.get());
      mtp_it != mtp_requests_.end()) {
    try {
      std::vector<std::shared_ptr<Request>> mtp_requests{mtp_it->second};
      mtp_cache_manager_->Deallocate(mtp_requests);
    } catch (...) {
    }
    mtp_it->second->CompleteCloseFromEngine(*this);
    mtp_requests_.erase(mtp_it);
  }
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
  request->CompleteCloseFromEngine(*this);
}

void Engine::ReclaimAbandonedRequests() {
  // ExternalRelease only publishes an atomic abandonment marker. The host's owner-thread boundary
  // can safely perform the normal removal sequence: logical scheduler removal, ready-notification
  // purge, and terminal close. Dynamic cache ownership is released immediately; a resident static
  // batch row can remain physically retained until its shared batch is recycled.
  if (!abandonment_pending_->exchange(false, std::memory_order_acq_rel)) {
    return;
  }

  while (true) {
    const auto abandoned = std::find_if(
        tracked_requests_.begin(), tracked_requests_.end(),
        [this](const std::shared_ptr<Request>& request) {
          return request &&
                 !IsClosed(request->status_) &&
                 request->BelongsTo(*this) &&
                 request->ExternalReferencesAbandoned();
        });
    if (abandoned == tracked_requests_.end()) {
      return;
    }

    auto request = *abandoned;
    // Recheck immediately before removal in case an external owner was reacquired before this
    // serialized boundary.
    if (request->ExternalReferencesAbandoned()) {
      CloseRequest(request);
    }
  }
}

void Engine::CompleteNonresidentClosedRequests() {
  tracked_requests_.erase(
      std::remove_if(
          tracked_requests_.begin(), tracked_requests_.end(),
          [this](const std::shared_ptr<Request>& request) {
            if (!request || !IsClosed(request->status_) ||
                !request->BelongsTo(*this) ||
                cache_manager_->IsResident(request)) {
              return false;
            }
            request->CompleteCloseFromEngine(*this);
            return true;
          }),
      tracked_requests_.end());
}

void Engine::ValidateRequestCanContinue(
    const std::shared_ptr<Request>& request,
    bool allow_nonresident) const {
  if (health_ == EngineHealth::Unhealthy) {
    std::rethrow_exception(fatal_error_);
  }
  if (!request->BelongsTo(*this)) {
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
    std::exception_ptr append_error,
    std::exception_ptr restore_error) {
  request->MarkFailedFromEngine(*this);
  std::string message = AddExceptionCause(
      "Continuation append failed and its Search state could not be restored.",
      append_error);
  try {
    CloseRequest(request);
  } catch (...) {
    message = AddExceptionCause(
        std::move(message) + " Closing the poisoned request also failed.",
        std::current_exception());
    request->CompleteCloseFromEngine(*this);
  }
  MarkUnhealthyAndThrow(
      StepOutcomeKind::FatalExecutionFailure,
      /*transaction_id=*/0,
      request.get(),
      std::move(message),
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
  CompleteNonresidentClosedRequests();
  ReclaimAbandonedRequests();
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
    // Scheduling may recycle an all-terminal static batch. Closed rows retain their Search,
    // parameters, host tokens, and length metadata only until that shared allocation is gone;
    // finish their physical close before executing the replacement batch.
    CompleteNonresidentClosedRequests();

    try {
      model_executor_->Decode(scheduled_requests);
      ++speculative_stats_.target_forward_passes;
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
          (step_results_[i].visible_token_count != 0 ||
           step_results_[i].done)) {
        AppendEventsFromStep(scheduled_requests[i], step_results_[i]);
      }
    }
    pending_events_.swap(staged_events_);
    pending_event_index_ = 0;
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
          std::make_exception_ptr(std::logic_error{
              "Invalid dynamic scheduler planning outcome."}));
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
    std::unique_ptr<MtpStep> mtp_step;
    const auto rollback_transaction = [&]() {
      // Request/search state and composite cache state are checkpointed separately. Both must be
      // restored so a retry observes exactly the state that existed before this Run() call. The
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
      // MTP drafting is optional acceleration inside a mandatory target transaction. A recoverable
      // head failure (cache pressure, shape mismatch, binding or session error) must not roll the
      // committed target pass back, or a persistent failure would repeat forever with no progress.
      // PrepareMtpStep restores every piece of MTP state before it rethrows, so the target step can
      // commit without drafts. Contract violations and failed MTP rollback stay fatal below.
      try {
        mtp_step = PrepareMtpStep(step_plan_, step_results_, scheduled_requests);
      } catch (const MtpRollbackError&) {
        throw;
      } catch (const std::logic_error&) {
        throw;
      } catch (...) {
        mtp_step.reset();
        ++speculative_stats_.standard_fallback_steps;
      }
      for (size_t i = 0; i < step_plan_.requests.size(); ++i) {
        if (step_results_[i].visible_token_count != 0 ||
            step_results_[i].done) {
          staged_event_order_.push_back(i);
        }
      }
      std::sort(
          staged_event_order_.begin(), staged_event_order_.end(),
          [this](size_t left, size_t right) {
            return step_plan_.requests[left].scheduling_order <
                   step_plan_.requests[right].scheduling_order;
          });
    } catch (const MtpRollbackError&) {
      const auto rollback_error = std::current_exception();
      rollback_transaction();
      ++transaction_metrics_.post_processing_aborts;
      MarkUnhealthyAndThrow(
          StepOutcomeKind::FatalExecutionFailure,
          step_plan_.transaction_id,
          nullptr,
          "MTP rollback failed and the Engine is no longer healthy.",
          rollback_error);
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
      if (mtp_step) {
        mtp_step->reservation->PrepareCommit();
      }
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

    // This is speculative preparation for the next step. Its no-throw boundary leaves failed
    // cursors dirty so mask construction retries when the next logits application needs it.
    scheduled_requests.ScheduleGuidanceMasks();

    staged_events_.clear();
    for (size_t i : staged_event_order_) {
      AppendEventsFromStep(
          step_plan_.requests[i].request, step_results_[i]);
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

void Engine::AppendEventsFromStep(
    const std::shared_ptr<Request>& request,
    const RequestStepResult& result) {
  const auto finish_turn = [&request, &result](EngineEvent& event) {
    event.flags |= EngineEventFlagTurnFinished;
    event.finish_reason = result.finish_reason;
    event.usage = {
        request->TurnPromptTokens(),
        request->TurnGeneratedTokens(),
        0};
  };

  for (size_t i = 0; i < result.visible_token_count; ++i) {
    EngineEvent event;
    event.request = request;
    event.turn_id = request->CurrentTurnId();
    event.flags = EngineEventFlagToken;
    event.token = result.visible_tokens[i];
    if (result.done && i + 1 == result.visible_token_count) {
      finish_turn(event);
    }
    staged_events_.push_back(std::move(event));
  }

  if (result.done && result.visible_token_count == 0) {
    EngineEvent event;
    event.request = request;
    event.turn_id = request->CurrentTurnId();
    finish_turn(event);
    staged_events_.push_back(std::move(event));
  }
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
        std::make_exception_ptr(std::logic_error{
            "Invalid unserviceable Request identity."}));
  }
  scheduler_->RemoveRequest(request);
  CloseMtpRequest(request);
  request->CompleteFailedTurnFromEngine(*this);

  EngineEvent event;
  event.request = request;
  event.turn_id = request->CurrentTurnId();
  event.flags = EngineEventFlagTurnFinished | EngineEventFlagFailed;
  event.finish_reason = GenerationFinishReason::Failed;
  event.error_code = EngineErrorCode::RequestUnserviceable;
  event.usage = {
      request->TurnPromptTokens(),
      request->TurnGeneratedTokens(),
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
    std::string message,
    std::exception_ptr error) {
  health_ = EngineHealth::Unhealthy;
  if (outcome == StepOutcomeKind::FatalExecutionFailure ||
      outcome == StepOutcomeKind::ExecutionContractFailure) {
    ++transaction_metrics_.fatal_execution_failures;
  }
  pending_events_.erase(
      pending_events_.begin(),
      pending_events_.begin() + static_cast<ptrdiff_t>(pending_event_index_));
  pending_event_index_ = 0;
  const auto error_code =
      outcome == StepOutcomeKind::ExecutionContractFailure
          ? EngineErrorCode::EngineContractFailure
          : EngineErrorCode::EngineExecutionFailure;
  for (const auto& request : tracked_requests_) {
    if (request && IsExecutable(request->status_)) {
      request->CompleteFailedTurnFromEngine(*this);
      EngineEvent event;
      event.request = request;
      event.turn_id = request->CurrentTurnId();
      event.flags = EngineEventFlagTurnFinished | EngineEventFlagFailed;
      event.finish_reason = GenerationFinishReason::Failed;
      event.error_code = error_code;
      event.usage = {
          request->TurnPromptTokens(),
          request->TurnGeneratedTokens(),
          0};
      const auto existing = std::find_if(
          pending_events_.rbegin(), pending_events_.rend(),
          [&request](const EngineEvent& pending) {
            return pending.request == request &&
                   pending.turn_id == request->CurrentTurnId();
          });
      if (existing == pending_events_.rend()) {
        pending_events_.push_back(std::move(event));
      } else {
        existing->flags |= event.flags;
        existing->finish_reason = event.finish_reason;
        existing->error_code = event.error_code;
        existing->usage = event.usage;
      }
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
  CompleteNonresidentClosedRequests();
  ReclaimAbandonedRequests();
  return pending_event_index_ < pending_events_.size() ||
         scheduler_->HasPendingRequests();
}

size_t Engine::MaxDraftTokensPerStep() const {
  ValidateOwnerThread();
  return cache_manager_->SupportsDynamicBatching() &&
                 model_executor_->SupportsDraftVerification()
             ? cache_manager_->MaxDraftTokensPerStep()
             : 0;
}

SpeculativeStats Engine::GetSpeculativeStats() const {
  // The counters are plain members mutated by Run(), so reading them from a monitoring thread
  // would be a data race.
  ValidateOwnerThread();
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
