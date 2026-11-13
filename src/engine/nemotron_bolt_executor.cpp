// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "nemotron_bolt_executor.h"

#include "../models/nemotron_bolt.h"
#include "fixed_slot_scheduler.h"
#include "token_producing_session.h"

namespace Generators {
namespace {

void AddStats(SpeculativeStats& total, const SpeculativeStats& value) {
  constexpr size_t SpeculativeStats::* fields[]{
      &SpeculativeStats::rounds, &SpeculativeStats::completed_rounds,
      &SpeculativeStats::draft_tokens_proposed, &SpeculativeStats::draft_tokens_evaluated,
      &SpeculativeStats::draft_tokens_accepted, &SpeculativeStats::correction_tokens,
      &SpeculativeStats::bonus_tokens, &SpeculativeStats::tokens_queued,
      &SpeculativeStats::tokens_discarded,
      &SpeculativeStats::tokens_buffered, &SpeculativeStats::draft_forward_passes,
      &SpeculativeStats::target_forward_passes, &SpeculativeStats::target_verify_forward_passes,
      &SpeculativeStats::full_accept_rounds, &SpeculativeStats::partial_accept_rounds,
      &SpeculativeStats::zero_accept_rounds};
  for (auto field : fields) total.*field += value.*field;
  for (size_t i = 0; i < total.acceptance_length_histogram.size(); ++i)
    total.acceptance_length_histogram[i] += value.acceptance_length_histogram[i];
}

struct NativeBoltExecutionState final : NemotronBoltExecutionState {
  explicit NativeBoltExecutionState(std::shared_ptr<NemotronBoltModel> model)
      : model{std::move(model)}, params{std::make_shared<GeneratorParams>(*this->model)} {
    params->search.max_length = this->model->config_->model.context_length;
    params->search.do_sample = false;
    params->search.min_length = 0;
    params->search.repetition_penalty = 1.0f;
    params->search.no_repeat_ngram_size = 0;
    ValidateNemotronBoltSearch(*params);
  }
  bool PrefixCachingEnabled() const override { return model->prefix_cache_ != nullptr; }
  size_t Prefill(std::span<const int32_t> tokens, int max_length, PrefixCacheMetrics& events) override {
    params->search.max_length = max_length;
    if (!state)
      state = std::make_unique<NemotronBoltState>(*model, *params);
    else
      state->RewindTo(0);
    state->Prefill(tokens, &events);
    return state->CachedPromptTokens();
  }
  int32_t NextToken(size_t public_length) override { return state->NextToken(public_length); }
  void Reset() override {
    if (state) state->RewindTo(0);
  }
  SpeculativeStats Stats() const override { return state ? state->Stats() : SpeculativeStats{}; }
  std::shared_ptr<NemotronBoltModel> model;
  std::shared_ptr<GeneratorParams> params;
  std::unique_ptr<NemotronBoltState> state;
};

struct BoltSession final : FixedSlotSession {
  BoltSession(std::shared_ptr<Model> model, std::unique_ptr<NemotronBoltExecutionState> state)
      : model{std::move(model)}, state{std::move(state)} {}
  size_t MaxRequestLength() const override { return model->config_->model.context_length; }
  size_t MaxBatchSize() const override { return 1; }
  const PrefixCacheMetrics* PrefixMetrics() const override {
    return state->PrefixCachingEnabled() ? &prefix_metrics : nullptr;
  }
  void ReleaseSlot(size_t) override {
    if (cleanup_error) std::rethrow_exception(cleanup_error);
    const bool had_owner = owner != nullptr;
    if (had_owner && state && !failed) {
      auto stats = state->Stats();
      stats.tokens_discarded += stats.tokens_buffered;
      stats.tokens_buffered = 0;
      try {
        state->Reset();
      } catch (...) {
        cleanup_error = std::current_exception();
        failed = true;
        throw;
      }
      AddStats(completed, stats);
    }
    completions.Retire(0);
    pending_cached_prefix = 0;
    owner = nullptr;
  }
  void ReleaseSlotForTeardown(size_t slot) noexcept override {
    try {
      ReleaseSlot(slot);
    } catch (...) {
      failed = true;
    }
  }
  std::shared_ptr<Model> model;
  std::unique_ptr<NemotronBoltExecutionState> state;
  const void* owner{};
  uint64_t turn{};
  bool failed{};
  std::exception_ptr cleanup_error;
  size_t pending_cached_prefix{};
  TokenProducingSession completions{1};
  SpeculativeStats completed;
  PrefixCacheMetrics prefix_metrics;
  size_t delivered{};
  size_t discarded{};
};

struct BoltExecutor final : ModelExecutor {
  explicit BoltExecutor(std::shared_ptr<BoltSession> session) : session{std::move(session)} {}
  bool SelectsOwnTokens() const override { return true; }
  void OnExecutionFailure() noexcept override {
    session->pending_cached_prefix = 0;
    session->failed = true;
  }
  void ValidateTurn(const EffectiveTurnPolicy& policy, const TurnOptions& options) const override {
    if (!policy.IsGreedy() || policy.repetition_penalty != 1.0f || policy.no_repeat_ngram_size != 0 ||
        policy.min_generated_tokens != 0 || options.seed || !options.guidance_type.empty() ||
        !options.guidance_data.empty() || options.delimited_guidance)
      throw std::runtime_error(
          "The nemotron_bolt backend requires greedy decoding without penalties, min_generated_tokens, seed, or guidance.");
  }
  void Decode(ScheduledRequests& requests, ExecutionContext&) override {
    if (session->failed) throw std::runtime_error("The nemotron_bolt execution context has failed.");
    std::vector<BackendPlannedRow> plan;
    for (size_t row = 0; row < requests.size(); ++row) {
      const auto& request = requests[row];
      if (!IsExecuting(request->status_)) continue;
      const bool begin = session->owner != request.get() || session->turn != request->CurrentTurnId();
      plan.push_back(session->completions.Plan(row, 0, request.get(), request->CurrentTurnId(),
                                               begin ? request->TokensCpu().size() : 0, begin));
    }
    requests.BeginBackendStep(plan);
    for (const auto& operation : plan) {
      const auto& request = requests[operation.request_row];
      BackendRowResult result;
      result.key = operation.key;
      if (session->owner != request.get() || session->turn != request->CurrentTurnId()) {
        if (session->owner) {
          auto stats = session->state->Stats();
          stats.tokens_discarded += stats.tokens_buffered;
          stats.tokens_buffered = 0;
          AddStats(session->completed, stats);
        }
        const auto max_length = static_cast<int>(
            request->TokensCpu().size() + std::min(request->RemainingTurnTokenBudget(),
                                                   request->MaxSessionTokens() - request->TokensCpu().size()));
        const size_t cached = session->state->Prefill(request->TokensCpu(), max_length, session->prefix_metrics);
        session->pending_cached_prefix = cached;
        session->owner = request.get();
        session->turn = request->CurrentTurnId();
        result.prefill = PrefillProgress{request->TokensCpu().size(), true, cached};
      }
      SelectedTokenBlock block;
      block.candidate_count = 1;
      block.candidates[0] = session->state->NextToken(request->CurrentSequenceLength());
      result.selected_tokens = block;
      session->completions.Collected(result);
      requests.StageBackendRow(operation.request_row, std::move(result));
    }
  }
  void OnStepCommitted(const ScheduledRequests& requests, std::span<const RequestStepResult>) override {
    for (const auto& ack : requests.CommitAcks()) {
      session->completions.Acknowledge(ack);
      if (session->pending_cached_prefix != 0) {
        ++session->prefix_metrics.hits;
        session->prefix_metrics.matched_tokens += std::exchange(session->pending_cached_prefix, 0);
      }
      session->delivered += ack.tokens_appended;
      session->discarded += ack.candidates_consumed - ack.tokens_appended;
    }
  }
  void AddSpeculativeStats(SpeculativeStats& stats) const override {
    AddStats(stats, session->completed);
    stats.effective_k = 0;
    if (session->owner && session->state) {
      const auto live = session->state->Stats();
      AddStats(stats, live);
      stats.effective_k = live.effective_k;
    }
    stats.tokens_emitted += session->delivered;
    stats.tokens_discarded += session->discarded;
    if (stats.rounds != 0)
      stats.mean_emitted_tokens_per_round = static_cast<float>(stats.tokens_emitted) / stats.rounds;
  }
  std::shared_ptr<BoltSession> session;
};

}  // namespace

EngineDependencies CreateNemotronBoltDependencies(std::shared_ptr<NemotronBoltModel> model) {
  return CreateNemotronBoltDependencies(model, std::make_unique<NativeBoltExecutionState>(model));
}

EngineDependencies CreateNemotronBoltDependencies(std::shared_ptr<Model> model,
                                                  std::unique_ptr<NemotronBoltExecutionState> state) {
  auto session = std::make_shared<BoltSession>(model, std::move(state));
  auto cache = std::make_shared<FixedSlotCacheManager>(model, session);
  auto scheduler = std::make_unique<FixedSlotScheduler>(model, cache);
  auto executor = std::make_unique<BoltExecutor>(std::move(session));
  return EngineDependencies{std::move(cache), std::move(scheduler), std::move(executor)};
}

}  // namespace Generators
