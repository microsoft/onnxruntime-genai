// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "nemotron_bolt_executor.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "../models/nemotron_bolt.h"
#include "fixed_slot_scheduler.h"
#include "token_producing_session.h"

namespace Generators {
namespace {

void AddStats(SpeculativeStats& total, const SpeculativeStats& value) {
  constexpr size_t SpeculativeStats::* fields[]{
      &SpeculativeStats::rounds,
      &SpeculativeStats::completed_rounds,
      &SpeculativeStats::draft_tokens_proposed,
      &SpeculativeStats::draft_tokens_evaluated,
      &SpeculativeStats::draft_tokens_accepted,
      &SpeculativeStats::correction_tokens,
      &SpeculativeStats::bonus_tokens,
      &SpeculativeStats::tokens_queued,
      &SpeculativeStats::tokens_discarded,
      &SpeculativeStats::tokens_buffered,
      &SpeculativeStats::draft_forward_passes,
      &SpeculativeStats::target_forward_passes,
      &SpeculativeStats::target_verify_forward_passes,
      &SpeculativeStats::full_accept_rounds,
      &SpeculativeStats::partial_accept_rounds,
      &SpeculativeStats::zero_accept_rounds,
  };
  for (auto field : fields) total.*field += value.*field;
  for (size_t i = 0; i < total.acceptance_length_histogram.size(); ++i)
    total.acceptance_length_histogram[i] += value.acceptance_length_histogram[i];
}

struct BoltSession final : FixedSlotSession {
  explicit BoltSession(std::shared_ptr<NemotronBoltModel> model_in)
      : model{std::move(model_in)},
        params{std::make_shared<GeneratorParams>(*model)},
        max_slots{std::max<size_t>(1, model->engine_max_batch_size_)},
        slots(max_slots),
        completions{max_slots} {
    if (!model->paged_attention_ || !model->batched_recurrent_ || !model->dynamic_prefill_ ||
        !model->has_logit_rows_ || !model->has_aux_rows_) {
      throw std::runtime_error(
          "nemotron_bolt Engine requires a PagedAttention dynamic-prefill target with banked recurrent state; "
          "use Generator for dense or single-bank packages.");
    }
    params->search.max_length = model->config_->model.context_length;
    if (!params->search.chunk_size && model->engine_max_scheduled_tokens_) {
      params->search.chunk_size = model->engine_max_scheduled_tokens_;
    }
    ValidateNemotronBoltSearch(*params);
  }

  size_t MaxRequestLength() const override {
    return static_cast<size_t>(model->config_->model.context_length);
  }

  size_t MaxBatchSize() const override { return max_slots; }

  bool SupportsChunkedPrefill() const override { return true; }

  size_t ChunkSize() const {
    return NemotronBoltPrefillChunkRows(params->search.chunk_size, model->capacity_);
  }

  void ReleaseSlot(size_t slot) override {
    auto& st = slots.at(slot);
    st = SlotRuntime{};
    completions.Retire(slot);
    if (batch_state && !failed) {
      batch_state->ResetSlot(slot);
    }
  }

  void ReleaseSlotForTeardown(size_t slot) noexcept override {
    try {
      ReleaseSlot(slot);
    } catch (...) {
      failed = true;
    }
  }

  struct PendingStep {
    bool active{};
    bool is_decode_step{};
    NemotronBoltEngineStep step{};
  };

  struct SlotRuntime {
    const void* owner{};
    uint64_t turn{};
    size_t prefilled{};
    bool decoding{};
    PendingStep pending_step{};
  };

  std::shared_ptr<NemotronBoltModel> model;
  std::shared_ptr<GeneratorParams> params;
  size_t max_slots{1};
  std::unique_ptr<NemotronBoltBatchState> batch_state;
  std::vector<SlotRuntime> slots;
  bool failed{};
  TokenProducingSession completions;
  SpeculativeStats committed_rounds{};
  size_t prefill_delivered{};
  size_t decode_delivered{};
  size_t discarded{};
};

struct BoltExecutor final : ModelExecutor {
  BoltExecutor(std::shared_ptr<BoltSession> session_in,
               std::shared_ptr<FixedSlotCacheManager> cache_in)
      : session{std::move(session_in)}, cache{std::move(cache_in)} {}

  bool SelectsOwnTokens() const override { return true; }

  void OnExecutionFailure() noexcept override { session->failed = true; }

  void ValidateTurn(const EffectiveTurnPolicy& policy, const TurnOptions& options) const override {
    if (!policy.IsGreedy() || policy.repetition_penalty != 1.0f || policy.no_repeat_ngram_size != 0 ||
        policy.min_generated_tokens != 0 || options.seed || !options.guidance_type.empty() ||
        !options.guidance_data.empty() || options.delimited_guidance) {
      throw std::runtime_error(
          "The nemotron_bolt backend requires greedy decoding without penalties, min_generated_tokens, seed, or guidance.");
    }
  }

  void Decode(ScheduledRequests& requests, ExecutionContext&) override {
    if (session->failed) throw std::runtime_error("The nemotron_bolt execution context has failed.");
    if (!session->batch_state) {
      session->batch_state = std::make_unique<NemotronBoltBatchState>(
          *session->model, *session->params, session->max_slots);
    }

    struct CandidateRow {
      size_t request_row{};
      size_t slot{};
      size_t used{};
      size_t remaining_budget{};
      size_t rem_prompt{};
      size_t planned_prompt{};
      bool is_decode{};
    };
    std::vector<CandidateRow> decode_candidates;
    std::vector<CandidateRow> prefill_candidates;
    decode_candidates.reserve(requests.size());
    prefill_candidates.reserve(requests.size());

    // First reclaim any slot whose turn finished or whose resident Request changed.
    for (size_t row = 0; row < requests.size(); ++row) {
      const auto& request = requests[row];
      const size_t slot = cache->SlotOf(request);
      auto& st = session->slots.at(slot);
      if (!IsExecuting(request->status_)) {
        if (st.owner != nullptr && (IsTurnComplete(request->status_) || IsClosed(request->status_))) {
          session->batch_state->ResetSlot(slot);
          st = BoltSession::SlotRuntime{};
        }
        continue;
      }
      if (st.owner != request.get() || st.turn != request->CurrentTurnId()) {
        session->batch_state->ResetSlot(slot);
        st = BoltSession::SlotRuntime{};
        st.owner = request.get();
        st.turn = request->CurrentTurnId();
      }
      const auto used = static_cast<size_t>(request->CurrentSequenceLength());
      const size_t session_limit = request->MaxSessionTokens();
      const size_t session_budget = used < session_limit ? session_limit - used : 0;
      const size_t remaining_budget = std::min(request->RemainingTurnTokenBudget(), session_budget);
      if (st.decoding) {
        decode_candidates.push_back(CandidateRow{row, slot, used, remaining_budget, 0, 0, true});
      } else {
        const size_t total_prompt = request->TokensCpu().size();
        const size_t rem_prompt = total_prompt - st.prefilled;
        prefill_candidates.push_back(CandidateRow{row, slot, used, remaining_budget, rem_prompt, 0, false});
      }
    }

    size_t free_blocks = session->batch_state->FreeBlockCount();
    std::vector<CandidateRow> admitted;
    admitted.reserve(decode_candidates.size() + prefill_candidates.size());

    // Decode-first admission: active decoders keep running without interruption.
    for (const auto& cand : decode_candidates) {
      const size_t needed =
          session->batch_state->BlocksNeededForSlot(cand.slot, cand.used + kBoltVerifyRows);
      if (needed <= free_blocks) {
        free_blocks -= needed;
        admitted.push_back(cand);
      }
    }

    // Distribute remaining chunk token budget and free KV blocks across prefilling slots.
    size_t prefill_budget = session->ChunkSize();
    for (auto cand : prefill_candidates) {
      if (prefill_budget == 0 && !admitted.empty()) break;
      const auto& st = session->slots[cand.slot];
      const size_t owned_blocks =
          (st.prefilled + kBoltPagedBlockSize - 1) / kBoltPagedBlockSize;
      const size_t max_addr_tokens = (owned_blocks + free_blocks) * kBoltPagedBlockSize;
      if (max_addr_tokens <= st.prefilled) continue;
      const size_t kv_room = max_addr_tokens - st.prefilled;
      const size_t token_cap = prefill_budget > 0 ? prefill_budget : cand.rem_prompt;
      const size_t chunk = std::min({cand.rem_prompt, token_cap, kv_room});
      if (chunk == 0) continue;
      const size_t needed = session->batch_state->BlocksNeededForSlot(cand.slot, st.prefilled + chunk);
      if (needed > free_blocks) continue;
      free_blocks -= needed;
      prefill_budget = chunk < prefill_budget ? prefill_budget - chunk : 0;
      cand.planned_prompt = chunk;
      admitted.push_back(cand);
    }

    if (admitted.empty() && (!decode_candidates.empty() || !prefill_candidates.empty())) {
      throw std::runtime_error("nemotron_bolt: PagedAttention KV block pool exhausted across active requests.");
    }

    std::sort(admitted.begin(), admitted.end(),
              [](const CandidateRow& a, const CandidateRow& b) { return a.request_row < b.request_row; });

    std::vector<BackendPlannedRow> plan;
    std::vector<NemotronBoltSlotTask> tasks;
    plan.reserve(admitted.size());
    tasks.reserve(admitted.size());

    for (const auto& cand : admitted) {
      const auto& request = requests[cand.request_row];
      const auto& st = session->slots[cand.slot];
      plan.push_back(session->completions.Plan(
          cand.request_row, cand.slot, request.get(), request->CurrentTurnId(), cand.planned_prompt,
          !cand.is_decode && cand.planned_prompt == cand.rem_prompt));
      tasks.push_back(NemotronBoltSlotTask{
          .slot = cand.slot,
          .full_prompt = request->TokensCpu(),
          .prefill_offset = st.prefilled,
          .prefill_tokens = cand.planned_prompt,
          .remaining_budget = cand.remaining_budget,
          .is_decode = cand.is_decode,
      });
    }

    requests.BeginBackendStep(plan);
    std::vector<NemotronBoltEngineStep> step_outputs(tasks.size());
    if (!tasks.empty()) {
      session->batch_state->StepBatch(tasks, step_outputs);
    }

    for (size_t i = 0; i < plan.size(); ++i) {
      const auto& cand = admitted[i];
      auto& st = session->slots[cand.slot];
      const auto& step = step_outputs[i];
      BackendRowResult result;
      result.key = plan[i].key;
      if (!cand.is_decode) {
        st.prefilled += step.prompt_tokens_processed;
        st.decoding = step.prompt_complete;
        result.prefill = PrefillProgress{step.prompt_tokens_processed, step.prompt_complete};
      }
      if (step.count > 0) {
        SelectedTokenBlock block;
        block.candidate_count = step.count;
        block.speculation = step.speculation;
        std::copy_n(step.tokens.begin(), step.count, block.candidates.begin());
        result.selected_tokens = std::move(block);
      }
      st.pending_step = BoltSession::PendingStep{true, cand.is_decode, step};
      session->completions.Collected(result);
      requests.StageBackendRow(cand.request_row, std::move(result));
    }
  }

  void OnStepCommitted(const ScheduledRequests& requests,
                       std::span<const RequestStepResult>) override {
    for (const auto& ack : requests.CommitAcks()) {
      session->completions.Acknowledge(ack);
      auto& slot_rt = session->slots.at(ack.key.slot.index);
      const auto pending = std::exchange(slot_rt.pending_step, BoltSession::PendingStep{});
      if (pending.active) {
        const size_t produced = pending.step.count;
        const size_t consumed = std::min(ack.candidates_consumed, produced);
        const size_t appended = ack.tokens_appended;
        if (pending.is_decode_step) {
          session->decode_delivered += appended;
        } else {
          session->prefill_delivered += appended;
        }
        if (produced >= appended) {
          session->discarded += produced - appended;
        }
        if (pending.is_decode_step && pending.step.speculative_round && consumed > 0) {
          const auto rs = GetNemotronBoltRoundStats(
              pending.step.drafts, std::span<const int32_t>{pending.step.tokens.data(), consumed});
          const size_t effective_accepted = std::min(rs.accepted, appended);
          auto& st = session->committed_rounds;
          ++st.rounds;
          ++st.completed_rounds;
          st.target_verify_forward_passes += pending.step.verify_forward_passes;
          st.draft_tokens_proposed += 7;
          st.draft_tokens_evaluated += rs.evaluated;
          st.draft_tokens_accepted += effective_accepted;
          ++st.acceptance_length_histogram[effective_accepted];
          if (effective_accepted == 7 && appended > 7) {
            ++st.bonus_tokens;
          } else if (appended > effective_accepted) {
            ++st.correction_tokens;
          }
          if (effective_accepted == 0) {
            ++st.zero_accept_rounds;
          } else if (effective_accepted == 7) {
            ++st.full_accept_rounds;
          } else {
            ++st.partial_accept_rounds;
          }
        }
      }
      if (ack.disposition != ContinuationDisposition::Continue || ack.turn_finished) {
        slot_rt.decoding = false;
      }
    }
  }

  void AddSpeculativeStats(SpeculativeStats& stats) const override {
    AddStats(stats, session->committed_rounds);
    stats.effective_k = 0;
    if (session->batch_state) {
      const auto live = session->batch_state->Stats();
      stats.target_forward_passes += live.target_forward_passes;
      stats.draft_forward_passes += live.draft_forward_passes;
      stats.effective_k = live.effective_k;
    }
    const size_t total_emitted = session->prefill_delivered + session->decode_delivered;
    stats.tokens_emitted += total_emitted;
    stats.tokens_queued += total_emitted + session->discarded;
    stats.tokens_discarded += session->discarded;
    if (stats.rounds != 0)
      stats.mean_emitted_tokens_per_round =
          static_cast<float>(session->decode_delivered) / static_cast<float>(stats.rounds);
  }

  std::shared_ptr<BoltSession> session;
  std::shared_ptr<FixedSlotCacheManager> cache;
};

}  // namespace

EngineDependencies CreateNemotronBoltDependencies(std::shared_ptr<NemotronBoltModel> model) {
  auto session = std::make_shared<BoltSession>(model);
  auto cache = std::make_shared<FixedSlotCacheManager>(model, session);
  auto scheduler = std::make_unique<FixedSlotScheduler>(model, cache);
  auto executor = std::make_unique<BoltExecutor>(std::move(session), cache);
  return EngineDependencies{std::move(cache), std::move(scheduler), std::move(executor)};
}

}  // namespace Generators
