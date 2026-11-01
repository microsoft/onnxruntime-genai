// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qwen38_graph_executor.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "../models/qwen38_graph.h"
#include "scheduler.h"

namespace Generators {

// ------------------------------------------------------------------------------------------------
// Session
// ------------------------------------------------------------------------------------------------
Qwen38GraphSession::Qwen38GraphSession(std::shared_ptr<Qwen38DecodeDevice> device, Options options)
    : device_{std::move(device)}, options_{std::move(options)}, completions_{device_ ? std::max<size_t>(1, device_->MaxBatchSize()) : 1} {
  if (!device_) {
    throw std::invalid_argument("Qwen38GraphSession requires a device.");
  }
  if (!device_->ContextIsHealthy()) {
    throw std::runtime_error("Cannot lease a poisoned token-producing execution context.");
  }
  // S=2 target steps consume token pairs: an odd chunk would shift the pairing of everything after it.
  options_.prefill_chunk_tokens = std::max<size_t>(2, options_.prefill_chunk_tokens & ~size_t{1});
  // A device with a fast prefill names its own chunk (any size is valid there); a longer chunk is one step without events,
  // so the preferred size also bounds how long cancellation and Run wait.
  if (const size_t preferred = device_->PrefillChunkTokens(); preferred > 0) {
    options_.prefill_chunk_tokens = preferred;
  }
  slot_states_.resize(MaxBatchSize());
}

Qwen38GraphSession::~Qwen38GraphSession() {
  ReleaseForTeardown();
}

size_t Qwen38GraphSession::MaxRequestLength() const {
  return device_->MaxSequenceLength();
}

size_t Qwen38GraphSession::MaxBatchSize() const {
  return std::max<size_t>(1, device_->MaxBatchSize());
}

void Qwen38GraphSession::Poison() noexcept {
  device_->MarkContextFailed();
}

void Qwen38GraphSession::Acknowledge(const TokenCommitAck& ack) {
  completions_.Acknowledge(ack);
  if (ack.disposition != ContinuationDisposition::Continue || ack.turn_finished) {
    // Stop useful work immediately. Already queued lookahead is retained until explicit retirement quiesces it.
    // A later turn replays Request's authoritative history; no physical-device-length equality is assumed here.
    if (MaxBatchSize() > 1) slot_states_.at(ack.key.slot.index).want_more = false;
  }
}

std::vector<BackendPlannedRow> Qwen38GraphSession::Plan(std::span<const Qwen38StepRequest> requests,
                                                        std::span<const size_t> request_rows) {
  if (requests.size() != request_rows.size()) throw std::logic_error("Backend plan row coverage differs from requests.");
  const bool resumes_paused_decoders = ResumesPausedDecoders(requests);
  std::vector<BackendPlannedRow> plan;
  plan.reserve(requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& request = requests[i];
    if (request.tokens.empty() || request.tokens.size() > MaxRequestLength() ||
        std::min(request.token_budget, request.session_budget) == 0) {
      throw std::logic_error("Fixed-slot request is outside the negotiated capacity or token budget.");
    }
    bool needs_prefill;
    size_t prefilled;
    if (MaxBatchSize() == 1) {
      const bool same_turn = request.owner == owner_ && request.turn_id == turn_id_;
      needs_prefill = !same_turn || !decoding_;
      prefilled = same_turn ? prefilled_ : 0;
    } else {
      const auto& state = slot_states_.at(request.slot);
      const bool same_turn = request.owner == state.owner && request.turn_id == state.turn_id;
      needs_prefill = !same_turn || !state.decoding;
      prefilled = same_turn ? state.prefilled : 0;
    }
    bool collected_cycle = MaxBatchSize() == 1 && !needs_prefill;
    for (size_t j = 0; batch_inflight_ && j < inflight_slots_.size(); ++j) {
      if (request.slot == inflight_slots_[j] && request.owner == inflight_owners_[j] &&
          request.turn_id == inflight_turns_[j] &&
          slot_states_.at(request.slot).epoch == inflight_epochs_[j]) collected_cycle = true;
    }
    // StepBatch() enqueues and collects a cycle for each paused decoder that wants more tokens.
    if (resumes_paused_decoders && slot_states_.at(request.slot).want_more) collected_cycle = true;
    // A resident decoder paused behind prompt ingestion did not submit work in this step.
    if (!needs_prefill && !collected_cycle) continue;
    const size_t remaining = needs_prefill ? request.tokens.size() - prefilled : 0;
    const size_t chunk = std::min(options_.prefill_chunk_tokens, remaining);
    plan.push_back(completions_.Plan(request_rows[i], request.slot, request.owner, request.turn_id,
                                     chunk, needs_prefill && chunk == remaining));
  }
  return plan;
}

bool Qwen38GraphSession::IsStopToken(int32_t token) const {
  return std::find(options_.stop_tokens.begin(), options_.stop_tokens.end(), token) != options_.stop_tokens.end();
}

void Qwen38GraphSession::RethrowDeferredError() {
  if (deferred_error_) {
    std::rethrow_exception(std::exchange(deferred_error_, nullptr));
  }
}

// Only a step consumes the failure (the Engine then turns it into a fatal error). A retried rewind or retirement must keep
// failing: clearing it here would let a later retry succeed on a poisoned device.
void Qwen38GraphSession::ReportDeferredError() const {
  if (deferred_error_) {
    std::rethrow_exception(deferred_error_);
  }
}

// Reports the decode status of cycles that were still unchecked when their owner left; nothing may be in flight.
void Qwen38GraphSession::CheckStatusIfPending() {
  if (!status_checked_) {
    status_checked_ = true;
    try {
      device_->CheckDecodeStatus();
    } catch (...) {
      // The check also runs outside a step (a rewind, the retirement of a Request), where RunStatic's fatal handler does
      // not see the error. The next Reset() erases the error bits, so keep the failure for the next step, as
      // ReleaseForTeardown() does, and mark the context failed: the caller still gets the exception.
      Poison();
      if (!deferred_error_) deferred_error_ = std::current_exception();
      throw;
    }
  }
}

// Reset() and ResetSlot() clear the error bits that cycles left in the device. A cycle nobody consumed (the look-ahead of a
// cancelled or stopped turn) reports them nowhere else, so whatever is about to clear the device first drains it and reads
// the status of every slot: Drain(), then CheckDecodeStatus(), as Begin() and Release() do. A no-op when the status was
// checked already.
void Qwen38GraphSession::DrainAndCheckStatus() {
  if (status_checked_) {
    return;
  }
  device_->Drain();
  CheckStatusIfPending();
}

void Qwen38GraphSession::ClearSlot(SlotState& slot) noexcept {
  ++slot.epoch;
  slot.owner = nullptr;
  slot.turn_id = 0;
  slot.prefilled = 0;
  slot.decoding = false;
  slot.want_more = false;
  slot.dirty = true;
  slot.pending = {};
}

void Qwen38GraphSession::Begin(const Qwen38StepRequest& request) {
  if (request.tokens.empty()) {
    throw std::logic_error("A Request must have at least one token before it can be decoded.");
  }
  if (options_.vocab_size > 0) {
    ValidateQwen38PromptTokens(request.tokens, options_.vocab_size);
  }
  // A cycle may still be in flight from a turn that stopped early (stop string, cancellation); drain and check its
  // decode status before Reset clears the device error bits.
  owner_ = nullptr;
  pending_step_ = {};
  if (deferred_error_) {
    std::rethrow_exception(std::exchange(deferred_error_, nullptr));
  }
  DrainAndCheckStatus();
  if (device_dirty_) {
    device_->Reset();
    device_dirty_ = false;
  }
  owner_ = request.owner;
  turn_id_ = request.turn_id;
  prefilled_ = 0;
  decoding_ = false;
}

void Qwen38GraphSession::BeginSlot(size_t slot, const Qwen38StepRequest& request) {
  if (request.tokens.empty()) {
    throw std::logic_error("A Request must have at least one token before it can be decoded.");
  }
  if (slot >= slot_states_.size()) {
    throw std::out_of_range("Qwen38GraphSession: slot index out of range.");
  }
  SlotState& st = slot_states_[slot];
  if (device_dirty_ || st.dirty) {
    // The turn that held the slot may have left error bits nobody reported yet: the reset below would clear them.
    DrainAndCheckStatus();
  }
  st.owner = nullptr;
  st.pending = {};
  if (device_dirty_) {
    device_->Reset();
    device_dirty_ = false;
    for (auto& s : slot_states_) {
      s.dirty = false;
    }
  } else if (st.dirty) {
    device_->ResetSlot(slot);
    st.dirty = false;
  }
  ++st.epoch;
  st.owner = request.owner;
  st.turn_id = request.turn_id;
  st.prefilled = 0;
  st.decoding = false;
  st.want_more = false;
}

void Qwen38GraphSession::Release() {
  device_->Drain();
  owner_ = nullptr;
  prefilled_ = 0;
  decoding_ = false;
  pending_step_ = {};
  batch_inflight_ = false;
  inflight_slots_.clear();
  inflight_owners_.clear();
  inflight_turns_.clear();
  inflight_epochs_.clear();
  for (size_t s = 0; s < slot_states_.size(); ++s) {
    ClearSlot(slot_states_[s]);
    completions_.Retire(s);
  }
  ReportDeferredError();
  CheckStatusIfPending();
}

void Qwen38GraphSession::ReleaseForTeardown() noexcept {
  try {
    Release();
  } catch (...) {
    Poison();
    if (!deferred_error_) {
      deferred_error_ = std::current_exception();
    }
  }
}

void Qwen38GraphSession::ReleaseSlot(size_t slot) {
  if (slot >= slot_states_.size()) throw std::out_of_range("Fixed-slot retirement index out of range.");
  if (MaxBatchSize() == 1) {
    Release();
    return;
  }
  ClearSlot(slot_states_[slot]);
  completions_.Retire(slot);
}

void Qwen38GraphSession::ReleaseSlotForTeardown(size_t slot) noexcept {
  try {
    ReleaseSlot(slot);
  } catch (...) {
    Poison();
    if (!deferred_error_) deferred_error_ = std::current_exception();
  }
}

Qwen38GraphSession::PendingStep Qwen38GraphSession::ProcessCycleRecord(const Qwen38CycleRecord& record, size_t limit,
                                                                       Qwen38StepResult& result, bool& ended) {
  const std::array<int32_t, 8> predictions{
      static_cast<int32_t>(record.pred0),
      static_cast<int32_t>(record.pred1),
      static_cast<int32_t>(record.pred2),
      static_cast<int32_t>(record.pred3),
      static_cast<int32_t>(record.pred4),
      static_cast<int32_t>(record.pred5),
      static_cast<int32_t>(record.pred6),
      static_cast<int32_t>(record.pred7),
  };
  const size_t produced = record.TokenCount();
  if (produced == 0 || produced > predictions.size() || record.accepted_count < 0 ||
      record.proposed_count < 0 || record.proposed_count > 7 ||
      (record.accepted_count > 0 && record.accepted_count > record.proposed_count)) {
    throw std::logic_error("Qwen backend returned a malformed cycle record.");
  }
  const std::array<int64_t, 8> raw{record.pred0, record.pred1, record.pred2, record.pred3,
                                   record.pred4, record.pred5, record.pred6, record.pred7};
  for (size_t i = 0; i < produced; ++i) {
    if (raw[i] < 0 || raw[i] > std::numeric_limits<int32_t>::max()) {
      throw std::logic_error("Qwen backend returned an invalid token representation.");
    }
  }
  size_t count = produced;
  bool stop_in_cycle = false;
  for (size_t i = 0; i < count; ++i) {
    if (IsStopToken(predictions[i])) {
      count = i + 1;
      stop_in_cycle = true;
      break;
    }
  }
  const size_t committed = std::min(count, limit);
  // A stop token beyond the limit is never committed, so it does not end the loop by itself.
  ended = (stop_in_cycle && committed == count) || committed == limit;
  // Deliver every target-validated candidate. Request decides the exact consumed/appended prefix;
  // the limit and EOS scan above only decide whether useful device lookahead may continue.
  for (size_t i = 0; i < produced; ++i) result.tokens[i] = predictions[i];
  result.count = produced;
  const size_t n_prop = record.proposed_count > 1 ? static_cast<size_t>(record.proposed_count) : 1;
  const size_t n_acc =
      record.accepted_count > 0 ? static_cast<size_t>(record.accepted_count) : (record.accepted ? 1 : 0);
  result.speculation = SpeculativeRowMetrics{n_prop, n_acc};
  return PendingStep{/*active=*/true, /*is_decode_cycle=*/true, /*mtp=*/device_->MtpEnabled(),
                     /*proposed=*/n_prop, /*accepted=*/n_acc, /*produced=*/produced, /*count_forward_pass=*/true};
}

Qwen38StepResult Qwen38GraphSession::Step(const Qwen38StepRequest& request) {
  if (MaxBatchSize() > 1 || request.slot != 0) {
    Qwen38StepResult result;
    StepBatch(std::span<const Qwen38StepRequest>{&request, 1}, std::span<Qwen38StepResult>{&result, 1});
    return result;
  }
  RethrowDeferredError();
  if (request.owner != owner_ || request.turn_id != turn_id_) {
    Begin(request);
  }
  const size_t limit = std::min(request.token_budget, request.session_budget);
  if (limit == 0) {
    throw std::logic_error("A request with no remaining budget was scheduled for decoding.");
  }

  Qwen38StepResult result;
  pending_step_ = {};
  if (!decoding_) {
    const size_t remaining = request.tokens.size() - prefilled_;
    const size_t chunk = std::min(options_.prefill_chunk_tokens, remaining);
    device_dirty_ = true;
    if (prefilled_ == 0) {
      // The device keeps the turn's prompt length for its chunks (Reset() and BeginDecode() drop it): the value belongs
      // to this device, so Engines running in one process never see each other's.
      device_->ExpectPrefillTokens(request.tokens.size());
    }
    device_->PrefillChunk(request.tokens.subspan(prefilled_, chunk));
    prefilled_ += chunk;
    ++work_metrics_.work_records;
    work_metrics_.prefill_tokens += chunk;
    result.prompt_tokens_processed = chunk;
    result.prompt_complete = prefilled_ == request.tokens.size();
    if (prefilled_ < request.tokens.size()) {
      return result;
    }
    // The first token is the target's prediction after the prompt. The head prefill, the first draft and the first verify
    // are enqueued before it is returned so that the next step finds the GPU already busy.
    const int32_t first = device_->PendingToken();
    const bool run_cycle = !IsStopToken(first) && limit > 1;
    device_->BeginDecode(run_cycle);
    if (run_cycle) {
      status_checked_ = false;
    }
    decoding_ = true;
    result.tokens[0] = first;
    result.count = 1;
    pending_step_ = PendingStep{/*active=*/true, /*is_decode_cycle=*/false, /*mtp=*/false,
                                /*proposed=*/0, /*accepted=*/0, /*produced=*/1};
    return result;
  }

  const Qwen38CycleRecord record = device_->WaitCycle();
  ++work_metrics_.work_records;
  ++work_metrics_.target_forward_passes;
  bool ended = false;
  const PendingStep pending = ProcessCycleRecord(record, limit, result, ended);
  // Enqueue the next cycle before returning: the GPU runs it while the Engine delivers these tokens. A stop string can
  // only be known after delivery, so one cycle may be wasted; Reset/Release wait for it.
  device_->Advance(record, /*want_more=*/!ended);
  if (ended) {
    status_checked_ = true;
    device_->CheckDecodeStatus();  // nothing is in flight now: a device that accumulated an error reports it here
  }
  pending_step_ = pending;
  return result;
}

void Qwen38GraphSession::OnStepCommitted(size_t slot, const void* owner, const RequestStepResult& result) {
  PendingStep* pending = nullptr;
  if (MaxBatchSize() == 1) {
    if (slot == 0 && owner == owner_) {
      pending = &pending_step_;
    }
  } else if (slot < slot_states_.size() && slot_states_[slot].owner == owner) {
    pending = &slot_states_[slot].pending;
  }
  if (pending == nullptr || !pending->active) {
    return;
  }
  const PendingStep step = std::exchange(*pending, PendingStep{});
  ApplyCommitted(step, result);
}

void Qwen38GraphSession::ApplyCommitted(const PendingStep& step, const RequestStepResult& result) {
  const size_t emitted = result.visible_token_count;
  stats_.tokens_emitted += emitted;
  if (!step.is_decode_cycle) {
    return;
  }

  ++stats_.rounds;
  ++stats_.completed_rounds;
  if (step.count_forward_pass) {
    ++stats_.target_forward_passes;
  }
  if (step.mtp) {
    const size_t n_prop = step.proposed > 0 ? step.proposed : 1;
    const size_t effective_accepted = std::min(step.accepted, emitted);
    // Count examined draft positions, including a terminal EOS which Request consumes without appending.
    const size_t consumed = result.candidates_consumed;
    const size_t n_eval =
        std::min(consumed, step.accepted) + ((consumed > step.accepted && step.accepted < n_prop) ? 1 : 0);
    if (step.count_forward_pass) {
      ++stats_.target_verify_forward_passes;
    }

    stats_.draft_forward_passes += n_prop;
    stats_.draft_tokens_proposed += n_prop;
    stats_.draft_tokens_evaluated += n_eval;
    if (effective_accepted < stats_.acceptance_length_histogram.size()) {
      ++stats_.acceptance_length_histogram[effective_accepted];
    }
    stats_.draft_tokens_accepted += effective_accepted;
    // The same classification as Engine::RecordSpeculativeCommit, by the drafts that were delivered; whether a correction
    // or bonus token followed is counted on its own.
    if (effective_accepted == 0) {
      ++stats_.zero_accept_rounds;
    } else if (effective_accepted == n_prop) {
      ++stats_.full_accept_rounds;
    } else {
      ++stats_.partial_accept_rounds;
    }
    if (effective_accepted == n_prop) {
      if (emitted > n_prop) {
        ++stats_.bonus_tokens;
      }
    } else if (emitted > effective_accepted) {
      ++stats_.correction_tokens;
    }
  }
  if (step.produced >= emitted) {
    stats_.tokens_discarded += step.produced - emitted;
  }
}

bool Qwen38GraphSession::ResumesPausedDecoders(std::span<const Qwen38StepRequest> requests) const {
  if (MaxBatchSize() == 1 || batch_inflight_) return false;
  bool wants_more = false;
  for (const auto& request : requests) {
    const SlotState& state = slot_states_.at(request.slot);
    if (request.owner != state.owner || request.turn_id != state.turn_id || !state.decoding) return false;
    wants_more = wants_more || state.want_more;
  }
  return wants_more;
}

void Qwen38GraphSession::EnqueueDecodeCycle(std::span<const Qwen38StepRequest> requests) {
  std::vector<size_t> active_slots;
  std::vector<const void*> active_owners;
  std::vector<uint64_t> active_turns;
  std::vector<uint64_t> active_epochs;
  for (const auto& req : requests) {
    const SlotState& st = slot_states_.at(req.slot);
    if (st.decoding && st.want_more) {
      active_slots.push_back(req.slot);
      active_owners.push_back(st.owner);
      active_turns.push_back(st.turn_id);
      active_epochs.push_back(st.epoch);
    }
  }
  if (active_slots.empty()) return;
  device_->EnqueueBatchDecode(active_slots);
  status_checked_ = false;
  batch_inflight_ = true;
  inflight_slots_ = std::move(active_slots);
  inflight_owners_ = std::move(active_owners);
  inflight_turns_ = std::move(active_turns);
  inflight_epochs_ = std::move(active_epochs);
}

void Qwen38GraphSession::StepBatch(std::span<const Qwen38StepRequest> requests, std::span<Qwen38StepResult> results) {
  if (results.size() < requests.size()) {
    throw std::invalid_argument("Qwen38GraphSession::StepBatch requires a result slot per request.");
  }
  for (size_t i = 0; i < requests.size(); ++i) {
    results[i] = Qwen38StepResult{};
  }
  if (requests.empty()) {
    return;
  }
  if (MaxBatchSize() == 1 && requests.size() == 1 && requests[0].slot == 0) {
    results[0] = Step(requests[0]);
    return;
  }
  RethrowDeferredError();

  for (const auto& req : requests) {
    const size_t limit = std::min(req.token_budget, req.session_budget);
    if (limit == 0) {
      throw std::logic_error("A request with no remaining budget was scheduled for decoding.");
    }
    slot_states_.at(req.slot).pending = {};
  }

  // The prompt the decoders were paused behind is gone (canceled): nothing would resume them, so the cycle that Plan()
  // announced is enqueued now and collected by phase 1 below.
  if (ResumesPausedDecoders(requests)) {
    EnqueueDecodeCycle(requests);
  }

  // Phase 1: if a batched decode cycle is in flight on the GPU, wait for it and commit its tokens.
  if (batch_inflight_) {
    std::vector<Qwen38CycleRecord> records(inflight_slots_.size());
    device_->WaitBatchDecode(inflight_slots_, records);
    ++work_metrics_.work_records;
    ++work_metrics_.target_forward_passes;
    batch_inflight_ = false;

    std::vector<uint8_t> want_more(inflight_slots_.size(), 0);
    bool counted_pass = false;
    for (size_t i = 0; i < inflight_slots_.size(); ++i) {
      const size_t s = inflight_slots_[i];
      size_t match_idx = requests.size();
      for (size_t r = 0; r < requests.size(); ++r) {
        if (requests[r].slot == s && requests[r].owner == inflight_owners_[i] &&
            requests[r].turn_id == inflight_turns_[i] &&
            slot_states_.at(s).epoch == inflight_epochs_[i]) {
          match_idx = r;
          break;
        }
      }
      if (match_idx < requests.size()) {
        const auto& req = requests[match_idx];
        const size_t limit = std::min(req.token_budget, req.session_budget);
        bool ended = false;
        PendingStep pending = ProcessCycleRecord(records[i], limit, results[match_idx], ended);
        // One target forward serves every slot of the batch.
        pending.count_forward_pass = !counted_pass;
        counted_pass = true;
        slot_states_[s].pending = pending;
        slot_states_[s].want_more = !ended;
        want_more[i] = !ended ? 1 : 0;
      } else if (s < slot_states_.size()) {
        ++work_metrics_.discarded_completions;
        slot_states_[s].want_more = false;
      }
    }

    bool any_needs_prefill = false;
    for (const auto& req : requests) {
      const SlotState& st = slot_states_.at(req.slot);
      if (req.owner != st.owner || req.turn_id != st.turn_id || !st.decoding) {
        any_needs_prefill = true;
        break;
      }
    }

    device_->AdvanceBatchDecode(inflight_slots_, records, want_more, /*defer_next_cycle=*/any_needs_prefill);

    if (!any_needs_prefill) {
      std::vector<size_t> next_slots;
      std::vector<const void*> next_owners;
      std::vector<uint64_t> next_turns;
      std::vector<uint64_t> next_epochs;
      for (size_t i = 0; i < inflight_slots_.size(); ++i) {
        if (want_more[i] != 0) {
          next_slots.push_back(inflight_slots_[i]);
          next_owners.push_back(inflight_owners_[i]);
          next_turns.push_back(inflight_turns_[i]);
          next_epochs.push_back(inflight_epochs_[i]);
        }
      }
      inflight_slots_ = std::move(next_slots);
      inflight_owners_ = std::move(next_owners);
      inflight_turns_ = std::move(next_turns);
      inflight_epochs_ = std::move(next_epochs);
      if (!inflight_slots_.empty()) {
        batch_inflight_ = true;
        status_checked_ = false;
      } else {
        status_checked_ = true;
        device_->CheckDecodeStatus();  // nothing is in flight now: a device that accumulated an error reports it here
      }
      return;
    }

    inflight_slots_.clear();
    inflight_owners_.clear();
    inflight_turns_.clear();
    inflight_epochs_.clear();
  }

  // Phase 2: prefill any slot that has not entered decoding yet.
  bool all_decoding = true;
  for (size_t r = 0; r < requests.size(); ++r) {
    const auto& req = requests[r];
    const size_t s = req.slot;
    SlotState& st = slot_states_.at(s);
    if (req.owner != st.owner || req.turn_id != st.turn_id) {
      BeginSlot(s, req);
    }
    if (!st.decoding) {
      const size_t remaining = req.tokens.size() - st.prefilled;
      const size_t chunk = std::min(options_.prefill_chunk_tokens, remaining);
      if (st.prefilled == 0) {
        // The length is the slot's own state in the device: slots that start their prompts one after the other never
        // overwrite what the others were told, and the executor never touches the process environment.
        device_->ExpectPrefillTokensSlot(s, req.tokens.size());
      }
      st.dirty = true;
      device_->PrefillChunkSlot(s, req.tokens.subspan(st.prefilled, chunk));
      st.prefilled += chunk;
      ++work_metrics_.work_records;
      work_metrics_.prefill_tokens += chunk;
      results[r].prompt_tokens_processed = chunk;
      results[r].prompt_complete = st.prefilled == req.tokens.size();
      if (st.prefilled < req.tokens.size()) {
        all_decoding = false;
        continue;
      }
      const size_t limit = std::min(req.token_budget, req.session_budget);
      const int32_t first = device_->PendingTokenSlot(s);
      const bool want_more = !IsStopToken(first) && limit > 1;
      device_->BeginDecodeSlot(s, want_more);
      if (want_more) {
        status_checked_ = false;
      }
      st.decoding = true;
      st.want_more = want_more;
      results[r].tokens[0] = first;
      results[r].count = 1;
      st.pending = PendingStep{/*active=*/true, /*is_decode_cycle=*/false, /*mtp=*/false,
                               /*proposed=*/0, /*accepted=*/0, /*produced=*/1, /*count_forward_pass=*/false};
    }
  }

  if (all_decoding) {
    EnqueueDecodeCycle(requests);
  }
}

void Qwen38GraphSession::AddStats(SpeculativeStats& stats) const {
  const auto add = [](size_t& total, size_t value) { total += value; };
  add(stats.rounds, stats_.rounds);
  add(stats.completed_rounds, stats_.completed_rounds);
  add(stats.draft_tokens_proposed, stats_.draft_tokens_proposed);
  add(stats.draft_tokens_evaluated, stats_.draft_tokens_evaluated);
  add(stats.draft_tokens_accepted, stats_.draft_tokens_accepted);
  add(stats.correction_tokens, stats_.correction_tokens);
  add(stats.bonus_tokens, stats_.bonus_tokens);
  add(stats.tokens_emitted, stats_.tokens_emitted);
  add(stats.tokens_discarded, stats_.tokens_discarded);
  add(stats.draft_forward_passes, stats_.draft_forward_passes);
  add(stats.target_forward_passes, stats_.target_forward_passes);
  add(stats.target_verify_forward_passes, stats_.target_verify_forward_passes);
  add(stats.full_accept_rounds, stats_.full_accept_rounds);
  add(stats.partial_accept_rounds, stats_.partial_accept_rounds);
  add(stats.zero_accept_rounds, stats_.zero_accept_rounds);
  for (size_t i = 0; i < stats.acceptance_length_histogram.size(); ++i) {
    add(stats.acceptance_length_histogram[i], stats_.acceptance_length_histogram[i]);
  }
}

// ------------------------------------------------------------------------------------------------
// Executor
// ------------------------------------------------------------------------------------------------
void Qwen38GraphExecutor::ValidateTurn(const EffectiveTurnPolicy& policy, const TurnOptions& options) const {
  constexpr const char* prefix = "The qwen38_flash_graph backend ";
  if (!policy.IsGreedy()) {
    throw std::runtime_error(std::string{prefix} +
                             "decodes greedily only (its device loop verifies drafts by argmax); set do_sample to false for the turn.");
  }
  if (policy.repetition_penalty != 1.0f) {
    throw std::runtime_error(std::string{prefix} + "does not support repetition_penalty.");
  }
  if (policy.no_repeat_ngram_size != 0) {
    throw std::runtime_error(std::string{prefix} + "does not support no_repeat_ngram_size.");
  }
  if (policy.min_generated_tokens != 0) {
    throw std::runtime_error(std::string{prefix} +
                             "does not support min_generated_tokens (the device loop cannot mask the end-of-sequence token).");
  }
  if (options.seed) {
    throw std::runtime_error(std::string{prefix} + "does not support a per-turn seed (decoding is greedy).");
  }
  if (!options.guidance_type.empty() || !options.guidance_data.empty() || options.delimited_guidance) {
    throw std::runtime_error(std::string{prefix} + "does not support constrained decoding (guidance).");
  }
}

void Qwen38GraphExecutor::Decode(ScheduledRequests& scheduled_requests, ExecutionContext&) {
  std::vector<Qwen38StepRequest> steps;
  std::vector<size_t> active_rows;
  steps.reserve(scheduled_requests.size());
  active_rows.reserve(scheduled_requests.size());
  for (size_t row = 0; row < scheduled_requests.size(); ++row) {
    const auto& request = scheduled_requests[row];
    if (!IsExecuting(request->status_)) continue;
    const auto used = static_cast<size_t>(request->CurrentSequenceLength());
    const size_t session_limit = request->MaxSessionTokens();
    Qwen38StepRequest step;
    step.slot = SlotOf(request, row);
    step.owner = request.get();
    step.turn_id = request->CurrentTurnId();
    step.tokens = request->TokensCpu();
    step.token_budget = request->RemainingTurnTokenBudget();
    step.session_budget = used < session_limit ? session_limit - used : 0;
    steps.push_back(step);
    active_rows.push_back(row);
  }

  auto plan = session_->Plan(steps, active_rows);
  scheduled_requests.BeginBackendStep(plan);
  if (steps.empty()) return;
  std::vector<Qwen38StepResult> results(steps.size());
  session_->StepBatch(steps, results);
  for (const auto& operation : plan) {
    const auto it = std::find(active_rows.begin(), active_rows.end(), operation.request_row);
    const auto& result = results.at(static_cast<size_t>(it - active_rows.begin()));
    BackendRowResult row;
    row.key = operation.key;
    if (result.prompt_tokens_processed > 0 || result.prompt_complete) {
      row.prefill = PrefillProgress{result.prompt_tokens_processed, result.prompt_complete};
    }
    if (result.count > 0) {
      SelectedTokenBlock tokens;
      if (result.count > tokens.candidates.size()) throw std::logic_error("Backend candidate block exceeds its negotiated width.");
      tokens.candidate_count = result.count;
      tokens.speculation = result.speculation;
      std::copy_n(result.tokens.begin(), result.count, tokens.candidates.begin());
      row.selected_tokens = std::move(tokens);
    }
    session_->Collected(row);
    scheduled_requests.StageBackendRow(operation.request_row, std::move(row));
  }
}

void Qwen38GraphExecutor::OnStepCommitted(const ScheduledRequests& scheduled_requests,
                                          std::span<const RequestStepResult> results) {
  // A progress-only completion is settled just like token work; public token policy remains Request's.
  for (const auto& ack : scheduled_requests.CommitAcks()) session_->Acknowledge(ack);
  for (size_t row = 0; row < scheduled_requests.size() && row < results.size(); ++row) {
    const auto& request = scheduled_requests[row];
    session_->OnStepCommitted(SlotOf(request, row), request.get(), results[row]);
  }
}

size_t Qwen38GraphExecutor::SlotOf(const std::shared_ptr<Request>& request, size_t row) const {
  return cache_manager_ ? cache_manager_->SlotOf(request) : row;
}

// ------------------------------------------------------------------------------------------------
// Assembly
// ------------------------------------------------------------------------------------------------
EngineDependencies CreateQwen38GraphDependencies(std::shared_ptr<Model> model,
                                                 std::shared_ptr<Qwen38DecodeDevice> device) {
  const auto& config = *model->config_;
  const bool multi_slot = device && device->MaxBatchSize() > 1;
  if (config.engine.dynamic_batching && !multi_slot) {
    throw std::runtime_error(
        "The qwen38_flash_graph model type runs on the Engine's static single-slot path; engine.dynamic_batching is not supported.");
  }
  const auto& qwen38 = config.model.qwen38_graph;
  // Not ValidateQwen38PrefillChunk: the model has checked the chunk against the configured kv_capacity, and kv_capacity is now
  // that of the loaded graph, which a context tier can leave below the chunk. The device clamps the chunk to it.
  ValidateQwen38PrefillChunkShape(qwen38);
  Qwen38GraphSession::Options options;
  options.prefill_chunk_tokens = EffectiveQwen38PrefillChunkTokens(qwen38);
  options.stop_tokens.assign(config.model.eos_token_id.begin(), config.model.eos_token_id.end());
  options.vocab_size = config.model.vocab_size;
  auto session = std::make_shared<Qwen38GraphSession>(std::move(device), std::move(options));
  auto cache_manager = std::make_shared<Qwen38GraphCacheManager>(model, session);
  auto scheduler = std::make_unique<FixedSlotScheduler>(model, cache_manager);
  auto executor = std::make_unique<Qwen38GraphExecutor>(std::move(session), cache_manager);
  return EngineDependencies{std::move(cache_manager), std::move(scheduler), std::move(executor)};
}

EngineDependencies CreateQwen38GraphDependencies(std::shared_ptr<Qwen38GraphModel> model) {
  auto device = model->AcquireDevice();
  return CreateQwen38GraphDependencies(std::shared_ptr<Model>{std::move(model)}, std::move(device));
}

}  // namespace Generators
