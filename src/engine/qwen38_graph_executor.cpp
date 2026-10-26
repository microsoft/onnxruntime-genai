// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qwen38_graph_executor.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "../models/qwen38_graph.h"
#include "scheduler.h"

namespace Generators {

// ------------------------------------------------------------------------------------------------
// Session
// ------------------------------------------------------------------------------------------------
Qwen38GraphSession::Qwen38GraphSession(std::shared_ptr<Qwen38DecodeDevice> device, Options options)
    : device_{std::move(device)}, options_{std::move(options)} {
  if (!device_) {
    throw std::invalid_argument("Qwen38GraphSession requires a device.");
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
  Release();
}

size_t Qwen38GraphSession::MaxRequestLength() const {
  return static_cast<size_t>(device_->Capacity() - device_->ReservedTail());
}

size_t Qwen38GraphSession::MaxBatchSize() const {
  return std::max<size_t>(1, device_->MaxBatchSize());
}

bool Qwen38GraphSession::IsStopToken(int32_t token) const {
  return std::find(options_.stop_tokens.begin(), options_.stop_tokens.end(), token) != options_.stop_tokens.end();
}

void Qwen38GraphSession::Begin(const Qwen38StepRequest& request) {
  if (request.tokens.empty()) {
    throw std::logic_error("A Request must have at least one token before it can be decoded.");
  }
  // A cycle may still be in flight from a turn that stopped early (stop string, cancellation); Reset waits for it.
  owner_ = nullptr;
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
  st.owner = nullptr;
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
  st.owner = request.owner;
  st.turn_id = request.turn_id;
  st.prefilled = 0;
  st.decoding = false;
  st.want_more = false;
}

void Qwen38GraphSession::Release() noexcept {
  device_->Drain();
  owner_ = nullptr;
  prefilled_ = 0;
  decoding_ = false;
  batch_inflight_ = false;
  inflight_slots_.clear();
  inflight_owners_.clear();
  inflight_turns_.clear();
  for (auto& st : slot_states_) {
    st.owner = nullptr;
    st.turn_id = 0;
    st.prefilled = 0;
    st.decoding = false;
    st.want_more = false;
    st.dirty = true;
  }
}

void Qwen38GraphSession::ReleaseSlot(size_t slot) noexcept {
  if (MaxBatchSize() == 1) {
    Release();
    return;
  }
  if (slot >= slot_states_.size()) {
    return;
  }
  SlotState& st = slot_states_[slot];
  st.owner = nullptr;
  st.turn_id = 0;
  st.prefilled = 0;
  st.decoding = false;
  st.want_more = false;
  st.dirty = true;
}

size_t Qwen38GraphSession::ProcessCycleRecord(const Qwen38CycleRecord& record, size_t limit,
                                              Qwen38StepResult& result, bool& ended,
                                              bool count_forward_pass) {
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
  ended = (stop_in_cycle && committed == count) || committed == limit;

  const bool mtp = device_->MtpEnabled();
  ++stats_.rounds;
  ++stats_.completed_rounds;
  if (count_forward_pass) {
    ++stats_.target_forward_passes;
  }
  if (mtp) {
    const size_t n_prop = record.proposed_count > 1 ? static_cast<size_t>(record.proposed_count) : 1;
    const size_t n_acc = record.accepted_count > 0 ? static_cast<size_t>(record.accepted_count) : (record.accepted ? 1 : 0);
    const size_t n_eval = std::min(n_prop, n_acc + 1);
    if (count_forward_pass) {
      ++stats_.target_verify_forward_passes;
    }
    stats_.draft_forward_passes += n_prop;
    stats_.draft_tokens_proposed += n_prop;
    stats_.draft_tokens_evaluated += n_eval;
    if (n_acc < stats_.acceptance_length_histogram.size()) {
      ++stats_.acceptance_length_histogram[n_acc];
    }
    stats_.draft_tokens_accepted += n_acc;
    if (n_acc == n_prop) {
      ++stats_.full_accept_rounds;
      stats_.bonus_tokens += 1;
    } else if (n_acc == 0) {
      ++stats_.zero_accept_rounds;
      stats_.correction_tokens += 1;
    } else {
      stats_.correction_tokens += 1;
    }
  }
  stats_.tokens_emitted += committed;
  stats_.tokens_discarded += produced - committed;
  for (size_t i = 0; i < committed; ++i) {
    result.tokens[i] = predictions[i];
  }
  result.count = committed;
  return committed;
}

Qwen38StepResult Qwen38GraphSession::Step(const Qwen38StepRequest& request) {
  if (MaxBatchSize() > 1 || request.slot != 0) {
    Qwen38StepResult result;
    StepBatch(std::span<const Qwen38StepRequest>{&request, 1}, std::span<Qwen38StepResult>{&result, 1});
    return result;
  }
  if (request.owner != owner_ || request.turn_id != turn_id_) {
    Begin(request);
  }
  const size_t limit = std::min(request.token_budget, request.session_budget);
  if (limit == 0) {
    throw std::logic_error("A request with no remaining budget was scheduled for decoding.");
  }

  Qwen38StepResult result;
  if (!decoding_) {
    const size_t remaining = request.tokens.size() - prefilled_;
    const size_t chunk = std::min(options_.prefill_chunk_tokens, remaining);
    if (prefilled_ == 0) {
      const std::string total_str = std::to_string(request.tokens.size());
#if defined(_WIN32)
      _putenv_s("ORT_QMOE_PREFILL_TOTAL_TOKENS", total_str.c_str());
#else
      setenv("ORT_QMOE_PREFILL_TOTAL_TOKENS", total_str.c_str(), 1);
#endif
    }
    device_dirty_ = true;
    device_->PrefillChunk(request.tokens.subspan(prefilled_, chunk));
    prefilled_ += chunk;
    if (prefilled_ < request.tokens.size()) {
      return result;
    }
    // The first token is the target's prediction after the prompt. The head prefill, the first draft and the first verify
    // are enqueued before it is returned so that the next step finds the GPU already busy.
    const int32_t first = device_->PendingToken();
    device_->BeginDecode(/*run_cycle=*/!IsStopToken(first) && limit > 1);
    decoding_ = true;
    result.tokens[0] = first;
    result.count = 1;
    stats_.tokens_emitted += 1;
    return result;
  }

  const Qwen38CycleRecord record = device_->WaitCycle();
  bool ended = false;
  ProcessCycleRecord(record, limit, result, ended, /*count_forward_pass=*/true);
  // Enqueue the next cycle before returning: the GPU runs it while the Engine delivers these tokens. A stop string can
  // only be known after delivery, so one cycle may be wasted; Reset/Release wait for it.
  device_->Advance(record, /*want_more=*/!ended);
  if (ended) {
    device_->CheckDecodeStatus();  // nothing is in flight now: a device that accumulated an error reports it here
  }
  return result;
}

void Qwen38GraphSession::StepBatch(std::span<const Qwen38StepRequest> requests,
                                   std::span<Qwen38StepResult> results) {
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

  for (const auto& req : requests) {
    const size_t limit = std::min(req.token_budget, req.session_budget);
    if (limit == 0) {
      throw std::logic_error("A request with no remaining budget was scheduled for decoding.");
    }
  }

  // Phase 1: If a batched decode cycle is in flight on the GPU, wait for it and commit its tokens.
  if (batch_inflight_) {
    std::vector<Qwen38CycleRecord> records(inflight_slots_.size());
    device_->WaitBatchDecode(inflight_slots_, records);
    batch_inflight_ = false;

    std::vector<uint8_t> want_more(inflight_slots_.size(), 0);
    bool counted_pass = false;
    for (size_t i = 0; i < inflight_slots_.size(); ++i) {
      const size_t s = inflight_slots_[i];
      ssize_t match_idx = -1;
      for (size_t r = 0; r < requests.size(); ++r) {
        if (requests[r].slot == s &&
            requests[r].owner == inflight_owners_[i] &&
            requests[r].turn_id == inflight_turns_[i]) {
          match_idx = static_cast<ssize_t>(r);
          break;
        }
      }
      if (match_idx >= 0) {
        const auto& req = requests[static_cast<size_t>(match_idx)];
        const size_t limit = std::min(req.token_budget, req.session_budget);
        bool ended = false;
        ProcessCycleRecord(records[i], limit, results[static_cast<size_t>(match_idx)], ended, !counted_pass);
        counted_pass = true;
        slot_states_[s].want_more = !ended;
        want_more[i] = !ended ? 1 : 0;
      } else if (s < slot_states_.size()) {
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
      for (size_t i = 0; i < inflight_slots_.size(); ++i) {
        if (want_more[i] != 0) {
          next_slots.push_back(inflight_slots_[i]);
          next_owners.push_back(inflight_owners_[i]);
          next_turns.push_back(inflight_turns_[i]);
        }
      }
      inflight_slots_ = std::move(next_slots);
      inflight_owners_ = std::move(next_owners);
      inflight_turns_ = std::move(next_turns);
      if (!inflight_slots_.empty()) {
        batch_inflight_ = true;
      } else {
        device_->CheckDecodeStatus();
      }
      return;
    }

    inflight_slots_.clear();
    inflight_owners_.clear();
    inflight_turns_.clear();
  }

  // Phase 2: Prefill any slot that has not entered decoding yet.
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
        const std::string total_str = std::to_string(req.tokens.size());
#if defined(_WIN32)
        _putenv_s("ORT_QMOE_PREFILL_TOTAL_TOKENS", total_str.c_str());
#else
        setenv("ORT_QMOE_PREFILL_TOTAL_TOKENS", total_str.c_str(), 1);
#endif
      }
      st.dirty = true;
      device_->PrefillChunkSlot(s, req.tokens.subspan(st.prefilled, chunk));
      st.prefilled += chunk;
      if (st.prefilled < req.tokens.size()) {
        all_decoding = false;
        continue;
      }
      const size_t limit = std::min(req.token_budget, req.session_budget);
      const int32_t first = device_->PendingTokenSlot(s);
      const bool want_more = !IsStopToken(first) && limit > 1;
      device_->BeginDecodeSlot(s, want_more);
      st.decoding = true;
      st.want_more = want_more;
      results[r].tokens[0] = first;
      results[r].count = 1;
      stats_.tokens_emitted += 1;
    }
  }

  if (all_decoding) {
    std::vector<size_t> active_slots;
    std::vector<const void*> active_owners;
    std::vector<uint64_t> active_turns;
    for (const auto& req : requests) {
      const SlotState& st = slot_states_.at(req.slot);
      if (st.decoding && st.want_more) {
        active_slots.push_back(req.slot);
        active_owners.push_back(st.owner);
        active_turns.push_back(st.turn_id);
      }
    }
    if (!active_slots.empty()) {
      device_->EnqueueBatchDecode(active_slots);
      batch_inflight_ = true;
      inflight_slots_ = std::move(active_slots);
      inflight_owners_ = std::move(active_owners);
      inflight_turns_ = std::move(active_turns);
    }
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
  add(stats.zero_accept_rounds, stats_.zero_accept_rounds);
  for (size_t i = 0; i < stats.acceptance_length_histogram.size(); ++i) {
    add(stats.acceptance_length_histogram[i], stats_.acceptance_length_histogram[i]);
  }
}

// ------------------------------------------------------------------------------------------------
// Cache manager
// ------------------------------------------------------------------------------------------------
Qwen38GraphCacheManager::Qwen38GraphCacheManager(std::shared_ptr<Model> model, std::shared_ptr<Qwen38GraphSession> session)
    : CacheManager{std::move(model)},
      session_{std::move(session)},
      max_slots_{session_ ? session_->MaxBatchSize() : 1},
      slots_(max_slots_, nullptr) {
  if (!session_) {
    throw std::invalid_argument("Qwen38GraphCacheManager requires a session.");
  }
}

bool Qwen38GraphCacheManager::CanAllocate(const std::vector<std::shared_ptr<Request>>& requests) const {
  if (requests.empty() || requests.size() > max_slots_) {
    return false;
  }
  size_t available = 0;
  for (const auto& resident : slots_) {
    if (!resident || IsTurnComplete(resident->status_) || IsClosed(resident->status_)) {
      ++available;
    }
  }
  return requests.size() <= available;
}

void Qwen38GraphCacheManager::Allocate(const std::vector<std::shared_ptr<Request>>& requests) {
  if (!CanAllocate(requests)) {
    throw std::runtime_error("The qwen38_flash_graph backend cannot allocate the requested batch.");
  }
  for (const auto& req : requests) {
    if (IsResident(req)) {
      continue;
    }
    ssize_t target_slot = -1;
    for (size_t s = 0; s < max_slots_; ++s) {
      if (!slots_[s] || IsClosed(slots_[s]->status_)) {
        target_slot = static_cast<ssize_t>(s);
        break;
      }
    }
    if (target_slot < 0) {
      for (size_t s = 0; s < max_slots_; ++s) {
        if (IsTurnComplete(slots_[s]->status_)) {
          target_slot = static_cast<ssize_t>(s);
          break;
        }
      }
    }
    if (target_slot < 0) {
      throw std::runtime_error("The qwen38_flash_graph backend has no free slot.");
    }
    const size_t s = static_cast<size_t>(target_slot);
    if (slots_[s]) {
      session_->ReleaseSlot(s);
    }
    slots_[s] = req;
  }
}

void Qwen38GraphCacheManager::Deallocate(std::vector<std::shared_ptr<Request>>& requests) {
  if (requests.empty()) {
    throw std::runtime_error("Cannot dynamically deallocate statically batched requests.");
  }
  for (const auto& req : requests) {
    if (!IsResident(req)) {
      throw std::runtime_error("Cannot dynamically deallocate statically batched requests.");
    }
  }
  for (const auto& req : requests) {
    for (size_t s = 0; s < max_slots_; ++s) {
      if (slots_[s] == req) {
        session_->ReleaseSlot(s);
        slots_[s].reset();
        break;
      }
    }
  }
}

void Qwen38GraphCacheManager::ValidateRewind(const std::shared_ptr<Request>&) const {
  // Each resident row owns its dedicated slot, so releasing it for a replay never affects a peer.
}

void Qwen38GraphCacheManager::ReleaseForRewind(const std::shared_ptr<Request>& request) {
  for (size_t s = 0; s < max_slots_; ++s) {
    if (slots_[s] == request) {
      session_->ReleaseSlot(s);
      slots_[s].reset();
      return;
    }
  }
}

void Qwen38GraphCacheManager::DetachRequestForTeardown(const std::shared_ptr<Request>& request) noexcept {
  for (size_t s = 0; s < max_slots_; ++s) {
    if (slots_[s] == request) {
      session_->ReleaseSlot(s);
      slots_[s].reset();
      return;
    }
  }
}

std::vector<std::shared_ptr<Request>> Qwen38GraphCacheManager::AllocatedRequests() const {
  std::vector<std::shared_ptr<Request>> allocated;
  allocated.reserve(max_slots_);
  for (const auto& req : slots_) {
    if (req) {
      allocated.push_back(req);
    }
  }
  return allocated;
}

bool Qwen38GraphCacheManager::IsResident(const std::shared_ptr<Request>& request) const {
  if (!request) return false;
  for (const auto& req : slots_) {
    if (req == request) return true;
  }
  return false;
}

size_t Qwen38GraphCacheManager::ResidentRequestCount() const {
  size_t count = 0;
  for (const auto& req : slots_) {
    if (req) ++count;
  }
  return count;
}

size_t Qwen38GraphCacheManager::SlotOf(const std::shared_ptr<Request>& request) const {
  for (size_t s = 0; s < max_slots_; ++s) {
    if (slots_[s] == request) return s;
  }
  return 0;
}

// ------------------------------------------------------------------------------------------------
// Scheduler
// ------------------------------------------------------------------------------------------------
namespace {

struct Qwen38GraphScheduler final : Scheduler {
  Qwen38GraphScheduler(std::shared_ptr<Model> model, std::shared_ptr<Qwen38GraphCacheManager> cache_manager)
      : Scheduler{model}, model_{std::move(model)}, cache_manager_{std::move(cache_manager)} {}

  void AddRequest(std::shared_ptr<Request> request) override {
    if (request->PrefillChunkSize().value_or(0) != 0) {
      throw std::runtime_error(
          "search.chunk_size requires dynamic batching; the static batch scheduler cannot chunk a "
          "prefill.");
    }
    requests_pool_.reserve(requests_pool_.size() + 1);
    auto sampling_state = CreateSamplingState(*request);
    request->CommitSamplingState(std::move(sampling_state));
    requests_pool_.push_back(std::move(request));
  }

  void RemoveRequest(std::shared_ptr<Request> request) override {
    if (cache_manager_->IsResident(request)) {
      std::vector<std::shared_ptr<Request>> to_deallocate{request};
      cache_manager_->Deallocate(to_deallocate);
    }
    requests_pool_.erase(
        std::remove(requests_pool_.begin(), requests_pool_.end(), request),
        requests_pool_.end());
  }

  void DetachRequestForTeardown(const std::shared_ptr<Request>& request) noexcept override {
    cache_manager_->DetachRequestForTeardown(request);
    requests_pool_.erase(
        std::remove(requests_pool_.begin(), requests_pool_.end(), request),
        requests_pool_.end());
  }

  ScheduledRequests Schedule() override {
    for (const auto& req : cache_manager_->AllocatedRequests()) {
      if (IsClosed(req->status_)) {
        std::vector<std::shared_ptr<Request>> closed_vec{req};
        cache_manager_->Deallocate(closed_vec);
        requests_pool_.erase(
            std::remove(requests_pool_.begin(), requests_pool_.end(), req),
            requests_pool_.end());
      } else if (IsQueued(req->status_)) {
        req->Schedule();
      }
    }

    std::vector<std::shared_ptr<Request>> requests_to_schedule;
    for (const auto& request : requests_pool_) {
      if (IsQueued(request->status_) && !cache_manager_->IsResident(request)) {
        requests_to_schedule.push_back(request);
      }
    }

    for (const auto& request : requests_to_schedule) {
      if (!cache_manager_->CanAllocate({request})) {
        break;
      }
      if (cache_manager_->ResidentRequestCount() >= cache_manager_->MaxBatchSize()) {
        for (const auto& resident : cache_manager_->AllocatedRequests()) {
          if (IsTurnComplete(resident->status_) || IsClosed(resident->status_)) {
            std::vector<std::shared_ptr<Request>> evicted{resident};
            cache_manager_->Deallocate(evicted);
            requests_pool_.erase(
                std::remove(requests_pool_.begin(), requests_pool_.end(), resident),
                requests_pool_.end());
            break;
          }
        }
      }
      cache_manager_->Allocate({request});
      request->Schedule();
    }

    ScheduledRequests scheduled_requests(cache_manager_->AllocatedRequests(), model_, GetBatchedSampler(),
                                         GetBatchedSamplingPlan());
    if (!scheduled_requests) {
      throw std::runtime_error("Unable to schedule requests: no requests available or all requests are completed.");
    }
    return scheduled_requests;
  }

  bool HasPendingRequests() const override {
    for (const auto& request : requests_pool_) {
      if (IsExecutable(request->status_)) {
        return true;
      }
    }
    return false;
  }

 private:
  std::shared_ptr<Model> model_;
  std::shared_ptr<Qwen38GraphCacheManager> cache_manager_;
  std::vector<std::shared_ptr<Request>> requests_pool_;
};

}  // namespace

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
  // Selecting its own tokens means the step always reports through the executor path, even when it commits none.
  scheduled_requests.StageExecutorTokens(0, {});
  if (session_->MaxBatchSize() == 1) {
    for (size_t row = 0; row < scheduled_requests.size(); ++row) {
      const auto& request = scheduled_requests[row];
      if (!IsExecuting(request->status_)) {
        continue;
      }
      const int64_t used = request->CurrentSequenceLength();
      const size_t session_limit = request->MaxSessionTokens();
      Qwen38StepRequest step;
      step.slot = 0;
      step.owner = request.get();
      step.turn_id = request->CurrentTurnId();
      step.tokens = request->TokensCpu();
      step.token_budget = request->RemainingTurnTokenBudget();
      step.session_budget = static_cast<size_t>(used) < session_limit ? session_limit - static_cast<size_t>(used) : 0;
      const Qwen38StepResult result = session_->Step(step);
      scheduled_requests.StageExecutorTokens(row, std::span<const int32_t>{result.tokens.data(), result.count});
      return;
    }
    return;
  }

  std::vector<Qwen38StepRequest> steps;
  std::vector<size_t> active_rows;
  steps.reserve(scheduled_requests.size());
  active_rows.reserve(scheduled_requests.size());
  for (size_t row = 0; row < scheduled_requests.size(); ++row) {
    const auto& request = scheduled_requests[row];
    if (!IsExecuting(request->status_)) {
      continue;
    }
    const int64_t used = request->CurrentSequenceLength();
    const size_t session_limit = request->MaxSessionTokens();
    Qwen38StepRequest step;
    step.slot = cache_manager_ ? cache_manager_->SlotOf(request) : row;
    step.owner = request.get();
    step.turn_id = request->CurrentTurnId();
    step.tokens = request->TokensCpu();
    step.token_budget = request->RemainingTurnTokenBudget();
    step.session_budget = static_cast<size_t>(used) < session_limit ? session_limit - static_cast<size_t>(used) : 0;
    steps.push_back(step);
    active_rows.push_back(row);
  }
  if (steps.empty()) {
    return;
  }
  std::vector<Qwen38StepResult> results(steps.size());
  session_->StepBatch(steps, results);
  for (size_t i = 0; i < active_rows.size(); ++i) {
    scheduled_requests.StageExecutorTokens(
        active_rows[i], std::span<const int32_t>{results[i].tokens.data(), results[i].count});
  }
}

// ------------------------------------------------------------------------------------------------
// Assembly
// ------------------------------------------------------------------------------------------------
EngineDependencies CreateQwen38GraphDependencies(std::shared_ptr<Model> model,
                                                 std::shared_ptr<Qwen38DecodeDevice> device) {
  const auto& config = *model->config_;
  if (config.engine.dynamic_batching && (!device || device->MaxBatchSize() <= 1)) {
    throw std::runtime_error(
        "The qwen38_flash_graph model type runs on the Engine's static single-slot path; engine.dynamic_batching is not supported.");
  }
  if (config.model.qwen38_graph.prefill_chunk_tokens < 2) {
    throw std::runtime_error("model.qwen38_graph.prefill_chunk_tokens must be at least 2.");
  }
  Qwen38GraphSession::Options options;
  options.prefill_chunk_tokens = static_cast<size_t>(config.model.qwen38_graph.prefill_chunk_tokens);
  options.stop_tokens.assign(config.model.eos_token_id.begin(), config.model.eos_token_id.end());
  auto session = std::make_shared<Qwen38GraphSession>(std::move(device), std::move(options));
  auto cache_manager = std::make_shared<Qwen38GraphCacheManager>(model, session);
  std::unique_ptr<Scheduler> scheduler = std::make_unique<Qwen38GraphScheduler>(model, cache_manager);
  auto executor = std::make_unique<Qwen38GraphExecutor>(std::move(session), cache_manager);
  return EngineDependencies{std::move(cache_manager), std::move(scheduler), std::move(executor)};
}

EngineDependencies CreateQwen38GraphDependencies(std::shared_ptr<Qwen38GraphModel> model) {
  auto device = model->AcquireDevice();
  return CreateQwen38GraphDependencies(std::shared_ptr<Model>{std::move(model)}, std::move(device));
}

}  // namespace Generators
