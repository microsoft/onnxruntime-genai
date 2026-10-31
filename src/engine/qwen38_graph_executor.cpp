// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qwen38_graph_executor.h"

#include <algorithm>
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
}

Qwen38GraphSession::~Qwen38GraphSession() {
  ReleaseForTeardown();
}

size_t Qwen38GraphSession::MaxRequestLength() const {
  return static_cast<size_t>(device_->Capacity() - Qwen38DecodeDevice::kReservedTail);
}

bool Qwen38GraphSession::IsStopToken(int32_t token) const {
  return std::find(options_.stop_tokens.begin(), options_.stop_tokens.end(), token) != options_.stop_tokens.end();
}

void Qwen38GraphSession::Begin(const Qwen38StepRequest& request) {
  if (request.tokens.empty()) {
    throw std::logic_error("A Request must have at least one token before it can be decoded.");
  }
  // A cycle may still be in flight from a turn that stopped early (stop string, cancellation); drain and check its
  // decode status before Reset clears the device error bits.
  owner_ = nullptr;
  pending_step_ = {};
  if (deferred_error_) {
    std::rethrow_exception(std::exchange(deferred_error_, nullptr));
  }
  if (!status_checked_) {
    device_->Drain();
    status_checked_ = true;
    device_->CheckDecodeStatus();
  }
  if (device_dirty_) {
    device_->Reset();
    device_dirty_ = false;
  }
  owner_ = request.owner;
  turn_id_ = request.turn_id;
  prefilled_ = 0;
  decoding_ = false;
}

void Qwen38GraphSession::Release() {
  device_->Drain();
  owner_ = nullptr;
  prefilled_ = 0;
  decoding_ = false;
  pending_step_ = {};
  if (deferred_error_) {
    std::rethrow_exception(std::exchange(deferred_error_, nullptr));
  }
  if (!status_checked_) {
    status_checked_ = true;
    device_->CheckDecodeStatus();
  }
}

void Qwen38GraphSession::ReleaseForTeardown() noexcept {
  try {
    Release();
  } catch (...) {
    if (!deferred_error_) {
      deferred_error_ = std::current_exception();
    }
  }
}

Qwen38StepResult Qwen38GraphSession::Step(const Qwen38StepRequest& request) {
  if (deferred_error_) {
    std::rethrow_exception(std::exchange(deferred_error_, nullptr));
  }
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
  const std::array<int32_t, 4> predictions{
      static_cast<int32_t>(record.pred0),
      static_cast<int32_t>(record.pred1),
      static_cast<int32_t>(record.pred2),
      static_cast<int32_t>(record.pred3),
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
  // A stop token beyond the limit is never committed, so it does not end the loop by itself.
  const bool ended = (stop_in_cycle && committed == count) || committed == limit;
  // Enqueue the next cycle before returning: the GPU runs it while the Engine delivers these tokens. A stop string can
  // only be known after delivery, so one cycle may be wasted; Reset/Release wait for it.
  device_->Advance(record, /*want_more=*/!ended);
  if (ended) {
    status_checked_ = true;
    device_->CheckDecodeStatus();  // nothing is in flight now: a device that accumulated an error reports it here
  }

  for (size_t i = 0; i < committed; ++i) {
    result.tokens[i] = predictions[i];
  }
  result.count = committed;
  const size_t n_prop = record.proposed_count > 1 ? static_cast<size_t>(record.proposed_count) : 1;
  const size_t n_acc =
      record.accepted_count > 0 ? static_cast<size_t>(record.accepted_count) : (record.accepted ? 1 : 0);
  pending_step_ = PendingStep{/*active=*/true, /*is_decode_cycle=*/true, /*mtp=*/device_->MtpEnabled(),
                              /*proposed=*/n_prop, /*accepted=*/n_acc, /*produced=*/produced};
  return result;
}

void Qwen38GraphSession::OnStepCommitted(const RequestStepResult& result) {
  if (!pending_step_.active) {
    return;
  }
  const PendingStep step = pending_step_;
  pending_step_ = {};
  const size_t emitted = result.visible_token_count;
  stats_.tokens_emitted += emitted;
  if (!step.is_decode_cycle) {
    return;
  }

  ++stats_.rounds;
  ++stats_.completed_rounds;
  ++stats_.target_forward_passes;
  if (step.mtp) {
    const size_t n_prop = step.proposed > 0 ? step.proposed : 1;
    const size_t effective_accepted = std::min(step.accepted, emitted);
    const size_t n_eval = std::min(n_prop, effective_accepted + 1);
    ++stats_.target_verify_forward_passes;
    stats_.draft_forward_passes += n_prop;
    stats_.draft_tokens_proposed += n_prop;
    stats_.draft_tokens_evaluated += n_eval;
    if (effective_accepted < stats_.acceptance_length_histogram.size()) {
      ++stats_.acceptance_length_histogram[effective_accepted];
    }
    stats_.draft_tokens_accepted += effective_accepted;
    if (effective_accepted == n_prop) {
      ++stats_.full_accept_rounds;
      if (emitted > n_prop) {
        ++stats_.bonus_tokens;
      }
    } else if (effective_accepted == 0) {
      ++stats_.zero_accept_rounds;
      if (emitted > 0) {
        ++stats_.correction_tokens;
      }
    } else if (emitted > effective_accepted) {
      ++stats_.correction_tokens;
    }
  }
  if (step.produced >= emitted) {
    stats_.tokens_discarded += step.produced - emitted;
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
    : CacheManager{std::move(model)}, session_{std::move(session)} {
  if (!session_) {
    throw std::invalid_argument("Qwen38GraphCacheManager requires a session.");
  }
}

bool Qwen38GraphCacheManager::CanAllocate(const std::vector<std::shared_ptr<Request>>& requests) const {
  // One slot: more than one row never fits, and the resident row can only be replaced once its turn is over.
  if (requests.size() != 1) {
    return false;
  }
  return !resident_ || IsTurnComplete(resident_->status_) || IsClosed(resident_->status_);
}

void Qwen38GraphCacheManager::Allocate(const std::vector<std::shared_ptr<Request>>& requests) {
  if (!CanAllocate(requests)) {
    throw std::runtime_error("The qwen38_flash_graph backend serves one Request at a time.");
  }
  if (resident_) {
    session_->ReleaseForTeardown();
  }
  resident_ = requests.front();
}

void Qwen38GraphCacheManager::Deallocate(std::vector<std::shared_ptr<Request>>& requests) {
  if (!resident_ || requests.size() != 1 || requests.front() != resident_) {
    throw std::runtime_error("Cannot dynamically deallocate statically batched requests.");
  }
  session_->Release();
  resident_.reset();
}

void Qwen38GraphCacheManager::ValidateRewind(const std::shared_ptr<Request>&) const {
  // The sole resident row owns the whole slot, so releasing it for a replay never affects a peer.
}

void Qwen38GraphCacheManager::ReleaseForRewind(const std::shared_ptr<Request>& request) {
  if (!IsResident(request)) {
    return;
  }
  session_->Release();
  resident_.reset();
}

void Qwen38GraphCacheManager::DetachRequestForTeardown(const std::shared_ptr<Request>& request) noexcept {
  if (!IsResident(request)) {
    return;
  }
  session_->ReleaseForTeardown();
  resident_.reset();
}

std::vector<std::shared_ptr<Request>> Qwen38GraphCacheManager::AllocatedRequests() const {
  return resident_ ? std::vector<std::shared_ptr<Request>>{resident_} : std::vector<std::shared_ptr<Request>>{};
}

bool Qwen38GraphCacheManager::IsResident(const std::shared_ptr<Request>& request) const {
  return resident_ && resident_ == request;
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
  // Selecting its own tokens means the step always reports through the executor path, even when it commits none.
  scheduled_requests.StageExecutorTokens(0, {});
  for (size_t row = 0; row < scheduled_requests.size(); ++row) {
    const auto& request = scheduled_requests[row];
    if (!IsExecuting(request->status_)) {
      continue;
    }
    const int64_t used = request->CurrentSequenceLength();
    const size_t session_limit = request->MaxSessionTokens();
    Qwen38StepRequest step;
    step.owner = request.get();
    step.turn_id = request->CurrentTurnId();
    step.tokens = request->TokensCpu();
    step.token_budget = request->RemainingTurnTokenBudget();
    step.session_budget = static_cast<size_t>(used) < session_limit ? session_limit - static_cast<size_t>(used) : 0;
    const Qwen38StepResult result = session_->Step(step);
    scheduled_requests.StageExecutorTokens(row, std::span<const int32_t>{result.tokens.data(), result.count});
    return;  // one resident row
  }
}

void Qwen38GraphExecutor::OnStepCommitted(const ScheduledRequests& scheduled_requests,
                                          std::span<const RequestStepResult> results) {
  for (size_t row = 0; row < scheduled_requests.size() && row < results.size(); ++row) {
    if (scheduled_requests[row].get() == session_->Owner()) {
      session_->OnStepCommitted(results[row]);
      return;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Assembly
// ------------------------------------------------------------------------------------------------
EngineDependencies CreateQwen38GraphDependencies(std::shared_ptr<Model> model,
                                                 std::shared_ptr<Qwen38DecodeDevice> device) {
  const auto& config = *model->config_;
  if (config.engine.dynamic_batching) {
    throw std::runtime_error(
        "The qwen38_flash_graph model type runs on the Engine's static single-slot path; engine.dynamic_batching is not supported.");
  }
  const auto& qwen38 = config.model.qwen38_graph;
  ValidateQwen38PrefillChunk(qwen38);
  Qwen38GraphSession::Options options;
  options.prefill_chunk_tokens = EffectiveQwen38PrefillChunkTokens(qwen38);
  options.stop_tokens.assign(config.model.eos_token_id.begin(), config.model.eos_token_id.end());
  auto session = std::make_shared<Qwen38GraphSession>(std::move(device), std::move(options));
  auto cache_manager = std::make_shared<Qwen38GraphCacheManager>(model, session);
  auto scheduler = Scheduler::Create(model, cache_manager);
  auto executor = std::make_unique<Qwen38GraphExecutor>(std::move(session));
  return EngineDependencies{std::move(cache_manager), std::move(scheduler), std::move(executor)};
}

EngineDependencies CreateQwen38GraphDependencies(std::shared_ptr<Qwen38GraphModel> model) {
  auto device = model->AcquireDevice();
  return CreateQwen38GraphDependencies(std::shared_ptr<Model>{std::move(model)}, std::move(device));
}

}  // namespace Generators
