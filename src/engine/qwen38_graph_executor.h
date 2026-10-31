// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <exception>
#include <memory>
#include <vector>

#include "../decoding/speculative_stats.h"
#include "../models/qwen38_decode_device.h"
#include "../span.h"
#include "engine.h"
#include "fixed_slot_scheduler.h"
#include "token_producing_session.h"

// Qwen-specific device-cycle orchestration. Reusable admission/residency lives in
// FixedSlotCacheManager/FixedSlotScheduler; completion identity and settlement live in
// TokenProducingSession. Request owns token policy and public commit/event semantics.

namespace Generators {

struct Qwen38GraphModel;

// What the session needs to know about the Request it serves, captured at the start of each Engine step.
struct Qwen38StepRequest {
  size_t slot{};                    // slot index in [0, MaxBatchSize())
  const void* owner{};              // the Request, as an identity
  uint64_t turn_id{};               // changes whenever the Request begins a new turn
  std::span<const int32_t> tokens;  // the Request's whole token history (prompt of every turn plus generated tokens)
  size_t token_budget{};            // generated tokens the turn may still produce
  size_t session_budget{};          // tokens the session may still hold
};

struct Qwen38StepResult {
  std::array<int32_t, kMaxGeneratedTokensPerStep> tokens{};
  size_t count{};  // 0 for a step that only advanced the prompt
  size_t prompt_tokens_processed{};
  bool prompt_complete{};
  SpeculativeRowMetrics speculation;
};

class Qwen38GraphSession : public FixedSlotSession {
 public:
  struct Options {
    size_t prefill_chunk_tokens{2048};  // S=2 step prefill: rounded down to an even number (pairs stay aligned), at least 2;
                                        // replaced by Qwen38DecodeDevice::PrefillChunkTokens() when the device names one
                                        // (the assembly passes EffectiveQwen38PrefillChunkTokens of the configuration)
    std::vector<int32_t> stop_tokens;   // EOS ids of the model: a cycle is not followed by another one after one of them
    int vocab_size{0};                  // > 0: validates all request token ids at turn start before mutating device state
  };

  Qwen38GraphSession(std::shared_ptr<Qwen38DecodeDevice> device, Options options);
  ~Qwen38GraphSession() override;
  Qwen38GraphSession(const Qwen38GraphSession&) = delete;
  Qwen38GraphSession& operator=(const Qwen38GraphSession&) = delete;

  // Longest session (prompt plus generated tokens) the device can hold.
  size_t MaxRequestLength() const override;

  // Maximum number of concurrent requests the device can hold.
  size_t MaxBatchSize() const override;

  // A new owner or turn announces its authoritative history length before bounded ingestion.

  Qwen38StepResult Step(const Qwen38StepRequest& request);

  // Runs one batched step across `requests` (in their assigned slots) and writes per-request results to `results`.
  void StepBatch(std::span<const Qwen38StepRequest> requests, std::span<Qwen38StepResult> results);

  // Immutable work selection: decode residents may be paused by the backend's serialized prefill barrier.
  std::vector<BackendPlannedRow> Plan(std::span<const Qwen38StepRequest> requests, std::span<const size_t> request_rows);
  void Collected(const BackendRowResult& result) { completions_.Collected(result); }
  void Acknowledge(const TokenCommitAck& ack);
  void Poison() noexcept;

  // Records the committed token counts of `slot` after Request::CommitExternalTokens applies EOS, limits, and stop
  // strings. Ignored when `owner` no longer owns the slot.
  void OnStepCommitted(size_t slot, const void* owner, const RequestStepResult& result);

  // The owner leaves (finished, rewound, replaced, or deallocated): drains work in flight, checks any unchecked decode
  // status, and forgets the owner.
  void Release();

  // No-throw variant used by destructors and teardown: captures any status-check failure to report on the next operation.
  void ReleaseForTeardown() noexcept;

  // The owner of one slot leaves. A device with a single slot is drained like Release(); with several slots the other
  // slots keep decoding, so only this slot is forgotten and its device state is cleared before it is reused.
  void ReleaseSlot(size_t slot) override;
  void ReleaseSlotForTeardown(size_t slot) noexcept override;

  void AddStats(SpeculativeStats& stats) const;
  const BackendWorkMetrics& WorkMetrics() const { return work_metrics_; }

 private:
  struct PendingStep {
    bool active{};
    bool is_decode_cycle{};
    bool mtp{};
    size_t proposed{};
    size_t accepted{};
    size_t produced{};
    bool count_forward_pass{true};  // a batched step runs one target forward for all of its slots
  };

  struct SlotState {
    const void* owner{};
    uint64_t turn_id{};
    uint64_t epoch{};
    size_t prefilled{};
    bool decoding{};
    bool want_more{};
    bool dirty{true};
    PendingStep pending{};
  };

  bool IsStopToken(int32_t token) const;
  void Begin(const Qwen38StepRequest& request);
  void BeginSlot(size_t slot, const Qwen38StepRequest& request);
  // Splits a cycle record into the tokens the Engine may commit and the statistics to apply once it has.
  PendingStep ProcessCycleRecord(const Qwen38CycleRecord& record, size_t limit, Qwen38StepResult& result, bool& ended);
  void ApplyCommitted(const PendingStep& step, const RequestStepResult& result);
  void ClearSlot(SlotState& slot) noexcept;
  void RethrowDeferredError();
  void ReportDeferredError() const;
  void CheckStatusIfPending();
  // The step before anything that clears the device state (Reset(), ResetSlot()), which also clears the error bits.
  void DrainAndCheckStatus();

  std::shared_ptr<Qwen38DecodeDevice> device_;
  Options options_;
  const void* owner_{};
  uint64_t turn_id_{};
  size_t prefilled_{};
  bool decoding_{};
  // Unknown until proven otherwise: the device may hold a previous Engine's state (it outlives its Engine), so the first
  // request always starts with a Reset.
  bool device_dirty_{true};
  bool status_checked_{true};
  std::exception_ptr deferred_error_;
  PendingStep pending_step_{};
  std::vector<SlotState> slot_states_;
  bool batch_inflight_{false};
  std::vector<size_t> inflight_slots_;
  std::vector<const void*> inflight_owners_;
  std::vector<uint64_t> inflight_turns_;
  std::vector<uint64_t> inflight_epochs_;
  SpeculativeStats stats_;
  TokenProducingSession completions_;
  BackendWorkMetrics work_metrics_;
};

// Compatibility name for existing embedders and tests.
using Qwen38GraphCacheManager = FixedSlotCacheManager;

struct Qwen38GraphExecutor final : ModelExecutor {
  Qwen38GraphExecutor(std::shared_ptr<Qwen38GraphSession> session,
                      std::shared_ptr<Qwen38GraphCacheManager> cache_manager = nullptr)
      : session_{std::move(session)}, cache_manager_{std::move(cache_manager)} {}

  void Decode(ScheduledRequests& scheduled_requests, ExecutionContext& context) override;
  bool SelectsOwnTokens() const override { return true; }
  void ValidateTurn(const EffectiveTurnPolicy& policy, const TurnOptions& options) const override;
  void AddSpeculativeStats(SpeculativeStats& stats) const override { session_->AddStats(stats); }
  void OnExecutionFailure() noexcept override { session_->Poison(); }
  void OnStepCommitted(const ScheduledRequests& scheduled_requests,
                       std::span<const RequestStepResult> results) override;

 private:
  std::shared_ptr<Qwen38GraphSession> session_;
  std::shared_ptr<Qwen38GraphCacheManager> cache_manager_;
};

// Assembles the Engine's collaborators for a qwen38_flash_graph model (leases the model's device engine).
EngineDependencies CreateQwen38GraphDependencies(std::shared_ptr<Qwen38GraphModel> model);

// The same assembly over an arbitrary device; used by tests with a scripted device.
EngineDependencies CreateQwen38GraphDependencies(std::shared_ptr<Model> model,
                                                 std::shared_ptr<Qwen38DecodeDevice> device);

}  // namespace Generators
