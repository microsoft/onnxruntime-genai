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

/**
 * @file qwen38_graph_executor.h
 * @brief Hosts the Qwen3.8-Flash CUDA-graph decode loop (MTP(1) draft -> verify -> commit cycles on the device) in the
 *        Engine's static path.
 *
 * The loop owns a fixed in-place KV store, a recurrent-state bank ring and a device-side accept/commit step, so it is a
 * single-slot backend: one resident Request, greedy decoding, no prefix cache. Reusable residency lives in
 * FixedSlotCacheManager; completion identity and settlement live in TokenProducingSession.
 */

namespace Generators {

struct Qwen38GraphModel;

// What the session needs to know about the Request it serves, captured at the start of each Engine step.
struct Qwen38StepRequest {
  const void* owner{};              // the Request, as an identity
  uint64_t turn_id{};               // changes whenever the Request begins a new turn
  std::span<const int32_t> tokens;  // the Request's whole token history (prompt of every turn plus generated tokens)
  size_t token_budget{};            // generated tokens the turn may still produce
  size_t session_budget{};          // tokens the session may still hold
};

struct Qwen38StepResult {
  std::array<int32_t, 4> tokens{};
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
  size_t MaxBatchSize() const override { return 1; }

  // Runs one step for `request`: a new owner or turn first clears the device, announces the history length to it
  // (Qwen38DecodeDevice::ExpectPrefillTokens) and starts ingesting the whole history; the step that ingests its last
  // chunk also returns the first generated token and enqueues the first cycle; every later step waits for the cycle in
  // flight, commits it and enqueues the next one before returning its tokens.
  Qwen38StepResult Step(const Qwen38StepRequest& request);

  BackendPlannedRow Plan(size_t request_row, const Qwen38StepRequest& request);
  void Collected(const BackendRowResult& result) { completions_.Collected(result); }
  void Acknowledge(const TokenCommitAck& ack);
  void Poison() noexcept;

  // Records the committed token counts after Request::CommitExternalTokens applies EOS, limits, and stop strings.
  void OnStepCommitted(const RequestStepResult& result);

  const void* Owner() const noexcept { return owner_; }

  // The owner leaves (finished, rewound, replaced, or deallocated): drains work in flight, checks any unchecked decode
  // status, and forgets the owner.
  void Release();

  // No-throw variant used by destructors and teardown: captures any status-check failure to report on the next operation.
  void ReleaseForTeardown() noexcept;

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
  };

  bool IsStopToken(int32_t token) const;
  void Begin(const Qwen38StepRequest& request);

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
  SpeculativeStats stats_;
  TokenProducingSession completions_{1};
  BackendWorkMetrics work_metrics_;
};

using Qwen38GraphCacheManager = FixedSlotCacheManager;

struct Qwen38GraphExecutor final : ModelExecutor {
  explicit Qwen38GraphExecutor(std::shared_ptr<Qwen38GraphSession> session) : session_{std::move(session)} {}

  void Decode(ScheduledRequests& scheduled_requests, ExecutionContext& context) override;
  bool SelectsOwnTokens() const override { return true; }
  void ValidateTurn(const EffectiveTurnPolicy& policy, const TurnOptions& options) const override;
  void AddSpeculativeStats(SpeculativeStats& stats) const override { session_->AddStats(stats); }
  void OnExecutionFailure() noexcept override { session_->Poison(); }
  void OnStepCommitted(const ScheduledRequests& scheduled_requests,
                       std::span<const RequestStepResult> results) override;

 private:
  std::shared_ptr<Qwen38GraphSession> session_;
};

// Assembles the Engine's collaborators for a qwen38_flash_graph model (leases the model's device engine).
EngineDependencies CreateQwen38GraphDependencies(std::shared_ptr<Qwen38GraphModel> model);

// The same assembly over an arbitrary device; used by tests with a scripted device.
EngineDependencies CreateQwen38GraphDependencies(std::shared_ptr<Model> model,
                                                 std::shared_ptr<Qwen38DecodeDevice> device);

}  // namespace Generators
