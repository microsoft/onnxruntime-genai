// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <limits>

#include "generator/generators.h"
#include "request_status.h"
#include "engine_invariants.h"
#include "step_plan.h"

/**
 * @file request.h
 * @brief Defines the request class that manages the state for each incoming user request.
 *        It handles the lifecycle of a request, from creation to completion.
 */

namespace Generators {

namespace test {
struct RequestGuidanceTestAccess;
}

struct Request;
struct ScheduledRequests;

struct RequestOptions {
  std::optional<size_t> max_session_tokens;
};

struct TurnOptions {
  std::weak_ptr<Request> request;
  std::optional<size_t> max_generated_tokens;

  void ValidateOwnerThread() const;
};

struct RequestStepResult {
  int32_t token{};
  bool token_appended{};
  bool done{};
  GenerationFinishReason finish_reason{GenerationFinishReason::None};
  std::array<int32_t, kMaxGeneratedTokensPerStep> visible_tokens{};
  size_t visible_token_count{};
};

struct RequestTurnAdmission {
  RequestStatus status{RequestStatus::Unassigned};
  size_t host_token_count{};
  int64_t prompt_sequence_length{};
  int64_t processed_sequence_length{};
  bool transaction_started{};
};

struct RequestTurnCounters {
  uint64_t prompt_tokens{};
  uint64_t generated_tokens{};
};

/**
 * @class Request
 * @brief Manages the state and lifecycle of a user request within the engine.
 *
 * The Request class tracks the progress of a user request, including its status,
 * input tokens, and generated outputs. It provides interfaces for adding tokens,
 * querying newly generated tokens, and interacting with the engine. Requests are
 * processed concurrently by the Engine, which dynamically batches them for efficient
 * model execution.
 */
struct Request : std::enable_shared_from_this<Request>,
                 LeakChecked<Request>,
                 ExternalRefCounted<Request> {
  /**
   * @brief Constructs a Request object with the given generator parameters.
   * @param params Shared pointer to GeneratorParams containing generation configuration.
   */
  Request(
      std::shared_ptr<GeneratorParams> params,
      size_t max_total_tokens,
      std::shared_ptr<std::atomic<bool>> abandonment_pending =
          std::make_shared<std::atomic<bool>>(false));
  ~Request();

  /**
   * @brief Updates the status of the request to Active and prepares it for processing.
   */
  void Schedule();

  uint64_t BeginTurn(
      std::span<const int32_t> tokens,
      std::optional<size_t> max_generated_tokens = std::nullopt);
  void ValidateOwnerThread() const;
  void AttachToEngine(std::shared_ptr<Engine> engine) noexcept;
  bool BelongsTo(const Engine& engine) const noexcept;
  bool IsAwaitingFirstTurn() const noexcept;
  bool IsRestartableCanceledTurn() const noexcept;
  void ValidateTurnAdmission(
      std::span<const int32_t> tokens,
      std::optional<size_t> max_generated_tokens) const;
  void ValidateContinuousDecodingSupport() const;
  void PrepareTurnAdmission(
      std::span<const int32_t> tokens,
      RequestTurnAdmission& admission);
  uint64_t CommitTurnAdmission(
      std::optional<size_t> max_generated_tokens,
      RequestTurnAdmission& admission);
  void RollbackTurnAdmission(RequestTurnAdmission& admission);
  bool CanCancelFromEngine(
      const Engine& engine,
      uint64_t turn_id) const;
  RequestTurnCounters CompleteCancelFromEngine(
      const Engine& engine,
      uint64_t turn_id) noexcept;
  // Publishes logical close without destroying runtime state that a resident static row still
  // needs. CompleteCloseFromEngine finishes both logical and physical close.
  void MarkClosedFromEngine(const Engine& engine) noexcept;
  void CompleteCloseFromEngine(const Engine& engine) noexcept;
  void MarkFailedFromEngine(const Engine& engine) noexcept;
  void CompleteFailedTurnFromEngine(const Engine& engine) noexcept;

  // Engine-hosted MTP head requests mirror a target request's committed tokens in a
  // scheduler-private shadow the Engine owns outright. They never sample and never enter the
  // public turn API, so they are admitted through this factory instead of BeginTurn.
  static std::shared_ptr<Request> CreateAuxiliaryDecoderRequest(
      std::shared_ptr<GeneratorParams> params,
      size_t max_total_tokens,
      std::shared_ptr<std::atomic<bool>> abandonment_pending,
      const std::shared_ptr<Engine>& engine,
      std::span<const int32_t> tokens);
  void AppendTokensForAuxiliaryDecoder(std::span<const int32_t> tokens);
  void AppendTokensForAuxiliaryDecoder(DeviceSpan<int32_t> tokens);
  void RewindAuxiliaryDecoderTo(size_t sequence_length);
  void CommitAuxiliaryDecoderStep() noexcept;

  /**
   * @brief Total tokens (prompt plus generated) this request may ever reach.
   */
  size_t MaxTotalTokens() const noexcept { return max_total_tokens_; }

  /**
   * @brief The speculative-decoding options this request was created with.
   */
  const Config::Speculative& SpeculativeOptions() const;

  /**
   * @brief Why the pending draft proposal cannot be verified, or nullptr when it can.
   */
  const char* DraftTokenValidationError() const noexcept;

  /**
   * @brief Returns a span of unprocessed tokens on the device.
   * @return DeviceSpan containing unprocessed token IDs.
   *
   * Unprocessed tokens are those tokens that have not been processed by
   * the model yet. They are used for token generation in the next step.
   */
  DeviceSpan<int32_t> UnprocessedTokens();

  /**
   * @brief Returns the unprocessed tokens from the host-side mirror of the sequence.
   * @return Span of unprocessed token IDs, valid only until the next call that appends to the
   *         sequence (CommitStep, CompleteGeneration, or BeginTurn). Copy it if it must
   *         outlive those.
   *
   * Same tokens as UnprocessedTokens(), but readable without copying them back from the device.
   * Building the next step's input ids is the hot path for this, and a device readback there costs
   * one full stream synchronization per request per step.
   */
  std::span<const int32_t> UnprocessedTokensCpu() const;

  /**
   * @brief Launches the generation of the next token based on the provided logits.
   * @param logits DeviceSpan containing logits for token generation.
   *
   * The work is only launched here. CompleteGeneration() must be called afterwards to pick up the
   * results and to update the request status. Splitting the two lets the engine launch every
   * scheduled request's token selection before it synchronizes with the device once.
   */
  void GenerateNextTokens(DeviceSpan<float> logits, bool guidance_applied = false);

  /**
   * @brief Proposes speculative draft tokens for this request's next step.
   * @param tokens Draft continuation of the sequence, in order. An empty span clears the proposal.
   *
   * The request must be ready to decode. The next step then runs 1 + tokens.size() rows, verifies
   * each draft against the target model's own prediction, and keeps the accepted prefix. Greedy
   * requests compare argmax tokens; sampled requests draw from each target row's bounded
   * top-k/top-p distribution.
   *
   * The proposal applies to the current turn's next committed decode step. A rolled back step
   * leaves it pending, while canceling the turn discards it.
   */
  void SetDraftTokens(std::span<const int32_t> tokens);

  /**
   * @brief Draft tokens proposed for the next step but not yet sent through the model.
   */
  size_t PendingDraftTokenCount() const noexcept { return draft_tokens_.size(); }

  /**
   * @brief Drafts of the step in flight that the target model accepted.
   */
  size_t AcceptedDraftTokenCount() const noexcept { return accepted_draft_count_; }

  /**
   * @brief Sequence length excluding any drafts staged by the step in flight.
   */
  int64_t CommittedSequenceLength() const;

  void AppendDraftsForTransaction(size_t draft_count);
  std::span<const int32_t> StagedDraftTokens() const;
  void CommitAcceptedDraftsForTransaction(size_t accepted_count);
  bool DraftVerificationCompletedGeneration() const noexcept {
    return draft_verification_completed_generation_;
  }
  bool IsStopToken(int32_t token) const;
  void RewindDraftsForTransaction(size_t accepted_count);
  void RecordSampledDraftAcceptance(size_t accepted_count);

  void ValidateEngineCompatibility() const;
  void SaveStateForTransaction();
  void SaveStateForNewTurnTransaction();
  void SaveStateForExternalSamplingTransaction();
  RequestStepResult ApplyLogitsForTransaction(DeviceSpan<float> logits,
                                              bool guidance_applied = false);
  void PrepareGenerationForTransaction(DeviceSpan<float> logits,
                                       bool guidance_applied = false);
  RequestStepResult StageGenerationForTransaction(
      const RequestStepPlan& plan);
  RequestStepResult StageDraftCompletionForTransaction();
  void RestoreStateForTransaction();
  void QueueStateRestoreForTransaction();
  void CompleteStateRestoreForTransaction();
  void CommitStateForTransaction();
  void CommitStep(const RequestStepPlan& plan,
                  const RequestStepResult& result) noexcept;

  /**
   * @brief Completes the generation started by GenerateNextTokens().
   *
   * Updates the host-side token mirror and the request status.
   */
  RequestStepResult CompleteGeneration();

  /**
   * @brief Checks if the current generation turn reached a stopping condition.
   * @return True in TurnComplete; the request may still be continued or closed.
   */
  bool IsTurnComplete() const;
  uint64_t CurrentTurnId() const noexcept { return current_turn_id_; }
  bool HasCurrentTurn() const noexcept { return has_current_turn_; }
  GenerationFinishReason FinishReason() const noexcept { return finish_reason_; }
  size_t TurnPromptTokens() const noexcept { return turn_prompt_tokens_; }
  size_t TurnGeneratedTokens() const noexcept { return turn_generated_tokens_; }
  size_t RemainingTurnTokenBudget() const noexcept {
    if (!turn_max_generated_tokens_) {
      return std::numeric_limits<size_t>::max();
    }
    return turn_generated_tokens_ < *turn_max_generated_tokens_
               ? *turn_max_generated_tokens_ - turn_generated_tokens_
               : 0;
  }

  RequestStatus Status() const noexcept { return status_; }

  void Close();
  bool Cancel(uint64_t turn_id);

  /**
   * @brief Checks if the request is in prefill mode.
   * @return True while the tokens the application supplied have not all been through the model.
   *
   * Stays true across every chunk of a chunked prefill, so callers that treat prefill steps
   * differently from decode steps (graph capture, for one) never mistake a trailing one token
   * prefill chunk for a decode step.
   */
  bool IsPrefill() const;

  /**
   * @brief Gets the current sequence length of the request.
   * @return The current sequence length.
   */
  int64_t CurrentSequenceLength() const;

  /**
   * @brief Captures an immutable snapshot of this request's progress counters.
   * @return A RequestStateSnapshot for invariant validation and state inspection.
   *
   * The snapshot copies out the request's status and sequence-length counters; it holds no
   * reference into the request and does not mutate any state.
   */
  RequestStateSnapshot Snapshot() const;

  /**
   * @brief Number of leading tokens of the sequence whose keys and values are already in the cache.
   *
   * This is the absolute position the next scheduled token will be written at, which is what the
   * decoder has to report to the model as the past sequence length.
   */
  int64_t ProcessedSequenceLength() const;

  /**
   * @brief Chooses the tokens this request contributes to the step that is about to run.
   *
   * Used by static batching before anything reads UnprocessedTokens(). Dynamic batching binds the
   * authoritative count from RequestStepPlan instead.
   */
  void ScheduleTokens();

  /**
   * @brief Binds the token count selected by a dynamic RequestStepPlan.
   */
  void BindScheduledTokenCount(size_t token_count);

  /**
   * @brief Tokens this request contributes to the next step.
   *
   * Equivalent to UnprocessedTokens().size() without constructing a device span.
   */
  size_t ScheduledTokenCount() const;

  /**
   * @brief True when this step's tokens run to the end of the sequence.
   *
   * Only then does the last logits row of this request predict a new token. A partial prefill chunk
   * ends in the middle of the prompt, so its logits are discarded.
   */
  bool IsChunkComplete() const;

  /**
   * @brief Moves the cursor past the tokens this step processed.
   */
  void AdvanceChunk();

  RequestStatus status_{RequestStatus::Unassigned};

  /**
   * @brief The search options this request was created with.
   */
  const Config::Search& SearchOptions() const;

  /**
   * @brief Binds this request's search to a one-element slot of a caller-owned next-token buffer.
   * @return True if the search accepted the slot, false if it must be sampled on its own.
   *
   * Used by ScheduledRequests to sample a whole batch of requests in one call. See
   * Search::BindNextTokensSlot.
   */
  bool BindNextTokensSlot(DeviceSpan<int32_t> slot);

  bool SupportsBatchedSampling() const;

  /**
   * @brief Runs everything token selection needs before the sampler: sequence bookkeeping,
   *        handing the logits to the search, and applying the logits processors.
   */
  void PrepareGeneration(DeviceSpan<float> logits, bool guidance_applied = false);
  bool HasGuidance() const { return guidance_logits_processor_ != nullptr; }
  std::span<const uint32_t> GetReadyGuidanceMask();

  /**
   * @brief Launches the per-sequence tail after a batched sampler has filled the bound slot.
   */
  void OnNextTokensSampled();

  /**
   * @brief Returns this request's persistent random state for the given batched sampler.
   */
  BatchedSamplerState& SamplingState(BatchedSampler& sampler);
  void CommitSamplingState(std::unique_ptr<BatchedSamplerState> state) noexcept;

 private:
  // The search sequence is partitioned at processed_sequence_length_: tokens before it already
  // have KV entries, and UnprocessedTokens() returns the scheduled prefix of [processed, current).
  // Host-side mirror of the full sequence (prompt + generated tokens). Kept in step with the
  // search's device sequence so that streaming and input-id preparation never read it back.
  std::vector<int32_t> tokens_host_;
  friend struct ScheduledRequests;
  friend struct test::RequestGuidanceTestAccess;

  void CompleteClose() noexcept;
  static DeviceSpan<int32_t> AllocateOnDevice(
      GeneratorParams& params,
      std::span<const int32_t> input_ids);
  static void ValidateAppendLength(
      size_t max_total_tokens,
      size_t current_sequence_length,
      size_t token_count);
  // Drops whatever the step in flight staged past the committed sequence, leaving the host mirror
  // exactly as long as the search after its own transaction rewind.
  void DiscardStagedDrafts() noexcept;

  int64_t processed_sequence_length_{};
  // Sequence length the application's tokens reach up to. Everything below it is prompt, so the
  // request is still prefilling while processed_sequence_length_ has not caught up with it.
  int64_t prompt_sequence_length_{};
  size_t scheduled_token_count_{};
  std::optional<size_t> turn_max_generated_tokens_;
  size_t turn_prompt_tokens_{};
  size_t turn_generated_tokens_{};
  const size_t max_total_tokens_;
  uint64_t current_turn_id_{};
  uint64_t next_turn_id_{1};
  bool has_current_turn_{};
  bool turn_id_exhausted_{};
  GenerationFinishReason finish_reason_{GenerationFinishReason::None};
  // Drafts proposed for the next step, the ones the step in flight staged onto the sequence, and
  // the leading part of those the target model accepted.
  std::vector<int32_t> draft_tokens_;
  size_t staged_draft_count_{};
  size_t accepted_draft_count_{};
  bool draft_verification_completed_generation_{};
  std::shared_ptr<GeneratorParams> params_;
  std::mt19937 rng_;
  std::mt19937 transaction_rng_;
  int64_t transaction_processed_sequence_length_{};
  size_t transaction_tokens_host_size_{};
  std::unique_ptr<Search> search_;
  std::unique_ptr<ConstrainedLogitsProcessor> guidance_logits_processor_;
  std::unique_ptr<ConstrainedLogitsProcessor> guidance_transaction_checkpoint_;
  std::unique_ptr<BatchedSamplerState> batched_sampler_state_;
  std::weak_ptr<Engine> engine_;
  const Engine* engine_identity_{};

  void ApplyLogitsProcessors(DeviceSpan<float> logits, bool guidance_applied);
  void SelectNextToken();
  void StageVisibleTokens(RequestStepResult& result,
                          size_t committed_count,
                          std::optional<int32_t> sampled_token) const;
  RequestStepResult StageGeneration(int64_t sequence_length_before);
  void CommitGuidanceToken(const RequestStepResult& result);
};

}  // namespace Generators
