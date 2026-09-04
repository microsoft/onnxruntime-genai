// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <limits>
#include <string>
#include <vector>

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
}  // namespace test

struct Request;
struct ScheduledRequests;
struct Tokenizer;
class StopStringController;

struct RequestOptions {
  std::optional<size_t> max_session_tokens;
};

struct TurnOptions {
  std::weak_ptr<Request> request;
  std::optional<size_t> max_generated_tokens;
  // Copied UTF-8 stop strings for this turn. Empty means stop strings are disabled (or, when set on
  // an already-configured turn, cleared). Bounds and UTF-8 validity are enforced by
  // OgaTurnOptionsSetStopStrings (via StopStringMatcher) before this is populated.
  std::vector<std::string> stop_strings;

  void ValidateOwnerThread() const;
};

struct RequestStepResult {
  int32_t token{};
  bool token_appended{};
  bool done{};
  GenerationFinishReason finish_reason{GenerationFinishReason::None};
  // Caller-facing index into the turn's stop-string list, or -1 unless this step's committing
  // token completed a match (finish_reason == StopString).
  int32_t matched_stop_string_index{-1};
  std::array<int32_t, kMaxGeneratedTokensPerStep> visible_tokens{};
  size_t visible_token_count{};
};

struct RequestTurnAdmission {
  // Declared (not defaulted) because pending_stop_controller's deleter needs the complete
  // StopStringController type, which this header only forward-declares; defined in request.cpp,
  // which includes stop_string_controller.h.
  ~RequestTurnAdmission();

  RequestStatus status{RequestStatus::Unassigned};
  size_t host_token_count{};
  int64_t prompt_sequence_length{};
  int64_t processed_sequence_length{};
  bool transaction_started{};
  // The complete stop controller this turn will have once committed (null when the turn has no
  // stop strings). The caller builds this -- including any tokenizer/stream construction, which
  // can throw -- before starting this admission attempt, so it is never itself a source of
  // mutation-after-partial-failure. It is moved into Request's live stop_controller_ only by
  // Request::CommitTurnAdmission(); Request::RollbackTurnAdmission() and this struct's own
  // destructor otherwise simply discard it, leaving whatever stop_controller_ the Request had
  // before this attempt (from a prior committed turn, or null) completely untouched.
  std::unique_ptr<StopStringController> pending_stop_controller;
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
  uint64_t BeginTurn(
      std::span<const int32_t> tokens,
      const TurnOptions& options);
  void ValidateOwnerThread() const;
  void AttachToEngine(std::shared_ptr<Engine> engine) noexcept;
  bool BelongsTo(const Engine& engine) const noexcept;
  bool IsAwaitingFirstTurn() const noexcept;
  bool IsRestartableCanceledTurn() const noexcept;
  void ValidateTurnAdmission(
      std::span<const int32_t> tokens,
      const TurnOptions& options) const;
  void ValidateContinuousDecodingSupport() const;
  void PrepareTurnAdmission(
      std::span<const int32_t> tokens,
      RequestTurnAdmission& admission);
  uint64_t CommitTurnAdmission(
      const TurnOptions& options,
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
   *
   * This is the single request-local check every speculative draft producer converges on: manual
   * Request::SetDraftTokens callers get the returned reason as a thrown error, and every automatic
   * in-Engine drafter (e.g. MTP's Engine::PrepareMtpStep) uses the same check to silently skip
   * proposing drafts for this request instead. An active decoded stop-string configuration is not
   * one of these reasons: draft verification observes target-accepted tokens through the request's
   * StopStringController in exactly the same committed order as the ordinary one-token path, so a
   * stop-enabled turn drafts and verifies normally.
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
   * Seeded sampled output is reproducible only when both the supplied drafts and scheduling path
   * are the same. Draft admission changes which random stream performs target sampling.
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
   * @brief Proposed draft positions logically resolved before this step's terminal boundary.
   *
   * This is the confirmed prefix plus, when applicable, one non-accepted proposal that ended
   * verification through rejection or EOS. It never counts a trailing replacement/bonus token,
   * which is not one of the proposed drafts. Always >= AcceptedDraftTokenCount().
   */
  size_t EvaluatedDraftTokenCount() const noexcept { return evaluated_draft_count_; }

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
  // Incrementally records one more sampled-path token that is not this round's finishing stage.
  // Called once per stage, strictly after that stage's StageGenerationForTransaction() call, for
  // every stage except the one that turns out to end the round (a stop match, the turn/context
  // limit, or the request's natural last stage) -- so accepted_draft_count_ always reflects prior
  // stages only, and the following stage's token_appended check (and any stop-string observation
  // nested in it) sees exactly one newly appended token, regardless of which stage that turns out
  // to be. Every stage recorded here is both accepted and evaluated (a non-finishing stage is
  // always a genuinely confirmed draft), so this advances accepted_draft_count_ and
  // evaluated_draft_count_ together. The finishing stage itself is never passed here, whether it is
  // a trailing replacement/bonus token (not one of the proposed drafts) or, when the turn/context
  // limit or a stop match lands exactly on the last proposed draft with no room left for a
  // replacement/bonus token, a genuinely confirmed draft -- see PromoteFinalStageAsAcceptedDraft
  // for that case.
  void AppendAcceptedSampledToken(int32_t token);
  // Called instead of AppendAcceptedSampledToken when the stage that turns out to end this round
  // (a stop-string match, the turn/context limit, or ordinary exhaustion of the proposed drafts) is
  // itself a genuinely target-confirmed draft rather than a trailing replacement/bonus token: folds
  // it into accepted_draft_count_/evaluated_draft_count_/tokens_host_ exactly like an earlier
  // accepted stage would, and suppresses CommitStep()'s own token_appended append of the same token
  // so it is recorded exactly once. Never called for a trailing replacement/bonus token, which
  // stays excluded from both counts exactly as it always has been.
  void PromoteFinalStageAsAcceptedDraft(RequestStepResult& result);
  // Called instead when the finishing stage is a proposed draft position that was logically
  // resolved but not accepted, either because it was rejected (and this stage contains its
  // replacement) or because it is EOS and ends without appending. A trailing bonus token past the
  // proposed range is not a proposal and must not call this. Advances only evaluated_draft_count_;
  // any appended replacement is committed normally through the ordinary token_appended path.
  void MarkFinalStageAsEvaluatedNonAcceptedDraft() noexcept { ++evaluated_draft_count_; }

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
  // Caller-facing index into the turn's stop-string list, valid only when FinishReason() ==
  // StopString. -1 otherwise, including after a cancellation or fatal failure that replaces an
  // undelivered result.
  int32_t MatchedStopStringIndex() const noexcept { return matched_stop_string_index_; }
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
  // Caller-facing stop-string match index committed by CommitStep(), or -1. Only ever written from
  // CommitStep() (after the commit boundary), so unlike stop_controller_ it needs no transactional
  // checkpoint: a rolled-back step never wrote to it.
  int32_t matched_stop_string_index_{-1};
  // Drafts proposed for the next step, the ones the step in flight staged onto the sequence, and
  // the leading part of those the target model accepted.
  std::vector<int32_t> draft_tokens_;
  size_t staged_draft_count_{};
  size_t accepted_draft_count_{};
  // Proposed draft positions whose target acceptance verification has actually examined this
  // round, independent of accept/reject outcome and always >= accepted_draft_count_. Reset and
  // advanced at exactly the same transaction boundaries as accepted_draft_count_ (see the reset
  // sites for that field); read by Engine::RecordSpeculativeCommit before CommitStep() resets it,
  // the same way accepted_draft_count_ is. See EvaluatedDraftTokenCount()'s doc comment above for
  // the exact semantics, and AppendAcceptedSampledToken/PromoteFinalStageAsAcceptedDraft/
  // MarkFinalStageAsEvaluatedNonAcceptedDraft/CommitAcceptedDraftsForTransaction for how each path
  // advances it.
  size_t evaluated_draft_count_{};
  bool draft_verification_completed_generation_{};
  // Set by CommitAcceptedDraftsForTransaction() when a stop-string match ends greedy draft
  // verification early, giving StopString precedence over the turn/context limit for the same
  // completing token. -1 when verification completed generation for any other reason (or has not
  // completed yet). The sampled/batched path never sets this: it observes stop matches through
  // the same StageGenerationForTransaction()/StageGeneration() path the ordinary one-token step
  // uses, one stage at a time, so it needs no separate bookkeeping here.
  //
  // StageDraftCompletionForTransaction() only ever reads this field (and accepted_draft_count_,
  // and Search's own state) to compute its result; it never resets or otherwise mutates it. Only
  // CommitStep()/RestoreStateForTransaction()/QueueStateRestoreForTransaction()/
  // DiscardStagedDrafts()/AppendDraftsForTransaction() reset it, at their own well-defined
  // transaction boundaries -- never in response to StageDraftCompletionForTransaction() being
  // called. This is deliberate: ScheduledRequests::GenerateNextTokensForTransaction() can call
  // StageDraftCompletionForTransaction() for the same request twice in one step when the
  // transaction also uses the batched sampler (once from the batched-sampling setup loop, once
  // unconditionally from the final per-request loop), and both calls must produce the exact same
  // result.
  int32_t draft_verification_stop_match_index_{-1};
  std::shared_ptr<GeneratorParams> params_;
  std::mt19937 rng_;
  std::mt19937 transaction_rng_;
  int64_t transaction_processed_sequence_length_{};
  size_t transaction_tokens_host_size_{};
  std::unique_ptr<Search> search_;
  std::unique_ptr<ConstrainedLogitsProcessor> guidance_logits_processor_;
  std::unique_ptr<ConstrainedLogitsProcessor> guidance_transaction_checkpoint_;
  // Null when the active turn has no stop strings (the no-stop fast path: no tokenizer stream, no
  // decode, no matcher work). Request::CommitTurnAdmission() installs the caller-prebuilt
  // controller only after successful admission, so a failed attempt leaves the previous state
  // untouched. Terminal completion releases it because finish metadata is stored separately.
  std::unique_ptr<StopStringController> stop_controller_;
  // Checkpoint for stop_controller_'s replayable token history, saved by every SaveState*
  // ForTransaction() and consumed by RestoreStateForTransaction()/CompleteStateRestoreForTransaction().
  size_t stop_controller_transaction_checkpoint_{};
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
