// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../generators.h"
#include "request_status.h"
#include "engine_invariants.h"
#include "step_plan.h"

/**
 * @file request.h
 * @brief Defines the request class that manages the state for each incoming user request.
 *        It handles the lifecycle of a request, from creation to completion.
 */

namespace Generators {

struct Request;
struct ScheduledRequests;
struct StaticBatchScheduler;

template <>
struct ExternalRefCountedTraits<Request> {
  static constexpr bool notify_external_reference_changes = true;
};

struct RequestStepResult {
  int32_t token{};
  bool token_appended{};
  bool done{};
};

// Every Engine Request has one sequence and one beam, so a chunk-complete step can append at most
// one generated-output index.
inline constexpr size_t kMaxGeneratedTokenIndicesPerStep = 1;

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
  Request(std::shared_ptr<GeneratorParams> params);

  /**
   * @brief Assigns this request to a specific engine for processing.
   * @param engine Shared pointer to the Engine to be used for processing this request.
   *
   * Once assigned, the request will finalize the prefill tokens and prepare for scheduling.
   */
  void Assign(std::shared_ptr<Engine> engine);

  /**
   * @brief Updates the status of the request to Active and prepares it for processing.
   */
  void Schedule();

  /**
   * @brief Adds initial input tokens before the request is submitted to an Engine.
   * @param tokens Span of token IDs to be added.
   *
   * This operation is legal only while the request is Unassigned. Use Continue()
   * to begin another turn after the current turn reaches TurnComplete.
   */
  void AddTokens(std::span<const int32_t> tokens);

  /**
   * @brief Queues another generation turn using resident model state.
   * @param tokens New input tokens to append after the completed turn.
   *
   * This operation is legal only from TurnComplete. It preserves unread generated
   * output, appends no input tokens to that output stream, and moves the request
   * back to Assigned (the queued state).
   */
  void Continue(std::span<const int32_t> tokens);

  /**
   * @brief Retrieves the next unseen token in the request.
   * @return The next unseen token ID.
   *
   * Newly generated tokens that have not been decoded by the calling application.
   * Applications looking to stream decode should call this method to get the next token
   *
   * while request.HasUnseenTokens():
   *     token = request.UnseenToken();
   *
   * Once an unseen token is seen, it is marked as seen and will not show up in
   * subsequent calls to this method.
   */
  int32_t UnseenToken();

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
   *         sequence (CommitStep, CompleteGeneration, Continue, or Assign). Copy it if it must
   *         outlive those.
   *
   * Same tokens as UnprocessedTokens(), but readable without copying them back from the device.
   * Building the next step's input ids is the hot path for this, and a device readback there costs
   * one full stream synchronization per request per step.
   */
  std::span<const int32_t> UnprocessedTokensCpu() const;

  /**
   * @brief Checks if there are any unseen tokens in the request.
   * @return True if there are unseen tokens, false otherwise.
   */
  bool HasUnseenTokens() const;

  /**
   * @brief Launches the generation of the next token based on the provided logits.
   * @param logits DeviceSpan containing logits for token generation.
   *
   * The work is only launched here. CompleteGeneration() must be called afterwards to pick up the
   * results and to update the request status. Splitting the two lets the engine launch every
   * scheduled request's token selection before it synchronizes with the device once.
   */
  void GenerateNextTokens(DeviceSpan<float> logits);

  /**
   * @brief Proposes speculative draft tokens for this request's next step.
   * @param tokens Draft continuation of the sequence, in order. An empty span clears the proposal.
   *
   * The next decode step then runs 1 + tokens.size() rows, verifies each draft against the target
   * model's own prediction, and keeps the accepted prefix. Greedy requests compare argmax tokens;
   * sampled requests draw from each target row's bounded top-k/top-p distribution.
   *
   * The proposal applies to the next step only. A committed step consumes it even when it could not
   * verify it (a prefill chunk, for one); a rolled back step leaves it pending.
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
  bool IsStopToken(int32_t token) const;
  void RewindDraftsForTransaction(size_t accepted_count);
  void RecordSampledDraftAcceptance(size_t accepted_count);

  void ValidateEngineCompatibility() const;
  void SaveStateForTransaction();
  void SaveStateForExternalSamplingTransaction();
  RequestStepResult ApplyLogitsForTransaction(DeviceSpan<float> logits);
  void PrepareGenerationForTransaction(DeviceSpan<float> logits);
  RequestStepResult StageGenerationForTransaction();
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
  void CompleteGeneration();

  /**
   * @brief Checks if the current generation turn reached a stopping condition.
   * @return True in TurnComplete; the request may still be continued or closed.
   */
  bool IsTurnComplete() const;

  RequestStatus Status() const noexcept { return status_; }

  /**
   * @brief Removes the request from its engine and moves it to terminal Closed.
   */
  void Remove();

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
  void PrepareGeneration(DeviceSpan<float> logits);

  /**
   * @brief Launches the per-sequence tail after a batched sampler has filled the bound slot.
   */
  void OnNextTokensSampled();

  /**
   * @brief Returns this request's persistent random state for the given batched sampler.
   */
  BatchedSamplerState& SamplingState(BatchedSampler& sampler);

  /**
   * @brief Retrieves the generator parameters associated with this request.
   * @return Shared pointer to GeneratorParams.
   */
  std::shared_ptr<GeneratorParams> Params();

  /**
   * @brief Sets the opaque data for user-defined purposes.
   * @param data Pointer to the opaque data.
   *
   * This data can be used by the application to store additional information
   * that may be useful for the application logic when new tokens are generated.
   * For example, the application could store a pointer to a user-defined structure
   * that contains the state of the application related to this request.
   * The data can be retrieved later using GetOpaqueData(). The stored data is not
   * used by the request or the engine and is solely for the application's use.
   * It is the application's responsibility to manage the lifetime of this data.
   */
  void SetOpaqueData(void* data);

  /**
   * @brief Gets the opaque data set by the user.
   * @return Pointer to the opaque data provided by the application.
   */
  void* GetOpaqueData();

 private:
  // The search sequence is partitioned at processed_sequence_length_: tokens before it already
  // have KV entries, and UnprocessedTokens() returns the scheduled prefix of [processed, current).
  // seen_sequence_length_ is the high-water sequence index of generated output consumed by the
  // application. Continuation input creates gaps, so it is not an unseen-token count.
  std::vector<int32_t> prefill_input_ids_;
  // Host-side mirror of the full sequence (prompt + generated tokens). Kept in step with the
  // search's device sequence so that streaming and input-id preparation never read it back.
  std::vector<int32_t> tokens_host_;
  std::vector<size_t> unseen_token_indices_;
  size_t next_unseen_token_index_{};
  int64_t seen_sequence_length_{};
  friend struct Engine;
  friend struct ExternalRefCounted<Request>;
  friend struct ScheduledRequests;
  friend struct StaticBatchScheduler;

  void CompleteClose();
  void OnFirstExternalReference() noexcept;
  void OnLastExternalReference() noexcept;
  bool IsExternallyAbandoned() const noexcept;
  // Runs before a scheduled step can execute. It keeps only useful consumed-prefix storage and
  // reserves every unseen-index append that the step can perform, so CommitStep stays noexcept.
  void PrepareForStep(size_t max_generated_token_indices);
  // Drops whatever the step in flight staged past the committed sequence, leaving the host mirror
  // exactly as long as the search after its own transaction rewind.
  void DiscardStagedDrafts() noexcept;
  // The Engine-hosted MTP head mirrors target-committed tokens in a scheduler-private request. It
  // never samples from this request; these helpers append decided tokens and advance its processed
  // boundary only after the auxiliary cache transaction commits.
  void AppendTokensForAuxiliaryDecoder(std::span<const int32_t> tokens);
  void AppendTokensForAuxiliaryDecoder(DeviceSpan<int32_t> tokens);
  void RewindAuxiliaryDecoderTo(size_t sequence_length);
  void CommitAuxiliaryDecoderStep() noexcept;

  int64_t processed_sequence_length_{};
  // Sequence length the application's tokens reach up to. Everything below it is prompt, so the
  // request is still prefilling while processed_sequence_length_ has not caught up with it.
  int64_t prompt_sequence_length_{};
  size_t scheduled_token_count_{};
  // Drafts proposed for the next step, the ones the step in flight staged onto the sequence, and
  // the leading part of those the target model accepted.
  std::vector<int32_t> draft_tokens_;
  size_t staged_draft_count_{};
  size_t accepted_draft_count_{};
  std::shared_ptr<GeneratorParams> params_;
  std::mt19937 rng_;
  std::mt19937 transaction_rng_;
  int64_t transaction_processed_sequence_length_{};
  size_t transaction_tokens_host_size_{};
  std::unique_ptr<Search> search_;
  std::unique_ptr<BatchedSamplerState> batched_sampler_state_;
  std::weak_ptr<Engine> engine_;
  std::atomic<bool> externally_abandoned_{false};

  void ApplyLogitsProcessors(DeviceSpan<float> logits);
  void SelectNextToken();
  RequestStepResult StageGeneration();
  // Sequence length recorded by PrepareGenerationForTransaction, so both the per-request and the
  // batched sampler path decide "did the sampler append a token" against the same boundary.
  int64_t sequence_length_before_sampling_{};

  void* opaque_data_{nullptr};  // Opaque data for user-defined purposes, can be set and retrieved by the application
};

}  // namespace Generators
