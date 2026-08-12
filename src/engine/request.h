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

struct RequestStepResult {
  int32_t token{};
  bool token_appended{};
  bool done{};
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
  Request(std::shared_ptr<GeneratorParams> params);

  /**
   * @brief Assigns this request to a specific engine for processing.
   * @param engine Shared pointer to the Engine to be used for processing this request.
   *
   * Once assigned, the request will finalize the prefill tokens and prepare for scheduling.
   */
  void Assign(std::shared_ptr<Engine> engine);

  /**
   * @brief Updates the status of the request to InProgress and prepares it for processing.
   */
  void Schedule();

  /**
   * @brief Adds a sequence of tokens to the request for processing.
   * @param tokens Span of token IDs to be added.
   */
  void AddTokens(std::span<const int32_t> tokens);

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
   *         sequence (CompleteGeneration, AddTokens or Assign). Copy it if it must outlive those.
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

  void ValidateEngineCompatibility() const;
  void SaveStateForTransaction();
  void SaveStateForExternalSamplingTransaction();
  RequestStepResult ApplyLogitsForTransaction(DeviceSpan<float> logits);
  void PrepareGenerationForTransaction(DeviceSpan<float> logits);
  RequestStepResult StageGenerationForTransaction(
      const RequestStepPlan& plan);
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
   * @brief Checks if the termination condition for the request has been met.
   * @return True if the request is done, false otherwise.
   */
  bool IsDone() const;

  /**
   * @brief Removes the request from being processed.
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
   * Called once per step, before anything reads UnprocessedTokens(). With a `search.chunk_size`
   * configured, a prompt longer than the chunk size is processed over several steps, which bounds
   * the number of tokens a single model run carries.
   */
  void ScheduleTokens();

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
  // Tokens of the current step, clamped to what is actually left to process.
  size_t ScheduledTokenCount() const;

  // The search sequence is partitioned at processed_sequence_length_: tokens before it already
  // have KV entries, and UnprocessedTokens() returns the scheduled prefix of [processed, current).
  // seen_sequence_length_ independently tracks tokens consumed by the application.
  std::vector<int32_t> prefill_input_ids_;
  // Host-side mirror of the full sequence (prompt + generated tokens). Kept in step with the
  // search's device sequence so that streaming and input-id preparation never read it back.
  std::vector<int32_t> tokens_host_;
  int64_t seen_sequence_length_{};
  int64_t processed_sequence_length_{};
  // Sequence length the application's tokens reach up to. Everything below it is prompt, so the
  // request is still prefilling while processed_sequence_length_ has not caught up with it.
  int64_t prompt_sequence_length_{};
  size_t scheduled_token_count_{};
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<Search> search_;
  std::unique_ptr<BatchedSamplerState> batched_sampler_state_;
  std::weak_ptr<Engine> engine_;

  void ApplyLogitsProcessors(DeviceSpan<float> logits);
  void SelectNextToken();
  RequestStepResult StageGeneration(int64_t sequence_length_before);

  void* opaque_data_{nullptr};  // Opaque data for user-defined purposes, can be set and retrieved by the application
};

}  // namespace Generators
