// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../generators.h"

/**
 * @file request.h
 * @brief Defines the request class that manages the state for each incoming user request.
 *        It handles the lifecycle of a request, from creation to completion.
 */

namespace Generators {

enum class RequestStatus {
  Unassigned,  // A request has been created but has not been added to the engine yet.
               // This is the state of a request when it is first created.
  Assigned,    // The request has been added to the engine and is waiting to be scheduled.
  InProgress,  // The request has been scheduled and is currently being processed.
  Completed,   // The request has been completed successfully.
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
   * @brief Checks if there are any unseen tokens in the request.
   * @return True if there are unseen tokens, false otherwise.
   */
  bool HasUnseenTokens() const;

  /**
   * @brief Generates the next set of tokens based on the provided logits.
   * @param logits DeviceSpan containing logits for token generation.
   */
  void GenerateNextTokens(DeviceSpan<float> logits);

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
   * @return True if the request is in prefill mode, false otherwise.
   */
  bool IsPrefill() const;

  /**
   * @brief Gets the current sequence length of the request.
   * @return The current sequence length.
   */
  int64_t CurrentSequenceLength() const;

  RequestStatus status_{RequestStatus::Unassigned};

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
  std::vector<int32_t> prefill_input_ids_;
  int64_t seen_sequence_length_{};
  int64_t processed_sequence_length_{};
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<Search> search_;
  std::weak_ptr<Engine> engine_;
  bool is_prefill_{true};

  void* opaque_data_{nullptr};  // Opaque data for user-defined purposes, can be set and retrieved by the application
};

}  // namespace Generators
