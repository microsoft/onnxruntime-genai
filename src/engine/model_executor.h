// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include "../models/decoder_only.h"
#include "decoders/decoder.h"
#include "scheduled_requests.h"
#include "cache_manager.h"
#include "execution_context.h"
#include "step_plan.h"

namespace Generators {

class ModelExecutionError : public std::runtime_error {
 public:
  ModelExecutionError(ExecutionFailureKind failure_kind, std::string message)
      : std::runtime_error{std::move(message)},
        failure_kind_{failure_kind} {}

  ExecutionFailureKind FailureKind() const { return failure_kind_; }

 private:
  ExecutionFailureKind failure_kind_;
};

/**
 * @struct ModelExecutor
 * @brief Runs the model for a scheduled batch and attaches the resulting decoder state so the batch
 *        can sample its next tokens.
 *
 * ModelExecutor is the abstract execution boundary the Engine drives each step. Production assembles
 * the real decoder-backed executor via Create; supplying an alternative implementation (for example
 * one that returns scripted outcomes or injects an execution failure) lets the Engine's execution
 * path be exercised without running a real model. Only Decode is part of the boundary because
 * Engine::Step drives decoding; encoding stays on the concrete executor until it becomes part of the
 * Engine execution path.
 */
struct ModelExecutor {
  static std::unique_ptr<ModelExecutor> Create(std::shared_ptr<Model> model,
                                               std::shared_ptr<CacheManager> cache_manager);

  void Decode(ScheduledRequests& scheduled_requests) {
    auto& context = scheduled_requests.CreateExecutionContext();
    Decode(scheduled_requests, context);
  }

  virtual void Decode(ScheduledRequests& scheduled_requests,
                      ExecutionContext& context) = 0;

  virtual ~ModelExecutor() = default;
};

/**
 * @struct DecoderModelExecutor
 * @brief The production ModelExecutor: decodes a scheduled batch by running the model's ONNX graph
 *        through a decoder for the model type.
 */
struct DecoderModelExecutor : ModelExecutor {
  DecoderModelExecutor(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager);
  DecoderModelExecutor(std::shared_ptr<Model> model,
                       std::shared_ptr<CacheManager> cache_manager,
                       std::unique_ptr<Decoder> decoder);

  void Encode(ScheduledRequests& scheduled_requests);

  void Decode(ScheduledRequests& scheduled_requests,
              ExecutionContext& context) override;

 private:
  std::shared_ptr<Model> model_;
  std::shared_ptr<CacheManager> cache_manager_;
  std::unique_ptr<Decoder> decoder_;
};

}  // namespace Generators
