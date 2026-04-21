// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// StreamingASRState: abstract interface for streaming ASR model states
// (Nemotron RNNT, Moonshine Streaming, etc.)
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace Generators {

/// Abstract interface for streaming ASR model states that bypass the standard
/// search/logits pipeline. Both NemotronSpeechState and MoonshineStreamingState
/// implement this interface so the Generator can drive them uniformly.
struct StreamingASRState {
  virtual ~StreamingASRState() = default;

  /// Advance by one token. The state runs the appropriate ONNX sessions
  /// and emits 0 or 1 token into its internal buffer.
  virtual std::span<const int32_t> StepToken() = 0;

  /// Returns the tokens produced by the last StepToken() call.
  virtual std::span<const int32_t> GetStepTokens() const = 0;

  /// Returns true when all tokens for the current chunk have been emitted.
  virtual bool IsChunkDone() const = 0;

  /// Total number of tokens emitted across all chunks.
  virtual size_t TokenCount() const = 0;
};

}  // namespace Generators
