// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// StreamingASR - abstract interface for streaming speech recognition.
#pragma once

#include "models/model.h"

namespace Generators {

/// Abstract base class for streaming ASR.
///
///
/// Usage:
///   auto model = CreateModel(env, "path/to/model");
///   auto asr = CreateStreamingASR(*model);
///   std::string text = asr->TranscribeChunk(audio_data, num_samples);
///   std::string full = asr->GetTranscript();
///   asr->Reset();
///
struct StreamingASR : LeakChecked<StreamingASR> {
  virtual ~StreamingASR() = default;

  /// Feed a chunk of raw PCM audio (mono, float32, sample rate depends on model).
  /// Returns newly transcribed text from this call.
  virtual std::string TranscribeChunk(const float* audio_data, size_t num_samples) = 0;

  /// Flush remaining buffered audio (call after last TranscribeChunk).
  /// Returns final transcribed text.
  virtual std::string Flush() = 0;

  /// Get the full transcript accumulated so far.
  virtual const std::string& GetTranscript() const = 0;

  /// Reset all streaming state for a new utterance.
  virtual void Reset() = 0;
};

std::unique_ptr<StreamingASR> CreateStreamingASR(Model& model);
}  // namespace Generators
