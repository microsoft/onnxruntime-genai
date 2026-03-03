// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// BatchASR - abstract interface for offline (non-streaming) speech recognition.
#pragma once

#include "models/model.h"

namespace Generators {

/// Abstract base class for batch (offline) ASR.
///
/// Unlike StreamingASR which processes audio chunk-by-chunk for real-time use,
/// BatchASR takes a complete audio buffer and returns the full transcript at once.
/// This is optimized for offline transcription of recorded audio files.
///
/// Usage:
///   auto model = CreateModel(env, "path/to/model");
///   auto asr = CreateBatchASR(*model);
///   std::string text = asr->Transcribe(audio_data, num_samples);
///
struct BatchASR : LeakChecked<BatchASR> {
  virtual ~BatchASR() = default;

  /// Transcribe a complete audio buffer (mono, float32, sample rate depends on model).
  /// Returns the full transcript for the given audio.
  /// Each call is independent - no state is carried between calls.
  virtual std::string Transcribe(const float* audio_data, size_t num_samples) = 0;
};

std::unique_ptr<BatchASR> CreateBatchASR(Model& model);

}  // namespace Generators
