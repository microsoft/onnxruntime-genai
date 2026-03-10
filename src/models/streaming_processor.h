// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// StreamingProcessor converts raw PCM audio chunks into mel features ready for the encoder.
#pragma once

#include "nemo_mel_spectrogram.h"
#include "nemotron_speech.h"

namespace Generators {

/// Stateful audio processor that converts raw PCM audio into mel spectrogram
/// tensors suitable for feeding into the Nemotron encoder via Generator.set_inputs().
struct StreamingProcessor : LeakChecked<StreamingProcessor> {
  explicit StreamingProcessor(Model& model);
  ~StreamingProcessor();

  /// Feed raw PCM audio (mono, float32, model sample rate).
  /// Returns a NamedTensors with the mel tensor when a full chunk is ready,
  /// or nullptr if more audio is needed.
  std::unique_ptr<NamedTensors> Process(const float* audio_data, size_t num_samples);

  /// Flush remaining buffered audio (pads to full chunk with silence).
  /// Returns final NamedTensors with mel, or nullptr if buffer is empty.
  std::unique_ptr<NamedTensors> Flush();

  /// Reset all streaming state for a new utterance.
  void Reset();

  /// Get the expected chunk size in samples.
  int GetChunkSamples() const { return cache_config_.chunk_samples; }

  /// Get the sample rate expected by this processor.
  int GetSampleRate() const { return cache_config_.sample_rate; }

 private:
  Model& model_;
  NemotronCacheConfig cache_config_;

  // Log-mel feature extraction
  nemo_mel::NemoStreamingMelExtractor mel_extractor_;

  // Mel pre-encode cache: ring buffer of last pre_encode_cache_size frames.
  // Stored time-major [cache_size, num_mels] with a circular write position.
  std::vector<float> mel_pre_encode_cache_;
  int cache_pos_{0};

  // Audio accumulation buffer for incoming PCM samples
  std::vector<float> audio_buffer_;

  /// Build mel tensor from one chunk of audio (exactly chunk_samples).
  /// Handles mel extraction, pre-encode cache prepend, and cache update.
  std::unique_ptr<OrtValue> BuildMelTensor(const float* audio_chunk, size_t chunk_samples);
};

std::unique_ptr<StreamingProcessor> CreateStreamingProcessor(Model& model);

}  // namespace Generators
