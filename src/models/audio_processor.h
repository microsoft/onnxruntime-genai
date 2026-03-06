// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// AudioProcessor - Streaming mel spectrogram extraction for Nemotron ASR models.
// Converts raw PCM audio chunks into mel features ready for the encoder.
#pragma once

#include "nemo_mel_spectrogram.h"
#include "nemotron_speech.h"

namespace Generators {

/// Stateful audio processor that converts raw PCM audio into mel spectrogram
/// tensors suitable for feeding into the Nemotron encoder via Generator.set_inputs().
///
/// Handles:
///   - Audio buffering (accumulates until a full chunk is available)
///   - Mel spectrogram extraction (NeMo-compatible)
///   - Pre-encode cache management (ring buffer of previous frames)
///
/// Usage:
///   auto processor = CreateAudioProcessor(*model);
///   auto mel = processor->Process(audio_data, num_samples);
///   if (mel) { generator.set_model_input("audio_features", mel); }
///
struct AudioProcessor : LeakChecked<AudioProcessor> {
  explicit AudioProcessor(Model& model);
  ~AudioProcessor();

  /// Feed raw PCM audio (mono, float32, model sample rate).
  /// Returns a mel tensor [1, total_frames, num_mels] when a full chunk is ready,
  /// or nullptr if more audio is needed.
  /// The tensor includes pre-encode cache frames prepended.
  std::unique_ptr<OrtValue> Process(const float* audio_data, size_t num_samples);

  /// Flush remaining buffered audio (pads to full chunk with silence).
  /// Returns final mel tensor, or nullptr if buffer is empty.
  std::unique_ptr<OrtValue> Flush();

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

std::unique_ptr<AudioProcessor> CreateAudioProcessor(Model& model);

}  // namespace Generators
