// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// StreamingASR — high-level streaming speech recognition API for RNNT models
// (Nemotron Speech Streaming, etc.)
#pragma once

#include "nemo_mel_spectrogram.h"  // from onnxruntime-extensions/shared/api
#include "models/model.h"
#include "models/nemotron_speech.h"

namespace Generators {

/// High-level streaming ASR interface.
/// Manages encoder cache and RNNT decoder state across audio chunks.
///
/// Usage:
///   auto model = CreateModel(env, "path/to/model");
///   auto asr = std::make_unique<StreamingASR>(*model);
///   std::string text = asr->TranscribeChunk(audio_data, num_samples);
///   std::string full = asr->GetTranscript();
///   asr->Reset();
///
struct StreamingASR : LeakChecked<StreamingASR> {
  StreamingASR(Model& model);
  ~StreamingASR();

  /// Feed a chunk of raw PCM audio (mono, 16kHz, float32).
  /// Each chunk is converted to mel, prepended with a 9-frame pre-encode cache
  /// (last 9 mel frames from the previous chunk), then fed to the encoder.
  /// This matches NeMo's CacheAwareStreamingAudioBuffer behavior.
  /// Returns newly transcribed text from this call.
  std::string TranscribeChunk(const float* audio_data, size_t num_samples);

  /// Flush remaining buffered audio (call after last TranscribeChunk).
  /// Processes any pending/buffered audio with silence padding.
  /// Returns final transcribed text.
  std::string Flush();

  /// Get the full transcript accumulated so far.
  const std::string& GetTranscript() const { return full_transcript_; }

  /// Reset all streaming state for a new utterance.
  void Reset();

 private:
  Model& model_;
  NemotronCacheConfig cache_config_;

  // Encoder ONNX session (borrowed from model)
  OrtSession* encoder_session_{};
  OrtSession* decoder_session_{};
  OrtSession* joiner_session_{};

  // Streaming state
  NemotronEncoderCache encoder_cache_;
  NemotronDecoderState decoder_state_;
  std::string full_transcript_;

  // Vocabulary
  std::vector<std::string> vocab_;
  bool vocab_loaded_{false};

  // Log-mel feature extraction (delegated to standalone nemo_mel::NemoStreamingMelExtractor)
  // Config values: Nemotron Speech defaults (128 mels, 512 FFT, 160 hop, 400 win, 16kHz, 0.97 preemph)
  nemo_mel::NemoStreamingMelExtractor mel_extractor_{
      nemo_mel::NemoMelConfig{128, 512, 160, 400, 16000, 0.97f, 5.96046448e-08f}};

  // Mel pre-encode cache: last pre_encode_cache_size mel frames from previous chunk.
  // Prepended to the current chunk's mel before feeding the encoder.
  // Size: kNumMels * pre_encode_cache_size floats (row-major: [kNumMels, cache_size]).
  std::vector<float> mel_pre_encode_cache_;
  bool is_first_chunk_{true};

  // Audio accumulation buffer for incoming PCM samples
  std::vector<float> audio_buffer_;

  static constexpr int kNumMels = 128;

// Debug: chunk counter for mel dump files
    int chunk_index_{0};

    void LoadVocab();
    std::string ProcessMelChunk(const std::vector<float>& mel_data, int num_frames);
    std::string RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len);

    // Save a float tensor as .npy file (row-major)
    static void SaveNpy(const std::string& path, const float* data,
                        const std::vector<int64_t>& shape);
};

std::unique_ptr<StreamingASR> CreateStreamingASR(Model& model);

}  // namespace Generators
