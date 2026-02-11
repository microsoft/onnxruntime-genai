// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// StreamingASR — high-level streaming speech recognition API for RNNT models
// (Nemotron Speech Streaming, etc.)
#pragma once

#include "models/model.h"
#include "models/nemotron_speech.h"
#include "models/nemotron_audio_processor.h"

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
  /// Returns newly transcribed text from this chunk.
  std::string TranscribeChunk(const float* audio_data, size_t num_samples);

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

  // Streaming state
  NemotronEncoderCache encoder_cache_;
  NemotronDecoderState decoder_state_;
  std::string full_transcript_;

  // Vocabulary
  std::vector<std::string> vocab_;
  bool vocab_loaded_{false};

  // Audio preprocessor — reads params from audio_processor_config.json.
  // Swap this for an ORT Extensions implementation when available.
  std::shared_ptr<NemotronAudioProcessor> audio_processor_;

  void LoadVocab();
  std::string RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len);
};

std::unique_ptr<StreamingASR> CreateStreamingASR(Model& model);

}  // namespace Generators
