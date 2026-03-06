// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../streaming_asr.h"
#include "nemo_mel_spectrogram.h"
#include "nemotron_speech.h"

namespace Generators {

/// Streaming ASR implementation for NeMo cache-aware FastConformer encoder
/// with RNNT greedy decoding.
struct NemoStreamingASR : StreamingASR {
  explicit NemoStreamingASR(Model& model);
  ~NemoStreamingASR() override;

  std::string TranscribeChunk(const float* audio_data, size_t num_samples) override;
  std::string Flush() override;
  const std::string& GetTranscript() const override { return full_transcript_; }
  void Reset() override;

 private:
  Model& model_;
  NemotronCacheConfig cache_config_;

  std::unique_ptr<OrtSession> encoder_session_;
  std::unique_ptr<OrtSession> decoder_session_;
  std::unique_ptr<OrtSession> joiner_session_;
  std::unique_ptr<OrtRunOptions> run_options_;

  // Streaming state
  NemotronEncoderCache encoder_cache_;
  NemotronDecoderState decoder_state_;
  std::string full_transcript_;

  // Vocabulary
  std::vector<std::string> vocab_;
  bool vocab_loaded_{false};

  // Log-mel feature extraction
  nemo_mel::NemoStreamingMelExtractor mel_extractor_;

  // Mel pre-encode cache: ring buffer of last pre_encode_cache_size frames.
  // Stored time-major [cache_size, num_mels] with a circular write position.
  // Prepended to the current chunk's mel before feeding the encoder.
  std::vector<float> mel_pre_encode_cache_;
  int cache_pos_{0};  // next write position in ring buffer [0, cache_size)

  // Audio accumulation buffer for incoming PCM samples
  std::vector<float> audio_buffer_;

  void LoadVocab();
  std::string TranscribeMelChunk(const std::vector<float>& mel_data, int num_frames);
  std::string RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len);
};

}  // namespace Generators
