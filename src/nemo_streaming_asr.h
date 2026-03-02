// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// NemoStreamingASR — streaming ASR for NeMo FastConformer + RNNT models.
#pragma once

#include "streaming_asr.h"
#include "nemo_mel_spectrogram.h"
#include "models/nemotron_speech.h"

namespace Generators {

/// Streaming ASR implementation for NeMo cache-aware FastConformer encoder
/// with RNNT (prediction network + joiner) greedy decoding.
///
/// Manages encoder cache, mel pre-encode cache, LSTM decoder state,
/// and vocabulary lookup across audio chunks.
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

  // Log-mel feature extraction
  nemo_mel::NemoStreamingMelExtractor mel_extractor_;

  // Mel pre-encode cache: last pre_encode_cache_size frames from previous chunk.
  // Prepended to the current chunk's mel before feeding the encoder.
  // Layout: [num_mels, cache_size] row-major.
  std::vector<float> mel_pre_encode_cache_;
  bool is_first_chunk_{true};

  // Audio accumulation buffer for incoming PCM samples
  std::vector<float> audio_buffer_;

  int chunk_index_{0};

  void LoadVocab();
  std::string TranscribeMelChunk(const std::vector<float>& mel_data, int num_frames);
  std::string RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len);
};

}  // namespace Generators
