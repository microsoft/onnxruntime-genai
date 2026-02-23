// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// ParakeetStreamingASR — streaming ASR for Parakeet FastConformer + TDT models.
#pragma once

#include "streaming_asr.h"
#include "nemo_mel_spectrogram.h"  // from onnxruntime-extensions/shared/api
#include "models/parakeet_speech.h"

namespace Generators {

/// Streaming ASR implementation for Parakeet FastConformer encoder (non-cache-aware)
/// with TDT (Token-and-Duration Transducer) greedy decoding.
///
/// Unlike RNNT which always advances by 1 encoder frame on blank, TDT predicts
/// a duration that controls how many encoder frames to advance after emitting a token.
///
/// Manages mel extraction, decoder LSTM state, and vocabulary lookup across audio chunks.
struct ParakeetStreamingASR : StreamingASR {
  explicit ParakeetStreamingASR(Model& model);
  ~ParakeetStreamingASR() override;

  std::string TranscribeChunk(const float* audio_data, size_t num_samples) override;
  std::string Flush() override;
  const std::string& GetTranscript() const override { return full_transcript_; }
  void Reset() override;

 private:
  Model& model_;
  ParakeetConfig config_;

  // ONNX sessions (borrowed from ParakeetSpeechModel)
  OrtSession* encoder_session_{};
  OrtSession* decoder_session_{};
  OrtSession* joiner_session_{};

  // Streaming decoder state (LSTM h/c maintained across chunks)
  ParakeetDecoderState decoder_state_;
  std::string full_transcript_;

  // Vocabulary
  std::vector<std::string> vocab_;
  bool vocab_loaded_{false};

  // Log-mel feature extraction (same as NeMo)
  nemo_mel::NemoStreamingMelExtractor mel_extractor_;

  // Audio accumulation buffer for incoming PCM samples
  std::vector<float> audio_buffer_;

  // Accumulated mel features across all chunks: [num_mels][frames...]
  // Stored per-mel-bin for easy normalization.
  std::vector<std::vector<float>> accumulated_mel_;
  int total_mel_frames_{0};

  // How many encoder output frames we have already decoded
  int64_t prev_decoded_frames_{0};

  int chunk_index_{0};

  void LoadVocab();
  std::string EncodeAndDecode();
  std::string RunTDTDecoder(OrtValue* encoder_output, int64_t encoded_len,
                            int64_t start_frame);
};

}  // namespace Generators
