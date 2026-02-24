// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// ParakeetStreamingASR — streaming ASR for Parakeet FastConformer + TDT models.
#pragma once

#include "streaming_asr.h"
#include "models/parakeet_speech.h"

namespace Generators {

/// Streaming ASR implementation for Parakeet FastConformer encoder (non-cache-aware)
/// with TDT (Token-and-Duration Transducer) greedy decoding.
///
/// Unlike RNNT which always advances by 1 encoder frame on blank, TDT predicts
/// a duration that controls how many encoder frames to advance after emitting a token.
///
/// Streaming algorithm (matches NeMo's speech_to_text_streaming_infer_rnnt.py):
///   - Buffer = [left_context | chunk | right_context]
///   - Encoder runs on full buffer each iteration
///   - Left context encoder frames are STRIPPED — only chunk frames are decoded
///   - Decoder LSTM state carries forward between chunks
///   - Each token is decoded exactly once (TDT durations handle frame advancement)
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

  // Streaming decoder state (LSTM h/c + decoder_output maintained across chunks)
  ParakeetDecoderState decoder_state_;
  bool decoder_initialized_{false};
  std::string full_transcript_;

  // Vocabulary
  std::vector<std::string> vocab_;
  bool vocab_loaded_{false};

  // Audio accumulation buffer — all audio received so far
  std::vector<float> all_audio_;

  // How many audio samples we have already processed (decoded) in previous chunks
  size_t processed_audio_samples_{0};

  int chunk_index_{0};

  void LoadVocab();
  void InitializeDecoderState();

  // Process a single streaming chunk: build [left|chunk|right] window,
  // compute mel, normalize, encode, decode chunk frames with TDT.
  std::string ProcessChunk(size_t chunk_start, size_t chunk_end, bool is_last);

  // TDT greedy decode on encoder frames [start_frame, end_frame).
  // Returns decoded text. Updates decoder_state_ in place.
  std::string RunTDTDecoder(OrtValue* encoder_output, int64_t encoded_len,
                            int64_t start_frame, int64_t end_frame);
};

}  // namespace Generators
