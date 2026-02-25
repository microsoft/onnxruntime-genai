// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// ParakeetStreamingASR — streaming ASR for Parakeet FastConformer + TDT models.
// Sliding window approach: every interval, encode last MAX_WINDOW of audio,
// decode with TDT greedy search, commit stable tokens based on timestamps.
#pragma once

#include "streaming_asr.h"
#include "nemo_mel.h"
#include "models/parakeet_speech.h"

namespace Generators {

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

  OrtSession* encoder_session_{};
  OrtSession* decoder_session_{};
  OrtSession* joiner_session_{};

  // Decoder integer input dtype: int32 for sherpa int8 models, int64 for FP32 NeMo exports
  ONNXTensorElementDataType decoder_int_dtype_{ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32};

  // Joiner layout: true if joiner expects [B, dim, T] (sherpa int8), false for [B, T, dim] (FP32)
  bool joiner_channel_first_{false};

  std::string full_transcript_;

  // Vocabulary
  std::vector<std::string> vocab_;
  bool vocab_loaded_{false};

  // Log-mel feature extraction via nemo_mel (NeMo-compatible, no kaldi dependency)

  // Audio sliding window buffer (last MAX_WINDOW seconds)
  std::vector<float> audio_buffer_;
  static constexpr float kMaxWindowSec = 8.0f;  // look-back window for encoder
  static constexpr float kStableDelaySec = 2.0f;
  static constexpr float kFrameSec = 0.08f; // encoder frame = 80ms

  // Committed tokens with timestamps
  struct TimestampedToken {
    int token_id;
    float abs_time;
  };
  std::vector<TimestampedToken> committed_tokens_;
  float total_audio_sec_{0.0f};

  int chunk_index_{0};

  void LoadVocab();

  // Run full encode + TDT decode on a segment, return tokens with abs timestamps
  std::vector<TimestampedToken> EncodeAndDecodeTDT(
      const float* audio, size_t num_samples, float window_start_sec);

  // Per-feature normalize mel in-place: [num_mels, num_frames]
  static void NormalizePerFeature(float* data, int num_mels, int num_frames);
};

}  // namespace Generators
