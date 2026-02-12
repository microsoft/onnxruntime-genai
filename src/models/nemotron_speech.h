// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Nemotron Speech Streaming ASR model support.
// Cache-aware streaming encoder + RNNT decoder_joint for real-time transcription.
#pragma once

#include "model.h"
#include "audio_features.h"

namespace Generators {

/// Configuration for Nemotron streaming encoder cache dimensions.
struct NemotronCacheConfig {
  int num_encoder_layers{24};
  int hidden_dim{1024};
  int left_context{70};   // Frame look-back for cache_last_channel
  int conv_context{8};    // Convolution context for cache_last_time
  int decoder_lstm_dim{640};
  int decoder_lstm_layers{2};
  int vocab_size{1024};
  int blank_id{1024};      // CTC blank / RNNT blank token
  int chunk_frames{56};    // Number of mel frames per chunk (560ms @ 16kHz with 10ms hop)
  int sample_rate{16000};
  int chunk_samples{8960}; // 560ms * 16000 = 8960 samples per chunk

  // Overlap-and-drop streaming config (O8b strategy).
  // overlap_mel_frames: consecutive encoder windows share this many mel frames.
  // drop_last_encoder_frames: discard last N encoder frames per chunk (boundary artifacts).
  // stride_samples = chunk_samples - overlap_mel_frames * hop_length.
  int overlap_mel_frames{8};         // 8 mel frames = 1280 samples overlap
  int drop_last_encoder_frames{1};   // Drop last 1 encoder frame per chunk
  static constexpr int kHopLength = 160;
  int stride_samples() const { return chunk_samples - overlap_mel_frames * kHopLength; }
};

/// Holds the rolling encoder cache state between streaming chunks.
struct NemotronEncoderCache {
  // cache_last_channel: [num_layers, 1, left_context, hidden_dim]
  std::unique_ptr<OrtValue> cache_last_channel;
  // cache_last_time: [num_layers, 1, hidden_dim, conv_context]
  std::unique_ptr<OrtValue> cache_last_time;
  // cache_last_channel_len: [1]
  std::unique_ptr<OrtValue> cache_last_channel_len;

  void Initialize(const NemotronCacheConfig& cfg, OrtAllocator& allocator);
  void Reset(const NemotronCacheConfig& cfg, OrtAllocator& allocator);
};

/// Holds the RNNT decoder LSTM hidden states between decoding steps.
struct NemotronDecoderState {
  // input_states_1 / input_states_2: [lstm_layers, 1, lstm_dim]
  std::unique_ptr<OrtValue> state_1;
  std::unique_ptr<OrtValue> state_2;
  int last_token{0};  // Last emitted non-blank token (for autoregressive feedback)

  void Initialize(const NemotronCacheConfig& cfg, OrtAllocator& allocator);
  void Reset(const NemotronCacheConfig& cfg, OrtAllocator& allocator);
};

// ─── Model ──────────────────────────────────────────────────────────────────

struct NemotronSpeechModel : Model {
  NemotronSpeechModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  // Three ONNX sessions: encoder, decoder (prediction network), joiner
  std::unique_ptr<OrtSession> session_encoder_;
  std::unique_ptr<OrtSession> session_decoder_;
  std::unique_ptr<OrtSession> session_joiner_;

  std::unique_ptr<OrtSessionOptions> encoder_session_options_;
  std::unique_ptr<OrtSessionOptions> decoder_session_options_;
  std::unique_ptr<OrtSessionOptions> joiner_session_options_;

  NemotronCacheConfig cache_config_;
};

// ─── State (streaming) ─────────────────────────────────────────────────────

struct NemotronSpeechState : State {
  NemotronSpeechState(const NemotronSpeechModel& model, const GeneratorParams& params,
                      DeviceSpan<int32_t> sequence_lengths);
  NemotronSpeechState(const NemotronSpeechState&) = delete;
  NemotronSpeechState& operator=(const NemotronSpeechState&) = delete;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  /// Run one streaming step: encode a chunk, then greedily decode with RNNT.
  /// Returns an empty DeviceSpan (transcription is retrieved via GetTranscript).
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  OrtValue* GetOutput(const char* name) override;

  /// Get the full transcript accumulated so far.
  const std::string& GetTranscript() const { return full_transcript_; }

  /// Reset streaming state for a new utterance.
  void ResetStreaming();

  /// Feed raw PCM audio (mono, 16kHz, float32) for streaming transcription.
  /// Returns the new text produced from this chunk.
  std::string TranscribeChunk(const float* audio_data, size_t num_samples);

 private:
  const NemotronSpeechModel& model_;

  // Encoder cache (persists across chunks)
  NemotronEncoderCache encoder_cache_;

  // Decoder state (persists across chunks)
  NemotronDecoderState decoder_state_;

  // Audio features (mel spectrogram) for current chunk
  std::unique_ptr<OrtValue> audio_features_;

  // Accumulated transcript
  std::string full_transcript_;
  std::string chunk_transcript_;

  // Tokenizer for RNNT decoding (sentencepiece-based)
  std::vector<std::string> vocab_;
  bool vocab_loaded_{false};

  // Internal helpers
  void RunEncoder(const float* audio_data, size_t num_samples);
  std::string RunRNNTDecoder(OrtValue* encoder_output, int64_t encoded_len);
  void LoadVocab();

  // Tensor holding last encoder output for GetOutput
  std::unique_ptr<OrtValue> last_encoder_output_;
  std::unique_ptr<OrtValue> last_encoded_len_;

  // Mel feature extractor (log-mel spectrogram)
  // We use ORT extensions if available, otherwise a simple built-in extraction
  // For nemotron: 80-dim mel, 10ms hop, 25ms window
  static constexpr int kNumMels = 128;
  static constexpr int kHopLength = 160;    // 10ms * 16kHz
  static constexpr int kWinLength = 400;    // 25ms * 16kHz
  static constexpr int kFFTSize = 512;
};

}  // namespace Generators
