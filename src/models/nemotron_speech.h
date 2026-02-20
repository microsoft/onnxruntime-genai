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
/// Populated from Config::Model at model load time via PopulateFromConfig().
struct NemotronCacheConfig {
  // All values populated from genai_config.json via PopulateFromConfig().
  // Encoder dimensions (from encoder.hidden_size / num_hidden_layers)
  int num_encoder_layers{};
  int hidden_dim{};
  int left_context{};
  int conv_context{};

  // Decoder LSTM dimensions (from decoder.hidden_size / num_hidden_layers)
  int decoder_lstm_dim{};
  int decoder_lstm_layers{};

  // Vocabulary
  int vocab_size{};
  int blank_id{};

  // Streaming chunk config
  int chunk_frames{};
  int sample_rate{16000};
  int chunk_samples{};
  int subsampling_factor{};
  int max_symbols_per_step{};

  // Mel spectrogram parameters (from speech.*)
  int num_mels{};
  int fft_size{};
  int hop_length{};
  int win_length{};
  float preemph{};
  float log_eps{};

  // Pre-encode cache
  int pre_encode_cache_size{};

  // Encoder I/O names (populated from genai_config.json)
  std::string enc_in_audio;
  std::string enc_in_length;
  std::string enc_in_cache_channel;
  std::string enc_in_cache_time;
  std::string enc_in_cache_channel_len;
  std::string enc_out_encoded;
  std::string enc_out_length;
  std::string enc_out_cache_channel;
  std::string enc_out_cache_time;
  std::string enc_out_cache_channel_len;

  // Decoder (prediction network) I/O names
  std::string dec_in_targets;
  std::string dec_in_target_length;
  std::string dec_in_states_1;
  std::string dec_in_states_2;
  std::string dec_out_outputs;
  std::string dec_out_prednet_lengths;
  std::string dec_out_states_1;
  std::string dec_out_states_2;

  // Joiner I/O names
  std::string join_in_encoder;
  std::string join_in_decoder;
  std::string join_out_logits;

  /// Populate from a Config object (reads encoder/decoder/joiner/speech sections).
  void PopulateFromConfig(const Config& config);
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

}  // namespace Generators
