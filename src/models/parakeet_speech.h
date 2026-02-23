// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Parakeet TDT Speech Model support.
// Non-cache-aware encoder + TDT (Token-and-Duration Transducer) decoder+joiner
// for real-time streaming transcription.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model.h"

namespace Generators {

/// Configuration for Parakeet TDT model.
/// Populated from Config::Model at model load time via PopulateFromConfig().
struct ParakeetConfig {
  // Encoder dimensions
  int num_encoder_layers{};
  int hidden_dim{};

  // Decoder LSTM dimensions
  int decoder_lstm_dim{};
  int decoder_lstm_layers{};

  // Vocabulary
  int vocab_size{};
  int blank_id{};

  // Streaming chunk config
  int sample_rate{16000};
  int chunk_samples{32000};  // Default 2 seconds for non-cache-aware encoder
  int subsampling_factor{8};
  int max_symbols_per_step{10};

  // Mel spectrogram parameters
  int num_mels{};
  int fft_size{};
  int hop_length{};
  int win_length{};
  float preemph{};
  float log_eps{};

  // TDT (Token-and-Duration Transducer) parameters
  std::vector<int> tdt_durations;      // e.g., {0, 1, 2, 3, 4}
  int tdt_num_extra_outputs{};         // Number of duration logit outputs (e.g., 5)

  // Encoder I/O names
  std::string enc_in_audio;
  std::string enc_in_length;
  std::string enc_out_encoded;
  std::string enc_out_length;

  // Decoder (prediction network) I/O names
  std::string dec_in_targets;
  std::string dec_in_target_length;
  std::string dec_in_states_1;       // h_in
  std::string dec_in_states_2;       // c_in
  std::string dec_out_outputs;       // decoder_output
  std::string dec_out_prednet_lengths;
  std::string dec_out_states_1;      // h_out
  std::string dec_out_states_2;      // c_out

  // Joiner I/O names
  std::string join_in_encoder;
  std::string join_in_decoder;
  std::string join_out_logits;

  /// Populate from a Config object (reads encoder/decoder/joiner/speech sections).
  void PopulateFromConfig(const Config& config);
};

/// Holds the TDT decoder LSTM hidden states between decoding steps.
struct ParakeetDecoderState {
  // h_in / c_in: [lstm_layers, 1, lstm_dim]
  std::unique_ptr<OrtValue> state_h;
  std::unique_ptr<OrtValue> state_c;
  int64_t last_token{0};  // Last emitted non-blank token (for autoregressive feedback)

  void Initialize(const ParakeetConfig& cfg, OrtAllocator& allocator);
  void Reset(const ParakeetConfig& cfg, OrtAllocator& allocator);
};

// ─── Model ──────────────────────────────────────────────────────────────────

struct ParakeetSpeechModel : Model {
  ParakeetSpeechModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  // Three ONNX sessions: encoder, decoder (prediction network), joiner
  std::unique_ptr<OrtSession> session_encoder_;
  std::unique_ptr<OrtSession> session_decoder_;
  std::unique_ptr<OrtSession> session_joiner_;

  std::unique_ptr<OrtSessionOptions> encoder_session_options_;
  std::unique_ptr<OrtSessionOptions> decoder_session_options_;
  std::unique_ptr<OrtSessionOptions> joiner_session_options_;

  ParakeetConfig parakeet_config_;
};

}  // namespace Generators
