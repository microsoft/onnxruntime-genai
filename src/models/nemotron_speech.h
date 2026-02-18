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
};

}  // namespace Generators
