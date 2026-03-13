// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Nemotron Speech Streaming ASR model support.
#pragma once

#include "model.h"
#include "audio_features.h"

namespace Generators {

struct NemotronCacheConfig {
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
  int sample_rate{};
  int chunk_samples{};
  int subsampling_factor{};
  int max_symbols_per_step{};

  // Mel spectrogram parameters
  int num_mels{};
  int fft_size{};
  int hop_length{};
  int win_length{};
  float preemph{};
  float log_eps{};

  // Pre-encode cache
  int pre_encode_cache_size{};

  // Encoder I/O names
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
  std::string dec_in_lstm_hidden;
  std::string dec_in_lstm_cell;
  std::string dec_out_outputs;
  std::string dec_out_prednet_lengths;
  std::string dec_out_lstm_hidden;
  std::string dec_out_lstm_cell;

  // Joiner I/O names
  std::string join_in_encoder;
  std::string join_in_decoder;
  std::string join_out_logits;

  void PopulateFromConfig(const Config& config);
};

/// Holds the rolling encoder cache state between streaming chunks.
struct NemotronEncoderCache {
  std::unique_ptr<OrtValue> cache_last_channel;
  std::unique_ptr<OrtValue> cache_last_time;
  std::unique_ptr<OrtValue> cache_last_channel_len;

  void Initialize(const NemotronCacheConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device);
  void Reset(const NemotronCacheConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device);
};

/// Holds the RNNT decoder LSTM hidden states between decoding steps.
struct NemotronDecoderState {
  std::unique_ptr<OrtValue> lstm_hidden_state;
  std::unique_ptr<OrtValue> lstm_cell_state;
  int last_token{0};  // Last emitted non-blank token (for autoregressive feedback)

  void Initialize(const NemotronCacheConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device);
  void Reset(const NemotronCacheConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device);
};

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

/// Sub-state for the streaming encoder.
struct NemotronEncoderSubState : State {
  NemotronEncoderSubState(const NemotronSpeechModel& model, const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  /// Set mel input and update registered input pointers.
  void SetMelInput(OrtValue* mel_tensor, int64_t total_mel_frames);

  /// Update registered input pointers after cache is modified.
  void UpdateCacheInputs();

 private:
  friend struct NemotronSpeechState;

  const NemotronSpeechModel& model_;
  NemotronEncoderCache cache_;
  std::unique_ptr<OrtValue> signal_length_;

  // Indices into inputs_/outputs_ vectors
  size_t mel_input_idx_{};
  size_t length_input_idx_{};
  size_t cache_channel_input_idx_{};
  size_t cache_time_input_idx_{};
  size_t cache_channel_len_input_idx_{};
};

/// Sub-state for the RNNT prediction network (decoder LSTM).
struct NemotronPredictionSubState : State {
  NemotronPredictionSubState(const NemotronSpeechModel& model, const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  /// Update LSTM state input pointers before each run.
  void UpdateInputs();

 private:
  friend struct NemotronSpeechState;

  const NemotronSpeechModel& model_;
  NemotronDecoderState lstm_state_;
  std::unique_ptr<OrtValue> targets_;
  std::unique_ptr<OrtValue> target_length_;

  size_t targets_input_idx_{};
  size_t target_length_input_idx_{};
  size_t lstm_hidden_input_idx_{};
  size_t lstm_cell_input_idx_{};
};

/// Sub-state for the joiner network.
struct NemotronJoinerSubState : State {
  NemotronJoinerSubState(const NemotronSpeechModel& model, const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  /// Update encoder/decoder frame input pointers before each run.
  void SetInputFrames(OrtValue* encoder_frame, OrtValue* decoder_frame);

 private:
  friend struct NemotronSpeechState;

  const NemotronSpeechModel& model_;

  size_t encoder_input_idx_{};
  size_t decoder_input_idx_{};
};

/// Orchestrator state for the full RNNT pipeline.
struct NemotronSpeechState : State {
  NemotronSpeechState(const NemotronSpeechModel& model, const GeneratorParams& params);
  ~NemotronSpeechState() override;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  std::span<const int32_t> StepToken();
  bool IsChunkDone() const { return chunk_done_; }
  std::span<const int32_t> GetStepTokens() const { return last_tokens_; }
  size_t TokenCount() const { return token_count_; }
  void ResetStreamingState();

  OrtValue* GetInput(const char* name) override;
  OrtValue* GetOutput(const char* name) override;

 private:
  const NemotronSpeechModel& nemotron_model_;
  NemotronCacheConfig cache_config_;

  std::unique_ptr<NemotronEncoderSubState> encoder_state_;
  std::unique_ptr<NemotronPredictionSubState> prediction_state_;
  std::unique_ptr<NemotronJoinerSubState> joiner_state_;

  // Current mel input
  std::shared_ptr<Tensor> current_mel_;

  // Encoder output persisted across StepToken calls
  std::unique_ptr<OrtValue> encoded_output_;
  int64_t encoded_len_{0};

  // Pre-allocated encoder frame for joiner input
  std::unique_ptr<OrtValue> encoder_frame_;

  // Decoder state machine
  int64_t time_step_{0};
  int symbol_step_{0};
  bool need_encoder_run_{false};
  bool chunk_done_{true};
  std::vector<int32_t> last_tokens_;
  size_t token_count_{};  // Total tokens emitted across all chunks

  void RunEncoder();
};

}  // namespace Generators
