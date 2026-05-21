// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Nemotron Speech Streaming ASR model support.
#pragma once

#include "model.h"
#include "audio_features.h"

namespace Generators {

struct NemotronConfig {
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
  float blank_penalty{};

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
  std::string enc_in_lang_id;
  std::string enc_out_encoded;
  std::string enc_out_length;
  std::string enc_out_cache_channel;
  std::string enc_out_cache_time;
  std::string enc_out_cache_channel_len;

  // Decoder (prediction network) I/O names
  std::string dec_in_targets;
  std::string dec_in_lstm_hidden;
  std::string dec_in_lstm_cell;
  std::string dec_out_outputs;
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

  void Initialize(const NemotronConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device);
  void Reset(const NemotronConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device);
};

/// Holds the RNNT decoder LSTM hidden states between decoding steps.
struct NemotronDecoderState {
  std::unique_ptr<OrtValue> lstm_hidden_state;
  std::unique_ptr<OrtValue> lstm_cell_state;
  int last_token{0};  // Last emitted non-blank token (for autoregressive feedback)

  void Initialize(const NemotronConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device);
  void Reset(const NemotronConfig& cfg, const SessionInfo& session_info, OrtAllocator& allocator, DeviceInterface& device);
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

  NemotronConfig nemotron_config_;
};

/// Sub-state for the streaming encoder.
struct NemotronEncoderSubState : State {
  NemotronEncoderSubState(const NemotronSpeechModel& model, const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  /// Set mel input and update registered input pointers.
  void SetMelInput(OrtValue* mel_tensor, int64_t total_mel_frames);

  /// After a Run, the new cache lives in the pre-allocated *_next_ buffers.
  /// Swap them with the input cache buffers so the next Run consumes them,
  /// then re-register both input and output pointers on the session.
  void RotateCaches();

  /// Re-register cache input pointers (called after RotateCaches or external resets).
  void UpdateCacheInputs();

  /// Re-register pre-allocated output pointers (called after RotateCaches).
  void UpdateOutputs();

  void SetLangId(int lang_id);

  bool HasLangIdInput() const { return has_lang_id_input_; }

  /// Accessors for pre-allocated outputs (filled in by ORT after Run).
  OrtValue* EncodedOutput() const { return encoded_out_.get(); }
  int64_t EncodedLength() const { return *output_length_->GetTensorData<int64_t>(); }

 private:
  friend struct NemotronSpeechState;

  const NemotronSpeechModel& model_;
  NemotronEncoderCache cache_;
  std::unique_ptr<OrtValue> signal_length_;
  std::unique_ptr<OrtValue> lang_id_tensor_;

  // Pre-allocated encoder outputs: encoded ([1,T,D] on device), output_length
  // ([1] on CPU), and the "next" cache tensors that will be swapped into the
  // input slots after each Run. Pre-allocating these avoids ORT falling back
  // to CPU placement when output slots are left null.
  std::unique_ptr<OrtValue> encoded_out_;
  std::unique_ptr<OrtValue> output_length_;
  std::unique_ptr<OrtValue> cache_last_channel_next_;
  std::unique_ptr<OrtValue> cache_last_time_next_;
  std::unique_ptr<OrtValue> cache_last_channel_len_next_;

  // Whether the encoder model has a "length" input
  bool has_length_input_{};
  // Whether the encoder model has a "lang_id" input (prompt-conditioned multilingual model)
  bool has_lang_id_input_{};

  // Indices into inputs_/outputs_ vectors
  size_t mel_input_idx_{};
  size_t length_input_idx_{};
  size_t cache_channel_input_idx_{};
  size_t cache_time_input_idx_{};
  size_t cache_channel_len_input_idx_{};
  size_t lang_id_input_idx_{};

  // Output slot indices (encoded, length, cache_channel_next, cache_time_next, cache_channel_len_next)
  size_t encoded_output_idx_{};
  size_t length_output_idx_{};
  size_t cache_channel_output_idx_{};
  size_t cache_time_output_idx_{};
  size_t cache_channel_len_output_idx_{};
};

/// Sub-state for the RNNT prediction network (decoder LSTM).
struct NemotronPredictionSubState : State {
  NemotronPredictionSubState(const NemotronSpeechModel& model, const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  /// Update LSTM state input pointers before each run.
  void UpdateInputs();

  /// Re-register pre-allocated output pointers (called after RotateLstmState).
  void UpdateOutputs();

  /// Swap the pre-allocated *_next_ LSTM state buffers into the input slots
  /// (called after Run when the emitted token was non-blank so the new LSTM
  /// state should persist).
  void RotateLstmState();

  OrtValue* DecoderOutput() const { return dec_output_.get(); }

 private:
  friend struct NemotronSpeechState;

  const NemotronSpeechModel& model_;
  NemotronDecoderState lstm_state_;
  std::unique_ptr<OrtValue> targets_;

  // Pre-allocated prediction-net outputs. `dec_output_` holds the decoder's
  // [1, lstm_dim] hidden vector; the *_next_ buffers form a ping-pong pair
  // with `lstm_state_` so ORT writes the new LSTM state directly to device.
  std::unique_ptr<OrtValue> dec_output_;
  std::unique_ptr<OrtValue> lstm_hidden_next_;
  std::unique_ptr<OrtValue> lstm_cell_next_;

  size_t targets_input_idx_{};
  size_t lstm_hidden_input_idx_{};
  size_t lstm_cell_input_idx_{};

  size_t dec_output_idx_{};
  size_t lstm_hidden_output_idx_{};
  size_t lstm_cell_output_idx_{};
};

/// Sub-state for the joiner network.
struct NemotronJoinerSubState : State {
  NemotronJoinerSubState(const NemotronSpeechModel& model, const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  /// Update encoder/decoder frame input pointers before each run.
  void SetInputFrames(OrtValue* encoder_frame, OrtValue* decoder_frame);

  OrtValue* LogitsOutput() const { return logits_.get(); }

 private:
  friend struct NemotronSpeechState;

  const NemotronSpeechModel& model_;

  // Pre-allocated logits output on the inference device so ORT writes there
  // directly (and we know the source device for the CPU copy in argmax).
  std::unique_ptr<OrtValue> logits_;

  size_t encoder_input_idx_{};
  size_t decoder_input_idx_{};
  size_t logits_output_idx_{};
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

  void SetLangId(int lang_id);

  OrtValue* GetInput(const char* name) override;
  OrtValue* GetOutput(const char* name) override;

 private:
  const NemotronSpeechModel& nemotron_model_;
  NemotronConfig nemotron_config_;

  std::unique_ptr<NemotronEncoderSubState> encoder_state_;
  std::unique_ptr<NemotronPredictionSubState> prediction_state_;
  std::unique_ptr<NemotronJoinerSubState> joiner_state_;

  // Current mel input
  std::shared_ptr<Tensor> current_mel_;

  // Encoder produces `encoded_len_` valid frames into encoder_state_'s
  // pre-allocated `encoded_out_` buffer; we just track the length here.
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
