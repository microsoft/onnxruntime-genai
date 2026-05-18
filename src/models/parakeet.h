// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Parakeet TDT speech recognition model.
//
// Mirrors the standard batch model structure (Model + State subclasses) so it can
// be driven by the standard Generator pipeline:
//
//     model = og.Model(config)
//     processor = model.create_multimodal_processor()
//     audios = og.Audios.open("audio.wav")
//     inputs = processor("", audios=audios)
//     params = og.GeneratorParams(model)
//     generator = og.Generator(model, params)
//     generator.set_inputs(inputs)
//     while not generator.is_done():
//         generator.generate_next_token()
//     transcription = processor.decode(generator.get_sequence(0))
//
// Internally the encoder is fed one chunk at a time with left + right
// context (matching the original NeMo streaming reference). The window
// length varies across chunks: the first chunk is clipped on the left
// (no past context available), the last chunk is clipped on the right
// (no more audio), and middle chunks span the full
// `left_context + chunk + right_context` extent. The encoder is therefore
// exported with a dynamic time dimension and is given a `signal_length`
// input each call. The TDT (Token-and-Duration Transducer) decoder runs
// against the encoder's chunk-proper frames and emits one symbol id per
// call to StepToken().

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model.h"
#include "transducer_state.h"

namespace Generators {

struct ParakeetTdtConfig {
  // Encoder dimensions
  int hidden_dim{};
  int num_encoder_layers{};

  // Decoder LSTM dimensions
  int decoder_lstm_dim{};
  int decoder_lstm_layers{};

  // Vocabulary
  int blank_id{};

  // Streaming chunk config
  int sample_rate{};
  int chunk_samples{};
  int subsampling_factor{};
  int max_symbols_per_step{};
  int left_context_samples{};
  int right_context_samples{};

  // TDT parameters
  std::vector<int> tdt_durations;

  // Encoder I/O names
  std::string enc_in_audio;
  std::string enc_in_length;
  std::string enc_out_encoded;
  std::string enc_out_length;

  // Decoder (prediction network) I/O names
  std::string dec_in_targets;
  std::string dec_in_targets_length;
  std::string dec_in_lstm_hidden_state;
  std::string dec_in_lstm_cell_state;
  std::string dec_out_outputs;
  std::string dec_out_outputs_length;
  std::string dec_out_lstm_hidden_state;
  std::string dec_out_lstm_cell_state;

  // Joiner I/O names
  std::string join_in_encoder;
  std::string join_in_decoder;
  std::string join_out_logits;

  void PopulateFromConfig(const Config& config);
};

struct ParakeetTdtModel : Model {
  ParakeetTdtModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_encoder_;
  std::unique_ptr<OrtSession> session_decoder_;
  std::unique_ptr<OrtSession> session_joiner_;

  std::unique_ptr<OrtSessionOptions> encoder_session_options_;
  std::unique_ptr<OrtSessionOptions> decoder_session_options_;
  std::unique_ptr<OrtSessionOptions> joiner_session_options_;

  ParakeetTdtConfig parakeet_config_;
};

struct ParakeetEncoderSubState : State {
  ParakeetEncoderSubState(const ParakeetTdtModel& model, const GeneratorParams& params);

  void SetInputs(OrtValue* mel_tensor, int64_t num_mel_frames);

  DeviceSpan<float> Run(int total_length,
                        DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

 private:
  const ParakeetTdtModel& model_;
  const ParakeetTdtConfig& cfg_;
  std::unique_ptr<OrtValue> signal_length_;
  size_t mel_input_idx_{};
  size_t length_input_idx_{};
};

struct ParakeetDecoderSubState : State {
  ParakeetDecoderSubState(const ParakeetTdtModel& model, const GeneratorParams& params);

  void ResetLstmState();

  void StepWithToken(int32_t token_id);

  std::unique_ptr<OrtValue> TakeDecoderOutput();

  DeviceSpan<float> Run(int total_length,
                        DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

 private:
  const ParakeetTdtModel& model_;
  const ParakeetTdtConfig& cfg_;
  std::unique_ptr<OrtValue> targets_;         // [1, 1] int64
  std::unique_ptr<OrtValue> targets_length_;  // [1] int64, always 1
  std::unique_ptr<OrtValue> state_h_;         // [layers, 1, dim] float
  std::unique_ptr<OrtValue> state_c_;         // [layers, 1, dim] float
  size_t targets_input_idx_{};
  size_t targets_length_input_idx_{};
  size_t state_h_input_idx_{};
  size_t state_c_input_idx_{};
};

struct ParakeetJoinerSubState : State {
  ParakeetJoinerSubState(const ParakeetTdtModel& model, const GeneratorParams& params);

  void SetInputFrames(OrtValue* encoder_frame, OrtValue* decoder_frame);

  DeviceSpan<float> Run(int total_length,
                        DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

 private:
  const ParakeetTdtModel& model_;
  const ParakeetTdtConfig& cfg_;
  size_t encoder_input_idx_{};
  size_t decoder_input_idx_{};
};

// State driving the streaming TDT decoder.

struct ParakeetTdtState : TransducerState {
  ParakeetTdtState(const ParakeetTdtModel& model, const GeneratorParams& params);

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  // TDT bypasses the search/logits pipeline entirely; the orchestrator state
  // has no meaningful Run() and instead exposes StepToken() below.
  DeviceSpan<float> Run(int total_length,
                        DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  // Run the TDT loop until a non-blank token is emitted, append it to the
  // accumulated transcript, and expose it via GetStepTokens(). Sets the
  // chunk-done flag when the audio has been fully consumed.
  void StepToken() override;

 private:
  // Encode the next audio chunk and update current_encoder_ / current_t_ /
  // current_end_frame_. Sets finished_ when the trailing chunk is encoded.
  void EncodeNextChunk();
  // Run the TDT loop until a non-blank token is emitted, or return blank_id
  // (== eos) when the whole utterance has been consumed.
  int32_t EmitNextToken();
  void InitializeDecoderState();

  const ParakeetTdtModel& model_;
  ParakeetTdtConfig cfg_;

  // Sub-states delegating to the three ORT sessions through the standard
  // State::Run(session) path
  std::unique_ptr<ParakeetEncoderSubState> encoder_state_;
  std::unique_ptr<ParakeetDecoderSubState> decoder_state_;
  std::unique_ptr<ParakeetJoinerSubState> joiner_state_;

  // Most recent decoder output (kept here, since the joiner consumes it on
  // every step but the decoder only runs on non-blank emits). Reshaped on
  // the fly into decoder_frame_ before each joiner call.
  std::unique_ptr<OrtValue> decoder_output_;

  std::vector<float> full_mel_;
  int total_mel_frames_{0};
  bool mel_loaded_{false};

  // Total audio length, in samples (set once when the mel arrives, since
  // total_audio = total_mel_frames_ * hop_length conceptually).
  size_t total_audio_{0};
  // Start of the next chunk-proper region, in audio-sample space. Advances
  // by chunk_samples_ each EncodeNextChunk() call (clamped at total_audio_).
  size_t next_chunk_start_{0};
  // True once the trailing chunk has been encoded; no more encoder runs.
  bool finished_{false};
  // True after the decoder LSTM state has been seeded with the SOS token
  // (lazy init on the first StepToken() call).
  bool initialized_{false};

  // Current encoder window (output of the most recent encoder run), staged
  // on CPU for per-frame slicing into the joiner. The encoder runs on
  // chunk + left context + right context, so the underlying tensor
  // [1, hidden_dim, T'] covers all three regions. The three indices below
  // carve out only the chunk-proper region for decoding; the context frames
  // exist so the encoder has enough receptive field at chunk boundaries.
  //
  //   encoder frames:  0 ........ a ........ b ........ T'
  //                    └─ left ──┘└── chunk ─┘└─ right ─┘
  //                                ↑          ↑          ↑
  //                            current_t_  current_end_  current_enc_time_
  //                            (start)      _frame_       (= T')
  std::vector<float> current_encoder_cpu_;  // [hidden_dim, T'] flattened
  int64_t current_enc_time_{0};
  int64_t current_t_{0};
  int64_t current_end_frame_{0};
  int symbols_this_frame_{0};

  // Per-step joiner inputs, allocated lazily on the first decoding step and
  // reused across every subsequent step to avoid per-frame allocator churn.
  // Shapes are fixed: encoder_frame_ = [1, 1, encoder_hidden_dim],
  // decoder_frame_ = [1, 1, decoder_lstm_dim].
  std::unique_ptr<OrtValue> encoder_frame_;
  std::unique_ptr<OrtValue> decoder_frame_;
};

}  // namespace Generators
