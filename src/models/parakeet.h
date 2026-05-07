// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Parakeet-TDT (Token-and-Duration Transducer) ASR model.
// Three ONNX sessions: encoder (FastConformer), decoder (LSTM prediction net), joiner.
// Long-form audio is handled by chunking with overlap inside the State; the customer-facing
// API mirrors Whisper:
//
//   inputs = processor(prompts=[""], audios=audios)
//   gen.set_inputs(inputs)
//   while not gen.is_done():
//       gen.generate_next_token()
//   text = processor.decode(gen.get_sequence(0))

#pragma once

#include <vector>
#include <string>
#include <memory>

#include "model.h"

namespace Generators {

struct ParakeetConfig {
  // From genai_config.json (model.* / parakeet block)
  int sample_rate{16000};
  int num_mels{128};
  int fft_size{512};
  int hop_length{160};
  int win_length{400};
  float preemph{0.0f};
  float log_eps{1e-10f};

  int subsampling_factor{8};

  int vocab_size{8192};       // real vocab tokens
  int blank_id{8192};         // blank index in joint logits
  int num_durations{5};       // |durations| from metadata
  std::vector<int> durations{0, 1, 2, 3, 4};
  int max_symbols_per_step{10};

  int decoder_lstm_layers{2};
  int decoder_lstm_dim{640};

  int encoder_hidden_dim{1024};

  // Long-form chunking (in seconds)
  float chunk_seconds{30.0f};
  float overlap_seconds{2.0f};

  // I/O names (sherpa-onnx convention; no Config fields needed)
  std::string enc_in_audio{"audio_signal"};
  std::string enc_in_length{"length"};
  std::string enc_out{"outputs"};
  std::string enc_out_length{"encoded_lengths"};

  std::string dec_in_targets{"targets"};
  std::string dec_in_target_length{"target_length"};
  std::string dec_in_states{"states.1"};
  std::string dec_in_states_init{"onnx::Slice_3"};
  std::string dec_out{"outputs"};
  std::string dec_out_length{"prednet_lengths"};
  std::string dec_out_states{"states"};
  std::string dec_out_states_extra{"162"};

  std::string join_in_encoder{"encoder_outputs"};
  std::string join_in_decoder{"decoder_outputs"};
  std::string join_out{"outputs"};

  void PopulateFromConfig(const Config& config);
};

struct ParakeetModel : Model {
  ParakeetModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_encoder_;
  std::unique_ptr<OrtSession> session_decoder_;
  std::unique_ptr<OrtSession> session_joiner_;

  std::unique_ptr<OrtSessionOptions> encoder_session_options_;
  std::unique_ptr<OrtSessionOptions> decoder_session_options_;
  std::unique_ptr<OrtSessionOptions> joiner_session_options_;

  ParakeetConfig parakeet_config_;
};

// State that integrates with the standard Generator/Search loop.
//
// On the first Run() (triggered by SetInputs supplying mel features as an extra input
// and a single dummy input_id), the state runs the encoder for every chunk, then begins
// the TDT greedy decode. Each subsequent Run() emits one real subword token by writing
// it into the logits tensor (one-hot at the chosen vocab id). When TDT decode is fully
// exhausted, Run() emits blank_id which is configured as eos_token_id, signalling
// search.IsDone().
struct ParakeetState : State {
  ParakeetState(const ParakeetModel& model, const GeneratorParams& params);
  ~ParakeetState() override;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int current_length,
                        DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

 private:
  // Run encoder on all queued chunks and concatenate their outputs.
  // Populates encoder_out_btc_ with shape [1, T_enc_total, hidden_dim].
  void RunEncoderAllChunks();

  // Run decoder LSTM with input token = last_token_; output stored in decoder_out_.
  // Returns true on success.
  void RunDecoder();

  // Run joiner on (encoder_frame_, decoder_out_) and return logits span [vocab+1+num_durations].
  std::span<const float> RunJoiner();

  // Advance one TDT step. Returns the emitted real token id, or blank_id if no more.
  int32_t StepOnce();

  // Write a one-hot logits tensor at token `tok_id`, return the device span used for search.
  DeviceSpan<float> EmitLogits(int32_t tok_id);

  const ParakeetModel& parakeet_model_;
  const ParakeetConfig& cfg_;

  // Mel features: [num_mels, total_frames] float32, set via SetExtraInputs.
  std::shared_ptr<Tensor> mel_input_;

  // Encoder pieces stitched across chunks: row-major [T_enc_total, hidden_dim].
  std::vector<float> encoder_out_btc_;
  int64_t T_enc_total_{0};

  // TDT decode state
  int32_t time_step_{0};            // current encoder frame index
  int32_t symbol_step_{0};          // emissions at current frame (capped by max_symbols_per_step)
  int32_t last_token_{-1};          // last non-blank token emitted (or -1 = SOS / use zero state)
  bool decode_started_{false};
  bool exhausted_{false};

  // Persistent decoder LSTM state [layers, 1, dim]
  std::unique_ptr<OrtValue> dec_states_;        // mutable
  std::unique_ptr<OrtValue> dec_states_zero_;   // frozen zeros for "onnx::Slice_3"

  // Last decoder output [1, dim, 1] kept until a new non-blank token forces re-run.
  std::unique_ptr<OrtValue> decoder_out_;       // shape [1, 640, 1]
  bool need_decoder_run_{true};

  // Pre-allocated joiner inputs (single frame, single token)
  std::unique_ptr<OrtValue> joiner_enc_frame_;  // [1, 1, hidden_dim]
  std::unique_ptr<OrtValue> joiner_dec_frame_;  // [1, 1, decoder_dim]

  // Output logits buffer the search reads via SetLogits().
  std::unique_ptr<OrtValue> logits_;            // [1, 1, vocab_with_blank]  (size = blank_id+1)
};

}  // namespace Generators
