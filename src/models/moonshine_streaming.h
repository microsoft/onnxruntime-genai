// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Moonshine Streaming ASR model support.
//
// Encoder-decoder model: the encoder consumes raw audio (already padded to a
// multiple of 80 samples) and emits encoder_hidden_states; the decoder is a
// two-graph pipeline (initial + with-past) with separate self / cross KV caches.
//
// Driven by the Generator via the shared TransducerState interface:
//   * the StreamingProcessor accumulates audio and runs the encoder on Flush()
//   * the resulting encoder_hidden_states tensor is passed in via SetInputs()
//   * each generate_next_token() call invokes StepToken() which runs one
//     decoder step (initial or with-past) and emits exactly one token.
#pragma once

#include "model.h"
#include "transducer_state.h"

namespace Generators {

struct MoonshineConfig {
  int sample_rate{16000};
  int chunk_samples{8960};   // 560ms at 16kHz
  int overlap_samples{5120}; // 320ms overlap

  int encoder_hidden_size{620};
  int num_decoder_layers{10};

  int bos_token_id{1};
  int eos_token_id{2};

  // Encoder output hidden-states tensor name (used by SetExtraInputs).
  std::string enc_out_hidden_states;

  // Decoder I/O names (initial decoder, no past KV)
  std::string dec_in_input_ids;
  std::string dec_in_encoder_hidden_states;

  // Decoder-with-past filename
  std::string dec_past_filename;

  void PopulateFromConfig(const Config& config);
};

struct MoonshineStreamingModel : Model {
  MoonshineStreamingModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_encoder_;
  std::unique_ptr<OrtSession> session_decoder_;
  std::unique_ptr<OrtSession> session_decoder_past_;

  std::unique_ptr<OrtSessionOptions> encoder_session_options_;
  std::unique_ptr<OrtSessionOptions> decoder_session_options_;
  std::unique_ptr<OrtSessionOptions> decoder_past_session_options_;

  MoonshineConfig moonshine_config_;
};

/// State for the Moonshine streaming encoder-decoder pipeline.
struct MoonshineStreamingState : TransducerState {
  MoonshineStreamingState(const MoonshineStreamingModel& model, const GeneratorParams& params);
  ~MoonshineStreamingState() override;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  // TransducerState interface: emit one decoder token (or mark chunk done).
  void StepToken() override;

  OrtValue* GetInput(const char* name) override;
  OrtValue* GetOutput(const char* name) override;

 private:
  const MoonshineStreamingModel& moonshine_model_;
  MoonshineConfig config_;

  // Accumulated encoder hidden states from processor
  std::shared_ptr<Tensor> encoder_hidden_states_tensor_;

  // Decoder state machine
  bool first_decode_{true};
  bool decode_mode_{false};

  // KV caches
  // self_kv[2*i]   = present_self_key_i
  // self_kv[2*i+1] = present_self_value_i
  // cross_kv similarly for cross attention.
  std::vector<std::unique_ptr<OrtValue>> self_kv_cache_;
  std::vector<std::unique_ptr<OrtValue>> cross_kv_cache_;

  // Working tensors
  std::unique_ptr<OrtValue> decoder_input_ids_;
  std::unique_ptr<OrtValue> logits_;

  void RunInitialDecoder();
  void RunDecoderWithPast();
};

}  // namespace Generators
