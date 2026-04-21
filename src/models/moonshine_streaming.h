// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Moonshine Streaming ASR model support.
#pragma once

#include "model.h"
#include "streaming_asr_state.h"

namespace Generators {

struct MoonshineConfig {
  int sample_rate{16000};
  int chunk_samples{8960};  // 560ms at 16kHz
  int overlap_samples{5120};  // 320ms overlap

  int encoder_hidden_size{620};
  int num_decoder_layers{10};
  int num_attention_heads{8};
  int head_size{64};
  int vocab_size{32768};

  int bos_token_id{1};
  int eos_token_id{2};

  // Encoder I/O names
  std::string enc_in_audio;
  std::string enc_in_attention_mask;
  std::string enc_out_hidden_states;

  // Decoder I/O names (initial decoder, no past KV)
  std::string dec_in_input_ids;
  std::string dec_in_encoder_hidden_states;

  // Decoder with past I/O names
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
struct MoonshineStreamingState : State, StreamingASRState {
  MoonshineStreamingState(const MoonshineStreamingModel& model, const GeneratorParams& params);
  ~MoonshineStreamingState() override;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  // StreamingASRState interface
  std::span<const int32_t> StepToken() override;
  std::span<const int32_t> GetStepTokens() const override { return last_tokens_; }
  bool IsChunkDone() const override { return chunk_done_; }
  size_t TokenCount() const override { return token_count_; }

  OrtValue* GetInput(const char* name) override;
  OrtValue* GetOutput(const char* name) override;

 private:
  const MoonshineStreamingModel& moonshine_model_;
  MoonshineConfig config_;

  // Accumulated encoder hidden states from processor
  std::shared_ptr<Tensor> encoder_hidden_states_tensor_;

  // Decoder state
  bool first_decode_{true};
  bool chunk_done_{true};
  bool decode_mode_{false};
  std::vector<int32_t> last_tokens_;
  size_t token_count_{0};
  size_t max_tokens_{448};  // Max tokens per chunk, from search.max_length
  int last_token_{1};  // BOS

  // Initial decoder outputs / decoder_with_past KV cache
  // self_kv[i] = present_self_key_i or present_self_value_i
  // cross_kv[i] = present_cross_key_i or present_cross_value_i
  std::vector<std::unique_ptr<OrtValue>> self_kv_cache_;
  std::vector<std::unique_ptr<OrtValue>> cross_kv_cache_;

  // Working tensors
  std::unique_ptr<OrtValue> decoder_input_ids_;
  std::unique_ptr<OrtValue> logits_;

  void RunInitialDecoder();
  void RunDecoderWithPast();
};

}  // namespace Generators
