// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "extra_inputs.h"
#include "position_inputs.h"

namespace Generators {

struct MarianModel : Model {
  MarianModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_encoder_;  // encoder_decoder_init.onnx
  std::unique_ptr<OrtSession> session_decoder_;  // decoder.onnx
};

struct MarianState : State {
  MarianState(const MarianModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params);
  MarianState(const MarianState&) = delete;
  MarianState& operator=(const MarianState&) = delete;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;

 private:
  const MarianModel& model_;
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices, int current_length, bool search_buffers);

  // Encoder IOs
  DefaultInputIDs encoder_input_ids_{*this};
  DefaultPositionInputs encoder_attention_mask_;
  std::unique_ptr<OrtValue> encoder_outputs_;


  // Decoder IOs
  DecoderInputIDs decoder_input_ids_{*this};
  DefaultPositionInputs attention_mask_;
  std::unique_ptr<OrtValue> encoder_hidden_states_;
  std::unique_ptr<OrtValue> rnn_states_prev_;
  std::unique_ptr<OrtValue> past_key_values_length_;
  std::unique_ptr<OrtValue> rnn_states_;
  std::vector<std::unique_ptr<OrtValue>> values_;
  // Logits logits_{*this};
  RNNLogits logits_{*this};
  // std::unique_ptr<OrtValue> logits_;
};
}  // namespace Generators