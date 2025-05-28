// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "model.h"
#include "input_ids.h"
#include "position_inputs.h"

namespace Generators {

struct MarianInputIDs {
  MarianInputIDs(State& state);
  MarianInputIDs(const MarianInputIDs&) = delete;
  MarianInputIDs& operator=(const MarianInputIDs&) = delete;

  void Add();

  void Update(DeviceSpan<int32_t> next_tokens);

  std::array<int64_t, 1> GetMarianInputsShape() const { return shape_; }
  const char* name_;

 private:
  State& state_;
  const Model& model_{state_.model_};
  size_t input_index_{~0U};

  std::array<int64_t, 1> shape_{};
  ONNXTensorElementDataType type_;
  std::unique_ptr<Tensor> value_;
  std::unique_ptr<Tensor> cast_value_;
};

struct MarianLogits {
  MarianLogits(State& state);
  virtual ~MarianLogits() = default;

  void Add();

  DeviceSpan<float> Get();
  void Update(const DeviceSpan<int32_t>& next_tokens, size_t new_kv_length);

 protected:
  State& state_;
  const Model& model_{state_.model_};
  size_t output_index_{~0U};

  std::array<int64_t, 2> shape_{};

  ONNXTensorElementDataType type_;
  std::unique_ptr<Tensor> output_raw_;
  std::unique_ptr<OrtValue> logits_of_last_token_fp32_;

  DeviceSpan<float> logits_;
};

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
  MarianInputIDs decoder_input_ids_{*this};

  DefaultPositionInputs attention_mask_;
  std::unique_ptr<OrtValue> encoder_hidden_states_;
  std::unique_ptr<Tensor> rnn_states_prev_;
  std::unique_ptr<OrtValue> past_key_values_length_;
  std::unique_ptr<Tensor> rnn_states_;
  std::vector<std::unique_ptr<OrtValue>> values_;
  MarianLogits logits_{*this};
};
}  // namespace Generators