// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "extra_inputs.h"
#include "position_inputs.h"

namespace Generators {

struct EncoderDecoderModel : Model {
  EncoderDecoderModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_encoder_;  // encoder_decoder_init.onnx
  std::unique_ptr<OrtSession> session_decoder_;  // decoder.onnx
};

struct EncoderState: State {
  EncoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params);
  EncoderState(const EncoderState&) = delete;
  EncoderState& operator=(const EncoderState&) = delete;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  CrossCache cross_cache_{*this};

  private:
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices, int current_length);

  friend struct EncoderDecoderState;
  const EncoderDecoderModel& model_;

  // DefaultInputIDs encoder_input_ids_{*this};
  // EncoderFeatures encoder_input_ids_{*this, model_.config_->model.encoder_decoder_init.inputs.input_features};

  // DefaultInputIDs encoder_attention_mask_{*this};
  std::unique_ptr<OrtValue> encoder_input_ids_;
  size_t encoder_input_ids_index {~0U};
  std::unique_ptr<OrtValue> encoder_attention_mask_;
  size_t encoder_attention_mask_index {~0U};
};


struct DecoderState: State{
  DecoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params);
  DecoderState(const DecoderState&) = delete;
  DecoderState& operator=(const DecoderState&) = delete;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;

 private:
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices, int current_length);

  friend struct EncoderDecoderState;
  const EncoderDecoderModel& model_;
  DefaultInputIDs input_ids_{*this};
  std::unique_ptr<OrtValue> encoder_attention_mask_;
  size_t encoder_attention_mask_index {~0U};
  Logits logits_{*this};
  DefaultKeyValueCache kv_cache_{*this};
  CrossCache cross_cache_{*this};

  // std::unique_ptr<OrtValue> input_ids_;
  // std::unique_ptr<OrtValue> past_decode_sequence_length_;
};


struct EncoderDecoderState : State {
  EncoderDecoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params);
  EncoderDecoderState(const EncoderDecoderState&) = delete;
  EncoderDecoderState& operator=(const EncoderDecoderState&) = delete;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  // OrtValue* GetOutput(const char* name) override;

 private:
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices, int current_length, bool search_buffers);


  const EncoderDecoderModel& model_;

  std::unique_ptr<EncoderState> encoder_state_;
  std::unique_ptr<CrossCache> cross_cache_;
  std::unique_ptr<DecoderState> decoder_state_;

  size_t encoder_input_ids_index {~0U};
};
}  // namespace Generators
