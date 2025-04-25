// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "extra_inputs.h"

namespace Generators {

struct StableDiffusion_Model : Model {
  StableDiffusion_Model(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_encoder_;  // encoder_decoder_init.onnx
  std::unique_ptr<OrtSession> session_decoder_;  // decoder.onnx
};

struct StableDiffusion_State : State {
  StableDiffusion_State(const StableDiffusion_Model& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params);
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  OrtValue* GetOutput(const char* name) override;

 private:
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices, int current_length, bool search_buffers);
  void Initialize(DeviceSpan<int32_t>& next_tokens, int total_length, DeviceSpan<int32_t> beam_indices);
  void Finalize() override;

  const StableDiffusion_Model& model_;
};
}  // namespace Generators
