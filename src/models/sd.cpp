// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "sd.h"
#include <vector>

namespace Generators {

// Constructor for StableDiffusion_Model
StableDiffusion_Model::StableDiffusion_Model(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model(std::move(config)) {
  // Initialize encoder and decoder sessions

  auto text_model_path = config_->config_path / "text_encoder" / "model.onnx";
  //session_encoder_ = CreateSession("encoder_decoder_init.onnx");
  //session_decoder_ = CreateSession("decoder.onnx");
}

// CreateState method for StableDiffusion_Model
std::unique_ptr<State> StableDiffusion_Model::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<StableDiffusion_State>(*this, sequence_lengths, params);
}

// Constructor for StableDiffusion_State
StableDiffusion_State::StableDiffusion_State(const StableDiffusion_Model& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params)
    : State(params, model), model_(model) {
  // Initialize state-specific resources
  // Initialize(sequence_lengths, params.total_length, params.beam_indices);
}

// Run method for StableDiffusion_State
DeviceSpan<float> StableDiffusion_State::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  UpdateInputsOutputs(next_tokens, next_indices, current_length, /*search_buffers=*/true);
  // Execute decoder session
  //model_.session_decoder_->Run(/*inputs=*/{}, /*outputs=*/{});
  return {};  // Return the output span (to be implemented)
}

// GetOutput method for StableDiffusion_State
OrtValue* StableDiffusion_State::GetOutput(const char* name) {
  // Retrieve output by name
  return nullptr;  // Replace with actual implementation
}

// UpdateInputsOutputs method for StableDiffusion_State
void StableDiffusion_State::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices, int current_length, bool search_buffers) {
  // Update input and output buffers for the decoder
  // ...implementation...
}

// Initialize method for StableDiffusion_State
void StableDiffusion_State::Initialize(DeviceSpan<int32_t>& next_tokens, int total_length, DeviceSpan<int32_t> beam_indices) {
  // Initialize state resources
  // ...implementation...
}

// Finalize method for StableDiffusion_State
void StableDiffusion_State::Finalize() {
  // Finalize state resources
  // ...implementation...
}

}  // namespace Generators
