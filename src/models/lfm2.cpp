// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "lfm2.h"

namespace Generators {
LFM2_Model::LFM2_Model(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_decoder_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());
  session_info_.Add(*session_decoder_);
}

std::unique_ptr<State> LFM2_Model::CreateState(DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params) const {
  return std::make_unique<LFM2_State>(*this, sequence_lengths_unk, params);
}

LFM2_State::LFM2_State(const LFM2_Model& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      cache_(CreateKeyValueCache(*this)),
      position_inputs_{CreatePositionInputs(*this, sequence_lengths_unk, model_.config_->model.decoder.inputs.attention_mask)} {
  input_ids_.Add();
  position_inputs_->Add();
  logits_.Add();
  cache_->Add();
}

void LFM2_State::SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) {
  extra_inputs_.Add(extra_inputs, model_.session_decoder_->GetInputNames());
}

DeviceSpan<float> LFM2_State::Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  UpdateInputsOutputs(next_tokens, next_indices, total_length);
  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }

  bool graph_capture_this_run = params_->use_graph_capture && input_ids_.GetShape()[1] == 1;
  State::Run(*model_.session_decoder_, graph_capture_this_run);

  return logits_.Get();
}

void LFM2_State::RewindTo(size_t index) {
  throw std::runtime_error("LFM2 does not support RewindTo. Conv state requires replaying all prior tokens.");
}

void LFM2_State::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int total_length) {
  input_ids_.Update(next_tokens);
  size_t new_length = static_cast<size_t>(input_ids_.GetShape()[1]);
  position_inputs_->Update(next_tokens, total_length, static_cast<int>(new_length));
  cache_->Update(beam_indices, total_length);
  logits_.Update(next_tokens, new_length);
}

}  // namespace Generators
