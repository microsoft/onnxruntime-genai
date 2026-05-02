// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "pipeline_config.h"

namespace Generators {

PipelineConfigModel::PipelineConfigModel(
    std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_decoder_ = CreateSession(ort_env, config_->model.decoder.filename,
                                   session_options_.get());
  session_info_.Add(*session_decoder_);
}

std::unique_ptr<State> PipelineConfigModel::CreateState(
    DeviceSpan<int32_t> sequence_lengths,
    const GeneratorParams& params) const {
  return std::make_unique<PipelineConfigState>(
      *this, sequence_lengths, params);
}

PipelineConfigState::PipelineConfigState(
    const PipelineConfigModel& model,
    DeviceSpan<int32_t> sequence_lengths,
    const GeneratorParams& params)
    : State{params, model},
      model_{model},
      kv_cache_(CreateKeyValueCache(*this)),
      position_inputs_{CreatePositionInputs(
          *this, sequence_lengths,
          model_.config_->model.decoder.inputs.attention_mask)} {
  input_ids_.Add();
  position_inputs_->Add();
  logits_.Add();
  if (kv_cache_) kv_cache_->Add();
}

void PipelineConfigState::SetExtraInputs(
    const std::vector<ExtraInput>& extra_inputs) {
  extra_inputs_.Add(extra_inputs,
                    model_.session_decoder_->GetInputNames());
}

DeviceSpan<float> PipelineConfigState::Run(
    int total_length, DeviceSpan<int32_t>& next_tokens,
    DeviceSpan<int32_t> next_indices) {
  UpdateInputsOutputs(next_tokens, next_indices, total_length);

  if (model_.config_->model.decoder.run_options.has_value()) {
    State::SetRunOptions(model_.config_->model.decoder.run_options.value());
  }

  bool graph_capture = params_->use_graph_capture &&
                       input_ids_.GetShape()[1] == 1;
  State::Run(*model_.session_decoder_, graph_capture);

  return logits_.Get();
}

void PipelineConfigState::UpdateInputsOutputs(
    DeviceSpan<int32_t>& next_tokens,
    DeviceSpan<int32_t> beam_indices,
    int total_length) {
  input_ids_.Update(next_tokens);
  position_inputs_->Update(next_tokens, total_length,
                           input_ids_.GetShape()[1]);
  if (kv_cache_) kv_cache_->Update(beam_indices, total_length);
}

void PipelineConfigState::RewindTo(size_t index) {
  if (kv_cache_) kv_cache_->RewindTo(index);
  position_inputs_->RewindTo(index);
}

}  // namespace Generators
