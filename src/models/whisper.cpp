// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "whisper.h"

namespace Generators {

Whisper_Model::Whisper_Model(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());
  session_encoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.encoder_decoder_init.filename)).c_str(), session_options_.get());

  InitDeviceAllocator(*session_decoder_);
  session_encoder_info_ = std::make_unique<SessionInfo>(*session_encoder_);
}

std::unique_ptr<State> Whisper_Model::CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<Whisper_State>(*this, sequence_lengths, params);
}

Whisper_State::Whisper_State(const Whisper_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  auto& inputs = const_cast<GeneratorParams::Whisper&>(std::get<GeneratorParams::Whisper>(params.inputs));

  encoder_input_ids_ = model_.ExpandInputs(inputs.input_features->ort_tensor_, params_->search.num_beams);

  auto hidden_states_type = model_.session_encoder_info_->GetOutputDataType("encoder_hidden_states");
  auto encoder_hidden_states_shape = std::array<int64_t, 3>{decoder_input_ids_.GetShape()[0], 1500, static_cast<int64_t>(model_.config_->model.decoder.num_key_value_heads) * model_.config_->model.decoder.head_size};
  encoder_hidden_states_ = OrtValue::CreateTensor(*model_.allocator_device_, encoder_hidden_states_shape, hidden_states_type);

  auto sequence_lengths = sequence_lengths_unk.GetCPU();
  for (int i = 0; i < decoder_input_ids_.GetShape()[0]; i++) {
    sequence_lengths[i] = static_cast<int32_t>(params_->sequence_length);
  }

  input_names_.push_back("encoder_input_ids");
  inputs_.push_back(encoder_input_ids_.get());
  decoder_input_ids_.name_ = "decoder_input_ids";
  decoder_input_ids_.Add();

  logits_.Add();
  output_names_.push_back("encoder_hidden_states");
  outputs_.push_back(encoder_hidden_states_.get());
  kv_cache_.AddEncoder();
  extra_inputs_.Add();
  cross_cache_.AddOutputs();
}

RoamingArray<float> Whisper_State::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  int batch_size = static_cast<int>(decoder_input_ids_.GetShape()[0]);

  switch (run_state_) {
    case RunState::Encoder_Decoder_Init:
      State::Run(*model_.session_encoder_, *model_.run_options_, batch_size);

      run_state_ = RunState::Decoder_First;
      return logits_.Get();

    case RunState::Decoder_First:
      ClearIO();

      decoder_input_ids_.name_ = model_.config_->model.decoder.inputs.input_ids.c_str();  // Set back to default name, since we overrode it above in the encoder step
      decoder_input_ids_.Add();
      logits_.Add();
      kv_cache_.Add();
      cross_cache_.AddInputs();
      run_state_ = RunState::Decoder;
      // Fall through

    case RunState::Decoder:
      UpdateInputs(next_tokens, next_indices, current_length);
      break;
  }

  State::Run(*model_.session_decoder_, *model_.run_options_, batch_size);
  return logits_.Get();
}

void Whisper_State::UpdateInputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> beam_indices, int current_length) {
  decoder_input_ids_.Update(next_tokens);
  kv_cache_.Update(beam_indices.GetCPU(), current_length);
}

}  // namespace Generators
