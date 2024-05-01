#include "../generators.h"
#include "whisper.h"

namespace Generators {

Whisper_Model::Whisper_Model(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_encoder_ = OrtSession::Create(ort_env, (config_->config_path / config_->model.encoder_decoder_init.filename).c_str(), session_options_.get());
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / config_->model.decoder.filename).c_str(), session_options_.get());

  InitDeviceAllocator(*session_decoder_);
  session_encoder_info_ = std::make_unique<SessionInfo>(*session_encoder_);
}

std::unique_ptr<State> Whisper_Model::CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<Whisper_State>(*this, sequence_lengths, params);
}

Whisper_State::Whisper_State(const Whisper_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params},
      model_{model} {
  auto& inputs = const_cast<GeneratorParams::Whisper&>(std::get<GeneratorParams::Whisper>(params.inputs));

  encoder_input_ids_ = model_.ExpandInputs(inputs.input_features, params_->search.num_beams);

  auto hidden_states_type = model_.session_encoder_info_->GetOutputDataType("encoder_hidden_states");
  encoder_hidden_states_ = OrtValue::CreateTensor(*model_.allocator_device_, std::array<int64_t, 3>{decoder_input_ids_.GetShape()[0], 1500, model_.config_->model.decoder.num_key_value_heads * model_.config_->model.decoder.head_size}, hidden_states_type);

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
  cross_cache_.AddOutputs();
}

RoamingArray<float> Whisper_State::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  switch (run_state_) {
    case RunState::Encoder_Decoder_Init:
      State::Run(*model_.session_encoder_, *model_.run_options_);
      run_state_=RunState::Decoder_First;
      return logits_.Get();

    case RunState::Decoder_First:
      ClearIO();

      decoder_input_ids_.name_ = model_.config_->model.decoder.inputs.input_ids.c_str();  // Set back to default name, since we overrode it above in the encoder step
      decoder_input_ids_.Add();
      logits_.Add();
      kv_cache_.Add();
      cross_cache_.AddInputs();
      run_state_ = RunState::Decoder;

    case RunState::Decoder:
      UpdateInputs(next_tokens, next_indices, current_length);
      break;
  }

  State::Run(*model_.session_decoder_, *model_.run_options_);
  return logits_.Get();
}

void Whisper_State::UpdateInputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> beam_indices, int current_length) {
  decoder_input_ids_.Update(next_tokens);
  kv_cache_.Update(beam_indices.GetCPU(), current_length);
}

}  // namespace Generators
