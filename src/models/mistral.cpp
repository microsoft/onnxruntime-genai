#include "../generators.h"
#include "../search.h"
#include "mistral.h"

namespace Generators {

Mistral_Model::Mistral_Model(std::unique_ptr<Config> config, OrtEnv& ort_env, const ProviderOptions* provider_options)
    : Model{std::move(config), ort_env, provider_options} {
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / config_->model_decoder).c_str(), session_options_.get());

  InitDeviceAllocator(*session_decoder_);
  InitModelParams();
}

std::unique_ptr<State> Mistral_Model::CreateState(RoamingArray<int32_t> sequence_lengths, const SearchParams& params) {
  return std::make_unique<Mistral_State>(*this, sequence_lengths, params);
}

void Mistral_Model::InitModelParams() {
  ValidateLogits(*session_decoder_->GetOutputTypeInfo(0));

  auto layer_count = (static_cast<int>(session_decoder_->GetOutputCount()) - 1) / 2;
  auto past_shape = session_decoder_->GetInputTypeInfo(3)->GetTensorTypeAndShapeInfo().GetShape();
  auto head_count = static_cast<int>(past_shape[1]);
  auto hidden_size = static_cast<int>(past_shape[3]);

  Unreferenced(layer_count, head_count, hidden_size);
  assert(config_->num_hidden_layers == layer_count);
  assert(config_->num_attention_heads == head_count);
  assert(config_->hidden_size == hidden_size);
}

Mistral_State::Mistral_State(Mistral_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const SearchParams& search_params)
    : State{search_params},
      model_{model},
      position_ids_{model, *this, sequence_lengths_unk} {
  input_ids_.Add();
  position_ids_.Add();
  logits_.Add();
  kv_cache_.Add();
}

RoamingArray<float> Mistral_State::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  if (first_run_)
    first_run_ = false;
  else
    UpdateInputs(next_tokens, next_indices, current_length);

  State::Run(*model_.session_decoder_);
  return logits_.Get();
}

void Mistral_State::UpdateInputs(RoamingArray<int32_t> next_tokens_unk, RoamingArray<int32_t> beam_indices, int current_length) {
  input_ids_.Update(next_tokens_unk);
  position_ids_.Update(current_length);
  logits_.Update();
  kv_cache_.Update(beam_indices.GetCPU(), current_length);
}

}  // namespace Generators
