#include "../generators.h"
#include "../search.h"
#include "whisper.h"
#include "debugging.h"

namespace Generators {

Whisper_Model::Whisper_Model(std::unique_ptr<Config> config, OrtEnv& ort_env, const ProviderOptions* provider_options)
    : Model{std::move(config), ort_env, provider_options} {

  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / config_->model_decoder).c_str(), session_options_.get());
  session_encoder_ = OrtSession::Create(ort_env, (config_->config_path / config_->model_encoder_decoder_init).c_str(), session_options_.get());

  InitDeviceAllocator(*session_decoder_);
  InitModelParams();
}

std::unique_ptr<State> Whisper_Model::CreateState(RoamingArray<int32_t> sequence_lengths, const SearchParams& params) {
  return std::make_unique<Whisper_State>(*this, sequence_lengths, params);
}

void Whisper_Model::InitModelParams() {
  // We could use this to determine the vocabulary size and if the logits has a width of 1
  auto logits_type_info = session_decoder_->GetOutputTypeInfo(0);
  auto& logits_tensor_info = logits_type_info->GetTensorTypeAndShapeInfo();
  auto logits_shape = logits_tensor_info.GetShape();
  assert(logits_shape.size() == 3);
  logits_uses_seq_len_ = logits_shape[1] == -1;
  auto vocab_size = static_cast<int>(logits_shape[2]);
  auto layer_count = (static_cast<int>(session_decoder_->GetOutputCount()) - 1) / 2;
  score_type_ = logits_tensor_info.GetElementType();

  auto past_shape = session_decoder_->GetInputTypeInfo(3)->GetTensorTypeAndShapeInfo().GetShape();
  auto head_count = static_cast<int>(past_shape[1]);
  auto hidden_size = static_cast<int>(past_shape[3]);

  assert(config_->vocab_size == vocab_size);
  assert(config_->num_hidden_layers == layer_count);
  assert(config_->num_attention_heads == head_count);
  assert(config_->hidden_size == hidden_size);
}

Whisper_State::Whisper_State(Whisper_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const SearchParams& search_params)
    : model_{model},
      search_params_{search_params} {

  auto& inputs = const_cast<SearchParams::Whisper&>(std::get<SearchParams::Whisper>(search_params.inputs));

  auto encoder_input_ids = ExpandInputs(inputs.input_features, search_params_.num_beams, *model_.allocator_device_, model_.device_type_, model_.cuda_stream_);
  encoder_hidden_states_ = OrtValue::CreateTensor<float>(*model_.allocator_device_, std::array<int64_t, 3>{decoder_input_ids_.input_ids_shape_[0], 1500, 384});

  auto sequence_lengths = sequence_lengths_unk.GetCPU();
  for (int i = 0; i < decoder_input_ids_.input_ids_shape_[0]; i++) {
    sequence_lengths[i] = static_cast<int32_t>(search_params_.sequence_length);
  }

  input_names_.push_back("encoder_input_ids");
  inputs_.push_back(encoder_input_ids.get());
  input_names_.push_back("decoder_input_ids");
  inputs_.push_back(decoder_input_ids_.input_ids_.get());

  output_names_.push_back("logits");
  outputs_.push_back(logits_.logits_.get());
  output_names_.push_back("encoder_hidden_states");
  outputs_.push_back(encoder_hidden_states_.get());

  for (int i = 0; i < kv_cache_.layer_count_ * 2; ++i) {
    outputs_.push_back(kv_cache_.presents_[i].get());
    output_names_.push_back(kv_cache_.output_name_strings_[i].c_str());
  }

  for (int i = 0; i < kv_cache_.layer_count_ * 2; ++i) {
    outputs_.push_back(kv_cache_.crosses_[i].get());
    output_names_.push_back(kv_cache_.output_cross_name_strings_[i].c_str());
  }

  model_.session_encoder_->Run(nullptr, input_names_.data(), inputs_.data(), input_names_.size(), output_names_.data(), outputs_.data(), output_names_.size());

  input_names_.clear();
  output_names_.clear();
  inputs_.clear();
  outputs_.clear();

  input_names_.push_back("input_ids");
  inputs_.push_back(nullptr);  // Placeholder, will be filled in by UpdateInputs

  output_names_.push_back("logits");
  outputs_.push_back(nullptr);  // Placeholder, will be filled in by UpdateInputs

  for (int i = 0; i < kv_cache_.layer_count_ * 2; ++i) {
    inputs_.push_back(nullptr);  // Placeholder, will be filled in by UpdateInputs
    input_names_.push_back(kv_cache_.input_name_strings_[i].c_str());
    outputs_.push_back(kv_cache_.presents_[i].get());
    output_names_.push_back(kv_cache_.output_name_strings_[i].c_str());
  }

  for (int i = 0; i < kv_cache_.layer_count_ * 2; ++i) {
    inputs_.push_back(kv_cache_.crosses_[i].get());
    input_names_.push_back(kv_cache_.input_cross_name_strings_[i].c_str());
  }
}

RoamingArray<float> Whisper_State::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {

  if (first_run_)
    first_run_ = false;
  else {
    UpdateInputs(next_tokens, next_indices, current_length);

#if 0
    printf("**Inputs:\r\n");
    DumpTensors(inputs_.data(), input_names_.data(), input_names_.size(), true);
    printf("**Outputs:\r\n");
    DumpTensors(outputs_.data(), output_names_.data(), output_names_.size(), false);
#endif

    model_.session_decoder_->Run(nullptr, input_names_.data(), inputs_.data(), input_names_.size(), output_names_.data(), outputs_.data(), output_names_.size());
  }
  return logits_.Get();
}

void Whisper_State::UpdateInputs(RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> beam_indices, int current_length) {
  decoder_input_ids_.Update(next_tokens);
  inputs_[0] = decoder_input_ids_.input_ids_.get();

  logits_.Update();
  outputs_[0]=logits_.logits_.get();

  kv_cache_.Update(beam_indices.GetCPU(), current_length);
  for (size_t i = 0; i < kv_cache_.layer_count_ * 2; i++) {
    inputs_[i + 1] = kv_cache_.pasts_[i].get();
    outputs_[i + 1] = kv_cache_.presents_[i].get();
  }
}

}  // namespace Generators
