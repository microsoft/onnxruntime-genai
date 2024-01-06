#include "../generators.h"
#include "../search.h"
#include "model.h"
#include "gpt.h"
#include "debugging.h"
#include <iostream>

namespace Generators {

Gpt_Model::Gpt_Model(Model& model, OrtEnv& ort_env, OrtSessionOptions& session_options)
    : model_{model} {

  session_decoder_ = OrtSession::Create(ort_env, (model_.config_.config_path / model_.config_.model_decoder).c_str(), &session_options);

  allocator_device_ = &model.allocator_cpu_;
#if USE_CUDA
  if (model.device_type_ == DeviceType::CUDA) {
    memory_info_cuda_ = OrtMemoryInfo::Create("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    allocator_cuda_ = Ort::Allocator::Create(*session_decoder_, *memory_info_cuda_);
    allocator_device_ = allocator_cuda_.get();
  }
#endif

  InitModelParams();
}

std::unique_ptr<State> Gpt_Model::CreateState(RoamingArray<int32_t> sequence_lengths, const SearchParams& params) {
  return std::make_unique<Gpt_State>(*this, sequence_lengths, params);
}

void Gpt_Model::InitModelParams() {
  // We could use this to determine the vocabulary size and if the logits has a width of 1
  auto logits_type_info = session_decoder_->GetOutputTypeInfo(0);
  auto& logits_tensor_info = logits_type_info->GetTensorTypeAndShapeInfo();
  auto logits_shape = logits_tensor_info.GetShape();
  assert(logits_shape.size() == 3);
  logits_uses_seq_len_ = logits_shape[1] == -1;
  vocab_size_ = static_cast<int>(logits_shape[2]);
  layer_count_ = static_cast<int>(session_decoder_->GetOutputCount()) - 1;
  score_type_ = logits_tensor_info.GetElementType();

  auto past_shape = session_decoder_->GetInputTypeInfo(3)->GetTensorTypeAndShapeInfo().GetShape();
  head_count_ = static_cast<int>(past_shape[2]);
  hidden_size_ = static_cast<int>(past_shape[4]);

  assert(model_.config_.vocab_size==vocab_size_);
  assert(model_.config_.num_hidden_layers == layer_count_);
  assert(model_.config_.num_attention_heads == head_count_);
  assert(model_.config_.hidden_size == hidden_size_);
}

Gpt_State::Gpt_State(Gpt_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const SearchParams& search_params)
    : search_params_{search_params},
      model_{&model},
      input_ids_ {model.model_, search_params, *model.allocator_device_},
      logits_{search_params, model.model_.device_type_, *model.allocator_device_, model.model_.cuda_stream_, model.score_type_, model.logits_uses_seq_len_},
      kv_cache_{search_params, model.model_.config_, *model.allocator_device_, model.model_.cuda_stream_, model.score_type_},
      position_ids_{model.model_, search_params, *model.allocator_device_, sequence_lengths_unk} {

  inputs_.push_back(input_ids_.input_ids_.get());
  input_names_.push_back("input_ids");
  inputs_.push_back(position_ids_.position_ids_.get());
  input_names_.push_back("position_ids");
  inputs_.push_back(position_ids_.attention_mask_.get());
  input_names_.push_back("attention_mask");

  output_names_.push_back("logits");
  outputs_.push_back(logits_.logits_.get());

  for (int i = 0; i < model_->layer_count_; i++) {
    inputs_.push_back(kv_cache_.empty_past_.get());
    input_names_.push_back(kv_cache_.input_name_strings_[i].c_str());
    outputs_.push_back(kv_cache_.presents_[i].get());
    output_names_.push_back(kv_cache_.output_name_strings_[i].c_str());
  }
}

RoamingArray<float> Gpt_State::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  if (first_run_)
    first_run_ = false;
  else
    UpdateInputs(next_tokens, next_indices, current_length);

#if 0
    printf("**Inputs:\r\n");
    DumpTensors(inputs_.data(), input_names_.data(), input_names_.size(), true);
    printf("**Outputs:\r\n");
    DumpTensors(outputs_.data(), output_names_.data(), output_names_.size(), false);
#endif

  model_->session_decoder_->Run(nullptr, input_names_.data(), inputs_.data(), input_names_.size(), output_names_.data(), outputs_.data(), output_names_.size());
  return logits_.Get();
}

void Gpt_State::UpdateInputs(RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> beam_indices, int current_length) {
  assert(search_params_.num_beams == 1 || !beam_indices.empty());  // We require beam_indices if we're a beam search

  input_ids_.Update(next_tokens);
  inputs_[0] = input_ids_.input_ids_.get();

  position_ids_.Update(current_length);
  inputs_[1] = position_ids_.position_ids_.get();
  inputs_[2] = position_ids_.attention_mask_.get();

  logits_.Update();
  outputs_[0]=logits_.logits_.get();

  kv_cache_.Update(beam_indices.GetCPU(), current_length);
  for (size_t i = 0; i < model_->layer_count_; i++) {
    inputs_[i + 3] = kv_cache_.pasts_[i].get();
    outputs_[i + 1] = kv_cache_.presents_[i].get();
  }
}

}  // namespace Generators
