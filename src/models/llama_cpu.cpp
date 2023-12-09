#include "../generators.h"
#include "../search.h"
#include "llama_cpu.h"
#include "debugging.h"
#include <iostream>

namespace Generators {

Llama_State::Llama_State(Llama_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const SearchParams& search_params)
 : model_{&model},
  search_params_{search_params} {

  assert(model.score_type_ == Ort::TypeToTensorType<float>::type);
  int64_t input_ids_shape[] = {search_params_.batch_size, search_params_.sequence_length};
  cpu_span<int32_t> sequence_lengths=sequence_lengths_unk;

  // Allocate position_ids and attention_mask based on shape of input_ids
  auto element_type = Ort::TypeToTensorType<int64_t>::type;
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();

  const OrtMemoryInfo& location = allocator.GetInfo();

  // Use original input_ids. This requires the input_ids for subgraph is also int32.
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  // To avoid cloning input_ids, we use const_cast here since this function does not change its content.
  input_ids_ = OrtValue::CreateTensor<int64_t>(allocator, input_ids_shape, std::size(input_ids_shape));
  auto *p_data=input_ids_->GetTensorMutableData<int64_t>();
  for (auto v : search_params_.input_ids)
    *p_data++ = v;
  position_ids_ = OrtValue::CreateTensor<int64_t>(allocator, input_ids_shape, std::size(input_ids_shape));

  int64_t position_shape[] = {search_params_.batch_size * search_params_.num_beams, 1};
  next_positions_ = Allocate<int64_t>(allocator, position_shape[0], next_positions_buffer_);
  memset(next_positions_.data(), 0, next_positions_.size_bytes());
  next_positions_tensor_ = OrtValue::CreateTensor<int64_t>(allocator.GetInfo(), next_positions_.data(), next_positions_.size(), position_shape, std::size(position_shape));

//   void* attn_mask_value = nullptr;  // TODO: Temporary hack until needed
// #if 0
//   attention_mask_;
//   if (attn_mask_value != nullptr) {
//     const Tensor& attn_mask = attn_mask_value->Get<Tensor>();
//     Tensor::InitOrtValue(element_type, input_ids_shape, const_cast<Tensor*>(&attn_mask)->MutableData<int32_t>(),
//                          allocator->Info(), attention_mask);
//   } else {
// #endif
  attention_mask_ = OrtValue::CreateTensor<int64_t>(allocator, input_ids_shape, std::size(input_ids_shape));

  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  int64_t* mask_data = attention_mask_->GetTensorMutableData<int64_t>();
  int64_t* position_data = position_ids_->GetTensorMutableData<int64_t>();
  const int32_t* word_id = search_params_.input_ids.data();
  int64_t* mask = mask_data;
  int64_t* position = position_data;
  for (int i = 0; i < search_params_.batch_size; i++) {
    int64_t abs_position = 0;
    for (int j = 0; j < search_params_.sequence_length; j++, word_id++, mask++, position++) {
      if (*word_id == model.config_.pad_token_id) {
        *mask = 0;
        *position = 0;
      } else {
        *mask = 1;
        *position = abs_position;
        abs_position++;
      }
    }

    for (int k = 0; k < search_params_.num_beams; k++) {
      sequence_lengths[i * search_params_.num_beams + k] = static_cast<int32_t>(abs_position);
    }
  }

  assert(search_params_.num_beams == 1);
  expanded_input_ids_ = std::move(input_ids_);
  expanded_position_ids_ = std::move(position_ids_);
  expanded_attention_mask_ = std::move(attention_mask_);

  for (auto* input : {expanded_input_ids_.get(), expanded_position_ids_.get(), expanded_attention_mask_.get()})
    inputs_.push_back(input);
  for (auto* name : {"input_ids", "position_ids", "attention_mask"})
    input_name_strings_.push_back(name);

  output_name_strings_.push_back("logits");

  auto past_type = Ort::TypeToTensorType<ScoreType>::type;
  // Initialize empty past state
  int64_t empty_past_shape[] = {search_params_.batch_size * search_params_.num_beams, model_->head_count_, 0, model_->hidden_size_};
  empty_past_ = OrtValue::CreateTensor(allocator, empty_past_shape, std::size(empty_past_shape), past_type);
  for (int i = 0; i < model_->layer_count_ * 2; i++)
    inputs_.push_back(empty_past_.get());

  // Initialize non empty past states
  int64_t past_shape[] = {search_params_.batch_size * search_params_.num_beams, model_->head_count_, input_ids_shape[1], model_->hidden_size_};
  pasts_.resize(model_->layer_count_ * 2);

  // The remaining inputs are past state.
  for (int i = 0; i < model_->layer_count_; ++i) {
    char string[32];
    snprintf(string, std::size(string), "past_key_values.%d.key", i);
    input_name_strings_.push_back(string);

    snprintf(string, std::size(string), "past_key_values.%d.value", i);
    input_name_strings_.push_back(string);
  }

  // Allocate space for logits (only works if we know the shape)
  {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, model_->logits_uses_seq_len_ ? input_ids_shape[1] : 1, model_->vocab_size_};
    logits_ = OrtValue::CreateTensor(allocator, logits_shape, std::size(logits_shape), past_type);
    outputs_.push_back(logits_.get());
  }

  {
    int64_t present_shape[] = {search_params_.batch_size * search_params_.num_beams, model_->head_count_, input_ids_shape[1], model_->hidden_size_};
    outputs_.reserve(model_->layer_count_ * 2);

    for (int i = 0; i < model_->layer_count_; ++i) {
      presents_.push_back(OrtValue::CreateTensor(allocator, present_shape, std::size(present_shape), past_type));
      outputs_.push_back(presents_.back().get());
      presents_.push_back(OrtValue::CreateTensor(allocator, present_shape, std::size(present_shape), past_type));
      outputs_.push_back(presents_.back().get());

      char string[32];
      snprintf(string, std::size(string), "present.%d.key", i);
      output_name_strings_.push_back(string);

      snprintf(string, std::size(string), "present.%d.value", i);
      output_name_strings_.push_back(string);
    }
  }

  for (auto& input_name : input_name_strings_)
    input_names_.push_back(input_name.c_str());
  for (auto& output_name : output_name_strings_)
    output_names_.push_back(output_name.c_str());
}

RoamingArray<float> Llama_State::Run(int current_length, RoamingArray<int32_t> next_tokens_unk, RoamingArray<int32_t> next_indices_unk) {
  cpu_span<int32_t> next_tokens=next_tokens_unk;
  cpu_span<int32_t> next_indices=next_indices_unk;
  assert(next_indices.empty());  // Llama doesn't support beam search

  if (first_run_)
    first_run_ = false;
  else
    UpdateInputs(next_tokens, current_length);

#if 0
    printf("**Inputs:\r\n");
    DumpTensors(inputs_.data(), input_names_.data(), input_names_.size(), true);
    printf("**Outputs:\r\n");
    DumpTensors(outputs_.data(), output_names_.data(), output_names_.size(), false);
#endif

  model_->session_decoder_->Run(nullptr, input_names_.data(), inputs_.data(), input_names_.size(), output_names_.data(), outputs_.data(), output_names_.size());

  auto type_shape = logits_->GetTensorTypeAndShapeInfo();
  auto shape = type_shape->GetShape();
  assert(type_shape->GetShape().size() == 3);

  return cpu_span<float>{logits_->GetTensorMutableData<ScoreType>(), type_shape->GetElementCount()};
}

void Llama_State::UpdateInputs(std::span<const int32_t> next_tokens, int current_length) {
  assert(search_params_.num_beams==1);
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();

  // The following updates inputs for subgraph

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(next_tokens.size());
  int64_t dims[] = {batch_beam_size, 1};
  std::unique_ptr<OrtValue> input_ids = OrtValue::CreateTensor<int64_t>(allocator, dims, std::size(dims));
  int64_t* input_ids_data = input_ids->GetTensorMutableData<int64_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    input_ids_data[i] = next_tokens[i];
  }
  expanded_input_ids_=std::move(input_ids);
  inputs_[0] = expanded_input_ids_.get();

  // Update position IDs
  inputs_[1] = next_positions_tensor_.get();
  {
    int64_t* position_data = next_positions_.data();
    for (int i = 0; i < batch_beam_size; i++) {
      position_data[i]=current_length-1;
    }
  }

  // Update attention mask
  const int64_t* old_mask_data = expanded_attention_mask_->GetTensorMutableData<int64_t>();
  int64_t mask_dims[] = {batch_beam_size, current_length};
  auto attention_mask = OrtValue::CreateTensor<int64_t>(allocator, mask_dims, std::size(mask_dims));
  int64_t* mask_data = attention_mask->GetTensorMutableData<int64_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < current_length - 1; j++) {
      mask_data[i * current_length + j] = old_mask_data[i * (current_length - 1) + j];
    }
    mask_data[i * current_length + current_length - 1] = 1;
  }
  expanded_attention_mask_=std::move(attention_mask);
  inputs_[2]=expanded_attention_mask_.get();

  // Update logits
  if (model_->logits_uses_seq_len_)
  {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, 1, model_->vocab_size_};
    logits_ = OrtValue::CreateTensor(allocator, logits_shape, std::size(logits_shape), Ort::TypeToTensorType<ScoreType>::type);
    outputs_[0]=logits_.get();
  }

  // feed present_* output to past_* inputs one by one
  int64_t present_shape[] = {batch_beam_size, model_->head_count_, current_length, model_->hidden_size_};

  for (size_t i = 0; i < model_->layer_count_ * 2; i++) {
    pasts_[i]=std::move(presents_[i]);
    inputs_[i + 3] = pasts_[i].get();

    presents_[i] = OrtValue::CreateTensor<float>(allocator, present_shape, std::size(present_shape));
    outputs_[i + 1] = presents_[i].get();
  }

}

} // Generators