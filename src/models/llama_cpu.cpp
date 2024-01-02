#include "../generators.h"
#include "../search.h"
#include "llama_cpu.h"
#include "debugging.h"

namespace Generators {

Llama_State::Llama_State(Llama_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const SearchParams& search_params)
    : model_{&model}, 
      search_params_{search_params},
      kv_cache_{search_params, model.config_, allocator_cpu_, model.cuda_stream_, model.score_type_, model.past_names_, model.present_names_} {

  assert(model.score_type_ == Ort::TypeToTensorType<float>::type);
  int64_t input_ids_shape[] = {search_params_.batch_size, search_params_.sequence_length};
  cpu_span<int32_t> sequence_lengths = sequence_lengths_unk;

  // Allocate position_ids and attention_mask based on shape of input_ids
  auto element_type = Ort::TypeToTensorType<int64_t>::type;

  // Use original input_ids. This requires the input_ids for subgraph is also int32.
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  // To avoid cloning input_ids, we use const_cast here since this function does not change its content.
  input_ids_ = OrtValue::CreateTensor<int64_t>(allocator_cpu_, input_ids_shape);
  auto* p_data = input_ids_->GetTensorMutableData<int64_t>();
  for (auto v : search_params_.input_ids)
    *p_data++ = v;
  position_ids_ = OrtValue::CreateTensor<int64_t>(allocator_cpu_, input_ids_shape);

  int64_t position_shape[] = {search_params_.batch_size * search_params_.num_beams, 1};
  next_positions_ = Allocate<int64_t>(allocator_cpu_, position_shape[0], next_positions_buffer_);
  memset(next_positions_.data(), 0, next_positions_.size_bytes());
  next_positions_tensor_ = OrtValue::CreateTensor<int64_t>(allocator_cpu_.GetInfo(), next_positions_, position_shape);

  attention_mask_ = OrtValue::CreateTensor<int64_t>(allocator_cpu_, input_ids_shape);

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

  expanded_input_ids_ = ExpandInputs(input_ids_, search_params_.num_beams, allocator_cpu_, DeviceType::CPU, nullptr);
  expanded_position_ids_ = ExpandInputs(position_ids_, search_params_.num_beams, allocator_cpu_, DeviceType::CPU, nullptr);
  expanded_attention_mask_ = ExpandInputs(attention_mask_, search_params_.num_beams, allocator_cpu_, DeviceType::CPU, nullptr);

  for (auto* input : {expanded_input_ids_.get(), expanded_position_ids_.get(), expanded_attention_mask_.get()})
    inputs_.push_back(input);
  for (auto* name : {"input_ids", "position_ids", "attention_mask"})
    input_names_.push_back(name);

  output_names_.push_back("logits");

  // Allocate space for logits (only works if we know the shape)
  {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, model_->logits_uses_seq_len_ ? input_ids_shape[1] : 1, model_->vocab_size_};
    logits_ = OrtValue::CreateTensor(allocator_cpu_, logits_shape, model_->score_type_);
    outputs_.push_back(logits_.get());
  }

  for (int i = 0; i < model_->layer_count_ * 2; ++i) {
    inputs_.push_back(kv_cache_.empty_past_.get());
    input_names_.push_back(kv_cache_.input_name_strings_[i].c_str());
    outputs_.push_back(kv_cache_.presents_[i].get());
    output_names_.push_back(kv_cache_.output_name_strings_[i].c_str());
  }
}

RoamingArray<float> Llama_State::Run(int current_length, RoamingArray<int32_t> next_tokens_unk, RoamingArray<int32_t> next_indices_unk) {
  cpu_span<int32_t> next_tokens = next_tokens_unk;
  cpu_span<int32_t> next_indices = next_indices_unk;

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

  auto type_shape = logits_->GetTensorTypeAndShapeInfo();
  auto shape = type_shape->GetShape();
  assert(type_shape->GetShape().size() == 3);

  return cpu_span<float>{logits_->GetTensorMutableData<ScoreType>(), type_shape->GetElementCount()};
}

void Llama_State::UpdateInputs(std::span<const int32_t> next_tokens, std::span<const int32_t> beam_indices, int current_length) {

  // The following updates inputs for subgraph

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(next_tokens.size());
  int64_t dims[] = {batch_beam_size, 1};
  std::unique_ptr<OrtValue> input_ids = OrtValue::CreateTensor<int64_t>(allocator_cpu_, dims);
  int64_t* input_ids_data = input_ids->GetTensorMutableData<int64_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    input_ids_data[i] = next_tokens[i];
  }
  expanded_input_ids_ = std::move(input_ids);
  inputs_[0] = expanded_input_ids_.get();

  // Update position IDs
  inputs_[1] = next_positions_tensor_.get();
  {
    int64_t* position_data = next_positions_.data();
    for (int i = 0; i < batch_beam_size; i++) {
      position_data[i] = current_length - 1;
    }
  }

  // Update attention mask
  const int64_t* old_mask_data = expanded_attention_mask_->GetTensorMutableData<int64_t>();
  int64_t mask_dims[] = {batch_beam_size, current_length};
  auto attention_mask = OrtValue::CreateTensor<int64_t>(allocator_cpu_, mask_dims);
  int64_t* mask_data = attention_mask->GetTensorMutableData<int64_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < current_length - 1; j++) {
      mask_data[i * current_length + j] = old_mask_data[i * (current_length - 1) + j];
    }
    mask_data[i * current_length + current_length - 1] = 1;
  }
  expanded_attention_mask_ = std::move(attention_mask);
  inputs_[2] = expanded_attention_mask_.get();

  // Update logits
  if (model_->logits_uses_seq_len_) {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, 1, model_->vocab_size_};
    logits_ = OrtValue::CreateTensor(allocator_cpu_, logits_shape, Ort::TypeToTensorType<ScoreType>::type);
    outputs_[0] = logits_.get();
  }

  kv_cache_.Update(beam_indices, current_length);
  for (size_t i = 0; i < model_->layer_count_ * 2; i++) {
    inputs_[i + 3] = kv_cache_.pasts_[i].get();
    outputs_[i + 1] = kv_cache_.presents_[i].get();
  }
}

}  // namespace Generators