#include "../generators.h"
#include "../search.h"
#include "gpt.h"
#include "debugging.h"
#include <iostream>

namespace Generators
{

template <typename T>
static void ExpandInputs(const OrtValue& input, int num_beams, OrtAllocator& allocator, std::unique_ptr<OrtValue>& expanded) {
  // Input shape (batch_size, sequence_length). The input is required with data type T.
  // Output shape (batch_size * num_beams, sequence_length)

  auto input_type_info = input.GetTensorTypeAndShapeInfo();
  auto input_shape = input_type_info->GetShape();
  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];

  int64_t dims[] = {batch_size * num_beams, sequence_length};

  auto element_type = input_type_info->GetElementType();
  assert(element_type == Ort::TypeToTensorType<T>::type);

  expanded = OrtValue::CreateTensor<T>(allocator, dims, std::size(dims));

  const T* input_data = input.GetTensorData<T>();
  T* expanded_data = expanded->GetTensorMutableData<T>();
  T* target = expanded_data;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_beams; j++) {
      memcpy(target, input_data + i * sequence_length, sizeof(T) * sequence_length);
      target += sequence_length;
    }
  }
}

Gpt::Gpt(OrtEnv& ort_env, const ORTCHAR_T* decode_path) {

  auto session_options = OrtSessionOptions::Create();
  session_decode_ = OrtSession::Create(ort_env, decode_path, session_options.get());

  // We could use this to determine the vocabulary size and if the logits has a width of 1
  auto logits_shape = session_decode_->GetOutputTypeInfo(0)->GetTensorTypeAndShapeInfo().GetShape();
  assert(logits_shape.size() == 3);
  model_params_.logits_uses_seq_len = logits_shape[1]==-1;
  model_params_.vocab_size = static_cast<int>(logits_shape[2]);
  model_params_.layer_count = static_cast<int>(session_decode_->GetOutputCount())-1;

  auto past_shape = session_decode_->GetInputTypeInfo(3)->GetTensorTypeAndShapeInfo().GetShape();
  model_params_.head_count = static_cast<int>(past_shape[2]);
  model_params_.hidden_size = static_cast<int>(past_shape[4]);
}

void Gpt::CreateInputs(std::span<int32_t> sequence_lengths, const SearchParams& search_params) {
  search_params_ = search_params;

  int64_t input_ids_shape[] = {search_params_.batch_size, search_params_.sequence_length};

  // Allocate position_ids and attention_mask based on shape of input_ids
  auto element_type = Ort::TypeToTensorType<int32_t>::type;
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();

  const OrtMemoryInfo& location = allocator.GetInfo();

  // Use original input_ids. This requires the input_ids for subgraph is also int32.
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  // To avoid cloning input_ids, we use const_cast here since this function does not change its content.
  input_ids_ = OrtValue::CreateTensor<int32_t>(allocator.GetInfo(), const_cast<int32_t*>(search_params_.input_ids.data()), input_ids_shape[0] * input_ids_shape[1], input_ids_shape, std::size(input_ids_shape));
  position_ids_ = OrtValue::CreateTensor<int32_t>(allocator, input_ids_shape, std::size(input_ids_shape));

  int64_t position_shape[] = {search_params_.batch_size * search_params_.num_beams, 1};
  next_positions_ = Allocate<int32_t>(allocator, position_shape[0], next_positions_buffer_);
  memset(next_positions_.data(), 0, next_positions_.size_bytes());
  next_positions_tensor_ = OrtValue::CreateTensor<int32_t>(allocator.GetInfo(), next_positions_.data(), next_positions_.size(), position_shape, std::size(position_shape));

  void* attn_mask_value = nullptr;  // TODO: Temporary hack until needed
#if 0
  attention_mask_;
  if (attn_mask_value != nullptr) {
    const Tensor& attn_mask = attn_mask_value->Get<Tensor>();
    Tensor::InitOrtValue(element_type, input_ids_shape, const_cast<Tensor*>(&attn_mask)->MutableData<int32_t>(),
                         allocator->Info(), attention_mask);
  } else {
#endif
  attention_mask_ = OrtValue::CreateTensor<int32_t>(allocator, input_ids_shape, std::size(input_ids_shape));

  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  int32_t* mask_data = attention_mask_->GetTensorMutableData<int32_t>();
  int32_t* position_data = position_ids_->GetTensorMutableData<int32_t>();
  const int32_t* word_id = search_params_.input_ids.data();
  int32_t* mask = mask_data;
  int32_t* position = position_data;
  for (int i = 0; i < search_params_.batch_size; i++) {
    int32_t abs_position = 0;
    for (int j = 0; j < search_params_.sequence_length; j++, word_id++, mask++, position++) {
      if (*word_id == search_params_.pad_token_id) {
        if (attn_mask_value == nullptr) {
          *mask = 0;
        }
        *position = 0;
      } else {
        if (attn_mask_value == nullptr) {
          *mask = 1;
        }
        *position = abs_position;
        abs_position++;
      }
    }

    for (int k = 0; k < search_params_.num_beams; k++) {
      sequence_lengths[i * search_params_.num_beams + k] = abs_position;
    }
  }

  // Expand (batch_size, sequence_length) to (batch_size * num_beams, sequence_length)
  // TODO(tianleiwu): Try expand outputs after first subgraph call instead. That may get better performance.
  if (search_params_.num_beams == 1) {
    expanded_input_ids_ = std::move(input_ids_);
    expanded_position_ids_ = std::move(position_ids_);
    expanded_attention_mask_ = std::move(attention_mask_);
  } else {
    ExpandInputs<int32_t>(*input_ids_, search_params_.num_beams, allocator, expanded_input_ids_);
    ExpandInputs<int32_t>(*position_ids_, search_params_.num_beams, allocator, expanded_position_ids_);
    ExpandInputs<int32_t>(*attention_mask_, search_params_.num_beams, allocator, expanded_attention_mask_);
  }

  for (auto* input : {expanded_input_ids_.get(), expanded_position_ids_.get(), expanded_attention_mask_.get()})
    inputs_.push_back(input);
  for (auto* name : {"input_ids", "position_ids", "attention_mask"})
    input_name_strings_.push_back(name);

  output_name_strings_.push_back("logits");

  auto past_type = Ort::TypeToTensorType<ScoreType>::type;
  if (!past_present_share_buffer_) {
    // Initialize empty past state
    int64_t empty_past_shape[] = {2, search_params_.batch_size * search_params_.num_beams, model_params_.head_count, 0, model_params_.hidden_size};
    empty_past_ = OrtValue::CreateTensor(allocator, empty_past_shape, std::size(empty_past_shape), past_type);
    for (int i = 0; i < model_params_.layer_count; i++)
      inputs_.push_back(empty_past_.get());

    // Initialize non empty past states
    int64_t past_shape[] = {2, search_params_.batch_size * search_params_.num_beams, model_params_.head_count, input_ids_shape[1], model_params_.hidden_size};
    pasts_.reserve(model_params_.layer_count);

    // The remaining inputs are past state.
    for (int i = 0; i < model_params_.layer_count; ++i) {
      pasts_.push_back(OrtValue::CreateTensor(allocator, past_shape, std::size(past_shape), past_type));

      char string[32];
      snprintf(string, std::size(string), "past_%d", i);
      input_name_strings_.push_back(string);
    }
  } else {
    assert(false);
  }

  // Allocate space for logits (only works if we know the shape)
  {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, model_params_.logits_uses_seq_len ? input_ids_shape[1] : 1, model_params_.vocab_size};
    logits_ = OrtValue::CreateTensor(allocator, logits_shape, std::size(logits_shape), past_type);
    outputs_.push_back(logits_.get());
  }

  {
    int64_t present_shape[] = {2, search_params_.batch_size * search_params_.num_beams, model_params_.head_count, input_ids_shape[1], model_params_.hidden_size};
    outputs_.reserve(model_params_.layer_count);

    for (int i = 0; i < model_params_.layer_count; ++i) {
      presents_.push_back(OrtValue::CreateTensor(allocator, present_shape, std::size(present_shape), past_type));
      outputs_.push_back(presents_.back().get());

      char string[32];
      snprintf(string, std::size(string), "present_%d", i);
      output_name_strings_.push_back(string);
    }
  }

  for (auto& input_name : input_name_strings_)
    input_names_.push_back(input_name.c_str());
  for (auto& output_name : output_name_strings_)
    output_names_.push_back(output_name.c_str());

  io_binding_decode_ = OrtIoBinding::Create(*session_decode_);

  for (size_t i=0;i<inputs_.size();i++)
    io_binding_decode_->BindInput(input_names_[i], *inputs_[i]);
  for (size_t i = 1; i < outputs_.size(); i++)
    io_binding_decode_->BindOutput(output_names_[i], *outputs_[i]);
}

std::span<const ScoreType> Gpt::GetLogits() {
  auto type_shape = logits_->GetTensorTypeAndShapeInfo();
  auto shape = type_shape->GetShape();
  assert(type_shape->GetShape().size() == 3);

  return {logits_->GetTensorData<ScoreType>(), type_shape->GetElementCount()};
}

void Gpt::Run(std::span<const int32_t> next_tokens, std::span<const int32_t> next_indices, int current_length) {
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

try {
    session_decode_->Run(nullptr, input_names_.data(), inputs_.data(), input_names_.size(), output_names_.data(), outputs_.data(), output_names_.size());
//    session_decode_->Run(nullptr, *io_binding_decode_);
//    logits_ = std::move(io_binding_decode_->GetOutputValues()[0]);
}
    catch (const Ort::Exception &e) {
      std::cout << e.what() << std::endl;
    }
}

void Gpt::UpdateInputs(std::span<const int32_t> next_tokens, std::span<const int32_t> beam_indices, int current_length) {
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();

  // The following updates inputs for subgraph

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(next_tokens.size());
  int64_t dims[] = {batch_beam_size, 1};
  std::unique_ptr<OrtValue> input_ids = OrtValue::CreateTensor<int32_t>(allocator, dims, std::size(dims));
  int32_t* input_ids_data = input_ids->GetTensorMutableData<int32_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    input_ids_data[i] = next_tokens[i];
  }
  expanded_input_ids_=std::move(input_ids);
  inputs_[0] = expanded_input_ids_.get();

  // Update position IDs
  inputs_[1] = next_positions_tensor_.get();
  {
    int32_t* position_data = next_positions_.data();
    for (int i = 0; i < batch_beam_size; i++) {
      position_data[i]=current_length-1;
    }
  }

  // Update attention mask
  const int32_t* old_mask_data = expanded_attention_mask_->GetTensorMutableData<int32_t>();
  int64_t mask_dims[] = {batch_beam_size, current_length};
  auto attention_mask = OrtValue::CreateTensor<int32_t>(allocator, mask_dims, std::size(mask_dims));
  int32_t* mask_data = attention_mask->GetTensorMutableData<int32_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < current_length - 1; j++) {
      mask_data[i * current_length + j] = old_mask_data[i * (current_length - 1) + j];
    }
    mask_data[i * current_length + current_length - 1] = 1;
  }
  expanded_attention_mask_=std::move(attention_mask);
  inputs_[2]=expanded_attention_mask_.get();

  // Update logits
  if (model_params_.logits_uses_seq_len)
  {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, 1, model_params_.vocab_size};
    logits_ = OrtValue::CreateTensor(allocator, logits_shape, std::size(logits_shape), Ort::TypeToTensorType<ScoreType>::type);
    outputs_[0]=logits_.get();
  }

#if 0
  if (past_present_share_buffer) {
    int32_t* past_seq_len_data = const_cast<int32_t*>(next_inputs.back().Get<Tensor>().Data<int32_t>());
    *past_seq_len_data = past_sequence_len;
    return Status::OK();
  }
#endif

  // feed present_* output to past_* inputs one by one
  int64_t present_shape[] = {2, batch_beam_size, model_params_.head_count, current_length, model_params_.hidden_size};

  if (beam_indices.empty()) {  // Update past state
    // If this is the first iteration it'll have an empty past, swap out the non empty past states for the future
    if (inputs_[3] == empty_past_.get()) {
      for (size_t i = 0; i < model_params_.layer_count; i++)
        inputs_[i + 3] = pasts_[i].get();
    }

    for (size_t i = 0; i < model_params_.layer_count; i++) {
      pasts_[i]=std::move(presents_[i]);
      inputs_[i + 3] = pasts_[i].get();

      presents_[i] = OrtValue::CreateTensor<float>(allocator, present_shape, std::size(present_shape));
      outputs_[i + 1] = presents_[i].get();
    }
  } else {
    for (size_t i = 0; i < model_params_.layer_count; i++) {
      PickPastState(allocator, i, beam_indices);

      presents_[i] = OrtValue::CreateTensor<float>(allocator, present_shape, std::size(present_shape));
      outputs_[i + 1] = presents_[i].get();
    }
  }
}

// Copy present state to past state
void Gpt::PickPastState(OrtAllocator& allocator, size_t index, std::span<const int32_t> beam_indices) {
  const OrtValue& present = *presents_[index];

  // shape is (2, batch_beam_size, 12, past_seq_len, 64)
  auto past_shape_info = present.GetTensorTypeAndShapeInfo();
  auto past_shape = past_shape_info->GetShape();
  auto block_size_per_beam = past_shape[2] * past_shape[3] * past_shape[4];
  auto past_key_size = past_shape[1] * past_shape[2] * past_shape[3] * past_shape[4];

  // Create a tensor with same shape.
  auto past = OrtValue::CreateTensor<ScoreType>(allocator, past_shape.data(), past_shape.size());

  auto past_span = std::span<ScoreType>(past->GetTensorMutableData<ScoreType>(), past_shape_info->GetElementCount());
  auto present_span = std::span<const ScoreType>(present.GetTensorData<ScoreType>(), past_shape_info->GetElementCount());
  for (size_t j = 0; j < beam_indices.size(); j++) {
    int32_t beam_index = beam_indices[j];
    std::span<const ScoreType> present_key = present_span.subspan(beam_index * block_size_per_beam, block_size_per_beam);
    std::span<const ScoreType> present_value = present_span.subspan(past_key_size + beam_index * block_size_per_beam, block_size_per_beam);

    std::span<ScoreType> past_key = past_span.subspan(j * block_size_per_beam, block_size_per_beam);
    std::span<ScoreType> past_value = past_span.subspan(past_key_size + j * block_size_per_beam, block_size_per_beam);
    copy(present_key, past_key);
    copy(present_value, past_value);
  }

  pasts_[index] = std::move(past);
  inputs_[index + 3] = pasts_[index].get();
}

}