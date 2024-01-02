#include "../generators.h"
#include "../search.h"
#include "gpt_cpu.h"
#include "debugging.h"

namespace Generators {

size_t GetOrtTypeSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return sizeof(Ort::Float16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return sizeof(Ort::BFloat16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return sizeof(double);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return sizeof(int8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return sizeof(uint8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return sizeof(int16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return sizeof(uint16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return sizeof(int32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return sizeof(uint32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return sizeof(int64_t); 
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return sizeof(uint64_t); 
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return sizeof(bool);
    default:
      throw std::runtime_error("Unsupported ONNXTensorElementDataType in GetTypeSize");
  }
}

std::unique_ptr<OrtValue> ExpandInputs(std::unique_ptr<OrtValue>& input, int num_beams, OrtAllocator& allocator, DeviceType device_type, cudaStream_t cuda_stream) {
  // Input shape (batch_size, sequence_length). The input is required with data type T.
  // Output shape (batch_size * num_beams, sequence_length)

  if (num_beams==1)
    return std::move(input);

  auto input_type_info = input->GetTensorTypeAndShapeInfo();
  auto element_type = input_type_info->GetElementType();
  auto element_size = GetOrtTypeSize(element_type);
  auto input_shape = input_type_info->GetShape();
  const int64_t batch_size = input_shape[0];
  const int64_t data_size_bytes = input_type_info->GetElementCount() * element_size / batch_size;

  input_shape[0] *= num_beams;

  auto expanded = OrtValue::CreateTensor(allocator, input_shape, element_type);

  auto input_data = reinterpret_cast<const uint8_t*>(input->GetTensorRawData());
  auto expanded_data = reinterpret_cast<uint8_t*>(expanded->GetTensorMutableRawData());
  auto target = expanded_data;

  switch (device_type) {
    case DeviceType::CPU:
      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_beams; j++) {
          memcpy(target, input_data + i * data_size_bytes, data_size_bytes);
          target += data_size_bytes;
        }
      }
      break;

#if USE_CUDA
    case DeviceType::CUDA:
      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_beams; j++) {
          cudaMemcpyAsync(target, input_data + i * data_size_bytes, data_size_bytes, cudaMemcpyHostToDevice, cuda_stream);
          target += data_size_bytes;
        }
      }
      break;
#endif
    default:
      throw std::runtime_error("ExpandInputs - Unsupported device type");
  }
  return expanded;
}

Gpt_State::Gpt_State(Gpt_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const SearchParams& search_params)
    : search_params_{search_params},
      model_{&model},
      kv_cache_{search_params, model.config_, allocator_cpu_, model.cuda_stream_, model.score_type_} {
  assert(model.score_type_ == Ort::TypeToTensorType<float>::type);
  int64_t input_ids_shape[] = {search_params_.batch_size, search_params_.sequence_length};
  cpu_span<int32_t> sequence_lengths = sequence_lengths_unk;

  // Allocate position_ids and attention_mask based on shape of input_ids
  auto element_type = Ort::TypeToTensorType<int32_t>::type;

  const OrtMemoryInfo& location_cpu = allocator_cpu_.GetInfo();

  // Use original input_ids. This requires the input_ids for subgraph is also int32.
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  // To avoid cloning input_ids, we use const_cast here since this function does not change its content.
  input_ids_ = OrtValue::CreateTensor<int32_t>(location_cpu, std::span<int32_t>(const_cast<int32_t*>(search_params_.input_ids.data()), input_ids_shape[0] * input_ids_shape[1]), input_ids_shape);
  position_ids_ = OrtValue::CreateTensor<int32_t>(allocator_cpu_, input_ids_shape);

  int64_t position_shape[] = {search_params_.batch_size * search_params_.num_beams, 1};
  next_positions_ = Allocate<int32_t>(allocator_cpu_, position_shape[0], next_positions_buffer_);
  memset(next_positions_.data(), 0, next_positions_.size_bytes());
  next_positions_tensor_ = OrtValue::CreateTensor<int32_t>(location_cpu, next_positions_, position_shape);

  attention_mask_ = OrtValue::CreateTensor<int32_t>(allocator_cpu_, input_ids_shape);

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
      if (*word_id == model_->config_.pad_token_id) {
        *mask = 0;
        *position = 0;
      } else {
        *mask = 1;
        *position = abs_position;
        abs_position++;
      }
    }

    for (int k = 0; k < search_params_.num_beams; k++) {
      sequence_lengths[i * search_params_.num_beams + k] = abs_position;
    }
  }

  // Expand (batch_size, sequence_length) to (batch_size * num_beams, sequence_length)
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

  for (int i = 0; i < model_->layer_count_; i++) {
    inputs_.push_back(kv_cache_.empty_past_.get());
    input_names_.push_back(kv_cache_.input_name_strings_[i].c_str());
    outputs_.push_back(kv_cache_.presents_[i].get());
    output_names_.push_back(kv_cache_.output_name_strings_[i].c_str());
  }
}

RoamingArray<float> Gpt_State::Run(int current_length, RoamingArray<int32_t> next_tokens_unk, RoamingArray<int32_t> next_indices_unk) {
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

  return cpu_span<float>{logits_->GetTensorMutableData<float>(), type_shape->GetElementCount()};
}

void Gpt_State::UpdateInputs(cpu_span<const int32_t> next_tokens, cpu_span<const int32_t> beam_indices, int current_length) {
  assert(search_params_.num_beams == 1 || !beam_indices.empty());  // We require beam_indices if we're a beam search
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(next_tokens.size());
  int64_t dims[] = {batch_beam_size, 1};
  std::unique_ptr<OrtValue> input_ids = OrtValue::CreateTensor<int32_t>(allocator, dims);
  int32_t* input_ids_data = input_ids->GetTensorMutableData<int32_t>();
  for (int i = 0; i < batch_beam_size; i++) {
    input_ids_data[i] = next_tokens[i];
  }
  expanded_input_ids_ = std::move(input_ids);
  inputs_[0] = expanded_input_ids_.get();

  // Update position IDs
  inputs_[1] = next_positions_tensor_.get();
  {
    int32_t* position_data = next_positions_.data();
    for (int i = 0; i < batch_beam_size; i++) {
      position_data[i] = current_length - 1;
    }
  }

  // Update attention mask
  const int32_t* old_mask_data = expanded_attention_mask_->GetTensorMutableData<int32_t>();
  int64_t mask_dims[] = {batch_beam_size, current_length};
  auto attention_mask = OrtValue::CreateTensor<int32_t>(allocator, mask_dims);
  int32_t* mask_data = attention_mask->GetTensorMutableData<int32_t>();
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
    logits_ = OrtValue::CreateTensor(allocator, logits_shape, Ort::TypeToTensorType<ScoreType>::type);
    outputs_[0] = logits_.get();
  }

  kv_cache_.Update(beam_indices, current_length);
  for (size_t i = 0; i < model_->layer_count_; i++) {
    inputs_[i + 3] = kv_cache_.pasts_[i].get();
    outputs_[i + 1] = kv_cache_.presents_[i].get();
  }
}

}  // namespace Generators