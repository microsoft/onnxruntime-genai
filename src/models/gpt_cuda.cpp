#include "../generators.h"
#include "gpt_cuda.h"
#include "debugging.h"

namespace Generators {

void ConvertFp16ToFp32(OrtAllocator& allocator, cudaStream_t stream, OrtValue& in, std::unique_ptr<OrtValue>& p_out);

template <typename T>
static void ExpandInputs(const OrtValue& input, int num_beams, OrtAllocator& allocator, std::unique_ptr<OrtValue>& expanded, cudaStream_t cuda_stream) {
  // Input shape (batch_size, sequence_length). The input is required with data type T.
  // Output shape (batch_size * num_beams, sequence_length)

  auto input_type_info = input.GetTensorTypeAndShapeInfo();
  auto input_shape = input_type_info->GetShape();
  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];

  int64_t dims[] = {batch_size * num_beams, sequence_length};

  auto element_type = input_type_info->GetElementType();
  assert(element_type == Ort::TypeToTensorType<T>::type);

  expanded = OrtValue::CreateTensor<T>(allocator, dims);

  const T* input_data = input.GetTensorData<T>();
  T* expanded_data = expanded->GetTensorMutableData<T>();
  T* target = expanded_data;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_beams; j++) {
      cudaMemcpyAsync(target, input_data + i * sequence_length, sizeof(T) * sequence_length, cudaMemcpyHostToDevice, cuda_stream);
      target += sequence_length;
    }
  }
}

Gpt_Cuda::Gpt_Cuda(Gpt_Model& model, RoamingArray<int32_t> sequence_lengths, const SearchParams& search_params)
    : search_params_{search_params}, 
      model_{&model},
      allocator_cpu_{Ort::Allocator::GetWithDefaultOptions()},
      memory_info_cuda_{OrtMemoryInfo::Create("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault)},
      allocator_cuda_{Ort::Allocator::Create(*model_->session_decoder_, *memory_info_cuda_)},
      kv_cache_{search_params, model.config_, *allocator_cuda_, model.cuda_stream_, model.score_type_} {
  // Allocate position_ids and attention_mask based on shape of input_ids
  auto element_type = Ort::TypeToTensorType<int32_t>::type;

  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  // To avoid cloning input_ids, we use const_cast here since this function does not change its content.
  int64_t input_ids_shape[] = {search_params_.batch_size, search_params_.sequence_length};
  input_ids_ = OrtValue::CreateTensor<int32_t>(*allocator_cuda_, input_ids_shape);
  auto input_ids_count = input_ids_shape[0] * input_ids_shape[1];
  auto* input_ids_data = input_ids_->GetTensorMutableData<int32_t>();

  // Copy input_ids into gpu memory. This requires the input_ids for subgraph is also int32.
  cudaMemcpy(input_ids_data, search_params_.input_ids.data(), input_ids_count * sizeof(int32_t), cudaMemcpyHostToDevice);

  position_ids_ = OrtValue::CreateTensor<int32_t>(*allocator_cuda_, input_ids_shape);

  int64_t position_shape[] = {search_params_.batch_size * search_params_.num_beams, 1};
  next_positions_ = Allocate<int32_t>(*allocator_cuda_, position_shape[0], next_positions_buffer_);
  cudaMemset(next_positions_.data(), 0, next_positions_.size_bytes());
  next_positions_tensor_ = OrtValue::CreateTensor<int32_t>(allocator_cuda_->GetInfo(), next_positions_, position_shape);

  attention_mask_ = OrtValue::CreateTensor<int32_t>(*allocator_cuda_, input_ids_shape);

  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  int32_t* mask_data = attention_mask_->GetTensorMutableData<int32_t>();
  int32_t* position_data = position_ids_->GetTensorMutableData<int32_t>();
  cuda::LaunchGpt_InitAttentionMask(mask_data, position_data, sequence_lengths.GetGPU().data(), input_ids_data, search_params_.batch_size, search_params_.num_beams, search_params_.sequence_length, search_params_.pad_token_id, model_->cuda_stream_);
  sequence_lengths.FlushGPUChanges();

  // Expand (batch_size, sequence_length) to (batch_size * num_beams, sequence_length)
  // TODO(tianleiwu): Try expand outputs after first subgraph call instead. That may get better performance.
  if (search_params_.num_beams == 1) {
    expanded_input_ids_ = std::move(input_ids_);
    expanded_position_ids_ = std::move(position_ids_);
    expanded_attention_mask_ = std::move(attention_mask_);
  } else {
    ExpandInputs<int32_t>(*input_ids_, search_params_.num_beams, *allocator_cuda_, expanded_input_ids_, model_->cuda_stream_);
    ExpandInputs<int32_t>(*position_ids_, search_params_.num_beams, *allocator_cuda_, expanded_position_ids_, model_->cuda_stream_);
    ExpandInputs<int32_t>(*attention_mask_, search_params_.num_beams, *allocator_cuda_, expanded_attention_mask_, model_->cuda_stream_);
  }

  for (auto* input : {expanded_input_ids_.get(), expanded_position_ids_.get(), expanded_attention_mask_.get()})
    inputs_.push_back(input);
  for (auto* name : {"input_ids", "position_ids", "attention_mask"})
    input_names_.push_back(name);

  output_names_.push_back("logits");

  // Allocate space for logits (only works if we know the shape)
  {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, model_->logits_uses_seq_len_ ? input_ids_shape[1] : 1, model_->vocab_size_};
    logits_ = OrtValue::CreateTensor(*allocator_cuda_, logits_shape, model_->score_type_);
    outputs_.push_back(logits_.get());
  }

  for (int i = 0; i < model_->layer_count_; i++) {
    inputs_.push_back(kv_cache_.empty_past_.get());
    input_names_.push_back(kv_cache_.input_name_strings_[i].c_str());
    outputs_.push_back(kv_cache_.presents_[i].get());
    output_names_.push_back(kv_cache_.output_name_strings_[i].c_str());
  }
}

RoamingArray<float> Gpt_Cuda::Run(int current_length, RoamingArray<int32_t> next_tokens_unk, RoamingArray<int32_t> next_indices_unk) {
  gpu_span<int32_t> next_tokens = next_tokens_unk;
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
  assert(type_shape->GetShape().size() == 3);

  if (model_->score_type_ == Ort::TypeToTensorType<Ort::Float16_t>::type) {
    ConvertFp16ToFp32(*allocator_cuda_, model_->cuda_stream_, *logits_, logits32_);
    return gpu_span<float>{logits32_->GetTensorMutableData<float>(), type_shape->GetElementCount()};
  }

  return gpu_span<float>{logits_->GetTensorMutableData<float>(), type_shape->GetElementCount()};
}

void Gpt_Cuda::UpdateInputs(std::span<const int32_t> next_tokens, std::span<const int32_t> beam_indices, int current_length) {
  assert(search_params_.num_beams == 1 || !beam_indices.empty());  // We require beam_indices if we're a beam search

  // The following updates inputs for subgraph

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(next_tokens.size());
  int64_t dims[] = {batch_beam_size, 1};
  std::unique_ptr<OrtValue> input_ids = OrtValue::CreateTensor<int32_t>(*allocator_cuda_, dims);
  int32_t* input_ids_data = input_ids->GetTensorMutableData<int32_t>();
  cudaMemcpyAsync(input_ids_data, next_tokens.data(), batch_beam_size * sizeof(int32_t), cudaMemcpyDeviceToDevice, model_->cuda_stream_);
  expanded_input_ids_ = std::move(input_ids);
  inputs_[0] = expanded_input_ids_.get();

  // Update position IDs
  inputs_[1] = next_positions_tensor_.get();
  cuda::LaunchGpt_UpdatePositionIds(next_positions_.data(), batch_beam_size, current_length, model_->cuda_stream_);

  // Update attention mask
  const int32_t* old_mask_data = expanded_attention_mask_->GetTensorMutableData<int32_t>();
  int64_t mask_dims[] = {batch_beam_size, current_length};
  auto attention_mask = OrtValue::CreateTensor<int32_t>(*allocator_cuda_, mask_dims);
  int32_t* mask_data = attention_mask->GetTensorMutableData<int32_t>();
  cuda::LaunchGpt_UpdateMask(mask_data, old_mask_data, batch_beam_size, current_length, model_->cuda_stream_);
  expanded_attention_mask_ = std::move(attention_mask);
  inputs_[2] = expanded_attention_mask_.get();

  // Update logits
  if (model_->logits_uses_seq_len_) {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, 1, model_->vocab_size_};
    logits_ = OrtValue::CreateTensor(*allocator_cuda_, logits_shape, model_->score_type_);
    outputs_[0] = logits_.get();
  }

  kv_cache_.Update(beam_indices, current_length);
  for (size_t i = 0; i < model_->layer_count_; i++) {
    inputs_[i + 3] = kv_cache_.pasts_[i].get();
    outputs_[i + 1] = kv_cache_.presents_[i].get();
  }
}

}  // namespace Generators
