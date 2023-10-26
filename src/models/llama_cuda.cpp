#include "../generators.h"
#include "../search.h"
#include "llama_cuda.h"
#include "debugging.h"
#include <iostream>
#include <numeric>

namespace Generators {

Llama_Cuda::Llama_Cuda(OrtEnv& ort_env, const ORTCHAR_T* decode_path, cudaStream_t cuda_stream)
 : cuda_stream_{cuda_stream},
  allocator_cpu_{Ort::Allocator::GetWithDefaultOptions()}
{ 
  auto session_options = OrtSessionOptions::Create();
  OrtCUDAProviderOptions cuda_options;
  cuda_options.has_user_compute_stream=true;
  cuda_options.user_compute_stream=cuda_stream;
  session_options->AppendExecutionProvider_CUDA(cuda_options);

  try {
    session_decode_ = OrtSession::Create(ort_env, decode_path, session_options.get());
    memory_info_cuda_ = OrtMemoryInfo::Create("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    allocator_cuda_ = Ort::Allocator::Create(*session_decode_, *memory_info_cuda_);

  } catch (const Ort::Exception &e) {
    std::cout << e.what() << std::endl;
    throw;
  }

  GetModelParams(model_params_, *session_decode_);
}

void Llama_Cuda::CreateInputs(std::span<int32_t> sequence_lengths, const SearchParams& search_params) {
  search_params_ = search_params;

  // Reset the state
  first_run_ = true;
  next_positions_buffer_.reset();

  pasts_.clear();
  input_name_strings_.clear();
  input_names_.clear();
  inputs_.clear();

  presents_.clear();
  output_name_strings_.clear();
  output_names_.clear();
  outputs_.clear();

  // Allocate position_ids and attention_mask based on shape of input_ids
  auto element_type = Ort::TypeToTensorType<int64_t>::type;

  // Use original input_ids. This requires the input_ids for subgraph is also int32.
  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  // To avoid cloning input_ids, we use const_cast here since this function does not change its content.
  int64_t input_ids_shape[] = {search_params_.batch_size, search_params_.sequence_length};
  input_ids_ = OrtValue::CreateTensor<int64_t>(*allocator_cuda_, input_ids_shape, std::size(input_ids_shape));
  auto input_ids_count = input_ids_shape[0] * input_ids_shape[1];
  std::vector<int64_t> cpu_input_ids(input_ids_count);
  auto *p_data = cpu_input_ids.data();
  for (auto v : search_params_.input_ids)
    *p_data++ = v;

  // Copy input_ids into gpu memory. This requires the input_ids for subgraph is also int32.
  auto* input_ids_data = input_ids_->GetTensorMutableData<int64_t>();
  cudaMemcpy(input_ids_data, cpu_input_ids.data(), input_ids_count*sizeof(int64_t), cudaMemcpyHostToDevice);

  position_ids_ = OrtValue::CreateTensor<int64_t>(*allocator_cuda_, input_ids_shape, std::size(input_ids_shape));

  int64_t position_shape[] = {search_params_.batch_size * search_params_.num_beams, 1};
  next_positions_ = Allocate<int64_t>(*allocator_cuda_, position_shape[0], next_positions_buffer_);
  
  std::vector<int64_t> cpu_position_ids(search_params_.sequence_length);
  std::iota(cpu_position_ids.begin(), cpu_position_ids.end(), 0);

  for (int i = 0; i < search_params_.batch_size * search_params_.num_beams; i++) {
    cudaMemcpy(next_positions_.data() + i * cpu_position_ids.size(), cpu_position_ids.data(), cpu_position_ids.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  }
  next_positions_tensor_ = OrtValue::CreateTensor<int64_t>(allocator_cuda_->GetInfo(), next_positions_.data(), next_positions_.size(), position_shape, std::size(position_shape));
  
  void* attn_mask_value = nullptr;
  attention_mask_ = OrtValue::CreateTensor<int64_t>(*allocator_cuda_, input_ids_shape, std::size(input_ids_shape));
  
  cuda_unique_ptr<int32_t> sequence_lengths_cuda = CudaMallocArray<int32_t>(sequence_lengths.size());

  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  int64_t* mask_data = attention_mask_->GetTensorMutableData<int64_t>();
  int64_t* position_data = position_ids_->GetTensorMutableData<int64_t>();
  cuda::LaunchGpt_InitAttentionMask(attn_mask_value ? nullptr : mask_data, position_data, sequence_lengths_cuda.get(), input_ids_data, search_params_.batch_size, search_params_.num_beams, search_params_.sequence_length, search_params_.pad_token_id, cuda_stream_);
  cudaMemcpy(sequence_lengths.data(), sequence_lengths_cuda.get(), sequence_lengths.size_bytes(), cudaMemcpyDeviceToHost);

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
  int64_t empty_past_shape[] = {search_params_.batch_size * search_params_.num_beams, model_params_.head_count, 0, model_params_.hidden_size};
  empty_past_ = OrtValue::CreateTensor(*allocator_cuda_, empty_past_shape, std::size(empty_past_shape), past_type);
  for (int i = 0; i < model_params_.layer_count * 2; i++)
    inputs_.push_back(empty_past_.get());

  // Initialize non empty past states
  int64_t past_shape[] = {search_params_.batch_size * search_params_.num_beams, model_params_.head_count, input_ids_shape[1], model_params_.hidden_size};
  pasts_.resize(model_params_.layer_count * 2);

  // The remaining inputs are past state.
  for (int i = 0; i < model_params_.layer_count; ++i) {
    char string[32];
    snprintf(string, std::size(string), "past_key_values.%d.key", i);
    input_name_strings_.push_back(string);

    snprintf(string, std::size(string), "past_key_values.%d.value", i);
    input_name_strings_.push_back(string);
  }

  // Allocate space for logits (only works if we know the shape)
  {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, model_params_.logits_uses_seq_len ? input_ids_shape[1] : 1, model_params_.vocab_size};
    logits_ = OrtValue::CreateTensor(*allocator_cuda_, logits_shape, std::size(logits_shape), past_type);
    outputs_.push_back(logits_.get());
  }

  {
    int64_t present_shape[] = {search_params_.batch_size * search_params_.num_beams, model_params_.head_count, input_ids_shape[1], model_params_.hidden_size};
    outputs_.reserve(model_params_.layer_count * 2);

    for (int i = 0; i < model_params_.layer_count; ++i) {
      presents_.push_back(OrtValue::CreateTensor(*allocator_cuda_, present_shape, std::size(present_shape), past_type));
      outputs_.push_back(presents_.back().get());
      presents_.push_back(OrtValue::CreateTensor(*allocator_cuda_, present_shape, std::size(present_shape), past_type));
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

std::span<const ScoreType> Llama_Cuda::GetLogits() {
  auto type_shape = logits_->GetTensorTypeAndShapeInfo();
  auto shape = type_shape->GetShape();
  assert(type_shape->GetShape().size() == 3);

  return {logits_->GetTensorData<ScoreType>(), type_shape->GetElementCount()};
}

void Llama_Cuda::Run(std::span<const int32_t> next_tokens, int current_length) {
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

  try {
    session_decode_->Run(nullptr, input_names_.data(), inputs_.data(), input_names_.size(), output_names_.data(), outputs_.data(), output_names_.size());
  }
  catch (const Ort::Exception &e) {
    std::cout << e.what() << std::endl;
  }
}

void Llama_Cuda::UpdateInputs(std::span<const int32_t> next_tokens, int current_length) {
  assert(search_params_.num_beams==1);

  // The following updates inputs for subgraph

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(next_tokens.size());
  int64_t dims[] = {batch_beam_size, 1};
  std::unique_ptr<OrtValue> input_ids = OrtValue::CreateTensor<int64_t>(*allocator_cuda_, dims, std::size(dims));
  
  std::vector<int32_t> cpu_next_tokens_int32(batch_beam_size);
  std::vector<int64_t> cpu_next_tokens_int64(batch_beam_size);
  cudaMemcpyAsync(cpu_next_tokens_int32.data(), next_tokens.data(), batch_beam_size * sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream_);
  auto *p_data = cpu_next_tokens_int64.data();
  for (auto v : cpu_next_tokens_int32)
    *p_data++ = v;

  int64_t* input_ids_data = input_ids->GetTensorMutableData<int64_t>();
  cudaMemcpyAsync(input_ids_data, cpu_next_tokens_int64.data(), batch_beam_size * sizeof(int64_t), cudaMemcpyHostToDevice, cuda_stream_);
  expanded_input_ids_=std::move(input_ids);
  inputs_[0] = expanded_input_ids_.get();

  // Update position IDs
  inputs_[1] = next_positions_tensor_.get();
  cuda::LaunchGpt_UpdatePositionIds(next_positions_.data(), batch_beam_size, current_length, cuda_stream_);

  // Update attention mask
  const int64_t* old_mask_data = expanded_attention_mask_->GetTensorMutableData<int64_t>();
  int64_t mask_dims[] = {batch_beam_size, current_length};
  auto attention_mask = OrtValue::CreateTensor<int64_t>(*allocator_cuda_, mask_dims, std::size(mask_dims));
  int64_t* mask_data = attention_mask->GetTensorMutableData<int64_t>();
  cuda::LaunchGpt_UpdateMask(mask_data, old_mask_data, batch_beam_size, current_length, cuda_stream_);
  expanded_attention_mask_=std::move(attention_mask);
  inputs_[2]=expanded_attention_mask_.get();

  // Update logits
  if (model_params_.logits_uses_seq_len)
  {
    int64_t logits_shape[] = {search_params_.batch_size * search_params_.num_beams, 1, model_params_.vocab_size};
    logits_ = OrtValue::CreateTensor(*allocator_cuda_, logits_shape, std::size(logits_shape), Ort::TypeToTensorType<ScoreType>::type);
    outputs_[0]=logits_.get();
  }

  // feed present_* output to past_* inputs one by one
  int64_t present_shape[] = {batch_beam_size, model_params_.head_count, current_length, model_params_.hidden_size};

  for (size_t i = 0; i < model_params_.layer_count * 2; i++) {
    pasts_[i]=std::move(presents_[i]);
    inputs_[i + 3] = pasts_[i].get();

    presents_[i] = OrtValue::CreateTensor<float>(*allocator_cuda_, present_shape, std::size(present_shape));
    outputs_[i + 1] = presents_[i].get();
  }

}

} // Generators