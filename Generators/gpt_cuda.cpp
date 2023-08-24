#include "Generators.h"
#include <cuda_runtime.h>
#include "gpt_cuda.h"

namespace Generators {

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
  ORT_ENFORCE(element_type == Ort::TypeToTensorType<T>::type);

  expanded = OrtValue::CreateTensor<T>(allocator, dims, std::size(dims));

  const T* input_data = input.GetTensorData<T>();
  T* expanded_data = expanded->GetTensorMutableData<T>();
  T* target = expanded_data;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_beams; j++) {
      cudaMemcpyAsync(target, input_data + i * sequence_length, sizeof(T) * SafeInt<size_t>(sequence_length), cudaMemcpyHostToDevice, cuda_stream);
      target += sequence_length;
    }
  }
}

Gpt_Cuda::Gpt_Cuda(OrtEnv& ort_env, const ORTCHAR_T* decode_path, cudaStream_t cuda_stream)
 : cuda_stream_{cuda_stream},
  allocator_cpu_{Ort::Allocator::GetWithDefaultOptions()}
{
  auto session_options = OrtSessionOptions::Create();
  OrtCUDAProviderOptions cuda_options;
  cuda_options.has_user_compute_stream=true;
  cuda_options.user_compute_stream=cuda_stream;
  session_options->AppendExecutionProvider_CUDA(cuda_options);

  session_decode_ = OrtSession::Create(ort_env, decode_path, session_options.get());

  memory_info_cuda_ = OrtMemoryInfo::Create("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
  allocator_cuda_ = Ort::Allocator::Create(*session_decode_, *memory_info_cuda_);
}

void Gpt_Cuda::CreateInputs(std::span<int32_t> sequence_lengths, const SearchParams& params) {
  params_ = params;

  // Allocate position_ids and attention_mask based on shape of input_ids
  auto element_type = Ort::TypeToTensorType<int32_t>::type;

  // Current shape is (batch_size, sequence_length)
  // Note that we will expand it to (batch_size * num_beams, sequence_length) later.
  // To avoid cloning input_ids, we use const_cast here since this function does not change its content.
  int64_t input_ids_shape[] = {params_.batch_size, params_.sequence_length};
  input_ids_ = OrtValue::CreateTensor<int32_t>(*allocator_cuda_, input_ids_shape, std::size(input_ids_shape));
  auto input_ids_count = input_ids_shape[0] * input_ids_shape[1];
  auto* input_ids_data = input_ids_->GetTensorMutableData<int32_t>();

  // Copy input_ids into gpu memory. This requires the input_ids for subgraph is also int32.
  cudaMemcpy(input_ids_data, params_.input_ids, input_ids_count*sizeof(int32_t), cudaMemcpyHostToDevice);

  position_ids_ = OrtValue::CreateTensor<int32_t>(*allocator_cuda_, input_ids_shape, std::size(input_ids_shape));

  int64_t position_shape[] = {params_.batch_size * params_.num_beams, 1};
  next_positions_ = AllocateBuffer<int32_t>(allocator_cuda_.get(), next_positions_buffer_, position_shape[0]);
  cudaMemset(next_positions_.data(), 0, next_positions_.size_bytes());
  next_positions_tensor_ = OrtValue::CreateTensor<int32_t>(allocator_cuda_->GetInfo(), next_positions_.data(), next_positions_.size(), position_shape, std::size(position_shape));

  void* attn_mask_value = nullptr;  // TODO: Temporary hack until needed
#if 0
  attention_mask_;
  if (attn_mask_value != nullptr) {
    const Tensor& attn_mask = attn_mask_value->Get<Tensor>();
    Tensor::InitOrtValue(element_type, input_ids_shape, const_cast<Tensor*>(&attn_mask)->MutableData<int32_t>(),
                         allocator->Info(), attention_mask);
  } else {
#endif
  attention_mask_ = OrtValue::CreateTensor<int32_t>(*allocator_cuda_, input_ids_shape, std::size(input_ids_shape));

  cuda_unique_ptr<int32_t> sequence_lengths_cuda = CudaMallocArray<int32_t>(sequence_lengths.size());

  // Set attention mask to be 0 for pad tokens, and 1 for all other tokens.
  // Set position id to be 0 for pad tokens, and accumulated sum of mask in a batch for other tokens
  int32_t* mask_data = attention_mask_->GetTensorMutableData<int32_t>();
  int32_t* position_data = position_ids_->GetTensorMutableData<int32_t>();
  LaunchGpt_InitAttentionMask(attn_mask_value ? nullptr : mask_data, position_data, sequence_lengths_cuda.get(), input_ids_data, params_.batch_size, params_.num_beams, params_.sequence_length, params_.pad_token_id, cuda_stream_);
  cudaMemcpy(sequence_lengths.data(), sequence_lengths_cuda.get(), sequence_lengths.size_bytes(), cudaMemcpyDeviceToHost);

  // Expand (batch_size, sequence_length) to (batch_size * num_beams, sequence_length)
  // TODO(tianleiwu): Try expand outputs after first subgraph call instead. That may get better performance.
  if (params_.num_beams == 1) {
    expanded_input_ids_ = std::move(input_ids_);
    expanded_position_ids_ = std::move(position_ids_);
    expanded_attention_mask_ = std::move(attention_mask_);
  } else {
    ExpandInputs<int32_t>(*input_ids_, params_.num_beams, *allocator_cuda_, expanded_input_ids_, cuda_stream_);
    ExpandInputs<int32_t>(*position_ids_, params_.num_beams, *allocator_cuda_, expanded_position_ids_, cuda_stream_);
    ExpandInputs<int32_t>(*attention_mask_, params_.num_beams, *allocator_cuda_, expanded_attention_mask_, cuda_stream_);
  }

  for (auto* input : {expanded_input_ids_.get(), expanded_position_ids_.get(), expanded_attention_mask_.get()})
    inputs_.push_back(input);
  for (auto* name : {"input_ids", "position_ids", "attention_mask"})
    input_name_strings_.push_back(name);

  output_name_strings_.push_back("logits");

  auto past_type = Ort::TypeToTensorType<ScoreType>::type;
  if (!past_present_share_buffer_) {
    // Initialize empty past state
    int64_t empty_past_shape[] = {2, params_.batch_size * params_.num_beams, c_num_heads, 0, c_head_size};
    empty_past_ = OrtValue::CreateTensor(*allocator_cuda_, empty_past_shape, std::size(empty_past_shape), past_type);
    for (int i = 0; i < c_counts; i++)
      inputs_.push_back(empty_past_.get());

    // Initialize non empty past states
    int64_t past_shape[] = {2, params_.batch_size * params_.num_beams, c_num_heads, input_ids_shape[1], c_head_size};
    // The remaining inputs are past state.
    for (int i = 0; i < c_counts; ++i) {
      pasts_[i] = OrtValue::CreateTensor(*allocator_cuda_, past_shape, std::size(past_shape), past_type);

      char string[32];
      snprintf(string, std::size(string), "past_%d", i);
      input_name_strings_.push_back(string);
    }
  } else {
    assert(false);
  }

  {
    int64_t logits_shape[] = {params_.batch_size * params_.num_beams, 1, c_vocab_size};
    logits_ = OrtValue::CreateTensor(*allocator_cuda_, logits_shape, std::size(logits_shape), past_type);
    outputs_.push_back(logits_.get());
  }
  {
    int64_t present_shape[] = {2, params_.batch_size * params_.num_beams, c_num_heads, input_ids_shape[1], c_head_size};

    for (int i = 0; i < c_counts; ++i) {
      presents_[i] = OrtValue::CreateTensor(*allocator_cuda_, present_shape, std::size(present_shape), past_type);
      outputs_.push_back(presents_[i].get());

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
}

void Gpt_Cuda::Run(std::span<const int32_t> next_tokens, std::span<const int32_t> next_indices, int current_length) {
  if (first_run_)
    first_run_ = false;
  else
    UpdateInputs(next_tokens, next_indices, current_length);

  io_binding_decode_->ClearBoundInputs();
  io_binding_decode_->ClearBoundOutputs();
  io_binding_decode_->SynchronizeInputs();
  io_binding_decode_->SynchronizeOutputs();

  for (size_t i = 0; i < inputs_.size(); i++)
    io_binding_decode_->BindInput(input_names_[i], *inputs_[i]);
  for (size_t i = 0; i < outputs_.size(); i++)
    io_binding_decode_->BindOutput(output_names_[i], *outputs_[i]);

//  session_decode_->Run(nullptr, input_names_.data(), inputs_.data(), input_names_.size(), output_names_.data(), outputs_.data(), output_names_.size());
  session_decode_->Run(nullptr, *io_binding_decode_);

#if 0
  printf("**Inputs:\r\n");
  DumpTensors(inputs_.data(), input_names_.data(), input_names_.size(), true);
  printf("**Outputs:\r\n");
  DumpTensors(outputs_.data(), output_names_.data(), output_names_.size(), true);
#endif
}

void Gpt_Cuda::UpdateInputs(std::span<const int32_t> next_tokens, std::span<const int32_t> beam_indices, int current_length) {
  // The following updates inputs for subgraph

  // Update input_ids with next tokens.
  int batch_beam_size = static_cast<int>(next_tokens.size());
  int64_t dims[] = {batch_beam_size, 1};
  std::unique_ptr<OrtValue> input_ids = OrtValue::CreateTensor<int32_t>(*allocator_cuda_, dims, std::size(dims));
  int32_t* input_ids_data = input_ids->GetTensorMutableData<int32_t>();
  cudaMemcpyAsync(input_ids_data, next_tokens.data(), batch_beam_size * sizeof(int32_t), cudaMemcpyDeviceToDevice, cuda_stream_);
  expanded_input_ids_ = std::move(input_ids);
  inputs_[0] = expanded_input_ids_.get();

  // Update position IDs
  inputs_[1] = next_positions_tensor_.get();
  LaunchGpt_UpdatePositionIds(next_positions_.data(), batch_beam_size, current_length, cuda_stream_);

  // Update attention mask
  const int32_t* old_mask_data = expanded_attention_mask_->GetTensorMutableData<int32_t>();
  int64_t mask_dims[] = {batch_beam_size, current_length};
  auto attention_mask = OrtValue::CreateTensor<int32_t>(*allocator_cuda_, mask_dims, std::size(mask_dims));
  int32_t* mask_data = attention_mask->GetTensorMutableData<int32_t>();
  LaunchGpt_UpdateMask(mask_data, old_mask_data, batch_beam_size, current_length, cuda_stream_);
  expanded_attention_mask_ = std::move(attention_mask);
  inputs_[2] = expanded_attention_mask_.get();

#if 0
  if (past_present_share_buffer) {
    int32_t* past_seq_len_data = const_cast<int32_t*>(next_inputs.back().Get<Tensor>().Data<int32_t>());
    *past_seq_len_data = past_sequence_len;
    return Status::OK();
  }
#endif

  // feed present_* output to past_* inputs one by one
  int64_t present_shape[] = {2, batch_beam_size, c_num_heads, current_length, c_head_size};

  if (beam_indices.empty()) {  // Update past state
    // If this is the first iteration it'll have an empty past, swap out the non empty past states for the future
    if (inputs_[3] == empty_past_.get()) {
      for (size_t i = 0; i < c_counts; i++)
        inputs_[i + 3] = pasts_[i].get();
    }

    for (size_t i = 0; i < c_counts; i++) {
      pasts_[i] = std::move(presents_[i]);
      inputs_[i + 3] = pasts_[i].get();

      presents_[i] = OrtValue::CreateTensor<float>(*allocator_cuda_, present_shape, std::size(present_shape));
      outputs_[i + 1] = presents_[i].get();
    }
  } else {
    for (size_t i = 0; i < c_counts; i++) {
      PickPastState(i, beam_indices);

      presents_[i] = OrtValue::CreateTensor<float>(*allocator_cuda_, present_shape, std::size(present_shape));
      outputs_[i + 1] = presents_[i].get();
    }
  }
}

// Copy present state to past state
void Gpt_Cuda::PickPastState(size_t index, std::span<const int32_t> beam_indices) {
  const OrtValue& present = *presents_[index];

  // shape is (2, batch_beam_size, 12, past_seq_len, 64)
  auto past_shape_info = present.GetTensorTypeAndShapeInfo();
  auto past_shape = past_shape_info->GetShape();
  auto block_size_per_beam = past_shape[2] * past_shape[3] * past_shape[4];
  auto past_key_size = past_shape[1] * past_shape[2] * past_shape[3] * past_shape[4];

  // Create a tensor with same shape.
  auto past = OrtValue::CreateTensor<ScoreType>(*allocator_cuda_, past_shape.data(), past_shape.size());

  std::span<ScoreType> past_span = std::span<ScoreType>(past->GetTensorMutableData<ScoreType>(), past_shape_info->GetElementCount());
  std::span<const ScoreType> present_span = std::span<const ScoreType>(present.GetTensorData<ScoreType>(), past_shape_info->GetElementCount());
  for (size_t j = 0; j < beam_indices.size(); j++) {
    int32_t beam_index = beam_indices[j];
    std::span<const ScoreType> present_key = present_span.subspan(beam_index * SafeInt<size_t>(block_size_per_beam), block_size_per_beam);
    std::span<const ScoreType> present_value = present_span.subspan(past_key_size + beam_index * SafeInt<size_t>(block_size_per_beam), block_size_per_beam);

    std::span<ScoreType> past_key = past_span.subspan(j * SafeInt<size_t>(block_size_per_beam), block_size_per_beam);
    std::span<ScoreType> past_value = past_span.subspan(past_key_size + j * SafeInt<size_t>(block_size_per_beam), block_size_per_beam);
    cudaMemcpyAsync(past_key.data(), present_key.data(), present_key.size_bytes(), cudaMemcpyDeviceToDevice, cuda_stream_);
    cudaMemcpyAsync(past_value.data(), present_value.data(), present_value.size_bytes(), cudaMemcpyDeviceToDevice, cuda_stream_);
  }

  pasts_[index] = std::move(past);
  inputs_[index + 3] = pasts_[index].get();
}

}
