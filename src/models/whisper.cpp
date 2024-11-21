// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "whisper.h"
#include "kernels.h"
#include <vector>

#if USE_CUDA
#include <cuda_fp16.h>
#endif

namespace Generators {

WhisperModel::WhisperModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_encoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.encoder.filename)).c_str(), session_options_.get());
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());

  InitDeviceAllocator(*session_decoder_);
  session_info_->Add(*session_encoder_);
}

std::unique_ptr<State> WhisperModel::CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<WhisperState>(*this, params, sequence_lengths);
}

AudioEncoderState::AudioEncoderState(const WhisperModel& model, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  // Add audio features
  audio_features_.Add();

  // Add encoder hidden states
  auto hidden_states_shape = std::array<int64_t, 3>{params_->BatchBeamSize(), audio_features_.GetShape()[2] / 2, model_.config_->model.encoder.hidden_size};
  hidden_states_ = OrtValue::CreateTensor(*model_.allocator_device_, hidden_states_shape, audio_features_.GetType());
  outputs_.push_back(hidden_states_.get());
  output_names_.push_back(model_.config_->model.encoder.outputs.hidden_states.c_str());
}

RoamingArray<float> AudioEncoderState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  int batch_size = static_cast<int>(audio_features_.GetShape()[0]);
  State::Run(*model_.session_encoder_, *model_.run_options_, batch_size);
  return MakeDummy();
}

WhisperDecoderState::WhisperDecoderState(const WhisperModel& model, const GeneratorParams& params, const int num_frames)
    : State{params, model},
      model_{model} {
  input_ids_.Add();
  logits_.Add();
  kv_cache_.Add();

  // Add past sequence length
  if (model_.session_info_->HasInput(model_.config_->model.decoder.inputs.past_sequence_length)) {
    auto past_sequence_length_type = model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.past_sequence_length);
    auto past_sequence_length_shape = std::array<int64_t, 1>{1};
    past_sequence_length_ = OrtValue::CreateTensor(model_.allocator_cpu_, past_sequence_length_shape, past_sequence_length_type);
    auto data = past_sequence_length_->GetTensorMutableData<int32_t>();
    *data = 0;

    input_names_.push_back(model_.config_->model.decoder.inputs.past_sequence_length.c_str());
    inputs_.push_back(past_sequence_length_.get());
  }
  
  // Add cache indirection
  if (model_.session_info_->HasInput(model_.config_->model.decoder.inputs.cache_indirection)) {
    auto cache_indirection_type = model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.cache_indirection);
    auto cache_indirection_shape = std::array<int64_t, 3>{params_->batch_size, params_->search.num_beams, params_->search.max_length};
    auto cache_indirection_size = cache_indirection_shape[0] * cache_indirection_shape[1] * cache_indirection_shape[2];
    cache_indirection_ = OrtValue::CreateTensor(*model_.allocator_device_, cache_indirection_shape, cache_indirection_type);
    cache_indirection_index_ = inputs_.size();

    input_names_.push_back(model_.config_->model.decoder.inputs.cache_indirection.c_str());
    inputs_.push_back(cache_indirection_.get());

#if USE_CUDA
    auto gpu_data = gpu_span<int32_t>{cache_indirection_->GetTensorMutableData<int32_t>(), static_cast<size_t>(cache_indirection_size)};
    CudaCheck() == cudaMemsetAsync(gpu_data.data(), 0, gpu_data.size_bytes(), params_->cuda_stream);
#else
    auto cpu_data = cpu_span<int32_t>{cache_indirection_->GetTensorMutableData<int32_t>(), static_cast<size_t>(cache_indirection_size)};
    memset(cpu_data.data(), 0, cpu_data.size_bytes());
#endif
  }

  // Add output QK
  std::string output_cross_qk_name = ComposeKeyValueName(model_.config_->model.decoder.outputs.output_cross_qk_names, 0);

  if (model_.session_info_->HasOutput(output_cross_qk_name)) {
    output_cross_qk_type_ = model_.session_info_->GetOutputDataType(output_cross_qk_name);
    output_cross_qk_shape_ = std::array<int64_t, 4>{params_->BatchBeamSize(), model_.config_->model.decoder.num_attention_heads, params_->sequence_length, num_frames / 2};
    output_cross_qk_index_ = outputs_.size();
    
    for (int i = 0; i < model_.config_->model.decoder.num_hidden_layers; i++) {
      output_cross_qk_.emplace_back(OrtValue::CreateTensor(*model_.allocator_device_, output_cross_qk_shape_, output_cross_qk_type_));
      output_cross_qk_names_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.output_cross_qk_names, i));

      output_names_.emplace_back(output_cross_qk_names_.back().c_str());
      outputs_.emplace_back(output_cross_qk_.back().get());
    }
  }
}

RoamingArray<float> WhisperDecoderState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
  State::Run(*model_.session_decoder_, *model_.run_options_, batch_size);
  return logits_.Get();
}

void WhisperDecoderState::UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens_unk, RoamingArray<int32_t> beam_indices, int current_length, bool first_update) {
  input_ids_.Update(next_tokens_unk);
  kv_cache_.Update(beam_indices.GetCPU(), current_length);
  logits_.Update();

  if (past_sequence_length_) {
    auto data = past_sequence_length_->GetTensorMutableData<int32_t>();
    *data = current_length - 1;
  }

  if (cache_indirection_ && params_->search.num_beams > 1 && !first_update) {
#if USE_CUDA
    gpu_span<int32_t> beam_indices_gpu = beam_indices.GetGPU();
    cuda_unique_ptr<int32_t> beam_indices_ptr;
    if (beam_indices_gpu.empty()) {
      beam_indices_ptr = CudaMallocArray<int32_t>(params_->batch_size, &beam_indices_gpu);
      std::vector<int32_t> beam_indices_cpu(params_->batch_size, 0);
      std::iota(beam_indices_cpu.begin(), beam_indices_cpu.end(), 0);
      CudaCheck() == cudaMemcpyAsync(beam_indices_gpu.data(), beam_indices_cpu.data(),
                                     beam_indices_cpu.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice, model_.cuda_stream_);
    }
    std::unique_ptr<OrtValue> new_cache_indirection;
    auto cache_indirection_type = model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.cache_indirection);
    auto cache_indirection_shape = std::array<int64_t, 3>{params_->batch_size, params_->search.num_beams, params_->search.max_length};
    new_cache_indirection = OrtValue::CreateTensor(*model_.allocator_device_, cache_indirection_shape, cache_indirection_type);

    cuda::UpdateCacheIndirectionKernelLauncher(new_cache_indirection->GetTensorMutableData<int32_t>(),
                                               cache_indirection_->GetTensorData<int32_t>(),
                                               beam_indices_gpu.data(),
                                               params_->batch_size,
                                               params_->search.num_beams,
                                               params_->sequence_length,
                                               params_->search.max_length,
                                               current_length,
                                               model_.cuda_stream_);

    cache_indirection_ = std::move(new_cache_indirection);
    inputs_[cache_indirection_index_] = cache_indirection_.get();
#endif
  }

  if (output_cross_qk_.size() && output_cross_qk_shape_[2] != 1) {
    // Resize output QKs from (batch_size, num_heads, sequence_length, total_sequence_length) for audio processing
    // to (batch_size, num_heads, 1, total_sequence_length) for token generation
    output_cross_qk_shape_[2] = 1;
    for (int i = 0; i < model_.config_->model.decoder.num_hidden_layers; i++) {
      output_cross_qk_[i] = OrtValue::CreateTensor(*model_.allocator_device_, output_cross_qk_shape_, output_cross_qk_type_);
      outputs_[output_cross_qk_index_ + i] = output_cross_qk_[i].get();
    }
  }
}

WhisperState::WhisperState(const WhisperModel& model, const GeneratorParams& params, RoamingArray<int32_t> sequence_lengths_unk)
    : State{params, model},
      model_{model} {
  encoder_state_ = std::make_unique<AudioEncoderState>(model, params);
  cross_cache_ = std::make_unique<Cross_Cache>(*this, encoder_state_->GetNumFrames() / 2);
  encoder_state_->AddCrossCache(cross_cache_);
  decoder_state_ = std::make_unique<WhisperDecoderState>(model, params, encoder_state_->GetNumFrames());
  decoder_state_->AddCrossCache(cross_cache_);

  transpose_k_cache_buffer_ = OrtValue::CreateTensor(*model_.allocator_device_, cross_cache_->GetShape(), cross_cache_->GetType());

  auto& inputs = const_cast<GeneratorParams::Whisper&>(std::get<GeneratorParams::Whisper>(params.inputs));
  if (decoder_state_->output_cross_qk_.size() && inputs.alignment_heads) {
#if USE_CUDA
    auto alignment_heads_type_and_shape_info = inputs.alignment_heads->ort_tensor_->GetTensorTypeAndShapeInfo();
    auto alignment_heads_type = alignment_heads_type_and_shape_info->GetElementType();  // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    auto alignment_heads_shape = alignment_heads_type_and_shape_info->GetShape();
    alignment_heads_ = OrtValue::CreateTensor(*model_.allocator_device_, alignment_heads_shape, alignment_heads_type);

    // Since alignment_heads is a user input, we need to copy from CPU to GPU
    auto alignment_heads_elements = alignment_heads_type_and_shape_info->GetElementCount();
    auto alignment_heads_element_size = SizeOf(alignment_heads_type);
    auto alignment_heads_data_size = alignment_heads_elements * alignment_heads_element_size;
    cudaMemcpyAsync(alignment_heads_->GetTensorMutableRawData(), inputs.alignment_heads->ort_tensor_->GetTensorRawData(), alignment_heads_data_size, cudaMemcpyHostToDevice, model_.cuda_stream_);

    auto cross_qk_search_buffer_shape = std::array<int64_t, 4>{params_->BatchBeamSize(), alignment_heads_shape[0], params_->search.max_length, encoder_state_->GetNumFrames() / 2};
    cross_qk_search_buffer_ = OrtValue::CreateTensor(*model_.allocator_device_, cross_qk_search_buffer_shape, decoder_state_->output_cross_qk_type_);

    // Allocate GPU buffer for storing output_cross_qk_{i} pointers
    cross_qk_ptrs_buffer_ = CudaMallocArray<void*>(model_.config_->model.decoder.num_hidden_layers);
    output_cross_qk_ptrs_gpu_ = gpu_span<void*>(cross_qk_ptrs_buffer_.get(), model_.config_->model.decoder.num_hidden_layers);
#else
    alignment_heads_ = std::move(inputs.alignment_heads->ort_tensor_);
#endif
  }
}

void WhisperState::TransposeKCaches(std::vector<std::unique_ptr<OrtValue>>& kv_caches) {
  // Transpose attention K caches for `DecoderMaskedMultiHeadAttention` kernel
  // Kernel is invoked for FP16 CUDA and FP32 CPU only
  auto kv_cache_info = kv_caches[0]->GetTensorTypeAndShapeInfo();
  auto kv_cache_type = kv_cache_info->GetElementType();

  // TODO: uncomment when kernel changes are ready
  bool fp32_cpu = false; // (kv_cache_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && model_.device_type_ == DeviceType::CPU);
  bool fp16_cuda = false;
#if USE_CUDA
  fp16_cuda = (kv_cache_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && model_.device_type_ == DeviceType::CUDA);
#endif

  if (!fp32_cpu && !fp16_cuda) {
    return;
  }

  auto kv_cache_dims = kv_cache_info->GetShape();
  auto kv_cache_element_size = SizeOf(kv_cache_type);
  auto kv_cache_data_size = kv_cache_info->GetElementCount() * kv_cache_element_size;
  auto temp_buffer = transpose_k_cache_buffer_->GetTensorMutableRawData();

#if USE_CUDA
  // Use pre-allocated temporary buffer since we need to reformat the `K` caches for
  // `DecoderMaskedMultiHeadAttention` and we need some extra memory to do so.
  //
  // Since the self attention K caches are of size (batch_size, num_heads, past_sequence_length, head_size),
  // the cross attention K caches are of size (batch_size, num_heads, num_frames / 2, head_size), and
  // past_sequence_length <= max_sequence_length < num_frames / 2, we have pre-allocated a temporary buffer that is the
  // size of a cross attention K cache. This lets us use the same temporary buffer for both
  // the self attention and cross attention K caches.

  // Transpose attention K caches for `DecoderMaskedMultiHeadAttention` kernel
  for (int i = 0; i < kv_caches.size(); i += 2) {
    auto dest_data = kv_caches[i]->GetTensorMutableRawData();

    // Treat the 'K' caches as if they are of shape [B, N, max_length, head_size / x, x]
    // and transpose each 'K' cache into [B, N, head_size / x, max_length, x], where x = 16 / sizeof(T)
    int chunk_size = static_cast<int>(16 / kv_cache_element_size);
    if (chunk_size != 4 && chunk_size != 8) {
      throw std::runtime_error("ReorderPastStatesKernelLauncher only supports float32 or float16 precision");
    }

    // Copy the original 'K' caches to a temporary buffer in order to
    // use the destination buffer to store the transposed 'K' caches
    cudaMemcpyAsync(temp_buffer, dest_data, kv_cache_data_size, cudaMemcpyDeviceToDevice, model_.cuda_stream_);

    // Transpose each 'K' cache
    cuda::ReorderPastStatesKernelLauncher(dest_data,
                                          temp_buffer,
                                          static_cast<int32_t>(kv_cache_dims[0]),
                                          static_cast<int32_t>(kv_cache_dims[1]),
                                          static_cast<int32_t>(kv_cache_dims[2]),
                                          static_cast<int32_t>(kv_cache_dims[3]),
                                          chunk_size,
                                          model_.cuda_stream_);
  }
#endif
}

template <typename T>
void WhisperState::UpdateCrossQKSearchBuffer(int current_length) {
  auto output_cross_qk_size = decoder_state_->output_cross_qk_.size();
  if (output_cross_qk_size && alignment_heads_) {
#if USE_CUDA
    // Collect a GPU array of T* pointers from the vector of OrtValues to pass to the kernel
    std::vector<void*> output_cross_qk_ptrs{output_cross_qk_size, nullptr};
    for (int i = 0; i < output_cross_qk_size; i++) {
      output_cross_qk_ptrs[i] = decoder_state_->output_cross_qk_[i]->GetTensorMutableData<T>();
    }
    cudaMemcpyAsync(output_cross_qk_ptrs_gpu_.data(), output_cross_qk_ptrs.data(), output_cross_qk_size * sizeof(void*), cudaMemcpyHostToDevice, model_.cuda_stream_);

    cuda::LaunchCopyCrossQKSingleDecodeStep(model_.cuda_stream_,
                                            cross_qk_search_buffer_->GetTensorMutableData<T>(),
                                            output_cross_qk_ptrs_gpu_.data(),
                                            current_length - params_->sequence_length,
                                            params_->BatchBeamSize(),
                                            model_.config_->model.decoder.num_hidden_layers,
                                            static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[1]),
                                            static_cast<int32_t>(alignment_heads_->GetTensorTypeAndShapeInfo()->GetShape()[0]),
                                            alignment_heads_->GetTensorData<int32_t>(),
                                            static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[3]),
                                            params_->search.max_length);
#endif
  }
}

template <typename T>
void WhisperState::FinalizeCrossQK(int current_length) {
  if (decoder_state_->output_cross_qk_.size() && alignment_heads_) {
#if USE_CUDA
    // Instantiate final output for cross QKs
    auto num_alignment_heads = alignment_heads_->GetTensorTypeAndShapeInfo()->GetShape()[0];
    auto cross_qk_shape = std::array<int64_t, 5>{params_->batch_size, params_->search.num_return_sequences, num_alignment_heads, current_length, encoder_state_->GetNumFrames() / 2};
    cross_qk_final_ = OrtValue::CreateTensor(*model_.allocator_device_, cross_qk_shape, decoder_state_->output_cross_qk_type_);

    cuda::LaunchFinalizeCrossQK(model_.cuda_stream_,
                                current_length - params_->sequence_length,
                                params_->sequence_length,
                                static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[0]),
                                params_->search.num_beams,
                                params_->search.max_length,
                                static_cast<int32_t>(num_alignment_heads),
                                static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[3]),
                                cross_qk_search_buffer_->GetTensorData<T>(),
                                cross_qk_final_->GetTensorMutableData<T>(),
                                params_->search.num_return_sequences,
                                decoder_state_->cache_indirection_->GetTensorData<int32_t>());
#endif
  }
}

RoamingArray<float> WhisperState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  if (encoder_state_->first_run_) {
    // Run encoder
    encoder_state_->Run(current_length, next_tokens, next_indices);

    // Run decoder-init
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    // TODO: Transpose the K caches only when the else branch is run for the first time.
    // Otherwise the GetOutput(present_key_self_{i}) method returns transposed K caches.
    TransposeKCaches(cross_cache_->GetValues());
    TransposeKCaches(decoder_state_->kv_cache_.GetPresents());

    if (decoder_state_->output_cross_qk_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
#if USE_CUDA
      UpdateCrossQKSearchBuffer<half>(current_length);
#endif
    } else {
      UpdateCrossQKSearchBuffer<float>(current_length);
    }

    return logits;
  } else {
    // Update inputs and outputs for decoder
    decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);

    // Run decoder-with-past
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    if (decoder_state_->output_cross_qk_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
#if USE_CUDA
      UpdateCrossQKSearchBuffer<half>(current_length);
#endif
    } else {
      UpdateCrossQKSearchBuffer<float>(current_length);
    }
    first_run_ = false;

    return logits;
  }
  // Not reached
  return MakeDummy();
}

void WhisperState::Finalize(int current_length) {
  if (decoder_state_->output_cross_qk_.size() && alignment_heads_) {
    if (decoder_state_->output_cross_qk_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
#if USE_CUDA
      FinalizeCrossQK<half>(current_length);
#endif
    } else {
      FinalizeCrossQK<float>(current_length);
    }
  }
}

OrtValue* WhisperState::GetOutput(const char* name) {
  // Check if output name is in encoder state's outputs
  for (size_t i = 0; i < encoder_state_->output_names_.size(); i++) {
    if (std::strcmp(encoder_state_->output_names_[i], name) == 0) {
      return encoder_state_->outputs_[i];
    }
  }

  // Check if output name is in decoder state's outputs
  for (size_t i = 0; i < decoder_state_->output_names_.size(); i++) {
    if (std::strcmp(decoder_state_->output_names_[i], name) == 0) {
      // Note: K caches will be transposed when returned
      return decoder_state_->outputs_[i];
    }
  }

  // cross_qk_final_ is an onnxruntime-genai maintained buffer that
  // is not part of the model's outputs, so we need to check for it here.
  if (std::strcmp("cross_qk", name) == 0) {
    return cross_qk_final_.get();
  }

  // cross_qk_search_buffer_ is an onnxruntime-genai maintained buffer that
  // is not part of the model's outputs, so we need to check for it here.
  if (std::strcmp("cross_qk_search", name) == 0) {
    return cross_qk_search_buffer_.get();
  }

  return State::GetOutput(name);
};

}  // namespace Generators
