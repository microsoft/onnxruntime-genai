// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "whisper.h"
#include "kernels.h"
#include <vector>

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

  // inputs_.push_back(audio_features_.get());
  // input_names_.push_back(model_.config_->model.encoder.inputs.audio_features.c_str());

  // // Get information from encoder model
  // auto audio_features_info = audio_features_->GetTensorTypeAndShapeInfo();
  // auto audio_features_shape = audio_features_info->GetShape();
  // auto model_type = audio_features_info->GetElementType();
  // num_frames_ = audio_features_shape[2];

  // Add encoder hidden states
  auto hidden_states_shape = std::array<int64_t, 3>{params_->BatchBeamSize(), audio_features_.GetShape()[2] / 2, model_.config_->model.encoder.hidden_size};
  hidden_states_ = OrtValue::CreateTensor(*model_.allocator_device_, hidden_states_shape, audio_features_.GetType());
  outputs_.push_back(hidden_states_.get());
  output_names_.push_back(model_.config_->model.encoder.outputs.hidden_states.c_str());

  // auto cross_attn_kv_cache_shape = std::array<int64_t, 4>(params_->BatchBeamSize(), model_.config_->model.decoder.num_attention_heads, num_frames_ / 2, model_.config_->model.decoder.head_size);
  // for (int i = 0; i < model_.config_->model.decoder.num_hidden_layers; i++) {
  //   // Cross attention K cache
  //   cross_cache.push_back(OrtValue::CreateTensor(*model_.allocator_device_, cross_attn_kv_cache_shape, model_type));
  //   state_.outputs_.push_back(cross_cache.back());
  //   cross_cache_names.emplace_back(ComposeKeyValueName(model_.config_->model.encoder.outputs.present_key_names, i));
  //   state_.output_names_.push_back(cross_cache_names.back());
    
  //   // Cross attention V cache
  //   cross_cache.push_back(OrtValue::CreateTensor(*model_.allocator_device_, cross_attn_kv_cache_shape, model_type));
  //   state_.outputs_.push_back(cross_cache.back());
  //   cross_cache_names.emplace_back(ComposeKeyValueName(model_.config_->model.encoder.outputs.present_value_names, i));
  //   state_.output_names_.push_back(cross_cache_names.back());
  // }
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
  // char qk_name[64];
  // snprintf(qk_name, std::size(qk_name), model_.config_->model.decoder.outputs.output_cross_qk_names.c_str(), 0);
  // output_cross_qk_name_ = std::string(qk_name);
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

  // // Add cross attention KV caches
  // auto cross_attn_kv_cache_shape = std::array<int64_t, 4>(params_->BatchBeamSize(), model_.config_->model.decoder.num_attention_heads, num_frames / 2, model_.config_->model.decoder.head_size);
  // for (int i = 0; i < model_.config_->model.decoder.num_hidden_layers; i++) {
  //   // Cross attention K cache
  //   state_.inputs_.push_back(cross_cache[2*i].get());
  //   state_.input_names_.push_back(cross_cache_names[2*i]);
    
  //   // Cross attention V cache
  //   state_.inputs_.push_back(cross_cache[2*i + 1].get());
  //   state_.input_names_.push_back(cross_cache_names[2*i + 1]);
  // }
}

RoamingArray<float> WhisperDecoderState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
  State::Run(*model_.session_decoder_, *model_.run_options_, batch_size);
  auto output = logits_.Get();
  auto& stream = Log("After WhisperDecoderState ran in WhisperDecoderState::Run");
  stream << std::endl;
  DumpCudaSpan(stream, std::span<const float>(output.GetGPU()));
  return output;
  // return MakeDummy();
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
    // Resize output QKs from (batch_size, num_heads, sequence_length, total_sequence_length)
    // to (batch_size, num_heads, 1, total_sequence_length) for token generation
    output_cross_qk_shape_[2] = 1;
    for (int i = 0; i < model_.config_->model.decoder.num_hidden_layers; i++) {
      auto new_output_cross_qk = OrtValue::CreateTensor(*model_.allocator_device_, output_cross_qk_shape_, output_cross_qk_type_);
      output_cross_qk_[i] = std::move(new_output_cross_qk);
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

    auto cross_qk_search_buffer_shape = std::array<int64_t, 4>{params_->BatchBeamSize(), alignment_heads_shape[0], params_->search.max_length, 1500};
    cross_qk_search_buffer_ = OrtValue::CreateTensor(*model_.allocator_device_, cross_qk_search_buffer_shape, decoder_state_->output_cross_qk_type_);

    // Allocate GPU buffer for storing output_cross_qk_{i} pointers
    cross_qk_ptrs_buffer_ = CudaMallocArray<float*>(model_.config_->model.decoder.num_hidden_layers);
    output_cross_qk_ptrs_gpu_ = gpu_span<float*>(cross_qk_ptrs_buffer_.get(), model_.config_->model.decoder.num_hidden_layers);
#else
    alignment_heads_ = std::move(inputs.alignment_heads->ort_tensor_);
#endif
  }

  // // TODO: Is this needed or can it be removed?
  // auto sequence_lengths = sequence_lengths_unk.GetCPU();
  // for (int i = 0; i < decoder_state_->input_ids_.GetShape()[0]; i++) {
  //   sequence_lengths[i] = static_cast<int32_t>(params_->sequence_length);
  // }

  // auto hidden_states_type = model_.session_info_->GetOutputDataType("encoder_hidden_states");
  // auto encoder_hidden_states_shape = std::array<int64_t, 3>{input_ids_.GetShape()[0], 1500, static_cast<int64_t>(model_.config_->model.decoder.num_attention_heads) * model_.config_->model.decoder.head_size};
  // encoder_hidden_states_ = OrtValue::CreateTensor(*model_.allocator_device_, encoder_hidden_states_shape, hidden_states_type);

  // input_names_.push_back("audio_features");
  // inputs_.push_back(audio_features_.get());

  // output_names_.push_back("encoder_hidden_states");
  // outputs_.push_back(encoder_hidden_states_.get());
  // cross_cache_.AddOutputs();

  // const auto kv_cache_indices = outputs_.size();
  // kv_cache_.AddEncoder();

  // {
  //   auto layer_count = model_.config_->model.decoder.num_hidden_layers;
  //   std::array<int64_t, 4> shape{params_->BatchBeamSize(), model_.config_->model.decoder.num_attention_heads, params_->sequence_length, model_.config_->model.decoder.head_size};
  //   auto type = model_.session_info_->GetOutputDataType(output_names_[kv_cache_indices]);

  //   for (int i = 0; i < layer_count * 2; i++) {
  //     init_presents_.emplace_back(OrtValue::CreateTensor(*model_.allocator_device_, shape, type));
  //     presents_.emplace_back(outputs_[kv_cache_indices + i]);
  //     outputs_[kv_cache_indices + i] = init_presents_.back().get();
  //   }
  // }
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

void WhisperState::UpdateCrossQKSearchBuffer(int current_length) {
  auto output_cross_qk_size = decoder_state_->output_cross_qk_.size();
  if (output_cross_qk_size && alignment_heads_) {
#if USE_CUDA
    // Collect a GPU array of float* pointers from the vector of OrtValues to pass to the kernel
    std::vector<float*> output_cross_qk_ptrs{output_cross_qk_size, nullptr};
    for (int i = 0; i < output_cross_qk_size; i++) {
      output_cross_qk_ptrs[i] = decoder_state_->output_cross_qk_[i]->GetTensorMutableData<float>();
    }
    cudaMemcpyAsync(output_cross_qk_ptrs_gpu_.data(), output_cross_qk_ptrs.data(), output_cross_qk_size * sizeof(output_cross_qk_ptrs[0]), cudaMemcpyHostToDevice, model_.cuda_stream_);

    cuda::LaunchCopyCrossQKSingleDecodeStep(model_.cuda_stream_,
                                            cross_qk_search_buffer_->GetTensorMutableData<float>(),
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

// RoamingArray<float> WhisperState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
//   if (encoder_state_->first_run_) {
//     // Run encoder
//     encoder_state_->Run(current_length, next_tokens, next_indices);
//     TransposeKCaches(cross_cache_->GetValues());
//   } else if (decoder_state_->first_run_) {
//     // Run decoder-init
//     decoder_state_->Run(current_length, next_tokens, next_indices);
//     // decoded_length_ = current_length;
//     TransposeKCaches(decoder_state_->kv_cache_.GetPresents());
//     return decoder_state_->logits_.Get();
//   } else {
//     // Run decoder-with-past
//     // Update inputs and outputs for decoder
//     decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length);
//     decoder_state_->Run(current_length, next_tokens, next_indices);
//     // decoded_length_ += 1;
//     UpdateCrossQKSearchBuffer(current_length);
//     return decoder_state_->logits_.Get();
//   }
//   return MakeDummy();
// }

RoamingArray<float> WhisperState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  if (encoder_state_->first_run_) {
    // Run encoder
    std::cout << "Running encoder" << std::endl;
    encoder_state_->Run(current_length, next_tokens, next_indices);
    // TransposeKCaches(cross_cache_->GetValues());

    // Run decoder-init
    std::cout << "Running decoder-init" << std::endl;
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);
    // decoded_length_ = current_length;

    // TODO: Transpose the K caches only when the else branch is run for the first time.
    // Otherwise the GetOutput(output_cross_qk_{i}) method returns transposed K caches.
    TransposeKCaches(cross_cache_->GetValues());
    TransposeKCaches(decoder_state_->kv_cache_.GetPresents());
    UpdateCrossQKSearchBuffer(current_length);
    return logits;
    // return decoder_state_->logits_.Get();
  } else {
    // Run decoder-with-past
    // Update inputs and outputs for decoder
    std::cout << "Running decoder-with-past" << std::endl;
    decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);
    // decoded_length_ += 1;
    UpdateCrossQKSearchBuffer(current_length);
    first_run_ = false;

    auto& stream = Log("After decoder-with-past ran inside WhisperState::Run");
    stream << std::endl;
    DumpCudaSpan(stream, std::span<const float>(logits.GetGPU()));
    return logits;
    // return decoder_state_->logits_.Get();
  }
  // Not reached
  return MakeDummy();
}

// RoamingArray<float> WhisperState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
//   if (encoder_state_->hidden_states_ == nullptr) {
//     // Run encoder
//     encoder_state_->Run(current_length, next_tokens, next_indices);
//     TransposeKCaches(cross_cache_->GetValues());
//     return MakeDummy();
//   } else if (first_run_) {
//     // Run decoder-init
//     decoder_state_->Run(current_length, next_tokens, next_indices);
//     decoded_length_ = current_length;
//     TransposeKCaches(decoder_state_->kv_cache_.GetPresents());
//     return decoder_state_->logits_.Get();
//   } else {
//     // Run decoder-with-past
//     // Update inputs and outputs for decoder
//     decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length);
//     decoder_state_->Run(current_length, next_tokens, next_indices);
//     decoded_length_ += 1;
//     UpdateCrossQKSearchBuffer();
//     return decoder_state_->logits_.Get();
//   }
//   // Not reached
//   return MakeDummy();
// }

// RoamingArray<float> WhisperState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
//   int batch_size = static_cast<int>(input_ids_.GetShape()[0]);

//   switch (run_state_) {
//     case RunState::Encoder_Decoder_Init:
//       State::Run(*model_.session_encoder_, *model_.run_options_, batch_size);

//       run_state_ = RunState::Decoder_First;
//       return logits_.Get();

//     case RunState::Decoder_First: {
//       auto src_shape_info = init_presents_[0]->GetTensorTypeAndShapeInfo();

//       const auto copy_data_size_all = src_shape_info->GetElementCount() * SizeOf(src_shape_info->GetElementType());

// #if USE_CUDA
//       const auto src_dims = src_shape_info->GetShape();
//       const auto src_element_type = src_shape_info->GetElementType();
//       const auto src_element_size = SizeOf(src_element_type);

//       auto dest_shape_info = presents_[0]->GetTensorTypeAndShapeInfo();
//       auto dest_dims = dest_shape_info->GetShape();
//       auto dest_element_type = dest_shape_info->GetElementType();
//       auto dest_element_size = SizeOf(dest_element_type);
//       auto dest_data_size = dest_shape_info->GetElementCount() * dest_element_size;

//       const auto copy_data_size = src_dims[2] * src_dims[3] * src_element_size;

//       // Allocate temporary buffer for when CUDA EP + FP16 precision is used because
//       // we need to reformat the `K` caches for `DecoderMaskedMultiHeadAttention`
//       // and we need some extra memory to do so.
//       //
//       // Since the self attention K caches are of size (batch_size, num_heads, past_sequence_length, head_size) with type 'float16',
//       // the cross attention K caches are of size (batch_size, num_heads, 1500, head_size) with type 'float32', and
//       // past_sequence_length <= 448 < 1500, we will allocate a temporary buffer that is the
//       // size of a cross attention K cache. This lets us use the same temporary buffer for both
//       // the self attention and cross attention K caches.

//       std::unique_ptr<OrtValue> temp_buffer;
//       auto self_attn_kv_cache_element_type = src_element_type;  // should be `float16` for the below case
//       if (self_attn_kv_cache_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && model_.device_type_ == DeviceType::CUDA) {
//         auto cross_attn_shape_info = outputs_[outputs_.size() - 1]->GetTensorTypeAndShapeInfo();
//         auto cross_attn_dims = cross_attn_shape_info->GetShape();
//         auto cross_attn_kv_cache_element_type = cross_attn_shape_info->GetElementType();  // should be `float32` for this case

//         temp_buffer = OrtValue::CreateTensor(*model_.allocator_device_, cross_attn_dims, cross_attn_kv_cache_element_type);
//       }
// #endif

//       // Copy over the hacked outputs to the real outputs
//       for (int i = 0; i < presents_.size(); i++) {
//         auto src_data = init_presents_[i]->GetTensorRawData();
//         auto dest_data = presents_[i]->GetTensorMutableRawData();

//         switch (model_.device_type_) {
// #if USE_CUDA
//           case DeviceType::CUDA:
//             if (self_attn_kv_cache_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
//               // CUDA EP + FP16 precision == `DecoderMaskedMultiHeadAttention` op is used
//               // This also means `past-present buffer sharing = true`

//               // Copy data from init_presents_[i] to presents_[i]
//               // from (batch_size, num_heads, past_sequence_length, head_size)
//               // to (batch_size, num_heads, max_sequence_length, head_size)
//               //
//               // Implemented as:
//               // real[:batch_size, :num_heads, :past_sequence_length, :head_size] = hacked
//               for (int b = 0; b < dest_dims[0] * dest_dims[1]; b++) {
//                 auto src_offset = b * src_dims[2] * src_dims[3];
//                 auto dest_offset = b * dest_dims[2] * dest_dims[3];

//                 src_offset *= src_element_size;
//                 dest_offset *= dest_element_size;
//                 cudaMemcpyAsync(reinterpret_cast<int8_t*>(dest_data) + dest_offset, reinterpret_cast<const int8_t*>(src_data) + src_offset, copy_data_size, cudaMemcpyDeviceToDevice, model_.cuda_stream_);
//               }

//               // Transpose self attention K caches for `DecoderMaskedMultiHeadAttention`
//               if (i % 2 == 0) {
//                 TransposeKCacheForDMMHA(dest_data, temp_buffer->GetTensorMutableRawData(), dest_dims,
//                                         dest_data_size, dest_element_size, model_.cuda_stream_);
//               }
//             } else {
//               cudaMemcpyAsync(dest_data, src_data, copy_data_size_all, cudaMemcpyDeviceToDevice, model_.cuda_stream_);
//             }
//             break;
// #endif
//           case DeviceType::CPU: {
//             memcpy(dest_data, src_data, copy_data_size_all);
//             break;
//           }

//           default:
//             throw std::runtime_error("Unsupported Device Type in WhisperState::Run");
//         }
//       }

// #if USE_CUDA
//       if (self_attn_kv_cache_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && model_.device_type_ == DeviceType::CUDA) {
//         // Transpose cross attention K caches for `DecoderMaskedMultiHeadAttention`

//         // Add +2 to start of loop to account for `logits` and `encoder_hidden_states` outputs
//         for (size_t i = 2 + init_presents_.size(); i < outputs_.size(); i += 2) {
//           auto dest_data = outputs_[i]->GetTensorMutableRawData();
//           dest_shape_info = outputs_[i]->GetTensorTypeAndShapeInfo();
//           dest_dims = dest_shape_info->GetShape();
//           dest_element_type = dest_shape_info->GetElementType();
//           dest_element_size = SizeOf(dest_element_type);
//           dest_data_size = dest_shape_info->GetElementCount() * dest_element_size;

//           TransposeKCacheForDMMHA(dest_data, temp_buffer->GetTensorMutableRawData(), dest_dims,
//                                   dest_data_size, dest_element_size, model_.cuda_stream_);
//         }
//       }
// #endif

//       ClearIO();

//       input_ids_.name_ = model_.config_->model.decoder.inputs.input_ids.c_str();  // Set back to default name, since we overrode it above in the encoder step
//       input_ids_.Add();
//       logits_.Add();
//       kv_cache_.Add();
//       cross_cache_.AddInputs();
//       run_state_ = RunState::Decoder;

//       if (model_.session_info_->HasInput("past_sequence_length")) {
//         past_sequence_length_ = OrtValue::CreateTensor<int32_t>(model_.allocator_cpu_, std::array<int64_t, 1>{1});
//         input_names_.push_back("past_sequence_length");
//         inputs_.push_back(past_sequence_length_.get());
//       }

//       if (model_.session_info_->HasInput("beam_width")) {
//         beam_width_ = OrtValue::CreateTensor<int32_t>(model_.allocator_cpu_, std::array<int64_t, 1>{1});
//         input_names_.push_back("beam_width");
//         inputs_.push_back(beam_width_.get());

//         auto data = beam_width_->GetTensorMutableData<int32_t>();
//         *data = params_->search.num_beams;
//       }

//       if (model_.session_info_->HasInput("cache_indirection")) {
// #if USE_CUDA
//         cache_indirection_ = OrtValue::CreateTensor<int32_t>(*model_.allocator_device_, std::array<int64_t, 3>{params_->batch_size, params_->search.num_beams, params_->search.max_length});
//         cache_indirection_index_ = inputs_.size();
//         input_names_.push_back("cache_indirection");
//         inputs_.push_back(cache_indirection_.get());

//         auto data = gpu_span<int32_t>{cache_indirection_->GetTensorMutableData<int32_t>(),
//                                       static_cast<size_t>(params_->batch_size) * params_->search.num_beams * params_->search.max_length};
//         CudaCheck() == cudaMemsetAsync(data.data(), 0, data.size_bytes(), params_->cuda_stream);
// #endif
//       }

//       if (model_.session_info_->HasOutput("output_cross_qk_0")) {
//         auto layer_count = model_.config_->model.decoder.num_hidden_layers;
//         auto type = model_.session_info_->GetOutputDataType("output_cross_qk_0");
//         std::array<int64_t, 4> shape{params_->BatchBeamSize(), model_.config_->model.decoder.num_attention_heads, 1, 1500};
//         for (int i = 0; i < layer_count; i++) {
//           char string[64];
//           snprintf(string, std::size(string), "output_cross_qk_%d", i);
//           output_cross_qk_names_.emplace_back(string);
//           output_cross_qk_.emplace_back(OrtValue::CreateTensor(*model_.allocator_device_, shape, type));

//           output_names_.emplace_back(output_cross_qk_names_.back().c_str());
//           outputs_.emplace_back(output_cross_qk_.back().get());
//         }
//       }

//       UpdateInputsOutputs(next_tokens, next_indices, current_length, false /* search_buffers */);

//       break;
//     }

//     case RunState::Decoder: {
//       bool search_buffers = true;
//       UpdateInputsOutputs(next_tokens, next_indices, current_length, search_buffers);
//       break;
//     }
//   }

//   State::Run(*model_.session_decoder_, *model_.run_options_, batch_size);
//   return logits_.Get();
// }

// void WhisperState::UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> beam_indices, int current_length) {
//   input_ids_.Update(next_tokens);
//   kv_cache_.Update(beam_indices.GetCPU(), current_length);
//   logits_.Update();

//   if (cache_indirection_) {
// #if USE_CUDA
//     gpu_span<int32_t> beam_indices_gpu = beam_indices.GetGPU();
//     cuda_unique_ptr<int32_t> beam_indices_ptr;
//     if (beam_indices_gpu.empty()) {
//       beam_indices_ptr = CudaMallocArray<int32_t>(params_->batch_size, &beam_indices_gpu);
//       std::vector<int32_t> beam_indices_cpu(params_->batch_size, 0);
//       std::iota(beam_indices_cpu.begin(), beam_indices_cpu.end(), 0);
//       CudaCheck() == cudaMemcpyAsync(beam_indices_gpu.data(), beam_indices_cpu.data(),
//                                      beam_indices_cpu.size() * sizeof(int32_t),
//                                      cudaMemcpyHostToDevice, model_.cuda_stream_);
//     }
//     std::unique_ptr<OrtValue> new_cache_indirection;
//     auto cache_indirection_type = model_.session_info_->GetInputDataType("cache_indirection");
//     auto cache_indirection_shape = std::array<int64_t, 3>{params_->batch_size, params_->search.num_beams, params_->search.max_length};
//     new_cache_indirection = OrtValue::CreateTensor(*model_.allocator_device_, cache_indirection_shape, cache_indirection_type);

//     cuda::UpdateCacheIndirectionKernelLauncher(new_cache_indirection->GetTensorMutableData<int32_t>(),
//                                                cache_indirection_->GetTensorData<int32_t>(),
//                                                beam_indices_gpu.data(),
//                                                params_->batch_size,
//                                                params_->search.num_beams,
//                                                params_->sequence_length,
//                                                params_->search.max_length,
//                                                current_length,
//                                                model_.cuda_stream_);

//     cache_indirection_ = std::move(new_cache_indirection);
//     inputs_[cache_indirection_index_] = cache_indirection_.get();
// #endif
//   }

//   if (output_cross_qk_.size() && alignment_heads_) {
// #if USE_CUDA
//     // Collect a GPU array of float* pointers from the vector of OrtValues to pass to the kernel
//     std::vector<float*> output_cross_qk_ptrs{output_cross_qk_.size(), nullptr};
//     for (int i = 0; i < output_cross_qk_.size(); i++) {
//       output_cross_qk_ptrs[i] = output_cross_qk_[i]->GetTensorMutableData<float>();
//     }
//     cudaMemcpyAsync(output_cross_qk_ptrs_gpu_.data(), output_cross_qk_ptrs.data(), output_cross_qk_.size() * sizeof(output_cross_qk_ptrs[0]), cudaMemcpyHostToDevice, model_.cuda_stream_);

//     auto output_cross_qk_dims = output_cross_qk_[0]->GetTensorTypeAndShapeInfo()->GetShape();
//     cuda::LaunchCopyCrossQKSingleDecodeStep(model_.cuda_stream_,
//                                             cross_qk_search_buffer_->GetTensorMutableData<float>(),
//                                             output_cross_qk_ptrs_gpu_.data(),
//                                             current_length - params_->sequence_length,
//                                             params_->BatchBeamSize(),
//                                             model_.config_->model.decoder.num_hidden_layers,
//                                             static_cast<int32_t>(output_cross_qk_dims[1]),
//                                             static_cast<int32_t>(alignment_heads_->GetTensorTypeAndShapeInfo()->GetShape()[0]),
//                                             alignment_heads_->GetTensorData<int32_t>(),
//                                             static_cast<int32_t>(output_cross_qk_dims[3]),
//                                             params_->search.max_length);
// #endif
//   }
// }

void WhisperState::Finalize() {
  if (decoder_state_->output_cross_qk_.size() && alignment_heads_) {
#if USE_CUDA
    // Instantiate final output for cross QKs
    auto num_alignment_heads = alignment_heads_->GetTensorTypeAndShapeInfo()->GetShape()[0];
    auto decoded_length = *(decoder_state_->past_sequence_length_->GetTensorMutableData<int32_t>()) + 1;
    auto cross_qk_shape = std::array<int64_t, 5>{params_->batch_size, params_->search.num_return_sequences, num_alignment_heads, decoded_length, encoder_state_->GetNumFrames() / 2};
    cross_qk_final_ = OrtValue::CreateTensor(*model_.allocator_device_, cross_qk_shape, decoder_state_->output_cross_qk_type_);

    cuda::LaunchFinalizeCrossQK(model_.cuda_stream_,
                                decoded_length - params_->sequence_length,
                                decoded_length,
                                static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[0]),
                                params_->search.num_beams,
                                params_->search.max_length,
                                static_cast<int32_t>(alignment_heads_->GetTensorTypeAndShapeInfo()->GetShape()[0]),
                                static_cast<int32_t>(decoder_state_->output_cross_qk_shape_[3]),
                                cross_qk_search_buffer_->GetTensorData<float>(),
                                cross_qk_final_->GetTensorMutableData<float>(),
                                params_->search.num_return_sequences,
                                decoder_state_->cache_indirection_->GetTensorData<int32_t>());
#endif
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

  // if (std::strcmp("encoder_hidden_states", name) == 0) {
  //   return encoder_state_->hidden_states_.get();
  // }
  // for (int i = 0; i < model_.config_->model.decoder.num_hidden_layers; i++) {
  //   // output_cross_qk_{i}
  //   std::string output_cross_qk_name = ComposeKeyValueName(model_.config_->model.decoder.outputs.output_cross_qk_names, i);
  //   if (std::strcmp(output_cross_qk_name.c_str(), name) == 0) {
  //     return cross_cache_->GetValues()[i].get();
  //   }
  // }

  // cross_qk_final_ is an onnxruntime-genai maintained buffer that
  // is not part of the model's outputs, so we need to check for it here.
  if (std::strcmp("cross_qk", name) == 0) {
    return cross_qk_final_.get();
  }
  // if (std::strcmp("logits", name) == 0) {
  //   return decoder_state_->logits_.GetRaw();
  // }

  return State::GetOutput(name);
};

}  // namespace Generators
