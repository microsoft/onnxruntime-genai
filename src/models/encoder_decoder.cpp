// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "encoder_decoder.h"
#include "kernels.h"
#include <vector>

#if USE_CUDA
#include <cuda_fp16.h>
#endif

namespace Generators {

EncoderDecoderModel::EncoderDecoderModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_encoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.encoder.filename)).c_str(), session_options_.get());
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());

  InitDeviceAllocator(*session_decoder_);
  session_info_->Add(*session_encoder_);
}

std::unique_ptr<State> EncoderDecoderModel::CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<EncoderDecoderState>(*this, params, sequence_lengths);
}

EncoderState::EncoderState(const EncoderDecoderModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      position_inputs_{model, *this, sequence_lengths} {
  input_ids_.Add();
  position_inputs_.Add();                           // adds attention_mask
}

RoamingArray<float> EncoderState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
    int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
    State::Run(*model_.session_encoder_, *model_.run_options_, batch_size);
    return MakeDummy();
}

DecoderState::DecoderState(const EncoderDecoderModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      position_inputs_{model, *this, sequence_lengths} {
  input_ids_.Add();
  position_inputs_.Add();
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

  // Add output QK
  std::string output_cross_qk_name = ComposeKeyValueName(model_.config_->model.decoder.outputs.output_cross_qk_names, 0);

  if (model_.session_info_->HasOutput(output_cross_qk_name)) {
    output_cross_qk_type_ = model_.session_info_->GetOutputDataType(output_cross_qk_name);
    output_cross_qk_shape_ = std::array<int64_t, 4>{params_->BatchBeamSize(), model_.config_->model.decoder.num_attention_heads, params_->sequence_length, 64};
    output_cross_qk_index_ = outputs_.size();

    for (int i = 0; i < model_.config_->model.decoder.num_hidden_layers; i++) {
      output_cross_qk_.emplace_back(OrtValue::CreateTensor(*model_.allocator_device_, output_cross_qk_shape_, output_cross_qk_type_));
      output_cross_qk_names_.emplace_back(ComposeKeyValueName(model_.config_->model.decoder.outputs.output_cross_qk_names, i));

      output_names_.emplace_back(output_cross_qk_names_.back().c_str());
      outputs_.emplace_back(output_cross_qk_.back().get());
    }
  }
}

RoamingArray<float> DecoderState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
  State::Run(*model_.session_decoder_, *model_.run_options_, batch_size);
  return logits_.Get();
}

void DecoderState::UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens_unk, RoamingArray<int32_t> beam_indices, int current_length, bool first_update) {
  input_ids_.Update(next_tokens_unk);
  kv_cache_.Update(beam_indices.GetCPU(), current_length);
  position_inputs_.Update(current_length);
  logits_.Update();
  if (past_sequence_length_) {
    auto data = past_sequence_length_->GetTensorMutableData<int32_t>();
    *data = current_length - 1;
  }
}

EncoderDecoderState::EncoderDecoderState(const EncoderDecoderModel& model, const GeneratorParams& params, RoamingArray<int32_t> sequence_lengths_unk)
    : State{params, model},
      model_{model} {
  encoder_state_ = std::make_unique<EncoderState>(model, sequence_lengths_unk, params);
  cross_cache_ = std::make_unique<Cross_Cache>(*this, 64);
  encoder_state_->AddCrossCache(cross_cache_);
  decoder_state_ = std::make_unique<DecoderState>(model, sequence_lengths_unk, params);
  decoder_state_->AddCrossCache(cross_cache_);

  transpose_k_cache_buffer_ = OrtValue::CreateTensor(*model_.allocator_device_, cross_cache_->GetShape(), cross_cache_->GetType());

}

void EncoderDecoderState::TransposeKCaches(std::vector<std::unique_ptr<OrtValue>>& kv_caches) {
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

RoamingArray<float> EncoderDecoderState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  if (encoder_state_->first_run_) {
    // Run encoder
    encoder_state_->Run(current_length, next_tokens, next_indices);

    // Run decoder
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);
    // TODO: Transpose the K caches only when the else branch is run for the first time.
    // Otherwise the GetOutput(present_key_self_{i}) method returns transposed K caches.
    TransposeKCaches(cross_cache_->GetValues());
    TransposeKCaches(decoder_state_->kv_cache_.GetPresents());

    return logits;
  } else {
    first_run_ = false;
    // Update inputs and outputs for decoder
    decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);

    // Run decoder
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);
    
    return logits;
  }
  // Not reached
  return MakeDummy();
}

OrtValue* EncoderDecoderState::GetInput(const char* name) {
  // Check if input name is in encoder state's inputs
  for (size_t i = 0; i < encoder_state_->input_names_.size(); i++) {
    if (std::strcmp(encoder_state_->input_names_[i], name) == 0) {
      return encoder_state_->inputs_[i];
    }
  }

  // Check if input name is in decoder state's inputs
  for (size_t i = 0; i < decoder_state_->input_names_.size(); i++) {
    if (std::strcmp(decoder_state_->input_names_[i], name) == 0) {
      return decoder_state_->inputs_[i];
    }
  }

  return State::GetInput(name);
};

OrtValue* EncoderDecoderState::GetOutput(const char* name) {
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

  return State::GetOutput(name);
};

}  // namespace Generators