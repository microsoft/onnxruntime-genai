// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "whisper.h"
#include <vector>
#include "kernels.h"

namespace Generators {

Whisper_Model::Whisper_Model(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_encoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.encoder_decoder_init.filename)).c_str(), session_options_.get());
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());

  InitDeviceAllocator(*session_decoder_);
  session_info_->Add(*session_encoder_);
}

std::unique_ptr<State> Whisper_Model::CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<Whisper_State>(*this, sequence_lengths, params);
}

Whisper_State::Whisper_State(const Whisper_Model& model, RoamingArray<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  auto& inputs = const_cast<GeneratorParams::Whisper&>(std::get<GeneratorParams::Whisper>(params.inputs));

  for (const auto& [name, value] : params.extra_inputs) {
    if (name == "encoder_input_ids") {
      encoder_input_ids_ = model_.ExpandInputs(value->ort_tensor_, params_->search.num_beams);
    }
  }
  if (encoder_input_ids_ == nullptr) {
    encoder_input_ids_ = model_.ExpandInputs(inputs.input_features->ort_tensor_, params_->search.num_beams);
  }

  if (encoder_input_ids_ == nullptr)
    throw std::runtime_error("encoder_input_ids must be provided in the extra inputs");

  auto hidden_states_type = model_.session_info_->GetOutputDataType("encoder_hidden_states");
  auto encoder_hidden_states_shape = std::array<int64_t, 3>{decoder_input_ids_.GetShape()[0], 1500, static_cast<int64_t>(model_.config_->model.decoder.num_attention_heads) * model_.config_->model.decoder.head_size};
  encoder_hidden_states_ = OrtValue::CreateTensor(*model_.allocator_device_, encoder_hidden_states_shape, hidden_states_type);

  auto sequence_lengths = sequence_lengths_unk.GetCPU();
  for (int i = 0; i < decoder_input_ids_.GetShape()[0]; i++) {
    sequence_lengths[i] = static_cast<int32_t>(params_->sequence_length);
  }

  input_names_.push_back("encoder_input_ids");
  inputs_.push_back(encoder_input_ids_.get());
  decoder_input_ids_.name_ = "decoder_input_ids";
  decoder_input_ids_.Add();

  logits_.Add();
  output_names_.push_back("encoder_hidden_states");
  outputs_.push_back(encoder_hidden_states_.get());

  const auto kv_cache_indices = outputs_.size();
  kv_cache_.AddEncoder();
  cross_cache_.AddOutputs();

  {
    auto layer_count = model_.config_->model.decoder.num_hidden_layers;
    std::array<int64_t, 4> shape{params_->BatchBeamSize(), model_.config_->model.decoder.num_attention_heads, params_->sequence_length, model_.config_->model.decoder.head_size};
    auto type = model_.session_info_->GetOutputDataType(output_names_[kv_cache_indices]);

    for (int i = 0; i < layer_count * 2; i++) {
      init_presents_.emplace_back(OrtValue::CreateTensor(*model_.allocator_device_, shape, type));
      presents_.emplace_back(outputs_[kv_cache_indices + i]);
      outputs_[kv_cache_indices + i] = init_presents_.back().get();
    }
  }
}

#if USE_CUDA
template <typename T>
void TransposeKCacheForDMMHA(T* dest_data,
                             T* temp_buffer,
                             std::vector<int64_t>& dest_dims,
                             int dest_data_size,
                             int dest_element_size,
                             cudaStream_t stream) {
  // Treat the 'K' caches as if they are of shape [B, N, max_length, head_size / x, x]
  // and transpose each 'K' cache into [B, N, head_size / x, max_length, x], where x = 16 / sizeof(T)
  int chunk_size = static_cast<int>(16 / dest_element_size);
  if (chunk_size != 4 && chunk_size != 8) {
    throw std::runtime_error("ReorderPastStatesKernelLauncher only supports float32 or float16 precision");
  }

  // Copy the original 'K' caches to a temporary buffer in order to
  // use the destination buffer to store the transposed 'K' caches
  cudaMemcpyAsync(temp_buffer, dest_data, dest_data_size, cudaMemcpyDeviceToDevice, stream);

  // Transpose each 'K' cache
  cuda::ReorderPastStatesKernelLauncher(dest_data,
                                        temp_buffer,
                                        dest_dims[0],
                                        dest_dims[1],
                                        dest_dims[2],
                                        dest_dims[3],
                                        chunk_size,
                                        stream);
}
#endif

RoamingArray<float> Whisper_State::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  int batch_size = static_cast<int>(decoder_input_ids_.GetShape()[0]);

  switch (run_state_) {
    case RunState::Encoder_Decoder_Init:
      State::Run(*model_.session_encoder_, *model_.run_options_, batch_size);

      run_state_ = RunState::Decoder_First;
      return logits_.Get();

    case RunState::Decoder_First:
      // Wrap below code in {} to avoid `note: crosses initialization of` warnings during compilation
      // that arise because variables are created in a switch statement
      {
        auto src_shape_info = init_presents_[0]->GetTensorTypeAndShapeInfo();

        const auto copy_data_size_all = src_shape_info->GetElementCount() * SizeOf(src_shape_info->GetElementType());

#if USE_CUDA
        const auto src_dims = src_shape_info->GetShape();
        const auto src_element_type = src_shape_info->GetElementType();
        const auto src_element_size = SizeOf(src_element_type);

        auto dest_shape_info = presents_[0]->GetTensorTypeAndShapeInfo();
        auto dest_dims = dest_shape_info->GetShape();
        auto dest_element_type = dest_shape_info->GetElementType();
        auto dest_element_size = SizeOf(dest_element_type);
        auto dest_data_size = dest_shape_info->GetElementCount() * dest_element_size;

        const auto copy_data_size = src_dims[2] * src_dims[3] * src_element_size;

        // Allocate temporary buffer for when CUDA EP + FP16 precision is used because
        // we need to reformat the `K` caches for `DecoderMaskedMultiHeadAttention`
        // and we need some extra memory to do so.
        //
        // Since the self attention K caches are of size (batch_size, num_heads, past_sequence_length, head_size) with type 'float16',
        // the cross attention K caches are of size (batch_size, num_heads, 1500, head_size) with type 'float32', and
        // past_sequence_length <= 448 < 1500, we will allocate a temporary buffer that is the
        // size of a cross attention K cache. This lets us use the same temporary buffer for both
        // the self attention and cross attention K caches.

        std::unique_ptr<OrtValue> temp_buffer;
        auto self_attn_kv_cache_element_type = src_element_type;  // should be `float16` for the below case
        if (self_attn_kv_cache_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && model_.device_type_ == DeviceType::CUDA) {
          auto num_layers = model_.config_->model.decoder.num_hidden_layers;
          auto cross_attn_shape_info = outputs_[outputs_.size() - 1]->GetTensorTypeAndShapeInfo();
          auto cross_attn_dims = cross_attn_shape_info->GetShape();
          auto cross_attn_kv_cache_element_type = cross_attn_shape_info->GetElementType();  // should be `float32` for this case

          temp_buffer = OrtValue::CreateTensor(*model_.allocator_device_, cross_attn_dims, cross_attn_kv_cache_element_type);
        }
#endif

        // Copy over the hacked outputs to the real outputs
        for (int i = 0; i < presents_.size(); i++) {
          auto src_data = init_presents_[i]->GetTensorRawData();
          auto dest_data = presents_[i]->GetTensorMutableRawData();

          switch (model_.device_type_) {
#if USE_CUDA
            case DeviceType::CUDA:
              if (self_attn_kv_cache_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                // CUDA EP + FP16 precision == `DecoderMaskedMultiHeadAttention` op is used
                // This also means `past-present buffer sharing = true`

                // Copy data from init_presents_[i] to presents_[i]
                // from (batch_size, num_heads, past_sequence_length, head_size)
                // to (batch_size, num_heads, max_sequence_length, head_size)
                //
                // Implemented as:
                // real[:batch_size, :num_heads, :past_sequence_length, :head_size] = hacked
                for (int b = 0; b < dest_dims[0] * dest_dims[1]; b++) {
                  auto src_offset = b * src_dims[2] * src_dims[3];
                  auto dest_offset = b * dest_dims[2] * dest_dims[3];

                  src_offset *= src_element_size;
                  dest_offset *= dest_element_size;
                  cudaMemcpyAsync(dest_data + dest_offset, src_data + src_offset, copy_data_size, cudaMemcpyDeviceToDevice, model_.cuda_stream_);
                }

                // Transpose self attention K caches for `DecoderMaskedMultiHeadAttention`
                if (i % 2 == 0) {
                  TransposeKCacheForDMMHA(dest_data, temp_buffer->GetTensorMutableRawData(), dest_dims,
                                          dest_data_size, dest_element_size, model_.cuda_stream_);
                }
              } else {
                cudaMemcpyAsync(dest_data, src_data, copy_data_size_all, cudaMemcpyDeviceToDevice, model_.cuda_stream_);
              }
              break;
#endif
            case DeviceType::CPU: {
              memcpy(dest_data, src_data, copy_data_size_all);
              break;
            }

            default:
              throw std::runtime_error("Unsupported Device Type in Whisper_State::Run");
          }
        }

#if USE_CUDA
        if (self_attn_kv_cache_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && model_.device_type_ == DeviceType::CUDA) {
          // Transpose cross attention K caches for `DecoderMaskedMultiHeadAttention`

          // Add +2 to start of loop to account for `logits` and `encoder_hidden_states` outputs
          for (int i = 2 + init_presents_.size(); i < outputs_.size(); i += 2) {
            auto dest_data = outputs_[i]->GetTensorMutableRawData();
            dest_shape_info = outputs_[i]->GetTensorTypeAndShapeInfo();
            dest_dims = dest_shape_info->GetShape();
            dest_element_type = dest_shape_info->GetElementType();
            dest_element_size = SizeOf(dest_element_type);
            dest_data_size = dest_shape_info->GetElementCount() * dest_element_size;

            TransposeKCacheForDMMHA(dest_data, temp_buffer->GetTensorMutableRawData(), dest_dims,
                                    dest_data_size, dest_element_size, model_.cuda_stream_);
          }
        }
#endif
      }

      ClearIO();

      decoder_input_ids_.name_ = model_.config_->model.decoder.inputs.input_ids.c_str();  // Set back to default name, since we overrode it above in the encoder step
      decoder_input_ids_.Add();
      logits_.Add();
      kv_cache_.Add();
      cross_cache_.AddInputs();
      run_state_ = RunState::Decoder;

      if (model_.session_info_->HasInput("past_sequence_length")) {
        past_sequence_length_ = OrtValue::CreateTensor<int32_t>(model_.allocator_cpu_, std::array<int64_t, 1>{1});
        input_names_.push_back("past_sequence_length");
        inputs_.push_back(past_sequence_length_.get());
      }

      if (model_.session_info_->HasInput("beam_width")) {
        beam_width_ = OrtValue::CreateTensor<int32_t>(model_.allocator_cpu_, std::array<int64_t, 1>{1});
        input_names_.push_back("beam_width");
        inputs_.push_back(beam_width_.get());

        auto data = beam_width_->GetTensorMutableData<int32_t>();
        *data = params_->search.num_beams;
      }

      if (model_.session_info_->HasInput("cache_indirection")) {
#if USE_CUDA
        cache_indirection_ = OrtValue::CreateTensor<int32_t>(*model_.allocator_device_, std::array<int64_t, 3>{params_->batch_size, params_->search.num_beams, params_->search.max_length});
        cache_indirection_index_ = inputs_.size();
        input_names_.push_back("cache_indirection");
        inputs_.push_back(cache_indirection_.get());

        auto data = gpu_span<int32_t>{cache_indirection_->GetTensorMutableData<int32_t>(),
                                      static_cast<size_t>(params_->batch_size) * params_->search.num_beams * params_->search.max_length};
        cudaMemsetAsync(data.data(), 0, data.size_bytes(), params_->cuda_stream);
#endif
      }

      if (model_.session_info_->HasOutput("output_cross_qk_0")) {
        auto layer_count = model_.config_->model.decoder.num_hidden_layers;
        auto type = model_.session_info_->GetOutputDataType("output_cross_qk_0");
        std::array<int64_t, 4> shape{params_->BatchBeamSize(), model_.config_->model.decoder.num_attention_heads, 1, 1500};
        for (int i = 0; i < layer_count; i++) {
          char string[64];
          snprintf(string, std::size(string), "output_cross_qk_%d", i);
          output_cross_qk_names_.emplace_back(string);
          output_cross_qk_.emplace_back(OrtValue::CreateTensor(*model_.allocator_device_, shape, type));

          output_names_.emplace_back(output_cross_qk_names_.back().c_str());
          outputs_.emplace_back(output_cross_qk_.back().get());
        }
      }

      // Fall through

    case RunState::Decoder:
      UpdateInputsOutputs(next_tokens, next_indices, current_length);
      break;
  }

  State::Run(*model_.session_decoder_, *model_.run_options_, batch_size);
  return logits_.Get();
}

void Whisper_State::UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> beam_indices, int current_length) {
  decoder_input_ids_.Update(next_tokens);
  kv_cache_.Update(beam_indices.GetCPU(), current_length);
  logits_.Update();

  if (past_sequence_length_) {
    auto data = past_sequence_length_->GetTensorMutableData<int32_t>();
    *data = current_length - 1;
  }

  if (cache_indirection_) {
#if USE_CUDA
    auto old_cache_indirection = gpu_span<int32_t>{cache_indirection_->GetTensorMutableData<int32_t>(),
                                                   static_cast<size_t>(params_->batch_size) *
                                                       params_->search.num_beams * params_->search.max_length};
    auto new_cache_indirection = OrtValue::CreateTensor<int32_t>(*model_.allocator_device_,
                                                                 cache_indirection_->GetTensorTypeAndShapeInfo()->GetShape());
    auto cache_indirection = gpu_span<int32_t>{new_cache_indirection->GetTensorMutableData<int32_t>(),
                                               static_cast<size_t>(params_->batch_size) * params_->search.num_beams *
                                                   params_->search.max_length};

    cuda::UpdateDecoderMaskedMultiHeadAttentionCacheIndirection(cache_indirection.data(),
                                                                old_cache_indirection.data(),
                                                                beam_indices.GetGPU().data(),
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
}

}  // namespace Generators
