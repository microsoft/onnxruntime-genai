// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "encoder_decoder.h"
#include <vector>

namespace Generators {

EncoderDecoderModel::EncoderDecoderModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_encoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.encoder_decoder_init.filename)).c_str(), session_options_.get());
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());

  std::cout<<"Encoder-Decoder Init session created" << (config_->config_path / fs::path(config_->model.encoder_decoder_init.filename)).c_str()<<std::endl;

  InitDeviceAllocator(*session_decoder_);
  session_info_->Add(*session_encoder_);
}

std::unique_ptr<State> EncoderDecoderModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<EncoderDecoderState>(*this, sequence_lengths, params);
}

EncoderState::EncoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model} {

  encoder_input_ids_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), std::array<int64_t, 2>{(params_->search.batch_size * params_->search.num_beams), 3}, model_.session_info_->GetInputDataType(model_.config_->model.encoder_decoder_init.inputs.input_features));
  encoder_input_ids_index = inputs_.size();
  input_names_.push_back("encoder_input_ids");
  inputs_.push_back(encoder_input_ids_.get());

  ByteWrapTensor(*model_.p_device_, *encoder_input_ids_).Zero();

  encoder_attention_mask_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), std::array<int64_t, 2>{(params_->search.batch_size * params_->search.num_beams), 3}, model_.session_info_->GetInputDataType(model_.config_->model.encoder_decoder_init.inputs.encoder_attention_mask));
  encoder_attention_mask_index = inputs_.size();
  input_names_.push_back("encoder_attention_mask");
  inputs_.push_back(encoder_attention_mask_.get());

  ByteWrapTensor(*model_.p_device_, *encoder_attention_mask_).Zero();

  cross_cache_.AddOutputs();
}

DeviceSpan<float> EncoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  State::Run(*model_.session_encoder_);
  return {};
}

DecoderState::DecoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model}  {
  std::cout<<"Inside of DecoderState constructor"<<std::endl;
  input_ids_.Add();
  std::cout<<"Input IDs added"<<std::endl;

  encoder_attention_mask_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), std::array<int64_t, 2>{(params_->search.batch_size * params_->search.num_beams), 3}, model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.encoder_attention_mask));
  encoder_attention_mask_index = inputs_.size();
  input_names_.push_back("encoder_attention_mask");
  inputs_.push_back(encoder_attention_mask_.get());

  ByteWrapTensor(*model_.p_device_, *encoder_attention_mask_).Zero();

  logits_.Add();
  kv_cache_.Add();
  cross_cache_.AddInputs();
}

DeviceSpan<float> DecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  State::Run(*model_.session_decoder_);
  return logits_.Get();
}

void DecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices, int current_length) {
  input_ids_.Update(next_tokens);
  // encoder_attention_mask_.Update(next_tokens);
  std::unique_ptr<OrtValue> new_encoder_attention_mask_;
  auto encoder_attention_mask_type = model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.encoder_attention_mask);
  auto encoder_attention_mask_shape = std::array<int64_t, 2>{(params_->search.batch_size * params_->search.num_beams), 3};
  new_encoder_attention_mask_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), encoder_attention_mask_shape, encoder_attention_mask_type);
  encoder_attention_mask_ = std::move(new_encoder_attention_mask_);
  inputs_[encoder_attention_mask_index] = encoder_attention_mask_.get();

  kv_cache_.Update(next_indices, current_length);
  size_t new_length = input_ids_.GetShape()[1];
  logits_.Update(next_tokens, new_length);
}

EncoderDecoderState::EncoderDecoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
    encoder_state_ = std::make_unique<EncoderState>(model_, sequence_lengths_unk, params);
    decoder_state_ = std::make_unique<DecoderState>(model_, sequence_lengths_unk, params);
}

DeviceSpan<float> EncoderDecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if(encoder_state_ -> first_run_) {
    // encoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length);
    encoder_state_->Run(current_length, next_tokens, next_indices);
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    return logits;
  }else{
    first_run_ = false;
    decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length);
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    return logits;
  }

  return {};
}


// EncoderDecoderState::EncoderDecoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
//     : State{params, model},
//       model_{model},
//       position_inputs_{model, *this, sequence_lengths_unk} {

//   // auto sequence_lengths = sequence_lengths_unk.CpuSpan();
//   // for (int i = 0; i < encoder_input_ids_.GetShape()[0]; i++) {
//   //   sequence_lengths[i] = 0;
//   // }
//   // sequence_lengths_unk.CopyCpuToDevice();

//   // input_names_.push_back("encoder_input_ids");
//   // inputs_.push_back(encoder_input_ids_.get());
//   encoder_input_ids_.name_ = "encoder_input_ids";
//   encoder_input_ids_.Add();
//   encoder_input_ids_.Update(sequence_lengths_unk);

//   encoder_attention_mask_.name_ = "encoder_attention_mask";
//   encoder_attention_mask_.Add();
//   encoder_attention_mask_.Update(sequence_lengths_unk);
//   // position_inputs_.Add();

//   const auto kv_cache_indices = outputs_.size();
//   // kv_cache_.AddEncoder();
//   cross_cache_.AddOutputs();
// }

// DeviceSpan<float> EncoderDecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
//   switch (run_state_) {
//     case RunState::Encoder_Decoder_Init:
//       State::Run(*model_.session_encoder_);

//       run_state_ = RunState::Decoder_First;
//       std::cout << "Encoder-Decoder Init completed" << std::endl;
//       return logits_.Get();

//     case RunState::Decoder_First: {
//       auto src_shape_info = init_presents_[0]->GetTensorTypeAndShapeInfo();

//       const auto copy_data_size_all = src_shape_info->GetElementCount() * Ort::SizeOf(src_shape_info->GetElementType());

//       // Copy over the hacked outputs to the real outputs
//       for (int i = 0; i < presents_.size(); i++) {
//         auto src_data = init_presents_[i]->GetTensorRawData();
//         auto dest_data = presents_[i]->GetTensorMutableRawData();

//         switch (model_.p_device_inputs_->GetType()) {
// #if 0  // USE_CUDA
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
//             throw std::runtime_error("Unsupported Device Type in Whisper_State::Run");
//         }
//       }

//       ClearIO();

//       decoder_input_ids_.name_ = model_.config_->model.decoder.inputs.input_ids.c_str();  // Set back to default name, since we overrode it above in the encoder step
//       decoder_input_ids_.Add();
//       logits_.Add();
//       kv_cache_.Add();
//       cross_cache_.AddInputs();
//       run_state_ = RunState::Decoder;

//       if (model_.session_info_->HasInput("past_sequence_length")) {
//         past_sequence_length_ = OrtValue::CreateTensor<int32_t>(model_.allocator_cpu_, std::array<int64_t, 1>{1});
//         input_names_.push_back("past_sequence_length");
//         inputs_.push_back(past_sequence_length_.get());
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

//   State::Run(*model_.session_decoder_);
//   return logits_.Get();
// }

// void EncoderDecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int current_length, bool search_buffers) {
//   encoder_input_ids_.Update(next_tokens);
//   position_inputs_.Update(next_tokens, current_length, static_cast<int>(encoder_input_ids_.GetShape()[1]));
//   decoder_input_ids_.Update(next_tokens);
//   kv_cache_.Update(beam_indices, current_length);
//   size_t new_length = decoder_input_ids_.GetShape()[1];
//   logits_.Update(next_tokens, new_length);

//   if (past_sequence_length_) {
//     auto data = past_sequence_length_->GetTensorMutableData<int32_t>();
//     *data = current_length - 1;
//   }

//   if (!search_buffers) {
//     // No need to update cache indirection and cross QK search buffers
//     // when preparing to run decoder for the first time.
//     if (cache_indirection_)
//       inputs_[cache_indirection_index_] = cache_indirection_.get();
//     return;
//   }
// }

// void EncoderDecoderState::Initialize(DeviceSpan<int32_t>& next_tokens, int total_length, DeviceSpan<int32_t> beam_indices) {
//   run_state_ = RunState::Encoder_Decoder_Init;
// }

}  // namespace Generators
