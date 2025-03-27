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

  InitDeviceAllocator(*session_decoder_);
  session_info_->Add(*session_encoder_);
}

std::unique_ptr<State> EncoderDecoderModel::CreateState(DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params) const {
  std::cout<<"Inside of EncoderDecoderModel::CreateState"<<std::endl;
  return std::make_unique<EncoderDecoderState>(*this, sequence_lengths_unk, params);
}

EncoderState::EncoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model} {
  std::cout<<"Inside of EncoderState constructor"<<std::endl;
  auto& inputs = const_cast<GeneratorParams::EncoderDecoder&>(std::get<GeneratorParams::EncoderDecoder>(params.encoderdecoder_inputs));
  // std::cout<<"Inputs = "<<inputs.input_features->ort_tensor_->GetTensorTypeAndShapeInfo().GetElementType()<<std::endl;
  // std::cout<<"Inputs = "<<inputs.encoder_attention_mask->ort_tensor_->GetTensorTypeAndShapeInfo().GetElementType()<<std::endl;
  encoder_input_ids_.Add();
  // auto encoder_input_ids_type = model_.session_info_->GetInputDataType("encoder_input_ids");
  // auto encoder_input_ids_shape = std::array<int64_t, 2>{1, 32128};
  // encoder_input_ids_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), encoder_input_ids_shape, encoder_input_ids_type);
  auto encoder_attention_mask_type = model_.session_info_->GetInputDataType("encoder_attention_mask");
  auto encoder_attention_mask_shape = std::array<int64_t, 2>{1, 32128};
  encoder_attention_mask_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), encoder_attention_mask_shape, encoder_attention_mask_type);
  input_names_.push_back("encoder_attention_mask");
  inputs_.push_back(encoder_attention_mask_.get());
  // input_names_.push_back("encoder_input_ids");
  // inputs_.push_back(encoder_input_ids_.get());
}

DeviceSpan<float> EncoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
    std::cout<<"Inside of EncoderState::Run"<<std::endl;
    State::Run(*model_.session_encoder_, false);
    std::cout<<"After State::Run"<<std::endl;
    return MakeDummy();
}

DecoderState::DecoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params)
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
}

DeviceSpan<float> DecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
  State::Run(*model_.session_decoder_, false);
  return logits_.Get();
}

void DecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens_unk, DeviceSpan<int32_t> beam_indices, int current_length, bool first_update) {
  input_ids_.Update(next_tokens_unk);
  size_t new_length = static_cast<size_t>(input_ids_.GetShape()[1]);
  kv_cache_.Update(beam_indices, current_length);
  position_inputs_.Update(next_tokens_unk, current_length, static_cast<int>(new_length));
  logits_.Update(next_tokens_unk, new_length);
  if (past_sequence_length_) {
    auto data = past_sequence_length_->GetTensorMutableData<int32_t>();
    *data = current_length - 1;
  }
}

EncoderDecoderState::EncoderDecoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    :State{params, model},
      model_{model}{

    encoder_state_ = std::make_unique<EncoderState>(model, sequence_lengths_unk, params);
    cross_cache_.AddInputs();
    cross_cache_.AddOutputs();
    // cross_cache_ = std::make_unique<CrossCache>(*this);
    // encoder_state_->AddCrossCache(cross_cache_);
    decoder_state_ = std::make_unique<DecoderState>(model, sequence_lengths_unk ,params);
    // decoder_state_->AddCrossCache(cross_cache_);
}

// EncoderDecoderState::EncoderDecoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
//   : State{params, model},
//   model_{model},
//   input_ids_{CreateInputIDs(*this)},
//   position_inputs_{CreatePositionInputs(*this, sequence_lengths_unk)} {

//   input_ids_->Add();
//   position_inputs_->Add();
//   logits_.Add();
//   kv_cache_.Add();
//   extra_inputs_.Add();

//   // auto& inputs = const_cast<GeneratorParams::EncoderDecoder&>(std::get<GeneratorParams::EncoderDecoder>(params.encoderdecoder_inputs));
//   // for (const auto& [name, value] : params.extra_inputs) {
//   //   if (name == "encoder_input_ids") {
//   //     input_ids_ = model_.ExpandInputs(value->ort_tensor_, params_->search.num_beams);
//   //   }
//   // }
//   // if (input_ids_ == nullptr) {
//   //   input_ids_ = model_.ExpandInputs(inputs.input_features->ort_tensor_, params_->search.num_beams);
//   // }

//   // if (input_ids_ == nullptr) {
//   //   throw std::runtime_error("encoder_input_ids must be provided in the extra inputs");
//   // }

//   // auto sequence_lengths = sequence_lengths_unk.CpuSpan();
//   // for (int i = 0; i < decoder_input_ids_.GetShape()[0]; i++) {
//   //   sequence_lengths[i] = 0;
//   // }
//   // sequence_lengths_unk.CopyCpuToDevice();

//   // input_names_.push_back("encoder_input_ids");
//   // inputs_.push_back(input_ids_.get());
//   decoder_input_ids_.name_ = "input_ids";
//   decoder_input_ids_.Add();

//   // logits_.Add();

//   // const auto kv_cache_indices = outputs_.size();
//   // kv_cache_.AddEncoder();
//   cross_cache_.AddInputs();
//   cross_cache_.AddOutputs();
//   for (auto inp:input_names_){
//     std::cout<<"Input Names = "<<inp<<std::endl;
//   }
// }

DeviceSpan<float> EncoderDecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if (encoder_state_->first_run_) {
    // Run encoder
    std::cout<<"Inside encoder run first run"<<std::endl;
    encoder_state_->Run(current_length, next_tokens, next_indices);

    // Run decoder-init
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    return logits;
  } else {
    first_run_ = false;
    // Update inputs and outputs for decoder
    decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);

    // Run decoder-with-past
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    return logits;
  }
  // Not reached
  return MakeDummy();
}

// DeviceSpan<float> EncoderDecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
//   // if (encoder_state_->first_run_) {
//   //   // Run encoder
//   //   encoder_state_->Run(current_length, next_tokens, next_indices);

//   //   // Run decoder
//   //   auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

//   //   return logits;
//   // } else {
//   //   first_run_ = false;
//   //   // Update inputs and outputs for decoder
//   //   decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length);

//   //   // Run decoder
//   //   auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);
    
//   //   return logits;
//   // }
//   // // Not reached
//   // return MakeDummy();
//   std::cout<<"Inside of EncoderDecoderState::Run"<<std::endl;
//   switch (run_state_) {
//     case RunState::Encoder_Decoder_Init:
//       std::cout<<"Inside of Encoder_Decoder_Init"<<std::endl;
//       State::Run(*model_.session_encoder_);

//       run_state_ = RunState::Decoder_First;
//       return logits_.Get();

//     case RunState::Decoder_First: {
//       std::cout<<"Inside of Decoder_First"<<std::endl;
//       auto src_shape_info = init_presents_[0]->GetTensorTypeAndShapeInfo();

//       const auto copy_data_size_all = src_shape_info->GetElementCount() * Ort::SizeOf(src_shape_info->GetElementType());

// #if 0  // USE_CUDA
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
//             throw std::runtime_error("Unsupported Device Type in EncoderDecoderState::Run");
//         }
//       }

// #if 0  // USE_CUDA
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

//       decoder_input_ids_.name_ = model_.config_->model.decoder.inputs.input_ids.c_str();  // Set back to default name, since we overrode it above in the encoder step
//       decoder_input_ids_.Add();
//       logits_.Add();
//       kv_cache_.Add();
//       cross_cache_.AddInputs();
//       run_state_ = RunState::Decoder;
//       UpdateInputsOutputs(next_tokens, next_indices, current_length);
//       break;
//     }

//     case RunState::Decoder: {
//       UpdateInputsOutputs(next_tokens, next_indices, current_length);
//       break;
//     }
//   }

//   State::Run(*model_.session_decoder_);
//   return logits_.Get();
// }

void EncoderDecoderState::UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> beam_indices, int current_length, bool first_update) {
  decoder_input_ids_.Update(next_tokens);
  kv_cache_.Update(beam_indices, current_length);
  size_t new_length = decoder_input_ids_.GetShape()[1];
  logits_.Update(next_tokens, new_length);

}


void EncoderDecoderState::Initialize(DeviceSpan<int32_t>& next_tokens, int total_length, DeviceSpan<int32_t> beam_indices) {
  run_state_ = RunState::Encoder_Decoder_Init;
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