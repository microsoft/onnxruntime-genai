#include "../generators.h"
#include "model.h"
#include "input_ids.h"
#include "kernels.h"

namespace Generators {

// NOW IS 0-INITIALIZED
InputIDs::InputIDs(const Model& model, State& state)
    : model_{model},
      state_{state} {
  name_ = model_.config_->model.decoder.inputs.input_ids.c_str();
  shape_ = {state_.params_->search.num_beams * state_.params_->batch_size, 0};
  type_ = model_.session_info_->GetInputDataType(name_);

  // If 64-bit, convert from 32-bit to 64-bit
  // if (type_ == Ort::TypeToTensorType<int64_t>) {
  //   value_ = OrtValue::CreateTensor(model.allocator_cpu_, shape_, type_);
  //   auto* p_data = value_->GetTensorMutableData<int64_t>();
  //   for (auto v : state_.params_->input_ids) {
  //     *p_data++ = v;
  //   }
  // } else {
  //   if (type_ != Ort::TypeToTensorType<int32_t>)
  //     throw std::runtime_error("InputIDs must be int64 or int32");
  //   value_ = OrtValue::CreateTensor<int32_t>(model.allocator_cpu_.GetInfo(), std::span<int32_t>(const_cast<int32_t*>(state_.params_->input_ids.data()), shape_[0] * shape_[1]), shape_);
  // }

  // value_ = OrtValue::CreateTensor(model.allocator_cpu_, shape_, type_); // TODO(aciddelgado): 0 initializing tensors allowed?

  // value_ = model_.ExpandInputs(value_, state_.params_->search.num_beams);
  // shape_[0] *= state_.params_->search.num_beams;

  if (state_.GetCapturedGraphInfo()) {
    sb_input_ids_ = state_.GetCapturedGraphInfo()->sb_input_ids_.get();

#if USE_DML
    if (model_.device_type_ == DeviceType::DML) {
      sb_input_ids_int32_ = state_.GetCapturedGraphInfo()->sb_input_ids_int32_.get();
    }
#endif
  }
}

void InputIDs::Add() {
  input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(value_.get());
  state_.input_names_.push_back(name_);
}

void InputIDs::Update(RoamingArray<int32_t> new_tokens) {
  // // Resize input_ids shape once if it doesn't match the decoder shape
  // if (shape_[1] != 1) {
  //   shape_[1] = 1;
  //   if (!sb_input_ids_) {
  //     value_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);

  // Resize input_ids shape to sequence_length of new_tokens
  size_t sequence_length = static_cast<size_t>(new_tokens.GetCPU().size()) / shape_[0];
  if (shape_[1] != sequence_length) {
    shape_[1] = sequence_length;
    if (!sb_input_ids_) {
      value_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);

#if USE_DML
      if (model_.device_type_ == DeviceType::DML) {
        value_int32_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
      }
#endif
    } else {
      value_ = sb_input_ids_->CreateTensorOnStaticBuffer(shape_, type_);

#if USE_DML
      if (model_.device_type_ == DeviceType::DML) {
        value_int32_ = sb_input_ids_int32_->CreateTensorOnStaticBuffer(shape_, Ort::TypeToTensorType<int32_t>);
      }
#endif
    }

    state_.inputs_[input_index_] = value_.get();
  }

  // Update input_ids with next tokens, converting from 32-bit to 64-bit
  if (type_ == Ort::TypeToTensorType<int64_t>) {
    switch (model_.device_type_) {
#if USE_CUDA
      case DeviceType::CUDA: {
        auto* data = value_->GetTensorMutableData<int64_t>();
        auto next_tokens = new_tokens.GetGPU();
        cuda::LaunchInt32ToInt64(next_tokens.data(), data, static_cast<int>(next_tokens.size()), model_.cuda_stream_);
      } break;
#endif

#if USE_DML
      case DeviceType::DML: {
        ComPtr<ID3D12Resource> source_resource;
        Ort::ThrowOnError(model_.GetOrtDmlApi()->GetD3D12ResourceFromAllocation(model_.allocator_device_, value_int32_->GetTensorMutableRawData(), &source_resource));

        auto source = std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(new_tokens.GetCPU().data()),
            new_tokens.GetCPU().size_bytes());

        model_.GetDmlUploadHeap()->BeginUploadToGpu(
            source_resource.Get(),
            0,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            source);

        DmlHelpers::DmlCastInputToOutput(
            model_.GetDmlExecutionContext(),
            *model_.allocator_device_,
            *value_int32_,
            value_,
            model_.GetDmlDevice(),
            model_.GetOrtDmlApi(),
            input_ids_cast_command_list_state_);
      } break;
#endif
      case DeviceType::CPU: {
        auto* data = value_->GetTensorMutableData<int64_t>();
        auto next_tokens = new_tokens.GetCPU();
        for (int b = 0; b < shape_[0]; b++) {
          for (int i = 0; i < shape_[1]; i++) {
            data[b * shape_[1] + i] = next_tokens[b * shape_[1] + i];
          }
        }
      }
    }
  } else {
    auto* data = value_->GetTensorMutableData<int32_t>();
#if USE_CUDA
    if (model_.device_type_ == DeviceType::CUDA)
      cudaMemcpyAsync(data, new_tokens.GetGPU().data(), shape_[0] * shape_[1] * sizeof(int32_t), cudaMemcpyDeviceToDevice, model_.cuda_stream_);
    else
#endif
      memcpy(data, new_tokens.GetCPU().data(), shape_[0] * shape_[1] * sizeof(int32_t));
  }
}

// Add tokens to the end of input ids tensor
// void InputIDs::AddInputTokens(RoamingArray<int32_t> tokens, bool is_first_tokens) {
//   switch (model_.device_type_) {
//     case DeviceType::CPU: {
//       break;
//     }
//     default:
//       throw std::runtime_error("Add Tokens not supported for device type " + to_string(model_.device_type_));
//   }
//   if (shape_[0] != 1) {
//     throw std::runtime_error("Add Tokens only supported for batch size 1, got " + std::to_string(shape_[0]));
//   }
//   auto tokens_cpu = tokens.GetCPU();
//   int start = is_first_tokens ? 0 : shape_[1];
//   int token_count = tokens_cpu.size();
//   shape_[1] = start + token_count;

//   std::unique_ptr<OrtValue> temp_value;
//   if (!sb_input_ids_) {
//     temp_value = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
//   } else {
//     temp_value = sb_input_ids_->CreateTensorOnStaticBuffer(shape_, type_);
//   }
//   if (type_ == Ort::TypeToTensorType<int64_t>) {
//     auto* data = temp_value->GetTensorMutableData<int64_t>();
//     auto next_tokens_cpu = next_tokens.GetCPU();
//     for (int i = 0; i < start; i++) {
//       data[i] = value_->GetTensorData<int64_t>()[i];
//     }
//     for (int i = 0; i < token_count; i++) {
//       data[start + i] = tokens_cpu[i];
//     }
//   } else {
//     auto* data = temp_value->GetTensorMutableData<int32_t>();
//     if (is_first_tokens) {
//       memcpy(data, value_->GetTensorData<int32_t>(), start * sizeof(int32_t));
//       data += start;
//     }
//     memcpy(data, tokens.GetCPU().data(), token_count * sizeof(int32_t));
//   }
//   value_ = std::move(temp_value);
//   state_.inputs_[input_index_] = value_.get();
// }

}  // namespace Generators
