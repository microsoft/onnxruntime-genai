#include "../generators.h"
#include "model.h"
#include "input_ids.h"
#include "kernels.h"

namespace Generators {

InputIDs::InputIDs(const Model& model, State& state)
    : model_{model},
      state_{state} {
  name_ = model_.config_->model.decoder.inputs.input_ids.c_str();
  shape_ = {state_.params_.batch_size, state_.params_.sequence_length};
  type_ = model_.session_info_->GetInputDataType(name_);

  // If 64-bit, convert from 32-bit to 64-bit
  if (type_ == Ort::TypeToTensorType<int64_t>::type) {
    value_ = OrtValue::CreateTensor(model.allocator_cpu_, shape_, type_);
    auto* p_data = value_->GetTensorMutableData<int64_t>();
    for (auto v : state_.params_.input_ids) {
      *p_data++ = v;
    }
  } else {
    if (type_ != Ort::TypeToTensorType<int32_t>::type)
      throw std::runtime_error("InputIDs must be int64 or int32");
    value_ = OrtValue::CreateTensor<int32_t>(model.allocator_cpu_.GetInfo(), std::span<int32_t>(const_cast<int32_t*>(state_.params_.input_ids.data()), shape_[0] * shape_[1]), shape_);
  }

  value_ = model_.ExpandInputs(value_, state_.params_.search.num_beams);
  shape_[0] *= state_.params_.search.num_beams;
}

void InputIDs::Add() {
  input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(value_.get());
  state_.input_names_.push_back(name_);
}

void InputIDs::Update(RoamingArray<int32_t> next_tokens_unk) {
  // Resize input_ids shape once if it doesn't match the decoder shape
  if (shape_[1] != 1) {
    shape_[1] = 1;
    value_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
    state_.inputs_[input_index_] = value_.get();
  }

  // Update input_ids with next tokens, converting from 32-bit to 64-bit
  if (type_ == Ort::TypeToTensorType<int64_t>::type) {
    auto* data = value_->GetTensorMutableData<int64_t>();
#if USE_CUDA
    if (model_.device_type_ == DeviceType::CUDA) {
      auto next_tokens = next_tokens_unk.GetGPU();
      cuda::LaunchInt32ToInt64(next_tokens.data(), data, static_cast<int>(next_tokens.size()), model_.cuda_stream_);
    } else
#endif
    {
      auto next_tokens = next_tokens_unk.GetCPU();
      for (int i = 0; i < shape_[0]; i++) {
        data[i] = next_tokens[i];
      }
    }
  } else {
    auto* data = value_->GetTensorMutableData<int32_t>();
#if USE_CUDA
    if (model_.device_type_ == DeviceType::CUDA)
      cudaMemcpyAsync(data, next_tokens_unk.GetGPU().data(), shape_[0] * sizeof(int32_t), cudaMemcpyDeviceToDevice, model_.cuda_stream_);
    else
#endif
      memcpy(data, next_tokens_unk.GetCPU().data(), shape_[0] * sizeof(int32_t));
  }
}

}  // namespace Generators
