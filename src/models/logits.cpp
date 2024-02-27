#include "../generators.h"
#include "model.h"
#include "logits.h"

namespace Generators {

Logits::Logits(const Model& model, State& state)
    : model_{model},
      state_{state},
      shape_{state_.search_params_.batch_size * state_.search_params_.num_beams, state_.search_params_.sequence_length, state_.search_params_.vocab_size},
      type_{model_.session_info_->GetOutputDataType(model_.config_->model.decoder.outputs.logits)} {
  value_ = OrtValue::CreateTensor(*model.allocator_device_, shape_, type_);

  if (model_.device_type_ == DeviceType::CPU && type_ != Ort::TypeToTensorType<float>::type)
    throw std::runtime_error("Model logits_type can only be float32 on CPU");
}

RoamingArray<float> Logits::Get() {
  size_t element_count = shape_[0] * shape_[1] * shape_[2];

#if USE_CUDA
  if (model_.device_type_ == DeviceType::CUDA) {
    if (type_ == Ort::TypeToTensorType<Ort::Float16_t>::type) {
      ConvertFp16ToFp32(*model_.allocator_device_, model_.cuda_stream_, *value_, value32_);
      return gpu_span<float>{value32_->GetTensorMutableData<float>(), element_count};
    }
    return gpu_span<float>{value_->GetTensorMutableData<float>(), element_count};
  }
#endif

  return cpu_span<float>{value_->GetTensorMutableData<float>(), element_count};
}

void Logits::Add() {
  output_index_ = state_.outputs_.size();

  state_.output_names_.push_back(model_.config_->model.decoder.outputs.logits.c_str());
  state_.outputs_.push_back(value_.get());
}

void Logits::Update() {
  // Resize the logits shape once if it doesn't match the decoder shape
  if (shape_[1] != 1) {
    shape_[1] = 1;
    value_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
    state_.outputs_[output_index_] = value_.get();
  }
}

}  // namespace Generators
