#include "../generators.h"
#include "model.h"
#include "logits.h"

namespace Generators {

Logits::Logits(Model& model, State& state)
    : model_{model},
      state_{state} {
  logits_shape_ = {state_.search_params_.batch_size * state_.search_params_.num_beams, state_.search_params_.sequence_length, state_.search_params_.vocab_size};
  logits_ = OrtValue::CreateTensor(*model.allocator_device_, logits_shape_, model_.config_->model.logits_type);

  if (model_.device_type_ == DeviceType::CPU && model_.config_->model.logits_type != Ort::TypeToTensorType<float>::type)
    throw std::runtime_error("Model logits_type can only be float32 on CPU");
}

RoamingArray<float> Logits::Get() {
  auto element_count = logits_->GetTensorTypeAndShapeInfo()->GetElementCount();

#if USE_CUDA
  if (model_.device_type_ == DeviceType::CUDA) {
    if (model_.config_->model.logits_type == Ort::TypeToTensorType<Ort::Float16_t>::type) {
      ConvertFp16ToFp32(*model_.allocator_device_, model_.cuda_stream_, *logits_, logits32_);
      return gpu_span<float>{logits32_->GetTensorMutableData<float>(), element_count};
    }
    return gpu_span<float>{logits_->GetTensorMutableData<float>(), element_count};
  }
#endif

  return cpu_span<float>{logits_->GetTensorMutableData<float>(), element_count};
}

void Logits::Add() {
  output_index_ = state_.outputs_.size();

  state_.output_names_.push_back("logits");
  state_.outputs_.push_back(logits_.get());
}

void Logits::Update() {
  // Resize the logits shape once if it doesn't match the decoder shape
  if (logits_shape_[1] != 1) {
    logits_shape_[1] = 1;
    logits_ = OrtValue::CreateTensor(*model_.allocator_device_, logits_shape_, model_.config_->model.logits_type);
    state_.outputs_[output_index_] = logits_.get();
  }
}

}  // namespace Generators
