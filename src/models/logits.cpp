#include "../generators.h"
#include "model.h"
#include "logits.h"

namespace Generators {

Logits::Logits(Model& model, State& state)
    : model_{model},
      state_{state} {

  logits_shape_ = {state_.search_params_.batch_size * state_.search_params_.num_beams, model_.logits_uses_seq_len_ ? state_.search_params_.sequence_length : 1, state_.search_params_.vocab_size};
  logits_ = OrtValue::CreateTensor(*model.allocator_device_, logits_shape_, model_.score_type_);
}

RoamingArray<float> Logits::Get() {
  auto type_shape = logits_->GetTensorTypeAndShapeInfo();

#if USE_CUDA
  if (model_.device_type_ == DeviceType::CUDA) {
    if (model_.score_type_ == Ort::TypeToTensorType<Ort::Float16_t>::type) {
      ConvertFp16ToFp32(*model_.allocator_device_, model_.cuda_stream_, *logits_, logits32_);
      return gpu_span<float>{logits32_->GetTensorMutableData<float>(), type_shape->GetElementCount()};
    }
    return gpu_span<float>{logits_->GetTensorMutableData<float>(), type_shape->GetElementCount()};
  }
#endif

  return cpu_span<float>{logits_->GetTensorMutableData<float>(), type_shape->GetElementCount()};
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
    logits_ = OrtValue::CreateTensor(*model_.allocator_device_, logits_shape_, model_.score_type_);
    state_.outputs_[output_index_] = logits_.get();
  }
}

}  // namespace Generators
