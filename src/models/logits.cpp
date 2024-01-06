#include "../generators.h"
#include "model.h"
#include "logits.h"

namespace Generators {

Logits::Logits(Model& model, const SearchParams& search_params)
    : model_{model} {

  logits_shape_ = {search_params.batch_size * search_params.num_beams, model_.logits_uses_seq_len_ ? search_params.sequence_length : 1, search_params.vocab_size};
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

void Logits::Update() {
  // Resize the logits shape once if it doesn't match the decoder shape
  if (logits_shape_[1] != 1) {
    logits_shape_[1] = 1;
    logits_ = OrtValue::CreateTensor(*model_.allocator_device_, logits_shape_, model_.score_type_);
  }
}

}  // namespace Generators
