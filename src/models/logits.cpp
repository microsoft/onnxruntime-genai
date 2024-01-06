#include "../generators.h"
#include "model.h"
#include "logits.h"

namespace Generators {

Logits::Logits(const SearchParams& search_params, DeviceType device_type, Ort::Allocator& allocator, cudaStream_t cuda_stream, ONNXTensorElementDataType score_type, bool uses_seq_length)
    : device_type_{device_type},
      cuda_stream_{cuda_stream},
      allocator_{allocator},
      score_type_{score_type} {

  logits_shape_ = {search_params.batch_size * search_params.num_beams, uses_seq_length ? search_params.sequence_length : 1, search_params.vocab_size};
  logits_ = OrtValue::CreateTensor(allocator_, logits_shape_, score_type_);
}

RoamingArray<float> Logits::Get() {
  auto type_shape = logits_->GetTensorTypeAndShapeInfo();

#if USE_CUDA
  if (device_type_ == DeviceType::CUDA) {
    if (score_type_ == Ort::TypeToTensorType<Ort::Float16_t>::type) {
      ConvertFp16ToFp32(allocator_, cuda_stream_, *logits_, logits32_);
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
    logits_ = OrtValue::CreateTensor(allocator_, logits_shape_, score_type_);
  }
}

}  // namespace Generators
