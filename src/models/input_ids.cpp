#include "../generators.h"
#include "model.h"
#include "input_ids.h"

namespace Generators {

InputIDs::InputIDs(Model& model, const SearchParams& search_params, Ort::Allocator& allocator)
    : model_{model},
      allocator_{allocator} {
  input_ids_shape_ = {search_params.batch_size, search_params.sequence_length};
  input_ids_ = OrtValue::CreateTensor<int32_t>(model.allocator_cpu_.GetInfo(), std::span<int32_t>(const_cast<int32_t*>(search_params.input_ids.data()), input_ids_shape_[0] * input_ids_shape_[1]), input_ids_shape_);
  input_ids_ = ExpandInputs(input_ids_, search_params.num_beams, allocator, model_.device_type_, model_.cuda_stream_);
  input_ids_shape_[0] *= search_params.num_beams;
}

void InputIDs::Update(RoamingArray<int32_t> next_tokens) {
  // Resize input_ids shape once if it doesn't match the decoder shape
  if (input_ids_shape_[1] != 1) {
    input_ids_shape_[1] = 1;
    input_ids_ = OrtValue::CreateTensor<int32_t>(allocator_, input_ids_shape_);
  }

  // Update input_ids with next tokens
  auto* input_ids_data = input_ids_->GetTensorMutableData<int32_t>();
#if USE_CUDA
  if (model_.device_type_ == DeviceType::CUDA)
    cudaMemcpyAsync(input_ids_data, next_tokens.GetGPU().data(), input_ids_shape_[0] * sizeof(int32_t), cudaMemcpyDeviceToDevice, model_.cuda_stream_);
  else
#endif
    memcpy(input_ids_data, next_tokens.GetCPU().data(), input_ids_shape_[0] * sizeof(int32_t));
}

}  // namespace Generators
