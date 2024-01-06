#include "../generators.h"
#include "model.h"
#include "input_ids.h"

namespace Generators {

template<typename T>
InputIDs<T>::InputIDs(Model& model, const SearchParams& search_params)
    : model_{model} {
  input_ids_shape_ = {search_params.batch_size, search_params.sequence_length};

  // If 64-bit, convert from 32-bit to 64-bit
  if constexpr (std::is_same_v<T, int64_t>) {
    input_ids_ = OrtValue::CreateTensor<int64_t>(model.allocator_cpu_, input_ids_shape_);
    auto* p_data = input_ids_->GetTensorMutableData<T>();
    for (auto v : search_params.input_ids)
      *p_data++ = v;
  } else {
    static_assert(std::is_same_v<T, int32_t>);
    input_ids_ = OrtValue::CreateTensor<T>(model.allocator_cpu_.GetInfo(), std::span<T>(const_cast<T*>(search_params.input_ids.data()), input_ids_shape_[0] * input_ids_shape_[1]), input_ids_shape_);
  }

  input_ids_ = ExpandInputs(input_ids_, search_params.num_beams, *model.allocator_device_, model.device_type_, model.cuda_stream_);
  input_ids_shape_[0] *= search_params.num_beams;
}

template<typename T>
void InputIDs<T>::Update(RoamingArray<int32_t> next_tokens_unk) {
  // Resize input_ids shape once if it doesn't match the decoder shape
  if (input_ids_shape_[1] != 1) {
    input_ids_shape_[1] = 1;
    input_ids_ = OrtValue::CreateTensor<T>(*model_.allocator_device_, input_ids_shape_);
  }

  auto* data = input_ids_->GetTensorMutableData<T>();
  // Update input_ids with next tokens, converting from 32-bit to 64-bit
  if constexpr (std::is_same_v<T, int64_t>) {
#if USE_CUDA
    if (model_.device_type_ == DeviceType::CUDA) {
      auto next_tokens = next_tokens_unk.GetGPU();
      cuda::LaunchInt32ToInt64(next_tokens.data(), data, static_cast<int>(next_tokens.size()), model_.cuda_stream_);
    } else
#endif
    {
      auto next_tokens = next_tokens_unk.GetCPU();
      for (int i = 0; i < input_ids_shape_[0]; i++)
        data[i] = next_tokens[i];
    }
  } else {
    static_assert(std::is_same_v<T, int32_t>);
#if USE_CUDA
    if (model_.device_type_ == DeviceType::CUDA)
      cudaMemcpyAsync(data, next_tokens_unk.GetGPU().data(), input_ids_shape_[0] * sizeof(T), cudaMemcpyDeviceToDevice, model_.cuda_stream_);
    else
#endif
      memcpy(data, next_tokens_unk.GetCPU().data(), input_ids_shape_[0] * sizeof(T));
  }
}

template InputIDs<int32_t>;
template InputIDs<int64_t>;

}  // namespace Generators
