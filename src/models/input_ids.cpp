#include "../generators.h"
#include "model.h"
#include "input_ids.h"

namespace Generators {

template <typename T>
InputIDs<T>::InputIDs(Model& model, State& state)
    : model_{model},
      state_{state} {
  shape_ = {state_.search_params_.batch_size, state_.search_params_.sequence_length};

  // If 64-bit, convert from 32-bit to 64-bit
  if constexpr (std::is_same_v<T, int64_t>) {
    value_ = OrtValue::CreateTensor<int64_t>(model.allocator_cpu_, shape_);
    auto* p_data = value_->GetTensorMutableData<T>();
    for (auto v : state_.search_params_.input_ids)
      *p_data++ = v;
  } else {
    static_assert(std::is_same_v<T, int32_t>);
    value_ = OrtValue::CreateTensor<T>(model.allocator_cpu_.GetInfo(), std::span<T>(const_cast<T*>(state_.search_params_.input_ids.data()), shape_[0] * shape_[1]), shape_);
  }

  value_ = model_.ExpandInputs(value_, state_.search_params_.num_beams);
  shape_[0] *= state_.search_params_.num_beams;
}

template <typename T>
void InputIDs<T>::Add() {
  input_index_ = state_.inputs_.size();

  state_.inputs_.push_back(value_.get());
  state_.input_names_.push_back(name_);
}

template <typename T>
void InputIDs<T>::Update(RoamingArray<int32_t> next_tokens_unk) {
  // Resize input_ids shape once if it doesn't match the decoder shape
  if (shape_[1] != 1) {
    shape_[1] = 1;
    value_ = OrtValue::CreateTensor<T>(*model_.allocator_device_, shape_);
    state_.inputs_[input_index_] = value_.get();
  }

  auto* data = value_->GetTensorMutableData<T>();
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
      for (int i = 0; i < shape_[0]; i++)
        data[i] = next_tokens[i];
    }
  } else {
    static_assert(std::is_same_v<T, int32_t>);
#if USE_CUDA
    if (model_.device_type_ == DeviceType::CUDA)
      cudaMemcpyAsync(data, next_tokens_unk.GetGPU().data(), shape_[0] * sizeof(T), cudaMemcpyDeviceToDevice, model_.cuda_stream_);
    else
#endif
      memcpy(data, next_tokens_unk.GetCPU().data(), shape_[0] * sizeof(T));
  }
}

template struct InputIDs<int32_t>;
template struct InputIDs<int64_t>;

}  // namespace Generators
