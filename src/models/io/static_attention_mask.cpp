// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "static_attention_mask.h"

#include <algorithm>

namespace Generators {

StaticAttentionMask::StaticAttentionMask(const Model& model, State& state, ONNXTensorElementDataType type, int mask_capacity)
    : AttentionMask{model, state, type},
      mask_capacity_{mask_capacity} {
  if (mask_capacity_ <= 0)
    throw std::runtime_error("StaticAttentionMask - mask capacity must be positive.");
}

void StaticAttentionMask::Initialize(std::unique_ptr<OrtValue> cpu_attention_mask) {
  const auto input_shape = cpu_attention_mask->GetTensorTypeAndShapeInfo()->GetShape();
  if (input_shape[1] > mask_capacity_)
    throw std::runtime_error("StaticAttentionMask - prompt exceeds the static mask capacity.");

  shape_ = {input_shape[0] * state_.params_->search.num_beams, mask_capacity_};
  mask_.CreateTensor(shape_, true);
  if (type_ == Ort::TypeToTensorType<int32_t>)
    CopyInitialMask<int32_t>(*cpu_attention_mask);
  else
    CopyInitialMask<int64_t>(*cpu_attention_mask);
}

template <typename T>
void StaticAttentionMask::CopyInitialMask(OrtValue& cpu_attention_mask) {
  auto output_span = mask_.GetDeviceSpan<T>();
  output_span.Zero();
  auto input_span = WrapTensor<T>(model_.p_device_inputs_->GetCpuFallbackDevice(), cpu_attention_mask);
  const auto input_shape = cpu_attention_mask.GetTensorTypeAndShapeInfo()->GetShape();
  const auto batch_size = input_shape[0];
  const auto prompt_length = input_shape[1];
  const auto num_beams = state_.params_->search.num_beams;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < num_beams; ++j) {
      auto output = output_span.subspan((i * num_beams + j) * mask_capacity_, prompt_length);
      output.CopyFrom(input_span.subspan(i * prompt_length, prompt_length));
    }
  }
}

void StaticAttentionMask::Update(int total_length, int new_length) {
  if (total_length > mask_capacity_)
    throw std::runtime_error("StaticAttentionMask - total_length exceeds the static mask capacity.");
  UpdateOnDevice(nullptr, total_length, new_length, mask_capacity_);
}

void StaticAttentionMask::RewindTo(size_t index) {
  const auto capacity = static_cast<size_t>(mask_capacity_);
  if (index > capacity)
    throw std::runtime_error("StaticAttentionMask::RewindTo - index exceeds mask capacity.");

  auto bytes = mask_.GetByteSpan();
  auto cpu_data = bytes.CpuSpan();
  const auto batch_beam_size = static_cast<size_t>(shape_[0]);
  auto fill = [&]<typename T>() {
    auto* data = reinterpret_cast<T*>(cpu_data.data());
    for (size_t i = 0; i < batch_beam_size; ++i) {
      std::fill_n(data + i * capacity, index, T{1});
      std::fill_n(data + i * capacity + index, capacity - index, T{0});
    }
  };
  if (type_ == Ort::TypeToTensorType<int32_t>)
    fill.template operator()<int32_t>();
  else
    fill.template operator()<int64_t>();
  bytes.CopyCpuToDevice();
}

}  // namespace Generators
