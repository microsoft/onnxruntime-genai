// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_mask.h"

#include "dynamic_attention_mask.h"
#include "static_attention_mask.h"

namespace Generators {

AttentionMask::AttentionMask(const Model& model, State& state, ONNXTensorElementDataType type)
    : model_{model}, state_{state}, type_{type}, mask_{model.p_device_inputs_, type} {}

void AttentionMask::UpdateOnDevice(Tensor* next_mask, int total_length, int new_length, int mask_capacity) {
  if (shape_[0] != 1 && !(total_length == 0 || new_length == 1))
    throw std::runtime_error("AttentionMask::Update - batch_size must be 1 for continuous decoding.");

  // A null destination updates the fixed-capacity mask in place.
  const bool in_place = next_mask == nullptr;
  if (!model_.p_device_inputs_->UpdateAttentionMask(
          next_mask ? next_mask->GetMutableRawData() : nullptr,
          mask_.GetMutableRawData(), static_cast<int>(shape_[0]),
          new_length, total_length, mask_capacity, in_place, type_)) {
    DeviceSpan<uint8_t> next_span;
    if (next_mask)
      next_span = next_mask->GetByteSpan();
    auto mask_span = mask_.GetByteSpan();
    model_.p_device_inputs_->GetCpuFallbackDevice().UpdateAttentionMask(
        next_mask ? next_span.CopyDeviceToCpu().data() : nullptr,
        mask_span.CopyDeviceToCpu().data(), static_cast<int>(shape_[0]),
        new_length, total_length, mask_capacity, in_place, type_);
    if (next_mask)
      next_span.CopyCpuToDevice();
    mask_span.CopyCpuToDevice();
  }
}

std::unique_ptr<AttentionMask> CreateAttentionMask(const Model& model, State& state, ONNXTensorElementDataType type,
                                                   std::optional<int> static_mask_capacity) {
  if (static_mask_capacity || state.params_->use_graph_capture ||
      (state.params_->IsPastPresentShareBufferEnabled(model.config_->model.type) &&
       model.p_device_inputs_->ShouldUseStaticPositionInputsForSharedBuffers(model.config_->model))) {
    return std::make_unique<StaticAttentionMask>(model, state, type, static_mask_capacity.value_or(state.params_->search.max_length));
  }
  return std::make_unique<DynamicAttentionMask>(model, state, type);
}

}  // namespace Generators
