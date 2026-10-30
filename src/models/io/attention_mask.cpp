// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_mask.h"

namespace Generators {

// Both mask policies use the same device operation and CPU fallback.
void AttentionMaskBuffer::UpdateMaskOnDevice(Tensor& mask, Tensor* next, const std::array<int64_t, 2>& shape,
                                             int total_length, int new_length, int max_length) {
  auto& device = *mask.p_device_;
  if (!device.UpdateAttentionMask(next ? next->GetMutableRawData() : nullptr, mask.GetMutableRawData(),
                                  static_cast<int>(shape[0]), new_length, total_length, max_length,
                                  next == nullptr, mask.GetType())) {
    auto mask_span = mask.GetByteSpan();
    auto next_span = next ? next->GetByteSpan() : DeviceSpan<uint8_t>{};
    device.GetCpuFallbackDevice().UpdateAttentionMask(
        next ? next_span.CopyDeviceToCpu().data() : nullptr, mask_span.CopyDeviceToCpu().data(),
        static_cast<int>(shape[0]), new_length, total_length, max_length, next == nullptr, mask.GetType());
    if (next)
      next_span.CopyCpuToDevice();
    else
      mask_span.CopyCpuToDevice();
  }
}

std::unique_ptr<AttentionMaskBuffer> CreateAttentionMaskBuffer(
    const Model& model, const GeneratorParams& params, DynamicMaskUpdate update) {
  if (params.use_graph_capture ||
      (params.IsPastPresentShareBufferEnabled(model.config_->model.type) &&
       model.p_device_inputs_->ShouldUseStaticPositionInputsForSharedBuffers(model.config_->model))) {
    return CreateStaticAttentionMaskBuffer(params.search.max_length,
                                           model.p_device_inputs_->GetType() == DeviceType::WEBGPU);
  }
  return CreateDynamicAttentionMaskBuffer(model, update);
}

}  // namespace Generators
