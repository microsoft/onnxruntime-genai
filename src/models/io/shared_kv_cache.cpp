// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_kv_cache.h"

#include "windowed_kv_cache.h"

namespace Generators {

namespace KeyValueCacheDetail {

std::unique_ptr<OrtValue> CreateLogicalKeyValueCacheOutputView(OrtValue& backing_tensor,
                                                               int logical_sequence_length) {
  auto type_and_shape = backing_tensor.GetTensorTypeAndShapeInfo();
  auto logical_shape = type_and_shape->GetShape();
  if (logical_shape.size() != 4) {
    throw std::runtime_error("Shared KV-cache tensors must have rank 4.");
  }
  if (logical_sequence_length < 0 || logical_sequence_length > logical_shape[2]) {
    throw std::runtime_error("Shared KV-cache logical length exceeds its backing allocation.");
  }

  logical_shape[2] = logical_sequence_length;
  const auto type = type_and_shape->GetElementType();
  const size_t capacity_bytes = type_and_shape->GetElementCount() * Ort::SizeOf(type);
  return OrtValue::CreateTensor(
      backing_tensor.GetTensorMemoryInfo(), backing_tensor.GetTensorMutableRawData(), capacity_bytes,
      logical_shape, type);
}

}  // namespace KeyValueCacheDetail

void SharedKeyValueCache::Update(DeviceSpan<int32_t> /*beam_indices*/, int total_length) {
  current_length_ = total_length;

  if (!Device().UsesLogicalKeyValueCacheOutputViews())
    return;

  const int output_length = state_.params_->use_graph_capture
                                ? state_.params_->search.max_length
                                : total_length;
  if (output_length == output_view_length_)
    return;

  std::vector<std::unique_ptr<OrtValue>> output_views;
  output_views.reserve(presents_.size());
  for (const auto& present : presents_) {
    output_views.push_back(
        KeyValueCacheDetail::CreateLogicalKeyValueCacheOutputView(*present, output_length));
  }

  for (size_t i = 0; i < output_views.size(); ++i)
    state_.outputs_[output_index_ + i] = output_views[i].get();
  output_views_ = std::move(output_views);
  output_view_length_ = output_length;
}

void SharedKeyValueCache::RewindTo(size_t index) {
  CheckWindowedKvCacheRewind(windowed_cache_size_, current_length_, index);
}

}  // namespace Generators
