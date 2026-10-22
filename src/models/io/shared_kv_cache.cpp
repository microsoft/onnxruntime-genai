// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shared_kv_cache.h"

#include "windowed_kv_cache.h"

#include <algorithm>

namespace Generators {

namespace KeyValueCacheDetail {

std::unique_ptr<OrtValue> CreateLogicalKeyValueCacheOutputView(OrtValue& backing_tensor,
                                                               int64_t logical_sequence_length) {
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

int64_t GetLogicalKeyValueCacheOutputLength(OrtValue& backing_tensor,
                                            int64_t output_sequence_dimension,
                                            int total_length,
                                            bool use_graph_capture) {
  const auto shape = backing_tensor.GetTensorTypeAndShapeInfo()->GetShape();
  if (shape.size() != 4) {
    throw std::runtime_error("Shared KV-cache tensors must have rank 4.");
  }

  const int64_t capacity = shape[2];
  if (output_sequence_dimension > 0) {
    return output_sequence_dimension;
  }
  return use_graph_capture ? capacity : std::min<int64_t>(total_length, capacity);
}

}  // namespace KeyValueCacheDetail

SharedKeyValueCache::SharedKeyValueCache(State& state)
    : DefaultKeyValueCacheBase{state} {
  output_sequence_dimensions_.reserve(output_name_strings_.size());
  for (const auto& output_name : output_name_strings_) {
    const auto output_shape = model_.session_info_.GetOutputShape(output_name);
    if (output_shape.size() != 4) {
      throw std::runtime_error("Shared KV-cache outputs must have rank 4.");
    }
    output_sequence_dimensions_.push_back(output_shape[2]);
  }
}

void SharedKeyValueCache::Update(DeviceSpan<int32_t> /*beam_indices*/, int total_length) {
  current_length_ = total_length;

  if (!Device().UsesLogicalKeyValueCacheOutputViews())
    return;

  std::vector<int64_t> output_lengths;
  output_lengths.reserve(presents_.size());
  for (size_t i = 0; i < presents_.size(); ++i) {
    output_lengths.push_back(KeyValueCacheDetail::GetLogicalKeyValueCacheOutputLength(
        *presents_[i], output_sequence_dimensions_[i], total_length,
        state_.params_->use_graph_capture));
  }

  if (output_lengths == output_view_lengths_)
    return;

  std::vector<std::unique_ptr<OrtValue>> output_views;
  output_views.reserve(presents_.size());
  for (size_t i = 0; i < presents_.size(); ++i) {
    output_views.push_back(
        KeyValueCacheDetail::CreateLogicalKeyValueCacheOutputView(*presents_[i], output_lengths[i]));
  }

  for (size_t i = 0; i < output_views.size(); ++i)
    state_.outputs_[output_index_ + i] = output_views[i].get();
  output_views_ = std::move(output_views);
  output_view_lengths_ = std::move(output_lengths);
}

void SharedKeyValueCache::RewindTo(size_t index) {
  CheckWindowedKvCacheRewind(windowed_cache_size_, current_length_, index);
}

}  // namespace Generators
