// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dynamic_kv_cache.h"

#include <algorithm>
#include <cassert>

namespace Generators {

void DynamicKeyValueCache::Update(DeviceSpan<int32_t> beam_indices, int total_length) {
  current_length_ = total_length;

  if (!is_first_update_) {
    for (int i = 0; i < layer_count_ * 2; i++) {
      if (beam_indices.empty()) {
        pasts_[i] = std::move(presents_[i]);
      } else {
        PickPastState(beam_indices, i);
      }
      state_.inputs_[input_index_ + i] = pasts_[i].get();
    }
  }

  if (!layer_shapes_.empty()) {
    for (int layer_idx = 0; layer_idx < layer_count_; ++layer_idx) {
      std::array<int64_t, 4> current_shape = layer_shapes_[layer_idx];
      const int max_cache_length = static_cast<int>(layer_shapes_[layer_idx][2]);
      current_shape[2] = (max_cache_length > 0) ? std::min(total_length, max_cache_length) : total_length;

      presents_[layer_idx * 2] = OrtValue::CreateTensor(Allocator(), current_shape, type_);
      state_.outputs_[output_index_ + layer_idx * 2] = presents_[layer_idx * 2].get();

      presents_[layer_idx * 2 + 1] = OrtValue::CreateTensor(Allocator(), current_shape, type_);
      state_.outputs_[output_index_ + layer_idx * 2 + 1] = presents_[layer_idx * 2 + 1].get();
    }
  } else {
    shape_[2] = total_length;
    for (int i = 0; i < layer_count_ * 2; i++) {
      presents_[i] = OrtValue::CreateTensor(Allocator(), shape_, type_);
      state_.outputs_[output_index_ + i] = presents_[i].get();
    }
  }

  is_first_update_ = false;
}

void DynamicKeyValueCache::RewindTo(size_t index) {
  if (shape_[2] <= static_cast<int>(index)) {
    throw std::runtime_error("Requested length of rewind is greater than the current length.");
  }

  is_first_update_ = true;
  if (index == 0) {
    for (int i = 0; i < layer_count_ * 2; i++) {
      pasts_[i] = nullptr;
      if (!empty_pasts_.empty()) {
        state_.inputs_[input_index_ + i] = empty_pasts_[i / 2].get();
      } else {
        state_.inputs_[input_index_ + i] = empty_past_.get();
      }
    }
  } else if (type_ == Ort::TypeToTensorType<float>) {
    RewindPastTensorsTo<float>(index);
  } else if (type_ == Ort::TypeToTensorType<int8_t>) {
    RewindPastTensorsTo<int8_t>(index);
  } else if (type_ == Ort::TypeToTensorType<uint8_t> || type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN) {
    RewindPastTensorsTo<uint8_t>(index);
  } else {
    RewindPastTensorsTo<Ort::Float16_t>(index);
  }
}

template <typename T>
void DefaultKeyValueCacheBase::RewindPastTensorsTo(size_t index) {
  assert(index > 0 && !past_present_share_buffer_);

  if (!layer_shapes_.empty()) {
    const int max_length = static_cast<int>(shape_[2]);
    if (static_cast<int>(index) > max_length) {
      throw std::runtime_error("Requested rewind length exceeds max_length.");
    }

    for (int i = 0; i < layer_count_ * 2; i++) {
      const int layer_idx = i / 2;
      const std::array<int64_t, 4> layer_shape = layer_shapes_[layer_idx];
      const int layer_max_cache = static_cast<int>(layer_shape[2]);
      const int actual_rewind_length = std::min(static_cast<int>(index), layer_max_cache);

      std::array<int64_t, 4> new_shape = layer_shape;
      new_shape[2] = actual_rewind_length;
      const auto batch_x_num_heads = new_shape[0] * new_shape[1];
      const auto new_length_x_head_size = new_shape[2] * new_shape[3];

      OrtValue& present = *presents_[i];
      const auto present_shape = present.GetTensorTypeAndShapeInfo()->GetShape();
      const auto old_length_x_head_size = present_shape[2] * new_shape[3];

      std::unique_ptr<OrtValue> past = OrtValue::CreateTensor(Allocator(), new_shape, type_);
      auto past_span = KeyValueCacheDetail::WrapKvCacheTensor<T>(Device(), *past, type_);
      auto present_span = KeyValueCacheDetail::WrapKvCacheTensor<T>(Device(), present, type_);

      for (int j = 0; j < batch_x_num_heads; j++) {
        auto present_data = present_span.subspan(j * old_length_x_head_size, new_length_x_head_size);
        auto past_data = past_span.subspan(j * new_length_x_head_size, new_length_x_head_size);
        past_data.CopyFrom(present_data);
      }
      pasts_[i] = std::move(past);
      state_.inputs_[input_index_ + i] = pasts_[i].get();
    }
  } else {
    assert(shape_[2] >= static_cast<int64_t>(index));
    std::array<int64_t, 4> new_shape = shape_;
    new_shape[2] = static_cast<int>(index);
    auto batch_x_num_heads = new_shape[0] * new_shape[1];
    auto new_length_x_head_size = new_shape[2] * new_shape[3];
    auto old_length_x_head_size = shape_[2] * new_shape[3];
    shape_[2] = new_shape[2];

    for (int i = 0; i < layer_count_ * 2; i++) {
      OrtValue& present = *presents_[i];
      std::unique_ptr<OrtValue> past = OrtValue::CreateTensor(Allocator(), shape_, type_);

      auto past_span = KeyValueCacheDetail::WrapKvCacheTensor<T>(Device(), *past, type_);
      auto present_span = KeyValueCacheDetail::WrapKvCacheTensor<T>(Device(), present, type_);

      for (int j = 0; j < batch_x_num_heads; j++) {
        auto present_data = present_span.subspan(j * old_length_x_head_size, new_length_x_head_size);
        auto past_data = past_span.subspan(j * new_length_x_head_size, new_length_x_head_size);
        past_data.CopyFrom(present_data);
      }
      pasts_[i] = std::move(past);
      state_.inputs_[input_index_ + i] = pasts_[i].get();
    }
  }
}

}  // namespace Generators
