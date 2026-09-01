// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kv_cache.h"

namespace Generators {

bool HasConvKeyValueCache(const Model& model);

// Combines attention KV cache layers with fixed-size convolution state layers.
struct ConvKeyValueCache : KeyValueCache {
  ConvKeyValueCache(State& state);

  void Add() override;
  void Update(DeviceSpan<int32_t> beam_indices, int total_length) override;
  void RewindTo(size_t index) override;

 private:
  template <typename ScoreType>
  void PickPastState(DeviceSpan<int32_t> beam_indices, int index);
  void PickPastState(DeviceSpan<int32_t> beam_indices, int index);

  template <typename T>
  void PickConvState(DeviceSpan<int32_t> beam_indices, int conv_index);

  DeviceInterface& Device() { return *model_.p_device_kvcache_; }
  Ort::Allocator& Allocator() { return model_.p_device_kvcache_->GetAllocator(); }

  State& state_;
  const Model& model_{state_.model_};

  // Layer type info
  std::vector<std::string> layer_types_;  // "conv" or "full_attention" per layer
  int layer_count_;

  // KV cache (only for attention layers)
  int kv_layer_count_{};
  std::vector<int> kv_layer_indices_;
  std::array<int64_t, 4> kv_shape_;
  ONNXTensorElementDataType kv_type_;
  std::unique_ptr<OrtValue> kv_empty_past_;
  std::vector<std::unique_ptr<OrtValue>> kv_pasts_, kv_presents_;
  std::vector<std::string> kv_input_name_strings_, kv_output_name_strings_;
  size_t kv_input_index_{~0U}, kv_output_index_{~0U};
  bool kv_is_first_update_{true};
  // When true, attention KV uses one fixed-size buffer for past and present (same OrtValue
  // bound to both). Enabled the same way as DefaultKeyValueCache: via
  // IsPastPresentShareBufferEnabled (genai search options) and/or a fixed kv seq_len
  // detected in the model graph by DetectAndConfigureFixedKvShape. Sequence capacity is
  // the detected fixed dim when present, else search.max_length.
  bool kv_share_buffer_{false};

  // Conv state cache (only for conv layers)
  int conv_layer_count_{};
  std::vector<int> conv_layer_indices_;
  std::vector<std::array<int64_t, 3>> conv_shapes_;  // Per-layer conv state shape [B, H, L]
  ONNXTensorElementDataType conv_type_;
  std::vector<std::unique_ptr<OrtValue>> conv_pasts_, conv_presents_;
  std::vector<std::string> conv_input_name_strings_, conv_output_name_strings_;
  size_t conv_input_index_{~0U}, conv_output_index_{~0U};
  bool conv_is_first_update_{true};
};

}  // namespace Generators
