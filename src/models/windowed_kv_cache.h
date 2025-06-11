// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kv_cache.h"

namespace Generators {

struct WindowedKeyValueCache : KeyValueCache {
  WindowedKeyValueCache(State& state);

  void Add() override;
  void AddEncoder() override {
    throw std::runtime_error("WindowedKeyValueCache does not support AddEncoder.");
  };

  void Update(DeviceSpan<int32_t> beam_indices, int current_length) override;

  bool IsPartialUpdateSupported() const override { return true; }

  void PartialUpdate(DeviceSpan<int32_t> beam_indices, int total_length,
                     std::span<const size_t> layer_indices_to_update) override;

  void RewindTo(size_t index) override {
    throw std::runtime_error("WindowedKeyValueCache does not support RewindTo.");
  }

 private:
  using CacheTensorShape = std::array<int64_t, 4>;

  struct LayerState {
    size_t window_index{0};
    size_t window_size{};
    size_t num_windows{};
    bool is_first_update{true};

    CacheTensorShape key_cache_shape_in{}, key_cache_shape_out{};
    CacheTensorShape value_cache_shape_in{}, value_cache_shape_out{};
  };

  static std::vector<LayerState> MakeInitialPerLayerStates(size_t layer_count,
                                                           size_t initial_window_size,
                                                           const CacheTensorShape& initial_key_cache_shape_in,
                                                           const CacheTensorShape& initial_key_cache_shape_out,
                                                           const CacheTensorShape& initial_value_cache_shape_in,
                                                           const CacheTensorShape& initial_value_cache_shape_out);

  void SlideLayer(size_t layer_idx);
  void TransitionLayerToTokenGeneration(size_t layer_idx);
  void UpdateLayer(DeviceSpan<int32_t> beam_indices, int total_length, size_t layer_idx);

  DeviceInterface& Device() { return *model_.p_device_kvcache_; }
  Ort::Allocator& Allocator() { return model_.p_device_kvcache_->GetAllocator(); }

  // Note: The KV cache may be partially updated by multiple threads. However, the updates should happen at a per-layer
  // granularity. Within a single layer's state, there should not be any shared state that needs to be synchronized
  // between multiple threads.

  State& state_;
  const Model& model_{state_.model_};
  const size_t layer_count_;

  std::vector<LayerState> per_layer_states_;

  size_t input_index_{~0U}, output_index_{~0U};

  ONNXTensorElementDataType type_;

  std::vector<std::unique_ptr<OrtValue>> key_caches_in_, value_caches_in_;
  std::vector<std::unique_ptr<OrtValue>> key_caches_out_, value_caches_out_;
  std::vector<std::string> input_name_strings_, output_name_strings_;

  const std::vector<size_t> all_layer_indices_;
};

}  // namespace Generators
