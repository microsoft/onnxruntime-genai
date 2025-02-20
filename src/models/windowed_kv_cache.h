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

  bool IsPartialTokenGenerationUpdateSupported() const override { return true; }

  void PartialTokenGenerationUpdate(DeviceSpan<int32_t> beam_indices, int total_length,
                                    std::span<const size_t> layer_indices_to_update) override;

  void RewindTo(size_t index) override {
    throw std::runtime_error("WindowedKeyValueCache does not support RewindTo.");
  }

 private:
  void SlideLayer(size_t layer_idx);
  void SlideAllLayers();
  void SlideLayers(std::span<const size_t> layer_indices);

  DeviceInterface& Device() { return *model_.p_device_kvcache_; }
  Ort::Allocator& Allocator() { return model_.p_device_kvcache_->GetAllocator(); }

  State& state_;
  const Model& model_{state_.model_};
  int layer_count_{};
  int window_size_{};
  size_t num_windows_{};
  size_t window_index_{};
  size_t input_index_{~0U}, output_index_{~0U};

  std::array<int64_t, 4> key_cache_shape_in_, key_cache_shape_out_;
  std::array<int64_t, 4> value_cache_shape_in_, value_cache_shape_out_;
  ONNXTensorElementDataType type_;

  std::vector<std::unique_ptr<OrtValue>> key_caches_in_, value_caches_in_;
  std::vector<std::unique_ptr<OrtValue>> key_caches_out_, value_caches_out_;
  std::vector<std::string> input_name_strings_, output_name_strings_;

  bool is_first_update_{true};
};

}  // namespace Generators
