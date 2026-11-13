// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "kv_cache.h"

namespace Generators {

int64_t DetectAndConfigureFixedKvShape(const SessionInfo& session_info,
                                       const std::vector<std::string>& input_name_strings,
                                       int layer_count,
                                       const Config::Search& search,
                                       bool& past_present_share_buffer,
                                       const char* cache_name);

int GetWindowedKeyValueCacheSize(const Model& model, const Config::Search& search, int max_length);

// Abstract base for exposed past/present KV-cache variants. It owns the common
// layer-name discovery, shape planning, tensor binding, beam-reorder, and rewind
// copy helpers; concrete variants implement their own update/rewind policy.
struct DefaultKeyValueCacheBase : KeyValueCache {
  DefaultKeyValueCacheBase(State& state);

  virtual ~DefaultKeyValueCacheBase() = default;

  void Add() override;
  void Update(DeviceSpan<int32_t> beam_indices, int total_length) override = 0;
  void RewindTo(size_t index) override = 0;

  auto& GetShape() const { return shape_; }
  auto& GetType() const { return type_; }
  auto& GetPresents() { return presents_; }

 protected:
  template <typename ScoreType>
  void PickPastState(DeviceSpan<int32_t> beam_indices, int index);
  void PickPastState(DeviceSpan<int32_t> beam_indices, int index);

  template <typename T>
  void RewindPastTensorsTo(size_t index);

  static void ValidatePastPresentLocations(const std::string& input_name, const std::string& output_name,
                                           const OrtMemoryInfo& input, const OrtMemoryInfo& output);
  void InitializeCacheAllocations(int kv_cache_quantization_bits);

  DeviceInterface& Device(size_t index = 0) {
    return cache_allocations_.empty() ? *model_.p_device_kvcache_ : *cache_allocations_[index].device;
  }
  Ort::Allocator& Allocator(size_t index) {
    return cache_allocations_.empty() ? model_.p_device_kvcache_->GetAllocator() : *cache_allocations_[index].allocator;
  }

  State& state_;
  const Model& model_{state_.model_};
  int layer_count_;
  size_t input_index_{~0U}, output_index_{~0U};
  bool past_present_share_buffer_;

  bool is_first_update_{true};

  std::array<int64_t, 4> shape_;
  ONNXTensorElementDataType type_;

  // Auto-discovered KV layer indices (sparse for hybrid models)
  std::vector<int> kv_layer_indices_;

  // Support for per-layer KV cache shapes (for models with alternating attention patterns)
  std::vector<std::array<int64_t, 4>> layer_shapes_;

  // Sequence capacity allocated for sliding-window layers, or 0 when no layer is windowed.
  // Only the most recent windowed_cache_size_ positions of those layers are retained, which bounds
  // how far RewindTo can go back.
  int windowed_cache_size_{0};
  int current_length_{0};  // Total sequence length seen by the most recent Update()

  // Allocators must outlive the tensors allocated through them.
  std::vector<CacheAllocation> cache_allocations_;
  std::vector<std::shared_ptr<OrtValue>> empty_pasts_;
  std::vector<std::unique_ptr<OrtValue>> pasts_, presents_;
  std::vector<std::string> input_name_strings_, output_name_strings_;
};

using DefaultKeyValueCache = DefaultKeyValueCacheBase;

bool ShouldUseSharedPastPresentKeyValueCache(State& state);

}  // namespace Generators
