// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../model.h"
#include <type_traits>

namespace Generators {

namespace KeyValueCacheDetail {
// KV cache tensors are copied/reordered as fixed-width elements. FLOAT8E4M3FN shares the
// 1-byte width of uint8_t and is moved as raw bytes, but WrapTensor<uint8_t> asserts on the
// tensor's element type, so wrap float8 tensors via ByteWrapTensor instead.
template <typename T>
DeviceSpan<T> WrapKvCacheTensor(DeviceInterface& device, OrtValue& value, ONNXTensorElementDataType type) {
  if constexpr (std::is_same_v<T, uint8_t>) {
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN) {
      return ByteWrapTensor(device, value);
    }
  }
  return WrapTensor<T>(device, value);
}
}  // namespace KeyValueCacheDetail

struct KeyValueCache {
  virtual ~KeyValueCache() = default;

  virtual void Add() = 0;

  virtual void Update(DeviceSpan<int32_t> beam_indices, int total_length) = 0;

  virtual void RewindTo(size_t index) = 0;

  // Note: PartialUpdate() is mainly for supporting DecoderOnlyPipelineState usage where we update
  // part of the KV cache after running part of the pipeline.
  // An alternative may be to have a dedicated KV cache per IntermediatePipelineState.

  virtual bool IsPartialUpdateSupported() const { return false; }

  virtual void PartialUpdate(DeviceSpan<int32_t> beam_indices, int total_length,
                             std::span<const size_t> layer_indices_to_update) {
    throw std::runtime_error("PartialUpdate is not supported.");
  }
};

std::string ComposeKeyValueName(const std::string& template_string, int index);

// Returns true when the runtime uses a non-rewindable windowed KV cache.
bool UsesNonRewindableWindowedKeyValueCache(
    const Model& model, const Config::Model::Decoder& decoder);

// Standard exposed past/present cache policy shared by EPs that do not provide a custom design.
std::unique_ptr<KeyValueCache> CreateStandardKeyValueCache(State& state);

// No-op binding policy for EP-managed stateful models.
std::unique_ptr<KeyValueCache> CreateModelManagedKeyValueCache(State& state);

}  // namespace Generators
