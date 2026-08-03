// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kv_cache.h"

namespace Generators {

// Fixed-capacity KV cache for decoder graphs that update past tensors in place
// with ONNX TensorScatter. The graph must expose cache_write_indices plus rank-4
// past/present key and value pairs with matching fixed non-batch dimensions.
//
// Usage:
//   1. Construct this in the decode State and call Add().
//   2. Call Initialize() once with compact KV tensors produced by prefill.
//   3. Before each single-token decode, call Update() with the new total length.
//
// Past and present bind to the same OrtValue, so Update() only changes
// cache_write_indices; it does not swap or reallocate cache tensors. Beam
// reordering and rewind are not supported.
struct TensorScatterKeyValueCache final : KeyValueCache {
  explicit TensorScatterKeyValueCache(State& state);

  void Add() override;
  void Update(DeviceSpan<int32_t> beam_indices, int total_length) override;
  void RewindTo(size_t index) override;

  void Initialize(const std::vector<std::unique_ptr<OrtValue>>& compact_values);

 private:
  State& state_;
  int cache_sequence_length_{};
  int layer_count_;
  std::string cache_write_indices_name_;
  ONNXTensorElementDataType cache_write_indices_type_;
  std::unique_ptr<Tensor> cache_write_indices_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::unique_ptr<OrtValue>> values_;
};

}  // namespace Generators
