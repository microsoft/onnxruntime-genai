// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantized_kv_cache.h"

#include <stdexcept>

namespace Generators {

// Compute the compressed KV cache head dimension for quantized KV caches.
// The quantizer packs each head into (1 + head_size / indices_per_word) u32 words: one fp32 scale followed by
// head_size values quantized to 4 or 8 bits (4-bit: 8 values/u32, 8-bit: 4 values/u32).
// The tensor dimension depends on the element type.
// If kv_cache_quantization_bits is enabled with an invalid head_size (< 8 or non-power-of-2),
// this throws instead of silently falling back.
//
// NOTE: this is the allocator side of a contract that ONNX shape inference cannot express
// (it still reports the uncompressed head_size). The same formula lives in the WebGPU kernel
// at onnxruntime/contrib_ops/webgpu/bert/turbo_quant_hadamard.h (see the "present-KV allocator
// contract" comment) and must be kept in sync; the kernel validates the resulting buffer size
// at runtime and fails with INVALID_ARGUMENT on a mismatch.
int64_t ComputeQuantizedKvCacheHeadSize(int head_size, int kv_cache_quantization_bits, ONNXTensorElementDataType type) {
  if (kv_cache_quantization_bits == 4 || kv_cache_quantization_bits == 8) {
    const bool is_power_of_two = head_size > 0 && (head_size & (head_size - 1)) == 0;
    if (head_size < 8 || !is_power_of_two) {
      throw std::runtime_error(
          "KV cache quantization requires head_size to be a power of 2 and >= 8, but got head_size=" +
          std::to_string(head_size) + ".");
    }
    // 4-bit packs 8 indices per u32; 8-bit packs 4 indices per u32. One extra u32 holds the scale.
    const int indices_per_word = (kv_cache_quantization_bits == 8) ? 4 : 8;
    const int compressed_u32_words = head_size / indices_per_word + 1;
    const int bytes_per_element = static_cast<int>(Ort::SizeOf(type));
    return static_cast<int64_t>(compressed_u32_words * (4 / bytes_per_element));  // 4 bytes per u32
  }
  return head_size;
}

}  // namespace Generators
