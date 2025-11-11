// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <dawn/webgpu_cpp.h>
#include <memory>

namespace Generators {
namespace WebGPU {

// WebGPU kernel for casting between different data types
// Available in both USE_WEBGPU=ON and OFF modes
class CastKernel {
 public:
  CastKernel() = default;
  ~CastKernel() = default;

  // Cast int32 to int64 (most common case for position ids)
  bool CastInt32ToInt64(wgpu::Device device, wgpu::Queue queue, void* input_data, void* output_data, size_t element_count);

  // Cast float16 to float32
  bool CastFloat16ToFloat32(wgpu::Device device, wgpu::Queue queue, void* input_data, void* output_data, size_t element_count);

 private:
  // Cached resources for int32 to int64 conversion
  wgpu::ComputePipeline int32_to_int64_pipeline_;

  // Cached resources for float16 to float32 conversion
  wgpu::ComputePipeline float16_to_float32_pipeline_;

  wgpu::Buffer constants_buffer_;

  void CreateInt32ToInt64Pipeline(wgpu::Device device);
  void CreateFloat16ToFloat32Pipeline(wgpu::Device device);
};

}  // namespace WebGPU
}  // namespace Generators
