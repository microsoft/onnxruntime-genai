// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#ifdef USE_WEBGPU
#include <dawn/webgpu_cpp.h>
#include <memory>

namespace Generators {
namespace WebGPU {

class CastKernel {
 public:
  CastKernel() = default;
  ~CastKernel() = default;

  void Initialize(wgpu::Device device, wgpu::Queue queue);

  // Cast int32 to int64 (most common case for position ids)
  bool CastInt32ToInt64(void* input_data, void* output_data, size_t element_count);
  
  // Cast float16 to float32
  bool CastFloat16ToFloat32(void* input_data, void* output_data, size_t element_count);

 private:
  wgpu::Device device_;
  wgpu::Queue queue_;
  
  // Cached resources for int32 to int64 conversion
  wgpu::ComputePipeline int32_to_int64_pipeline_;
  wgpu::BindGroup int32_to_int64_bind_group_;
  
  // Cached resources for float16 to float32 conversion
  wgpu::ComputePipeline float16_to_float32_pipeline_;
  wgpu::BindGroup float16_to_float32_bind_group_;
  
  wgpu::Buffer constants_buffer_;
  
  bool initialized_ = false;
  bool int32_to_int64_bind_group_initialized_ = false;
  bool float16_to_float32_bind_group_initialized_ = false;
  
  void CreateInt32ToInt64Pipeline();
  void CreateFloat16ToFloat32Pipeline();
};

}  // namespace WebGPU
}  // namespace Generators

#endif  // USE_WEBGPU
