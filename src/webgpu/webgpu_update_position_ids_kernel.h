// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <dawn/webgpu_cpp.h>
#include <memory>
#include <stdexcept>
#include <string>
#include "../models/onnxruntime_api.h"

// WebGPU kernel for updating position IDs efficiently on GPU
// Available in both USE_WEBGPU=ON and OFF modes
template <typename T>
class WebGPUUpdatePositionIdsKernel {
 public:
  WebGPUUpdatePositionIdsKernel() = default;
  ~WebGPUUpdatePositionIdsKernel() = default;

  // Update position IDs using WebGPU compute shader (continuous decoding only)
  void UpdatePositionIds(
      wgpu::Device device,
      wgpu::Queue queue,
      T* position_ids,
      int batch_beam_size,
      int total_length,
      int new_kv_length);

 private:
  struct Constants {
    uint32_t total_length;
    uint32_t new_kv_length;
  };

  wgpu::ComputePipeline pipeline_;
  wgpu::Buffer constants_buffer_;
  wgpu::BindGroup bind_group_;
  bool initialized_ = false;
  bool bind_group_initialized_ = false;

  void InitializePipeline(wgpu::Device device);
  std::string GetShaderSource();
};
