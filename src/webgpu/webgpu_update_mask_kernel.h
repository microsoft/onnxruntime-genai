// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#ifdef USE_WEBGPU
#include <dawn/webgpu_cpp.h>
#include <memory>
#include <stdexcept>
#include <string>
#include "../models/onnxruntime_api.h"

// WebGPU kernel for updating attention masks efficiently on GPU
template <typename T>
class WebGPUUpdateMaskKernel {
 public:
  WebGPUUpdateMaskKernel() = default;
  ~WebGPUUpdateMaskKernel() = default;

  // Update attention mask using WebGPU compute shader
  void UpdateMask(
      wgpu::Device device,
      wgpu::Queue queue,
      T* next_mask_data,
      T* mask_data,
      int batch_beam_size,
      int new_kv_length,
      int total_length,
      int max_length,
      bool update_only);

 private:
  struct Constants {
    uint32_t batch_beam_size;
    uint32_t new_kv_length;
    uint32_t total_length;
    uint32_t max_length;
  };

  wgpu::ComputePipeline pipeline_;
  wgpu::Buffer constants_buffer_;
  wgpu::BindGroup bind_group_;
  bool initialized_ = false;
  bool bind_group_initialized_ = false;

  void InitializePipeline(wgpu::Device device);
  std::string GetShaderSource();
};

#endif  // USE_WEBGPU
