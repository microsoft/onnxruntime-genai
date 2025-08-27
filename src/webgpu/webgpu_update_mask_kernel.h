#pragma once

#ifdef USE_WEBGPU

#include <memory>
#include <stdexcept>
#include <string>
#include "../models/onnxruntime_api.h"
#include <dawn/webgpu_cpp.h>

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
    uint32_t update_only;
    uint32_t padding[3];  // Align to 16 bytes
  };

  wgpu::ComputePipeline pipeline_;
  bool initialized_ = false;

  void InitializePipeline(wgpu::Device device);
  std::string GetShaderSource();
};

#endif  // USE_WEBGPU
