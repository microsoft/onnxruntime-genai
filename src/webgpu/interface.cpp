// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../search.h"
#include "interface.h"
#include "../models/graph_executor.h"

namespace Generators {
namespace WebGPU {

static Ort::Allocator* ort_allocator_{};
const char* device_label = "WebGPU";

struct WebGPUMemory final : DeviceBuffer {
  WebGPUMemory(size_t size) : owned_{true} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size_in_bytes_));
  }

  WebGPUMemory(void* p, size_t size) : owned_{false} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(p);
  }

  ~WebGPUMemory() override {
    if (owned_)
      ort_allocator_->Free(p_device_);
    if (p_cpu_)
      free(p_cpu_);
  }

  const char* GetType() const override { return device_label; }

  void AllocateCpu() override {
    if (!p_cpu_)
      p_cpu_ = static_cast<uint8_t*>(malloc(size_in_bytes_));
  }

  void CopyDeviceToCpu() override {
    if (!ort_allocator_) {
      throw std::runtime_error("WebGPU allocator not initialized");
    }

    AllocateCpu();

    // Get WebGPU allocator's memory info
    const OrtMemoryInfo* webgpu_mem_info = nullptr;
    Ort::ThrowOnError(Ort::api->AllocatorGetInfo(ort_allocator_, &webgpu_mem_info));

    // Create source tensor (WebGPU device memory) - treat as 1D uint8 array
    int64_t shape_val = static_cast<int64_t>(size_in_bytes_);
    std::span<const int64_t> shape{&shape_val, 1};
    auto src_tensor = OrtValue::CreateTensor(*webgpu_mem_info, p_device_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Create CPU memory info and destination tensor
    auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto dst_tensor = OrtValue::CreateTensor(*cpu_mem_info, p_cpu_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Use ORT C API's CopyTensors (synchronous copy, stream = nullptr)
    OrtValue* src_ptrs[] = {src_tensor.get()};
    OrtValue* dst_ptrs[] = {dst_tensor.get()};
    Ort::ThrowOnError(Ort::api->CopyTensors(&GetOrtEnv(), src_ptrs, dst_ptrs, nullptr, 1));
  }

  void CopyCpuToDevice() override {
    if (!ort_allocator_) {
      throw std::runtime_error("WebGPU allocator not initialized");
    }
    assert(p_cpu_);

    // Get WebGPU allocator's memory info
    const OrtMemoryInfo* webgpu_mem_info = nullptr;
    Ort::ThrowOnError(Ort::api->AllocatorGetInfo(ort_allocator_, &webgpu_mem_info));

    // Create source tensor (CPU memory) - treat as 1D uint8 array
    int64_t shape_val = static_cast<int64_t>(size_in_bytes_);
    std::span<const int64_t> shape{&shape_val, 1};
    auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto src_tensor = OrtValue::CreateTensor(*cpu_mem_info, p_cpu_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Create destination tensor (WebGPU device memory)
    auto dst_tensor = OrtValue::CreateTensor(*webgpu_mem_info, p_device_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Use ORT C API's CopyTensors (synchronous copy, stream = nullptr)
    OrtValue* src_ptrs[] = {src_tensor.get()};
    OrtValue* dst_ptrs[] = {dst_tensor.get()};
    Ort::ThrowOnError(Ort::api->CopyTensors(&GetOrtEnv(), src_ptrs, dst_ptrs, nullptr, 1));
  }

  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    if (!ort_allocator_) {
      throw std::runtime_error("WebGPU allocator not initialized");
    }

    // Get WebGPU allocator's memory info
    const OrtMemoryInfo* webgpu_mem_info = nullptr;
    Ort::ThrowOnError(Ort::api->AllocatorGetInfo(ort_allocator_, &webgpu_mem_info));

    // Create tensors with explicit byte offsets into each buffer, then use CopyTensors.
    int64_t shape_val = static_cast<int64_t>(size_in_bytes);
    std::span<const int64_t> shape{&shape_val, 1};
    auto dst_tensor = OrtValue::CreateTensor(*webgpu_mem_info, p_device_, size_in_bytes_, begin_dest, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    std::unique_ptr<OrtValue> src_tensor;
    if (source.GetType() == device_label) {
      src_tensor = OrtValue::CreateTensor(*webgpu_mem_info, source.p_device_, source.size_in_bytes_, begin_source, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    } else {
      auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
      src_tensor = OrtValue::CreateTensor(*cpu_mem_info, source.p_cpu_, source.size_in_bytes_, begin_source, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    }

    const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
    std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
    GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
  }

  void Zero() override {
    if (!ort_allocator_) {
      throw std::runtime_error("WebGPU allocator not initialized");
    }

    // Allocate zeroed CPU memory
    std::vector<uint8_t> zero_buffer(size_in_bytes_, 0);

    // Get WebGPU allocator's memory info
    const OrtMemoryInfo* webgpu_mem_info = nullptr;
    Ort::ThrowOnError(Ort::api->AllocatorGetInfo(ort_allocator_, &webgpu_mem_info));

    // Create source tensor (CPU memory with zeros) - treat as 1D uint8 array
    int64_t shape_val = static_cast<int64_t>(size_in_bytes_);
    std::span<const int64_t> shape{&shape_val, 1};
    auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto src_tensor = OrtValue::CreateTensor(*cpu_mem_info, zero_buffer.data(), size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Create destination tensor (WebGPU device memory)
    auto dst_tensor = OrtValue::CreateTensor(*webgpu_mem_info, p_device_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Use ORT C API's CopyTensors to copy zeros to GPU (synchronous copy, stream = nullptr)
    OrtValue* src_ptrs[] = {src_tensor.get()};
    OrtValue* dst_ptrs[] = {dst_tensor.get()};
    Ort::ThrowOnError(Ort::api->CopyTensors(&GetOrtEnv(), src_ptrs, dst_ptrs, nullptr, 1));
  }

  bool owned_;
};

struct InterfaceImpl : DeviceInterface {
  InterfaceImpl() {
  }

  DeviceType GetType() const override { return DeviceType::WEBGPU; }

  void InitOrt(const OrtApi& /*api*/, Ort::Allocator& allocator) override {
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;

    // Cache memory info for reuse
    Ort::ThrowOnError(Ort::api->AllocatorGetInfo(ort_allocator_, &webgpu_mem_info_));
    cpu_mem_info_ = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  }

  Ort::Allocator& GetAllocator() override {
    return *ort_allocator_;
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<WebGPUMemory>(size);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<WebGPUMemory>(p, size);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return std::make_unique<GreedySearch_Cpu>(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return std::make_unique<BeamSearch_Cpu>(params); }

  void Synchronize() override {}  // Nothing to do?

  bool Cast(void* input, void* output, ONNXTensorElementDataType input_type, ONNXTensorElementDataType output_type, size_t element_count) override {
    if (!ort_allocator_) {
      throw std::runtime_error("WebGPU allocator not initialized");
    }

    // WebGPU-specific session configuration
    static const char* webgpu_config_key = "ep.webgpuexecutionprovider.enableInt64";
    static const char* webgpu_config_value = "1";
    std::vector<const char*> session_config_keys = {webgpu_config_key};
    std::vector<const char*> session_config_values = {webgpu_config_value};

    // Use the generalized ExecuteCastOp helper with WebGPU session config
    ExecuteCastOp(
        input,
        output,
        input_type,
        output_type,
        element_count,
        "WebGPU",
        webgpu_mem_info_,
        session_config_keys,
        session_config_values);

    return true;
  }

  bool UpdateAttentionMask(void* next_mask_data, void* mask_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, ONNXTensorElementDataType type) override {
    // Only support continuous decoding for batch_beam_size == 1 and static mask handling
    if (batch_beam_size != 1 || !update_only) {
      return false;  // Fall back to CPU implementation
    }

    if (!ort_allocator_) {
      throw std::runtime_error("WebGPU allocator not initialized");
    }

    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      std::vector<int32_t> cpu_data(new_kv_length, 1);  // Fill new portion with 1s

      std::array<int64_t, 2> shape = {static_cast<int64_t>(batch_beam_size), static_cast<int64_t>(new_kv_length)};
      auto src_tensor = OrtValue::CreateTensor(*cpu_mem_info_, cpu_data.data(), new_kv_length * sizeof(int32_t), shape, type);

      size_t destination_offset = (total_length - new_kv_length) * sizeof(int32_t);
      auto dst_tensor = OrtValue::CreateTensor(*webgpu_mem_info_, mask_data, max_length * sizeof(int32_t), destination_offset, shape, type);

      const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
      std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
      GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
    } else {
      std::vector<int64_t> cpu_data(new_kv_length, 1);  // Fill new portion with 1s

      std::array<int64_t, 2> shape = {static_cast<int64_t>(batch_beam_size), static_cast<int64_t>(new_kv_length)};
      auto src_tensor = OrtValue::CreateTensor(*cpu_mem_info_, cpu_data.data(), new_kv_length * sizeof(int64_t), shape, type);

      size_t destination_offset = (total_length - new_kv_length) * sizeof(int64_t);
      auto dst_tensor = OrtValue::CreateTensor(*webgpu_mem_info_, mask_data, max_length * sizeof(int64_t), destination_offset, shape, type);

      const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
      std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
      GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
    }

    return true;
  }

 private:
  const OrtMemoryInfo* webgpu_mem_info_ = nullptr;
  std::unique_ptr<OrtMemoryInfo> cpu_mem_info_;
};

}  // namespace WebGPU

DeviceInterface* GetWebGPUInterface() {
  static std::unique_ptr<DeviceInterface> g_device = std::make_unique<WebGPU::InterfaceImpl>();
  return g_device.get();
}

}  // namespace Generators
