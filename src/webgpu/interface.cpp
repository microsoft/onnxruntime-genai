// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../search.h"
#include "interface.h"
#include "../models/graph_executor.h"

namespace Generators {
namespace WebGPU {

namespace {
const char* device_label = "WebGPU";
const char* label_cpu = "cpu";
}  // namespace

struct WebGPUMemory final : DeviceBuffer {
  WebGPUMemory(size_t size, Ort::Allocator* allocator, const OrtMemoryInfo* memory_info)
      : owned_{true}, ort_allocator_{allocator}, ort_memory_info_{memory_info} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size_in_bytes_));
  }

  WebGPUMemory(void* p, size_t size, Ort::Allocator* allocator, const OrtMemoryInfo* memory_info)
      : owned_{false}, ort_allocator_{allocator}, ort_memory_info_{memory_info} {
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

    // Create source tensor (WebGPU device memory) - treat as 1D uint8 array
    int64_t shape_val = static_cast<int64_t>(size_in_bytes_);
    std::span<const int64_t> shape{&shape_val, 1};
    auto src_tensor = OrtValue::CreateTensor(*ort_memory_info_, p_device_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Create CPU memory info and destination tensor
    auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto dst_tensor = OrtValue::CreateTensor(*cpu_mem_info, p_cpu_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Use ORT C++ wrapper for CopyTensors (synchronous copy, stream = nullptr)
    const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
    const std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
    GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
  }

  void CopyCpuToDevice() override {
    if (!ort_allocator_) {
      throw std::runtime_error("WebGPU allocator not initialized");
    }
    assert(p_cpu_);

    // Create source tensor (CPU memory) - treat as 1D uint8 array
    int64_t shape_val = static_cast<int64_t>(size_in_bytes_);
    std::span<const int64_t> shape{&shape_val, 1};
    auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto src_tensor = OrtValue::CreateTensor(*cpu_mem_info, p_cpu_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Create destination tensor (WebGPU device memory)
    auto dst_tensor = OrtValue::CreateTensor(*ort_memory_info_, p_device_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Use ORT C++ wrapper for CopyTensors (synchronous copy, stream = nullptr)
    const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
    const std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
    GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
  }

  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    if (!ort_allocator_) {
      throw std::runtime_error("WebGPU allocator not initialized");
    }

    // Fast path: WebGPU-to-WebGPU copy with zero offsets
    // NOTE: p_device_ is a WGPUBuffer handle (cast to uint8_t*), not a memory pointer.
    // We cannot use pointer arithmetic (p_device_ + offset) to create sub-buffer views.
    // OrtValue::CreateTensor expects the actual buffer handle, not an offset pointer.
    if (source.GetType() == device_label && begin_source == 0 && begin_dest == 0) {
      // Full buffer copy using CopyTensors (no offsets)
      int64_t shape_val = static_cast<int64_t>(size_in_bytes);
      std::span<const int64_t> shape{&shape_val, 1};
      auto src_tensor = OrtValue::CreateTensor(*ort_memory_info_, source.p_device_, size_in_bytes, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      auto dst_tensor = OrtValue::CreateTensor(*ort_memory_info_, p_device_, size_in_bytes, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

      // Use ORT C++ wrapper for CopyTensors for GPU-to-GPU copy
      const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
      const std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
      GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
    } else if (strcmp(source.GetType(), label_cpu) == 0 && begin_source == 0 && begin_dest == 0) {
      // Fast path: CPU-to-WebGPU copy with zero offsets
      // IMPORTANT: Only use this path for actual CPU buffers. For other device types
      // (CUDA/DML/QNN), source.p_device_ is a device handle, not a CPU pointer.
      // Full buffer copy using CopyTensors (no offsets)
      int64_t shape_val = static_cast<int64_t>(size_in_bytes);
      std::span<const int64_t> shape{&shape_val, 1};
      auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
      auto src_tensor = OrtValue::CreateTensor(*cpu_mem_info, source.p_device_, size_in_bytes, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      auto dst_tensor = OrtValue::CreateTensor(*ort_memory_info_, p_device_, size_in_bytes, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

      // Use ORT C++ wrapper for CopyTensors for CPU-to-GPU copy
      const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
      const std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
      GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
    } else {
      // Fallback: Copy through CPU for:
      // - WebGPU-to-WebGPU copies with non-zero offsets (buffer handles don't support offset arithmetic)
      // - Cross-device copies with non-zero offsets
      CopyThroughCpu(*this, begin_dest, source, begin_source, size_in_bytes);
    }
  }

  void Zero() override {
    if (!ort_allocator_) {
      throw std::runtime_error("WebGPU allocator not initialized");
    }

    // Allocate zeroed CPU memory
    std::vector<uint8_t> zero_buffer(size_in_bytes_, 0);

    // Create source tensor (CPU memory with zeros) - treat as 1D uint8 array
    int64_t shape_val = static_cast<int64_t>(size_in_bytes_);
    std::span<const int64_t> shape{&shape_val, 1};
    auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto src_tensor = OrtValue::CreateTensor(*cpu_mem_info, zero_buffer.data(), size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Create destination tensor (WebGPU device memory)
    auto dst_tensor = OrtValue::CreateTensor(*ort_memory_info_, p_device_, size_in_bytes_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    // Use ORT C++ wrapper for CopyTensors to copy zeros to GPU (synchronous copy, stream = nullptr)
    const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
    const std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
    GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
  }

  bool owned_;
  Ort::Allocator* ort_allocator_;
  const OrtMemoryInfo* ort_memory_info_;
};

struct InterfaceImpl : DeviceInterface {
  InterfaceImpl() {
  }

  DeviceType GetType() const override { return DeviceType::WEBGPU; }

  void InitOrt(const OrtApi& /*api*/, Ort::Allocator& allocator) override {
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;
    // Cache the memory info to avoid repeated GetInfo calls
    ort_memory_info_ = &ort_allocator_->GetInfo();
  }

 private:
  Ort::Allocator* ort_allocator_{};
  const OrtMemoryInfo* ort_memory_info_{};

 public:
  Ort::Allocator& GetAllocator() override {
    return *ort_allocator_;
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<WebGPUMemory>(size, ort_allocator_, ort_memory_info_);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<WebGPUMemory>(p, size, ort_allocator_, ort_memory_info_);
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
        ort_memory_info_,
        session_config_keys,
        session_config_values);

    return true;
  }

  // Compact attention mask: write total_length into a [batch_beam_size, 1] tensor on WebGPU.
  // This avoids GPU->CPU->GPU round-trips by only doing a single CPU->GPU copy.
  bool UpdateCompactAttentionMask(void* mask_data, int batch_beam_size, int total_length, ONNXTensorElementDataType type) override {
    if (!ort_allocator_) {
      throw std::runtime_error("WebGPU allocator not initialized");
    }

    // Prepare the values on CPU
    size_t elem_size = (type == Ort::TypeToTensorType<int32_t>) ? sizeof(int32_t) : sizeof(int64_t);
    size_t byte_count = batch_beam_size * elem_size;
    std::vector<uint8_t> cpu_buffer(byte_count);

    if (type == Ort::TypeToTensorType<int32_t>) {
      auto* data = reinterpret_cast<int32_t*>(cpu_buffer.data());
      for (int i = 0; i < batch_beam_size; i++)
        data[i] = static_cast<int32_t>(total_length);
    } else {
      auto* data = reinterpret_cast<int64_t*>(cpu_buffer.data());
      for (int i = 0; i < batch_beam_size; i++)
        data[i] = static_cast<int64_t>(total_length);
    }

    // Single CPU->GPU copy (no GPU->CPU read needed since we know the value)
    int64_t shape_val = static_cast<int64_t>(byte_count);
    std::span<const int64_t> shape{&shape_val, 1};
    auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto src_tensor = OrtValue::CreateTensor(*cpu_mem_info, cpu_buffer.data(), byte_count, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    auto dst_tensor = OrtValue::CreateTensor(*ort_memory_info_, mask_data, byte_count, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
    const std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
    GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);

    return true;
  }
};

}  // namespace WebGPU

DeviceInterface* GetWebGPUInterface() {
  static std::unique_ptr<DeviceInterface> g_device = std::make_unique<WebGPU::InterfaceImpl>();
  return g_device.get();
}

}  // namespace Generators
