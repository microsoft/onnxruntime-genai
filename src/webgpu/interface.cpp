// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../search.h"
#include "interface.h"
#include "cast_model_builder.h"
#include <unordered_map>
#include <mutex>

namespace Generators {
namespace WebGPU {

static Ort::Allocator* ort_allocator_{};
const char* device_label = "WebGPU";

// Cache for Cast inference sessions, keyed by input and output types
struct CastSessionCache {
  std::unordered_map<uint64_t, std::unique_ptr<OrtSession>> sessions;
  std::mutex mutex;

  static uint64_t MakeKey(ONNXTensorElementDataType input_type, ONNXTensorElementDataType output_type) {
    return (static_cast<uint64_t>(input_type) << 32) | static_cast<uint64_t>(output_type);
  }

  OrtSession* GetOrCreate(ONNXTensorElementDataType input_type, ONNXTensorElementDataType output_type) {
    uint64_t key = MakeKey(input_type, output_type);

    std::lock_guard<std::mutex> lock(mutex);

    auto it = sessions.find(key);
    if (it != sessions.end()) {
      return it->second.get();
    }

    // Create new session with Cast model
    auto model_bytes = CreateCastModelBytes(input_type, output_type);

    auto session_options = OrtSessionOptions::Create();
    session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Add WebGPU execution provider
    if (ort_allocator_) {
      const OrtMemoryInfo* mem_info = nullptr;
      Ort::ThrowOnError(Ort::api->AllocatorGetInfo(ort_allocator_, &mem_info));

      // Append WebGPU execution provider with no options
      session_options->AppendExecutionProvider("WebGPU", nullptr, nullptr, 0);
    }

    auto session = OrtSession::Create(GetOrtEnv(), model_bytes.data(), model_bytes.size(), session_options.get());

    auto* result = session.get();
    sessions[key] = std::move(session);
    return result;
  }
};

static CastSessionCache g_cast_session_cache;

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

    // Fast path: WebGPU-to-WebGPU copy with zero offsets
    // NOTE: p_device_ is a WGPUBuffer handle (cast to uint8_t*), not a memory pointer.
    // We cannot use pointer arithmetic (p_device_ + offset) to create sub-buffer views.
    // OrtValue::CreateTensor expects the actual buffer handle, not an offset pointer.
    if (source.GetType() == device_label && begin_source == 0 && begin_dest == 0) {
      // Get WebGPU allocator's memory info
      const OrtMemoryInfo* webgpu_mem_info = nullptr;
      Ort::ThrowOnError(Ort::api->AllocatorGetInfo(ort_allocator_, &webgpu_mem_info));

      // Full buffer copy using CopyTensors (no offsets)
      int64_t shape_val = static_cast<int64_t>(size_in_bytes);
      std::span<const int64_t> shape{&shape_val, 1};
      auto src_tensor = OrtValue::CreateTensor(*webgpu_mem_info, source.p_device_, size_in_bytes, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
      auto dst_tensor = OrtValue::CreateTensor(*webgpu_mem_info, p_device_, size_in_bytes, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

      // Use ORT C API's CopyTensors for GPU-to-GPU copy
      OrtValue* src_ptrs[] = {src_tensor.get()};
      OrtValue* dst_ptrs[] = {dst_tensor.get()};
      Ort::ThrowOnError(Ort::api->CopyTensors(&GetOrtEnv(), src_ptrs, dst_ptrs, nullptr, 1));
    } else {
      // Fallback: Copy through CPU for:
      // - WebGPU-to-WebGPU copies with non-zero offsets (buffer handles don't support offset arithmetic)
      // - Cross-device copies (e.g., CPU to WebGPU or vice versa)
      CopyThroughCpu(*this, begin_dest, source, begin_source, size_in_bytes);
    }
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

    // Get or create cached session for this type pair
    OrtSession* session = g_cast_session_cache.GetOrCreate(input_type, output_type);
    if (!session) {
      return false;
    }

    // Get WebGPU allocator's memory info
    const OrtMemoryInfo* webgpu_mem_info = nullptr;
    Ort::ThrowOnError(Ort::api->AllocatorGetInfo(ort_allocator_, &webgpu_mem_info));

    // Create input tensor with proper shape [element_count]
    int64_t shape_val = static_cast<int64_t>(element_count);
    std::span<const int64_t> shape{&shape_val, 1};

    // Create input OrtValue
    auto input_tensor = OrtValue::CreateTensor(*webgpu_mem_info, input, element_count * GetElementSize(input_type), shape, input_type);

    // Create output OrtValue
    auto output_tensor = OrtValue::CreateTensor(*webgpu_mem_info, output, element_count * GetElementSize(output_type), shape, output_type);

    // Use IOBinding for efficient execution
    auto io_binding = OrtIoBinding::Create(*session);

    // Bind input
    io_binding->BindInput("input", *input_tensor);

    // Bind output
    io_binding->BindOutput("output", *output_tensor);

    // Run inference
    session->Run(nullptr, *io_binding);

    // Synchronize to ensure completion
    io_binding->SynchronizeOutputs();

    return true;
  }

 private:
  // Helper to get element size for a given ONNX data type
  static size_t GetElementSize(ONNXTensorElementDataType type) {
    switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return sizeof(float);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return sizeof(uint8_t);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return sizeof(int8_t);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return sizeof(uint16_t);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return sizeof(int16_t);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return sizeof(int32_t);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return sizeof(int64_t);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return sizeof(bool);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return 2;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return sizeof(double);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return sizeof(uint32_t);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return sizeof(uint64_t);
      default:
        throw std::runtime_error("Unsupported ONNX data type");
    }
  }
};

}  // namespace WebGPU

DeviceInterface* GetWebGPUInterface() {
  static std::unique_ptr<DeviceInterface> g_device = std::make_unique<WebGPU::InterfaceImpl>();
  return g_device.get();
}

}  // namespace Generators
