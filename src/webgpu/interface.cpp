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
  WebGPUMemory(size_t size, Ort::Allocator& allocator, const OrtMemoryInfo& memory_info)
      : owned_{true}, ort_allocator_{&allocator}, ort_memory_info_{&memory_info} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size_in_bytes_));
  }

  WebGPUMemory(void* p, size_t size, Ort::Allocator& allocator, const OrtMemoryInfo& memory_info)
      : owned_{false}, ort_allocator_{&allocator}, ort_memory_info_{&memory_info} {
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
  InterfaceImpl(OrtEnv& env) : env_{env} {
  }

  DeviceType GetType() const override { return DeviceType::WEBGPU; }

  // Plugin-mode detection. In general the presence of a matching OrtEpDevice signals a plugin EP
  // (a shared allocator is a separate concern). WebGPU is the exception: ORT registers a built-in
  // ("internal") WebGPU plugin EP that creates an OrtEpDevice but provides NO shared allocator, and
  // genai must run that through the legacy trivial-session path rather than plugin mode. So WebGPU
  // — and only WebGPU — additionally requires a device-local shared allocator to be present.
  std::vector<const OrtEpDevice*> FindMyEpDevices() const override {
    auto [devices, shared] = ResolveEp();
    return shared.HasDeviceAllocator() ? devices : std::vector<const OrtEpDevice*>{};
  }

  // §12.1 legacy path: called by EnsureDeviceOrtInit when WebGPU is not a plugin EP.
  // Sets ort_allocator_ so EnsureAllocator() is a no-op in legacy mode.
  void InitOrt(const OrtApi& /*api*/, Ort::Allocator& allocator) override {
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;
    // Cache the memory info to avoid repeated GetInfo calls
    ort_memory_info_ = &ort_allocator_->GetInfo();
    if (g_log.enabled)
      Log("webgpu", "Using legacy internal WebGPU EP");
  }

 private:
  // The env this interface belongs to (created before it, destroyed after it per the reverse-order
  // teardown), so the reference is valid for the interface's whole lifetime.
  OrtEnv& env_;
  // Lazily fetched on first use via EnsureAllocator(); memoized for the env cycle.
  Ort::Allocator* ort_allocator_{};
  const OrtMemoryInfo* ort_memory_info_{};

  // Find this EP's plugin devices and resolve their shared allocators in one place (used by both
  // FindMyEpDevices and EnsureAllocator).
  struct ResolvedEp {
    std::vector<const OrtEpDevice*> devices;
    EpSharedAllocators shared;
  };

  ResolvedEp ResolveEp() const {
    static constexpr const char* kEpNames[] = {"WebGpuExecutionProvider"};
    auto devices = FindEpDevicesByName(env_, kEpNames);
    auto shared = ResolveEpSharedAllocators(env_, devices);
    return {std::move(devices), std::move(shared)};
  }

  // §6/§12.1: Ensure ort_allocator_ is set before use.
  // - Legacy path (V1): InitOrt() already set ort_allocator_; returns immediately.
  // - Plugin path (V2): fetches the env's device-local shared allocator on first call.
  void EnsureAllocator() {
    if (ort_allocator_) return;  // Set by InitOrt() in legacy mode, or by a prior plugin-mode call.
    auto shared = ResolveEp().shared;
    if (!shared.HasDeviceAllocator())
      throw std::runtime_error(
          "WebGPU EP does not provide a device-local shared allocator. "
          "Call OgaRegisterExecutionProviderLibrary with the WebGPU plugin EP before loading models.");
    ort_allocator_ = shared.device_allocator;
    ort_memory_info_ = shared.device_mem_info;
    if (g_log.enabled)
      Log("webgpu", "Using plugin-EP WebGPU EP shared allocator");
  }
  // Reusable CPU staging buffers for UpdateAttentionMask, pre-filled with 1s.
  // Content is always all 1s so sharing across generators is safe; only upload_bytes
  // worth of data is copied each call, regardless of buffer capacity.
  std::vector<int32_t> mask_staging_buffer_i32_;
  std::vector<int64_t> mask_staging_buffer_i64_;

 public:
  Ort::Allocator& GetAllocator() override {
    EnsureAllocator();
    return *ort_allocator_;
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    EnsureAllocator();
    return std::make_shared<WebGPUMemory>(size, *ort_allocator_, *ort_memory_info_);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    EnsureAllocator();
    return std::make_shared<WebGPUMemory>(p, size, *ort_allocator_, *ort_memory_info_);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return std::make_unique<GreedySearch_Cpu>(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return std::make_unique<BeamSearch_Cpu>(params); }

  void Synchronize() override {}  // Nothing to do?

  bool UpdateAttentionMask([[maybe_unused]] void* next_mask_data, void* mask_data, int batch_beam_size, [[maybe_unused]] int new_kv_length, int total_length, [[maybe_unused]] int max_length, bool update_only, ONNXTensorElementDataType type) override {
    EnsureAllocator();
    if (batch_beam_size != 1 || !update_only) {
      return false;  // Fall back to CPU for multi-beam or non-static mask
    }
    if (type != Ort::TypeToTensorType<int32_t> && type != Ort::TypeToTensorType<int64_t>) {
      return false;  // Unsupported mask type; fall back to CPU handling.
    }
    // For batch_beam_size == 1 with static mask (update_only=true, no padding),
    // the mask is always all 1s for attended positions.
    size_t num_elements = static_cast<size_t>(total_length);
    size_t upload_bytes;
    void* staging_data;

    // Use the correctly typed staging buffer. Each grows monotonically and
    // only newly extended positions need to be filled with 1.
    if (type == Ort::TypeToTensorType<int32_t>) {
      if (mask_staging_buffer_i32_.size() < num_elements) {
        mask_staging_buffer_i32_.resize(num_elements, static_cast<int32_t>(1));
      }
      staging_data = mask_staging_buffer_i32_.data();
      upload_bytes = num_elements * sizeof(int32_t);
    } else {
      if (mask_staging_buffer_i64_.size() < num_elements) {
        mask_staging_buffer_i64_.resize(num_elements, static_cast<int64_t>(1));
      }
      staging_data = mask_staging_buffer_i64_.data();
      upload_bytes = num_elements * sizeof(int64_t);
    }

    int64_t shape_val = static_cast<int64_t>(upload_bytes);
    std::span<const int64_t> shape{&shape_val, 1};
    static const auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto src_tensor = OrtValue::CreateTensor(*cpu_mem_info, staging_data, upload_bytes, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    auto dst_tensor = OrtValue::CreateTensor(*ort_memory_info_, mask_data, upload_bytes, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
    const std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
    GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);

    return true;
  }

  bool Cast(void* input, void* output, ONNXTensorElementDataType input_type, ONNXTensorElementDataType output_type, size_t element_count) override {
    EnsureAllocator();

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
        GetType(),
        "WebGpuExecutionProvider",
        ort_memory_info_,
        session_config_keys,
        session_config_values);

    return true;
  }

  void ShapeInitSessionProviderOptions(Config::ProviderOptions& init_options,
                                       const Config::ProviderOptions* user_options) const override {
    if (!user_options) return;

    // Forward only global/singleton WebGPU options to the init session so that the
    // process-wide WebGpuContext singleton is initialized with the correct settings.
    // Per-session options (preferredLayout, enableGraphCapture, sessionBufferPoolGenerations,
    // enableInt64, multiRotaryCacheConcatOffset, forceCpuNodeNames, enablePIXCapture) are
    // excluded because they are meaningless for the trivial initialization model.
    // Keep this list in sync with ParseWebGpuContextConfig in
    // onnxruntime/core/providers/webgpu/webgpu_provider_factory.cc.
    constexpr std::array<std::string_view, 14> kWebGpuGlobalOptions = {
        "deviceId",
        "webgpuInstance",
        "webgpuDevice",
        "dawnProcTable",
        "dawnBackendType",
        "powerPreference",
        "validationMode",
        "preserveDevice",
        "maxStorageBufferBindingSize",
        "maxNumPendingDispatches",
        "storageBufferCacheMode",
        "uniformBufferCacheMode",
        "queryResolveBufferCacheMode",
        "defaultBufferCacheMode",
    };
    for (const auto& opt : user_options->options) {
      if (std::find(kWebGpuGlobalOptions.begin(), kWebGpuGlobalOptions.end(), opt.first) != kWebGpuGlobalOptions.end()) {
        init_options.options.emplace_back(opt);
      }
    }
  }
};

}  // namespace WebGPU

std::unique_ptr<DeviceInterface> CreateWebGPUInterface(OrtEnv& env) {
  return std::make_unique<WebGPU::InterfaceImpl>(env);
}

}  // namespace Generators
