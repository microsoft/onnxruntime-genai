// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "config_utils.h"
#include "search.h"
#include "interface.h"
#include "models/graph_builder.h"
#include "models/graph_executor.h"
#include "models/io/kv_cache.h"
#include "state_update_replay.h"

#include <charconv>
#include <limits>
#include <mutex>

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
  std::mutex replay_mutex_;
  std::unique_ptr<OrtSession> replay_session_;
  // Reusable CPU staging buffers for UpdateAttentionMask, pre-filled with 1s.
  // Content is always all 1s so sharing across generators is safe; only upload_bytes
  // worth of data is copied each call, regardless of buffer capacity.
  std::vector<int32_t> mask_staging_buffer_i32_;
  std::vector<int64_t> mask_staging_buffer_i64_;

 public:
  Ort::Allocator& GetAllocator() override {
    return *ort_allocator_;
  }

  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override {
    try {
      return OrtMemoryInfo::Create("WebGPU_Buf", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    } catch (const Ort::Exception& e) {
      // WebGPU memory type name changed from "WebGPU_Buffer" to "WebGPU_Buf" in ORT 1.24.3.
      // Try the old name before giving up.
      try {
        return OrtMemoryInfo::Create("WebGPU_Buffer", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
      } catch (const Ort::Exception& fallback_e) {
        throw std::runtime_error(
            "Failed to create memory info for WebGPU. "
            "Primary name 'WebGPU_Buf' error: " +
            std::string(e.what()) +
            "; fallback 'WebGPU_Buffer' error: " + std::string(fallback_e.what()));
      }
    }
  }

  std::string GetExecutionProviderName() const override { return "WebGPU"; }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<WebGPUMemory>(size, ort_allocator_, ort_memory_info_);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<WebGPUMemory>(p, size, ort_allocator_, ort_memory_info_);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return std::make_unique<GreedySearch_Cpu>(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return std::make_unique<BeamSearch_Cpu>(params); }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) override {
    return CreateStandardKeyValueCache(state);
  }

  bool ShouldZeroKeyValueCacheTensors() const override { return false; }
  bool SupportsOffsetTensorViews() const override { return false; }
  bool SupportsTransactionalFixedState() const override { return true; }
  bool SupportsCompactStateReplay() const override { return true; }

  int GetKeyValueCacheQuantizationBits(const Config::SessionOptions& session_options) const override {
    return GetKvCacheQuantizationBits(session_options, to_string(GetType()));
  }

  void Synchronize() override {}  // Nothing to do?

  void ReplayStateUpdates(const StateUpdateReplayDesc* descriptors, size_t count) override {
    auto* cpu = GetDeviceInterface(DeviceType::CPU);
    std::vector<StateUpdateReplayDesc> staged;
    staged.reserve(count);
    const auto stage_input = [cpu](DeviceSpan<const uint8_t> source) {
      if (source.empty()) {
        return DeviceSpan<const uint8_t>{};
      }
      auto destination = cpu->Allocate<uint8_t>(source.size());
      destination.CopyFrom(source);
      DeviceSpan<const uint8_t> view = destination;
      return view;
    };

    for (size_t index = 0; index < count; ++index) {
      const auto& source = descriptors[index];
      staged.push_back(StateUpdateReplayDesc{
          stage_input(source.source_state),
          cpu->Allocate<uint8_t>(source.destination_state.size()),
          stage_input(source.value),
          stage_input(source.decay),
          stage_input(source.key),
          stage_input(source.delta),
          source.channel_count,
          source.state_width,
          source.key_width,
          source.key_head_count,
          source.capacity,
          source.kept_count,
          source.element_size,
          source.kind,
      });
    }

    ReplayStateUpdatesOnCpu(staged.data(), staged.size());
    for (size_t index = 0; index < count; ++index) {
      auto destination = descriptors[index].destination_state;
      destination.CopyFrom(staged[index].destination_state);
    }
    Synchronize();
  }

  void GetAvailableMemory(size_t& /*free_bytes*/, size_t& /*total_bytes*/) override {
    throw std::runtime_error(
        "WebGPU does not expose available device memory. Set "
        "engine.dynamic_batching.num_blocks explicitly for PagedAttention models.");
  }

  bool UpdateAttentionMask([[maybe_unused]] void* next_mask_data, void* mask_data, int batch_beam_size, [[maybe_unused]] int new_kv_length, int total_length, [[maybe_unused]] int max_length, bool update_only, ONNXTensorElementDataType type) override {
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
        GetType(),
        "WebGpuExecutionProvider",
        ort_memory_info_,
        session_config_keys,
        session_config_values);

    return true;
  }

  bool UpdatePositionIds(void* position_ids, int batch_beam_size, int total_length, int new_kv_length, ONNXTensorElementDataType type) override {
    if (batch_beam_size != 1) {
      return false;
    }
    if (new_kv_length <= 0 || total_length < new_kv_length) {
      return false;
    }
    if (type != Ort::TypeToTensorType<int32_t> && type != Ort::TypeToTensorType<int64_t>) {
      return false;
    }

    int start = total_length - new_kv_length;
    if (type == Ort::TypeToTensorType<int32_t>) {
      UploadPositionIds<int32_t>(position_ids, start, new_kv_length);
    } else {
      UploadPositionIds<int64_t>(position_ids, start, new_kv_length);
    }

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
    constexpr std::array<std::string_view, 15> kWebGpuGlobalOptions = {
        "deviceId",
        "adapterIndex",
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

 private:
  friend void RunGatedDeltaNetStateReplay(const StateUpdateReplayDesc& descriptor);

  void EnsureReplaySession() {
    if (replay_session_) return;
    const auto& init_options =
        GetOrtGlobals()->device_allocators_[static_cast<int>(DeviceType::WEBGPU)].session_options_;
    if (!init_options || !ort_memory_info_ || ort_memory_info_->GetDeviceId() != 0) {
      throw std::runtime_error("WebGPU state replay requires an initialized context-0 allocator.");
    }
    constexpr const char* context_key = "ep.webgpuexecutionprovider.deviceId";
    if (init_options->HasConfigEntry(context_key)) {
      size_t length = 0;
      Ort::ThrowOnError(Ort::api->GetSessionConfigEntry(init_options.get(), context_key, nullptr, &length));
      std::string value(length, '\0');
      Ort::ThrowOnError(Ort::api->GetSessionConfigEntry(init_options.get(), context_key, value.data(), &length));
      value.resize(length - 1);
      int context_id = -1;
      const auto result = std::from_chars(value.data(), value.data() + value.size(), context_id);
      if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || context_id != 0) {
        throw std::runtime_error("WebGPU state replay supports only context 0.");
      }
    }

    ModelConfig config("GatedDeltaNetStateReplay");
    for (const char* name : {"source_state", "capsule", "destination_state"}) {
      config.inputs.emplace_back(name, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{-1});
    }
    config.inputs.emplace_back("metadata", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, std::vector<int64_t>{11});
    config.outputs.emplace_back("replayed_state", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{-1});
    auto model = GraphBuilder::Build(config, "com.microsoft", 1);
    auto options = init_options->Clone();
    options->AddConfigEntry("ep.webgpuexecutionprovider.enableGraphCapture", "0");
    OrtSession* session = nullptr;
    Ort::ThrowOnError(Ort::GetModelEditorApi().CreateSessionFromModel(
        &GetOrtEnv(), model.get(), options.get(), &session));
    replay_session_.reset(session);
  }

  void RunStateReplay(const StateUpdateReplayDesc& descriptor) {
    std::lock_guard<std::mutex> lock(replay_mutex_);
    if (descriptor.kind != StateUpdateReplayKind::GatedDeltaNet ||
        descriptor.element_size != sizeof(float)) {
      throw std::invalid_argument("WebGPU state replay requires an FP32 GatedDeltaNet descriptor.");
    }
    const auto extent_bytes = [](std::initializer_list<uint64_t> dimensions) {
      uint64_t elements = 1;
      for (const auto dimension : dimensions) {
        if (!dimension || dimension > std::numeric_limits<uint32_t>::max() / elements) {
          throw std::invalid_argument("WebGPU state replay geometry exceeds the supported indexing range.");
        }
        elements *= dimension;
      }
      if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::invalid_argument("WebGPU state replay extent exceeds the supported byte range.");
      }
      return static_cast<size_t>(elements) * sizeof(float);
    };
    const auto backing = [&](const auto& view, size_t required_bytes) {
      auto buffer = std::dynamic_pointer_cast<WebGPUMemory>(view.BackingBuffer());
      if (!buffer || !ort_allocator_ || buffer->ort_allocator_ != ort_allocator_ ||
          !buffer->p_device_ || buffer->size_in_bytes_ % sizeof(float) ||
          buffer->size_in_bytes_ / sizeof(float) > std::numeric_limits<uint32_t>::max() ||
          view.ByteOffset() % sizeof(float) ||
          view.ByteOffset() > buffer->size_in_bytes_ ||
          view.size() > buffer->size_in_bytes_ - view.ByteOffset() ||
          view.size() < required_bytes) {
        throw std::invalid_argument("WebGPU state replay requires valid FP32 views of this device's full backings.");
      }
      return buffer;
    };
    const auto state_bytes = extent_bytes(
        {descriptor.channel_count, descriptor.state_width, descriptor.key_width});
    auto source = backing(descriptor.source_state, state_bytes);
    auto destination = backing(descriptor.destination_state, state_bytes);
    auto capsule = backing(descriptor.decay, extent_bytes({descriptor.capacity, descriptor.channel_count}));
    auto key_backing = backing(descriptor.key, extent_bytes(
                                                   {descriptor.capacity, descriptor.key_head_count, descriptor.key_width}));
    auto delta_backing = backing(descriptor.delta, extent_bytes(
                                                       {descriptor.capacity, descriptor.channel_count, descriptor.state_width}));
    if (capsule != key_backing || capsule != delta_backing) {
      throw std::invalid_argument("WebGPU state replay capsule sections must share one backing owner.");
    }

    EnsureReplaySession();
    const auto tensor = [&](const WebGPUMemory& buffer) {
      const std::array<int64_t, 1> shape{static_cast<int64_t>(buffer.size_in_bytes_ / sizeof(float))};
      return OrtValue::CreateTensor(*ort_memory_info_, buffer.p_device_, buffer.size_in_bytes_,
                                    shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    };
    auto source_tensor = tensor(*source);
    // Preserve Tensor-owner identity when logical source/capsule views share a backing.
    auto capsule_tensor = source == capsule ? nullptr : tensor(*capsule);
    auto destination_tensor = tensor(*destination);
    std::array<int64_t, 11> metadata{
        static_cast<int64_t>(descriptor.source_state.ByteOffset() / sizeof(float)),
        static_cast<int64_t>(descriptor.destination_state.ByteOffset() / sizeof(float)),
        static_cast<int64_t>(descriptor.decay.ByteOffset() / sizeof(float)),
        static_cast<int64_t>(descriptor.key.ByteOffset() / sizeof(float)),
        static_cast<int64_t>(descriptor.delta.ByteOffset() / sizeof(float)),
        static_cast<int64_t>(descriptor.channel_count),
        static_cast<int64_t>(descriptor.state_width),
        static_cast<int64_t>(descriptor.key_width),
        static_cast<int64_t>(descriptor.key_head_count),
        descriptor.capacity, descriptor.kept_count};
    const std::array<int64_t, 1> metadata_shape{11};
    auto cpu_memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto metadata_tensor = OrtValue::CreateTensor(*cpu_memory_info, metadata.data(), sizeof(metadata),
                                                  metadata_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    auto binding = OrtIoBinding::Create(*replay_session_);
    binding->BindInput("source_state", *source_tensor);
    binding->BindInput("capsule", capsule_tensor ? *capsule_tensor : *source_tensor);
    binding->BindInput("destination_state", *destination_tensor);
    binding->BindInput("metadata", *metadata_tensor);
    binding->BindOutput("replayed_state", *destination_tensor);
    // The operator performs checked queue completion before Run returns.
    replay_session_->Run(nullptr, *binding);
  }

  template <typename T>
  void UploadPositionIds(void* position_ids, int start, int new_kv_length) {
    // For the common single-token decode, use a stack variable to avoid heap allocation
    T stack_val;
    std::vector<T> heap_buf;
    T* cpu_data;
    if (new_kv_length == 1) {
      stack_val = static_cast<T>(start);
      cpu_data = &stack_val;
    } else {
      heap_buf.resize(new_kv_length);
      for (int i = 0; i < new_kv_length; i++) {
        heap_buf[i] = static_cast<T>(start + i);
      }
      cpu_data = heap_buf.data();
    }

    size_t byte_size = static_cast<size_t>(new_kv_length) * sizeof(T);
    int64_t shape_val = static_cast<int64_t>(byte_size);
    std::span<const int64_t> shape{&shape_val, 1};
    static const auto cpu_mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto src_tensor = OrtValue::CreateTensor(*cpu_mem_info, cpu_data, byte_size, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    auto dst_tensor = OrtValue::CreateTensor(*ort_memory_info_, position_ids, byte_size, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    const std::vector<const OrtValue*> src_ptrs = {src_tensor.get()};
    const std::vector<OrtValue*> dst_ptrs = {dst_tensor.get()};
    GetOrtEnv().CopyTensors(src_ptrs, dst_ptrs, nullptr);
  }
};

void RunGatedDeltaNetStateReplay(const StateUpdateReplayDesc& descriptor) {
  auto& device = static_cast<InterfaceImpl&>(*GetDeviceInterface(DeviceType::WEBGPU));
  device.RunStateReplay(descriptor);
}

}  // namespace WebGPU

std::unique_ptr<DeviceInterface> CreateWebGPUInterface() {
  return std::make_unique<WebGPU::InterfaceImpl>();
}

}  // namespace Generators
