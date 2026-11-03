// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "config_utils.h"
#include "search.h"
#include "interface.h"
#include "models/env_utils.h"
#include "models/graph_builder.h"
#include <iostream>
#include <limits>
#include "models/graph_executor.h"
#include "models/io/kv_cache.h"

namespace Generators {
namespace WebGPU {

namespace {
bool WebGpuGreedyArgMaxEnabled() {
  static const bool enabled = []() {
    bool value = false;
    GetEnv("ORT_GENAI_WEBGPU_GREEDY_ARGMAX", value);
    if (value) {
      bool shader = false;
      GetEnv("ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE", shader);
      if (!shader)
        throw std::runtime_error("ORT_GENAI_WEBGPU_GREEDY_ARGMAX=1 requires ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE=1");
    }
    return value;
  }();
  return enabled;
}
}


namespace {
const char* device_label = "WebGPU";
const char* label_cpu = "cpu";
}  // namespace


static bool WebGpuGreedyArgMaxTraceEnabled();
static bool WebGpuGpuSliceCopyEnabled() {
  static const bool enabled = []() {
    bool value = false;
    GetEnv("ORT_GENAI_WEBGPU_GPU_SLICE_COPY", value);
    if (value && !WebGpuGreedyArgMaxEnabled())
      throw std::runtime_error("ORT_GENAI_WEBGPU_GPU_SLICE_COPY=1 requires ORT_GENAI_WEBGPU_GREEDY_ARGMAX=1");
    return value;
  }();
  return enabled;
}

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
    } else if (source.GetType() == device_label && begin_dest == 0 && begin_source != 0 &&
               dynamic_cast<WebGPUMemory*>(&source) != nullptr &&
               WebGpuGpuSliceCopyEnabled() && size_in_bytes != 0 &&
               p_device_ != source.p_device_ && begin_source % 4 == 0 && size_in_bytes % 4 == 0 &&
               source.size_in_bytes_ % 4 == 0 &&
               source.size_in_bytes_ / 4 <= std::numeric_limits<uint32_t>::max()) {
      if (size_in_bytes > size_in_bytes_ || begin_source > source.size_in_bytes_ ||
          size_in_bytes > source.size_in_bytes_ - begin_source)
        throw std::runtime_error("WebGPU sliced copy exceeds source or destination buffer bounds");
      const int64_t source_words = static_cast<int64_t>(source.size_in_bytes_ / 4);
      const int64_t output_words = static_cast<int64_t>(size_in_bytes / 4);
      int64_t start = static_cast<int64_t>(begin_source / 4);
      int64_t end = start + output_words;
      ModelConfig config("WebGpuSliceCopyU32");
      config.inputs.push_back(TensorConfig("source", Ort::TypeToTensorType<uint32_t>, {-1}));
      config.inputs.push_back(TensorConfig("starts", Ort::TypeToTensorType<int64_t>, {1}));
      config.inputs.push_back(TensorConfig("ends", Ort::TypeToTensorType<int64_t>, {1}));
      config.outputs.push_back(TensorConfig("output", Ort::TypeToTensorType<uint32_t>, {-1}));
      ExecutionParams params(DeviceType::WEBGPU, "WebGpuExecutionProvider", ort_memory_info_);
      params.inputs.push_back(TensorSpec(source.p_device_, Ort::TypeToTensorType<uint32_t>,
                                        {source_words}, source.size_in_bytes_));
      params.inputs.push_back(TensorSpec(&start, Ort::TypeToTensorType<int64_t>, {1}, sizeof(start)));
      params.inputs.push_back(TensorSpec(&end, Ort::TypeToTensorType<int64_t>, {1}, sizeof(end)));
      params.outputs.push_back(TensorSpec(p_device_, Ort::TypeToTensorType<uint32_t>,
                                         {output_words}, size_in_bytes));
      params.session_config_keys = {"ep.webgpuexecutionprovider.enableInt64", "session.disable_cpu_ep_fallback"};
      params.session_config_values = {"1", "1"};
      GraphExecutor::Execute(config, params);
      if (WebGpuGreedyArgMaxTraceEnabled())
        std::cerr << "WEBGPU_GPU_SLICE_COPY source_bytes=" << source.size_in_bytes_
                  << " offset_bytes=" << begin_source << " copied_bytes=" << size_in_bytes << std::endl;
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



static bool WebGpuGreedyArgMaxTraceEnabled() {
  static const bool enabled = []() {
    bool value = false;
    GetEnv("ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE", value);
    return value;
  }();
  return enabled;
}


static bool WebGpuGreedyFusedMinLengthEnabled() {
  static const bool enabled = []() {
    bool value = false;
    GetEnv("ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH", value);
    return value;
  }();
  return enabled;
}

struct LazyWebGpuLogits final : DeviceBuffer {
  explicit LazyWebGpuLogits(DeviceSpan<float> logits) : logits_{logits} {
    p_device_ = reinterpret_cast<uint8_t*>(logits_.Span().data());
    size_in_bytes_ = logits_.size() * sizeof(float);
  }
  const char* GetType() const override { return device_label; }
  void AllocateCpu() override {
    if (!cpu_ready_) {
      const auto cpu = logits_.CopyDeviceToCpu();
      p_cpu_ = reinterpret_cast<uint8_t*>(cpu.data());
      cpu_ready_ = true;
    }
    cpu_accessed_ = true;
  }
  void CopyDeviceToCpu() override {
    // Materialized CPU scores are authoritative for this search step.
    AllocateCpu();
  }
  void CopyCpuToDevice() override {
    if (!cpu_ready_)
      throw std::runtime_error("WebGPU greedy CPU scores have not been materialized");
    logits_.CopyCpuToDevice();
    cpu_accessed_ = true;
  }
  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size) override {
    CopyThroughCpu(*this, begin_dest, source, begin_source, size);
  }
  void Zero() override {
    logits_.Zero();
    cpu_ready_ = cpu_accessed_ = false;
    p_cpu_ = nullptr;
  }
  bool CpuAccessed() const { return cpu_accessed_; }
 private:
  DeviceSpan<float> logits_;
  bool cpu_ready_{}, cpu_accessed_{};
};

struct GreedySearch_WebGpu final : GreedySearch_Cpu {
  GreedySearch_WebGpu(const GeneratorParams& params, DeviceInterface& device)
      : GreedySearch_Cpu{params}, device_{device} {}
  void SetLogits(DeviceSpan<float> logits) override {
    lazy_.reset();
    gpu_min_length_applied_ = false;
    masked_tokens_ready_ = false;
    if (logits.empty())
      throw std::runtime_error("WebGPU greedy requires nonempty logits");
    if (logits.CpuSpan().data() == logits.Span().data()) {
      GreedySearch_Cpu::SetLogits(logits);
      return;
    }
    lazy_ = std::make_shared<LazyWebGpuLogits>(logits);
    next_token_scores_ = DeviceSpan<float>{std::shared_ptr<DeviceBuffer>{lazy_}};
  }

  void ApplyMinLength(int min_length, std::optional<int> sequence_length = std::nullopt) override {
    const int length = sequence_length.value_or(sequences_.GetSequenceLength());
    if (length >= min_length)
      return;
    const int vocab = params_->config.model.vocab_size;
    const int rows = params_->search.batch_size;
    const auto& eos = params_->config.model.eos_token_id;
    if (!lazy_ || lazy_->CpuAccessed() || vocab < 32 || vocab > 1048576) {
      GreedySearch_Cpu::ApplyMinLength(min_length, sequence_length);
      return;
    }
    if (gpu_min_length_applied_) {
      if (cached_eos_ != eos)
        GreedySearch_Cpu::ApplyMinLength(min_length, sequence_length);
      return;
    }
    if (next_token_scores_.size() != static_cast<size_t>(rows) * vocab)
      throw std::runtime_error("WebGPU min-length logits shape mismatch");
    if (eos_mask_.size() != static_cast<size_t>(vocab) || cached_eos_ != eos) {
      std::vector<uint8_t> mask(vocab, 0);
      for (auto token : eos) {
        if (token >= 0 && token < vocab)
          mask[token] = 1;
      }
      eos_mask_ = device_.Allocate<uint8_t>(static_cast<size_t>(vocab));
      eos_mask_.CopyFromCpu(mask);
      cached_eos_ = eos;
    }
    if (lowest_.empty()) {
      const float lowest = std::numeric_limits<float>::lowest();
      lowest_ = device_.Allocate<float>(1);
      lowest_.CopyFromCpu(std::span<const float>{&lowest, 1});
    }
    if (masked_.size() != static_cast<size_t>(rows) * vocab)
      masked_ = device_.Allocate<float>(static_cast<size_t>(rows) * vocab);
    const bool fused = WebGpuGreedyFusedMinLengthEnabled() &&
                       rows <= 65535 / ((vocab + 2047) / 2048);
    ModelConfig config(fused ? "WebGpuGreedyWhereArgMax" : "Where");
    config.inputs.push_back(TensorConfig("condition", Ort::TypeToTensorType<bool>, {-1}));
    config.inputs.push_back(TensorConfig("lowest", Ort::TypeToTensorType<float>, {1}));
    config.inputs.push_back(TensorConfig("scores", Ort::TypeToTensorType<float>, {-1, -1}));
    config.outputs.push_back(TensorConfig("masked", Ort::TypeToTensorType<float>, {-1, -1}));
    if (fused) {
      if (masked_tokens_.size() != static_cast<size_t>(rows))
        masked_tokens_ = device_.Allocate<int64_t>(static_cast<size_t>(rows));
      config.outputs.push_back(TensorConfig("tokens", Ort::TypeToTensorType<int64_t>, {-1}));
      config.attributes.push_back(AttributeValue::Int("axis", -1));
      config.attributes.push_back(AttributeValue::Int("keepdims", 0));
      config.attributes.push_back(AttributeValue::Int("select_last_index", 0));
    }
    auto memory = device_.GetMemoryInfo();
    ExecutionParams params(DeviceType::WEBGPU, "WebGpuExecutionProvider", memory.get());
    params.inputs.push_back(TensorSpec(eos_mask_.Span().data(), Ort::TypeToTensorType<bool>,
                                      {vocab}, static_cast<size_t>(vocab)));
    params.inputs.push_back(TensorSpec(lowest_.Span().data(), Ort::TypeToTensorType<float>, {1}, sizeof(float)));
    params.inputs.push_back(TensorSpec(next_token_scores_.Span().data(), Ort::TypeToTensorType<float>,
                                      {rows, vocab}, static_cast<size_t>(rows) * vocab * sizeof(float)));
    params.outputs.push_back(TensorSpec(masked_.Span().data(), Ort::TypeToTensorType<float>,
                                       {rows, vocab}, static_cast<size_t>(rows) * vocab * sizeof(float)));
    if (fused) {
      params.outputs.push_back(TensorSpec(masked_tokens_.Span().data(), Ort::TypeToTensorType<int64_t>,
                                         {rows}, static_cast<size_t>(rows) * sizeof(int64_t)));
      params.session_config_keys = {"ep.webgpuexecutionprovider.enableInt64"};
      params.session_config_values = {"1"};
    }
    GraphExecutor::Execute(config, params);
    masked_tokens_ready_ = fused;
    if (fused && WebGpuGreedyArgMaxTraceEnabled())
      std::cerr << "WEBGPU_GREEDY_MIN_LENGTH_FUSED" << std::endl;
    lazy_ = std::make_shared<LazyWebGpuLogits>(masked_);
    next_token_scores_ = DeviceSpan<float>{std::shared_ptr<DeviceBuffer>{lazy_}};
    gpu_min_length_applied_ = true;
    if (WebGpuGreedyArgMaxTraceEnabled())
      std::cerr << "WEBGPU_GREEDY_MIN_LENGTH rows=" << rows << " vocabulary=" << vocab << std::endl;
  }

  void SelectTop() override {
    const int vocab = params_->config.model.vocab_size;
    const int rows = params_->search.batch_size;
    if (!lazy_ || lazy_->CpuAccessed() || vocab < 32 || vocab > 1048576 ||
        rows > 65535 / ((vocab + 2047) / 2048)) {
      if (WebGpuGreedyArgMaxTraceEnabled())
        std::cerr << "WEBGPU_GREEDY_ARGMAX_CPU_FALLBACK" << std::endl;
      GreedySearch_Cpu::SelectTop();
      return;
    }
    if (next_token_scores_.size() != static_cast<size_t>(rows) * vocab)
      throw std::runtime_error("WebGPU greedy logits shape does not match batch/vocabulary");
    std::vector<int32_t> tokens(rows);
    if (masked_tokens_ready_) {
      const auto cached = masked_tokens_.CopyDeviceToCpu();
      for (size_t row = 0; row < static_cast<size_t>(rows); ++row) {
        if (cached[row] < 0 || cached[row] >= vocab)
          throw std::runtime_error("WebGPU cached ArgMax returned an invalid token ID");
        tokens[row] = static_cast<int32_t>(cached[row]);
      }
      if (WebGpuGreedyArgMaxTraceEnabled())
        std::cerr << "WEBGPU_GREEDY_ARGMAX_CACHED" << std::endl;
    } else if (!device_.ArgMax(next_token_scores_.Span().data(), Ort::TypeToTensorType<float>,
                               rows, vocab, tokens.data())) {
      throw std::runtime_error("WebGPU greedy ArgMax was not admitted");
    }
    if (WebGpuGreedyArgMaxTraceEnabled())
      std::cerr << "WEBGPU_GREEDY_ARGMAX rows=" << rows << " vocabulary=" << vocab
                << " token_bytes=" << rows * sizeof(int64_t) << std::endl;
    for (size_t row = 0; row < static_cast<size_t>(rows); ++row) {
      if (!PadIfAlreadyEOS(row))
        SetNextToken(row, tokens[row]);
    }
    if (!done_)
      AppendNextTokensToSequences();
  }
 private:
  DeviceInterface& device_;
  std::shared_ptr<LazyWebGpuLogits> lazy_;
  DeviceSpan<uint8_t> eos_mask_;
  DeviceSpan<float> lowest_, masked_;
  std::vector<int32_t> cached_eos_;
  DeviceSpan<int64_t> masked_tokens_;
  bool gpu_min_length_applied_{}, masked_tokens_ready_{};
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

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    if (WebGpuGreedyArgMaxEnabled()) return std::make_unique<GreedySearch_WebGpu>(params, *this);
    return std::make_unique<GreedySearch_Cpu>(params);
  }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return std::make_unique<BeamSearch_Cpu>(params); }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) override {
    return CreateStandardKeyValueCache(state);
  }

  bool ShouldZeroKeyValueCacheTensors() const override { return false; }
  bool SupportsOffsetTensorViews() const override { return false; }
  bool SupportsTransactionalFixedState() const override { return true; }

  int GetKeyValueCacheQuantizationBits(const Config::SessionOptions& session_options) const override {
    return GetKvCacheQuantizationBits(session_options, to_string(GetType()));
  }

  void Synchronize() override {}  // Nothing to do?

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


  bool ArgMax(const void* logits, ONNXTensorElementDataType type, int rows, int vocab,
              int32_t* out_tokens) override {
    if (!WebGpuGreedyArgMaxEnabled())
      return false;
    if (type != Ort::TypeToTensorType<float> && type != Ort::TypeToTensorType<Ort::Float16_t>)
      return false;
    if (!logits || !out_tokens || rows <= 0 || vocab <= 0)
      throw std::runtime_error("Invalid WebGPU ArgMax input/output dimensions");
    if (vocab < 32 || vocab > 1048576 || rows > 65535 / ((vocab + 2047) / 2048))
      return false;
    if (!ort_allocator_)
      throw std::runtime_error("WebGPU allocator not initialized");
    ModelConfig config("ArgMax");
    config.inputs.push_back(TensorConfig("input", type, {-1, -1}));
    config.outputs.push_back(TensorConfig("output", Ort::TypeToTensorType<int64_t>, {-1}));
    config.attributes.push_back(AttributeValue::Int("axis", -1));
    config.attributes.push_back(AttributeValue::Int("keepdims", 0));
    config.attributes.push_back(AttributeValue::Int("select_last_index", 0));
    auto tokens = Allocate<int64_t>(static_cast<size_t>(rows));
    ExecutionParams params(GetType(), "WebGpuExecutionProvider", ort_memory_info_);
    params.inputs.push_back(TensorSpec(const_cast<void*>(logits), type, {rows, vocab},
                                      static_cast<size_t>(rows) * vocab * Ort::SizeOf(type)));
    params.outputs.push_back(TensorSpec(tokens.Span().data(), Ort::TypeToTensorType<int64_t>,
                                       {rows}, static_cast<size_t>(rows) * sizeof(int64_t)));
    params.session_config_keys = {"ep.webgpuexecutionprovider.enableInt64"};
    params.session_config_values = {"1"};
    GraphExecutor::Execute(config, params);
    const auto cpu_tokens = tokens.CopyDeviceToCpu();
    for (size_t row = 0; row < static_cast<size_t>(rows); ++row) {
      if (cpu_tokens[row] < 0 || cpu_tokens[row] >= vocab)
        throw std::runtime_error("WebGPU ArgMax returned an invalid token ID");
      out_tokens[row] = static_cast<int32_t>(cpu_tokens[row]);
    }
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

 private:
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

}  // namespace WebGPU

std::unique_ptr<DeviceInterface> CreateWebGPUInterface() {
  return std::make_unique<WebGPU::InterfaceImpl>();
}

}  // namespace Generators
