﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "sequences.h"
#include "models/model.h"
#include "search.h"
#include "cuda/interface.h"
#if USE_CUDA
#include "cuda/search_cuda.h"
#include "models/kernels.h"
#endif

#if _WIN32
EXTERN_C IMAGE_DOS_HEADER __ImageBase;

std::string CurrentModulePath() {
  char path[MAX_PATH];
  GetModuleFileNameA((HINSTANCE)&__ImageBase, path, _countof(path));

  char absolute_path[MAX_PATH];
  char* name;
  GetFullPathNameA(path, _countof(path), absolute_path, &name);

  auto idx = std::distance(absolute_path, name);
  auto out_path = std::string(absolute_path);
  out_path.resize(idx);

  return out_path;
}
#endif

namespace Generators {

#if USE_CUDA
// TODO: Remove once we remove all dependencies
void OnCudaError(cudaError_t error) { assert(false); }
#endif

static bool _ = (Ort::InitApi(), false);

OrtGlobals::OrtGlobals()
    : env_{OrtEnv::Create(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR)} {
  auto arena_config = OrtArenaCfg::Create(0, -1, -1, -1);
  Ort::Allocator& allocator_cpu{Ort::Allocator::GetWithDefaultOptions()};
  env_->CreateAndRegisterAllocator(allocator_cpu.GetInfo(), *arena_config);
}

// Ensure Shutdown() has been called before process exit
struct ValidateShutdown {
  ~ValidateShutdown() {
    if (GetOrtGlobals()) {
      std::cerr << "OGA Error: Shutdown must be called before process exit, please check the documentation for the proper API to call to ensure clean shutdown." << std::endl;
      std::abort();
    }
  }
};

std::unique_ptr<OrtGlobals>&
GetOrtGlobals() {
  static auto globals = std::make_unique<OrtGlobals>();
  static auto validate = std::make_unique<ValidateShutdown>();  // Must be after the above line so the destructor runs before the above destructor
  return globals;
}

// Used by Shutdown() to display the counts and types of any leaked objects
template <typename... Types>
bool LeakTypeList<Types...>::Dump() {
  ((LeakChecked<Types>::Count() != 0 ? std::cerr << "OGA Error: " << LeakChecked<Types>::Count() << " instances of " << typeid(Types).name() << " were leaked." << std::endl : std::cerr), ...);
  return ((LeakChecked<Types>::Count() != 0) || ...);
}

void Shutdown() {
  if (LeakTypes::Dump()) {
    std::cerr << "    Please see the documentation for the API being used to ensure proper cleanup." << std::endl;
    std::abort();
  }

  GetOrtGlobals().reset();  // Delete now because on process exit is too late
}

OrtEnv& GetOrtEnv() {
  return *GetOrtGlobals()->env_;
}

struct GenaiInterfaceImpl : GenaiInterface {
#if _WIN32
  void* HeapAllocate(size_t size) override { return std::malloc(size); }
  void HeapFree(void* p) override { std::free(p); }
#endif

  Generators::LogItems& GetLogItems() override { return g_log; }
  std::ostream& operator_leftshift(std::ostream& stream, Generators::SGR sgr_code) override { return stream << sgr_code; }
  std::ostream& Log(std::string_view label, std::string_view text = {}) override { return Log(label, text); }

  void DumpSpan(std::ostream& stream, std::span<const float> values) override { return DumpSpan(stream, values); }
  virtual void DumpSpan(std::ostream& stream, std::span<const int> values) override { return DumpSpan(stream, values); }
} g_genai;

const char* label_cpu = "cpu";

struct CpuMemory : DeviceMemoryBase {
  CpuMemory(size_t size) {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = new uint8_t[size_in_bytes_];
  }

  ~CpuMemory() override {
    delete[] p_device_;
  }

  const char* GetType() const override { return label_cpu; }
  bool IsCpuAccessible() const override { return true; }
  void GetOnCpu() override { assert(false); }  // Should never be called, as p_cpu_ is always valid
  void CopyFromDevice(size_t begin_dest, DeviceMemoryBase& source, size_t begin_source, size_t size_in_bytes) override {
    if (GetType() == label_cpu)
      memcpy(p_device_ + begin_dest, source.p_device_ + begin_source, size_in_bytes);
    else
      throw std::runtime_error("CpuMemory::CopyFromDevice not implemented for " + std::string(source.GetType()));
  }
};

struct CpuInterface : DeviceInterface {
  std::shared_ptr<DeviceMemoryBase> AllocateBase(size_t size, bool cpu_accessible) override {
    assert(cpu_accessible == true);
    return std::make_shared<CpuMemory>(size);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return nullptr; }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return nullptr; }
} g_cpu;

DeviceInterface& GetCpuDeviceInterface() {
  return g_cpu;
}

#if USE_CUDA
CudaInterface* GetCudaInterface() {
// Load the shared library onnxruntime-genai-cuda.dll
// This is a workaround to avoid linking the CUDA library to the generator library
// The CUDA library is only needed for the CUDA allocator
#ifdef _WIN32
  static std::unique_ptr<void, void (*)(void*)> cuda_library{LoadLibrary((CurrentModulePath() + "onnxruntime-genai-cuda.dll").c_str()),
                                                             [](void* h) { FreeLibrary(reinterpret_cast<HMODULE>(h)); }};
#else
  static std::unique_ptr<void, void (*)(void*)> cuda_library{dlopen((Ort::GetCurrentModuleDir() + "/libonnxruntime-genai-cuda.so").c_str(), RTLD_NOW | RTLD_DEEPBIND),
                                                             [](void* h) { dlclose(h); }};
#endif

  if (!cuda_library)
    throw std::runtime_error("Cuda interface not available");

  Generators::CudaInterface* GetInterface(GenaiInterface * p_genai);
  static CudaInterface* cuda_interface{[] {
#ifdef _WIN32
    auto get_cuda_fn = reinterpret_cast<decltype(&GetInterface)>(GetProcAddress(reinterpret_cast<HMODULE>(cuda_library.get()), "GetInterface"));
#else
    auto get_cuda_fn = reinterpret_cast<decltype(&GetInterface)>(dlsym(cuda_library.get(), "GetInterface"));
#endif
    return get_cuda_fn(&g_genai);
  }()};

  return cuda_interface;
}

namespace cuda {
void LaunchInt32ToInt64(const int32_t* input, int64_t* output, int count, cudaStream_t stream) { GetCudaInterface()->Int32ToInt64(input, output, count, stream); }
void LaunchFp16ToFp32(const uint16_t* input, float* output, int count, cudaStream_t stream) { GetCudaInterface()->Fp16ToFp32(input, output, count, stream); }
void LaunchFp32ToFp16(const float* input, uint16_t* output, int count, cudaStream_t stream) { GetCudaInterface()->Fp32ToFp16(input, output, count, stream); }
template <>
void Launch_UpdatePositionIds<int32_t>(int32_t* position_ids, int batch_beam_size, cudaStream_t stream) { GetCudaInterface()->Launch_UpdatePositionIds(position_ids, batch_beam_size, stream); }
template <>
void Launch_UpdatePositionIds<int64_t>(int64_t* position_ids, int batch_beam_size, cudaStream_t stream) { GetCudaInterface()->Launch_UpdatePositionIds(position_ids, batch_beam_size, stream); }
template <>
void Launch_UpdateAttentionMask<int32_t>(int32_t* mask_data, const int32_t* old_mask_data, int batch_beam_size, int current_length, int max_length, bool update_only, cudaStream_t stream) { GetCudaInterface()->Launch_UpdateAttentionMask(mask_data, old_mask_data, batch_beam_size, current_length, max_length, update_only, stream); }
template <>
void Launch_UpdateAttentionMask<int64_t>(int64_t* mask_data, const int64_t* old_mask_data, int batch_beam_size, int current_length, int max_length, bool update_only, cudaStream_t stream) { GetCudaInterface()->Launch_UpdateAttentionMask(mask_data, old_mask_data, batch_beam_size, current_length, max_length, update_only, stream); }
void LaunchHandleEOSArray(float* batch_logits, int batch_beam_size, int vocab_size, const int32_t* eos_token_ids, int eos_token_ids_count, cudaStream_t stream) { GetCudaInterface()->LaunchHandleEOSArray(batch_logits, batch_beam_size, vocab_size, eos_token_ids, eos_token_ids_count, stream); }
void UpdateCacheIndirectionKernelLauncher(int32_t* tgt_indir_cache, const int32_t* src_indir_cache, const int32_t* beam_ids, int batch_size, int beam_width, int input_seq_length, int max_seq_length, int current_length, cudaStream_t stream) { GetCudaInterface()->UpdateCacheIndirectionKernelLauncher(tgt_indir_cache, src_indir_cache, beam_ids, batch_size, beam_width, input_seq_length, max_seq_length, current_length, stream); }
void ReorderPastStatesKernelLauncher(void* out_buffer, const void* in_buffer, int batch_size, int num_heads, int max_length, int head_size, int chunk_size, cudaStream_t stream) { GetCudaInterface()->ReorderPastStatesKernelLauncher(out_buffer, in_buffer, batch_size, num_heads, max_length, head_size, chunk_size, stream); }
template <>
void LaunchCopyCrossQKSingleDecodeStep<float>(cudaStream_t stream, float* cross_qk_buffer_data, float** qk_layer_pointers, int token_index, int batch_beam_size, int num_layers, int num_heads, int num_alignment_heads, const int* alignment_heads, int frames, int max_length) { GetCudaInterface()->LaunchCopyCrossQKSingleDecodeStep(stream, cross_qk_buffer_data, qk_layer_pointers, token_index, batch_beam_size, num_layers, num_heads, num_alignment_heads, alignment_heads, frames, max_length); }
template <>
void LaunchFinalizeCrossQK<float>(cudaStream_t stream, int iteration_number, int context_decoding_len, int batch_size, int num_beams, int max_length, int num_alignment_heads, int frames_of_k, const float* cross_qk_buffer_data, float* cross_qk_output, int num_return_sequences, const int* cache_indir_data) { GetCudaInterface()->LaunchFinalizeCrossQK(stream, iteration_number, context_decoding_len, batch_size, num_beams, max_length, num_alignment_heads, frames_of_k, cross_qk_buffer_data, cross_qk_output, num_return_sequences, cache_indir_data); }
}  // namespace cuda
#endif

std::string to_string(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::CUDA:
      return "CUDA";
    case DeviceType::DML:
      return "DirectML";
    case DeviceType::WEBGPU:
      return "WebGpu";
  }
  throw std::runtime_error("Unknown device type");
}

GeneratorParams::GeneratorParams(const Config& config) : config{config} {
}

GeneratorParams::GeneratorParams(const Model& model)
    : config{*model.config_.get()},
      device_type{model.device_type_},
      cuda_stream{model.cuda_stream_},
      is_cuda_graph_enabled_{IsCudaGraphEnabled(model.config_->model.decoder.session_options)} {
  use_cuda_graph = is_cuda_graph_enabled_;
  if (use_cuda_graph) {
    max_batch_size = 1;  // set it to 1 by default
  }
}

void GeneratorParams::TryGraphCapture(int max_bs) {
  if (!is_cuda_graph_enabled_ || device_type == DeviceType::CPU) {
    // no-op
    return;
  }

  if (DeviceType::CUDA == device_type || DeviceType::DML == device_type) {
    if (max_bs == 0) {
      throw std::runtime_error("Graph capture is enabled, but max_batch_size is not set.");
    }
    use_cuda_graph = true;
    max_batch_size = max_bs;
  } else {
    throw std::runtime_error("CUDA graph is not supported on this device");
  }
}

void GeneratorParams::SetInputs(const NamedTensors& named_tensors) {
  for (const auto& [name, tensor] : named_tensors) {
    if (name == Config::Defaults::InputIdsName) {
      input_ids = std::span<const int32_t>(tensor->ort_tensor_->GetTensorMutableData<int32_t>(),
                                           tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount());
      batch_size = static_cast<int>(tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape()[0]);
      sequence_length = static_cast<int>(input_ids.size()) / batch_size;
    } else {
      // If the nominal name is found in the map, use the graph name.
      // Else, use the nominal name as the graph name.
      [[maybe_unused]] const auto [graph_name, found] = config.GetGraphName(name);
      extra_inputs.push_back({graph_name, tensor});
    }
  }
}

std::unique_ptr<Generator> CreateGenerator(const Model& model, const GeneratorParams& params) {
  return std::make_unique<Generator>(model, params);
}

std::unique_ptr<Search> CreateSearch(const GeneratorParams& params) {
#if USE_CUDA
  if (params.device_type == DeviceType::CUDA) {
    if (params.search.num_beams > 1)
      return GetCudaInterface()->CreateBeam(params);
    return GetCudaInterface()->CreateGreedy(params);
  }
#endif

  if (params.search.num_beams > 1) {
    return std::make_unique<BeamSearch_Cpu>(params);
  }
  return std::make_unique<GreedySearch_Cpu>(params);
}

Generator::Generator(const Model& model, const GeneratorParams& params) : model_{model.shared_from_this()} {
  if (params.search.max_length == 0)
    throw std::runtime_error("search max_length is 0");
  if (params.search.max_length > model.config_->model.context_length)
    throw std::runtime_error("max_length (" + std::to_string(params.search.max_length) + ") cannot be greater than model context_length (" + std::to_string(model.config_->model.context_length) + ")");
  if (params.batch_size < 1)
    throw std::runtime_error("batch_size must be 1 or greater, is " + std::to_string(params.batch_size));
  if (params.config.model.vocab_size < 1)
    throw std::runtime_error("vocab_size must be 1 or greater, is " + std::to_string(params.config.model.vocab_size));
  if (params.sequence_length >= params.search.max_length)
    throw std::runtime_error("input sequence_length (" + std::to_string(params.sequence_length) + ") is >= max_length (" + std::to_string(params.search.max_length) + ")");
  if (params.input_ids.empty() || params.input_ids.data() == nullptr)
    throw std::runtime_error("input_ids not set in GeneratorParams");

  search_ = CreateSearch(params);
  state_ = model.CreateState(search_->GetSequenceLengths(), params);
}

void Generator::ComputeLogits() {
  if (computed_logits_)
    throw std::runtime_error("ComputeLogits called again without calling GenerateNextToken first");

  auto logits = state_->Run(search_->GetSequenceLength(), search_->GetNextTokens(), search_->GetNextIndices());
  if (g_log.enabled && g_log.model_logits) {
    auto& stream = Log("model_logits");
    DumpSpan(stream, logits.GetCPU());
    stream << std::endl;
  }
  search_->SetLogits(logits);
  computed_logits_ = true;

  auto& search = search_->params_->search;
  search_->ApplyMinLength(search.min_length);
  search_->ApplyRepetitionPenalty(search.repetition_penalty);
}

bool Generator::IsDone() const {
  if (computed_logits_)
    throw std::runtime_error("IsDone() can't be called in the middle of processing logits");

  bool is_done = search_->IsDone();
  if (is_done) {
    state_->Finalize();
  }

  return is_done;
}

void Generator::GenerateNextToken() {
  if (!computed_logits_)
    throw std::runtime_error("Must call ComputeLogits before GenerateNextToken");
  computed_logits_ = false;
  auto& search = search_->params_->search;

  if (g_log.enabled && g_log.generate_next_token) {
    auto& stream = Log("generate_next_token");
    stream << SGR::Fg_Green << "do_sample: " << SGR::Reset << search.do_sample << ' '
           << SGR::Fg_Green << "top_k: " << SGR::Reset << search.top_k << ' '
           << SGR::Fg_Green << "top_p: " << SGR::Reset << search.top_p << ' '
           << SGR::Fg_Green << "temperature: " << SGR::Reset << search.temperature << ' '
           << SGR::Fg_Cyan << "sequence length: " << SGR::Reset << search_->GetSequenceLength()
           << std::endl;
  }

  if (!search.do_sample || search.top_k == 1) {
    search_->SelectTop();
    return;
  }

  // The user explicitly called TopK_TopP on a beam search
  if (search.num_beams != 1)
    throw std::runtime_error("TopK and TopP cannot be used with a beam search");

  // Sanity checks
  if (search.top_p < 0.0f || search.top_p > 1.0f)
    throw std::runtime_error("top_p must be between 0.0 and 1.0");
  if (search.top_k < 0)
    throw std::runtime_error("top_k must be 0 or greater");

  if (search.top_p > 0.0f && search.top_p < 1.0f && search.top_k > 1) {
    search_->SampleTopKTopP(search.top_k, search.top_p, search.temperature);
  } else if (search.top_k > 1) {
    search_->SampleTopK(search.top_k, search.temperature);
  } else {
    assert(search.top_k == 0);
    search_->SampleTopP(search.top_p, search.temperature);
  }
}

DeviceMemorySpan<int32_t> Generator::GetSequence(size_t index) const {
  return search_->GetSequence(index);
}

TokenSequences Generate(const Model& model, const GeneratorParams& params) {
  auto generator = CreateGenerator(model, params);

  while (!generator->IsDone()) {
    generator->ComputeLogits();
    generator->GenerateNextToken();
  }

  TokenSequences result;

  for (int i = 0; i < params.batch_size * params.search.num_return_sequences; i++) {
    auto sequence = generator->search_->GetSequence(i);
    auto sequence_cpu = sequence.CpuSpan();

    auto& v = result.emplace_back();
    v.assign(sequence_cpu.begin(), sequence_cpu.end());
  }
  return result;
}

}  // namespace Generators

#if USE_CUDA
cudaError_t cudaStreamCreate(cudaStream_t* stream) { return Generators::GetCudaInterface()->cudaStreamCreate(stream); }
cudaError_t cudaStreamDestroy(cudaStream_t stream) { return Generators::GetCudaInterface()->cudaStreamDestroy(stream); }
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) { return Generators::GetCudaInterface()->cudaMemcpyAsync(dst, src, count, kind, stream); }
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) { return Generators::GetCudaInterface()->cudaMemcpy(dst, src, count, kind); }
cudaError_t cudaMemsetAsync(void* ptr, int value, size_t count, cudaStream_t stream) { return Generators::GetCudaInterface()->cudaMemsetAsync(ptr, value, count, stream); }
cudaError_t cudaMemset(void* ptr, int value, size_t count) { return Generators::GetCudaInterface()->cudaMemset(ptr, value, count); }
cudaError_t cudaMalloc(void** ptr, size_t size) { return Generators::GetCudaInterface()->cudaMalloc(ptr, size); }
cudaError_t cudaFree(void* ptr) { return Generators::GetCudaInterface()->cudaFree(ptr); }
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) { return Generators::GetCudaInterface()->cudaHostAlloc(ptr, size, flags); }
cudaError_t cudaFreeHost(void* ptr) { return Generators::GetCudaInterface()->cudaFreeHost(ptr); }
#endif
