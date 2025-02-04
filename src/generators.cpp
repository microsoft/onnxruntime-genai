// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "sequences.h"
#include "models/env_utils.h"
#include "models/model.h"
#include "models/decoder_only.h"
#include "search.h"
#include "cpu/interface.h"
#include "cuda/interface.h"
#if USE_CUDA
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

void ThrowErrorIfSessionTerminated(bool is_session_terminated) {
  if (is_session_terminated)
    throw std::runtime_error("Session in Terminated state, exiting!");
}

namespace Generators {

#if USE_CUDA
// TODO: Remove once we remove all dependencies
void OnCudaError(cudaError_t error) { assert(false); }
#endif

static bool _ = (Ort::InitApi(), false);

static OrtLoggingLevel GetDefaultOrtLoggingLevel() {
  bool ort_verbose_logging = false;
  GetEnvironmentVariable("ORTGENAI_ORT_VERBOSE_LOGGING", ort_verbose_logging);
  return ort_verbose_logging ? OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE : OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
}

OrtGlobals::OrtGlobals()
    : env_{OrtEnv::Create(GetDefaultOrtLoggingLevel())} {
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
  void DumpSpan(std::ostream& stream, std::span<const int> values) override { return DumpSpan(stream, values); }

  void Sequences_AfterAppendNextTokens(Sequences* p_this, DeviceSpan<int32_t> next_tokens, size_t batch_beam_size) override { return p_this->AfterAppendNextTokens(next_tokens, batch_beam_size); }
  void Sequences_RewindTo(Sequences* p_this, size_t new_length) override { return p_this->RewindTo(new_length); }
} g_genai;

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

  if (!cuda_library) {
    throw std::runtime_error("Cuda interface not available.");
  }

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
void LaunchExpandAndInt32ToInt64(const int32_t* src, int64_t* dst, int num_beams, int batch_size, int sequence_length, cudaStream_t stream) { GetCudaInterface()->LaunchExpandAndInt32ToInt64(src, dst, num_beams, batch_size, sequence_length, stream); }
void LaunchExpand(const int32_t* src, int32_t* dst, int num_beams, int batch_size, int sequence_length, cudaStream_t stream) { GetCudaInterface()->LaunchExpand(src, dst, num_beams, batch_size, sequence_length, stream); }
template <>
void Launch_UpdatePositionIds<int32_t>(int32_t* position_ids, int batch_beam_size, int total_length, int new_kv_length, cudaStream_t stream) { GetCudaInterface()->Launch_UpdatePositionIds(position_ids, batch_beam_size, total_length, new_kv_length, stream); }
template <>
void Launch_UpdatePositionIds<int64_t>(int64_t* position_ids, int batch_beam_size, int total_length, int new_kv_length, cudaStream_t stream) { GetCudaInterface()->Launch_UpdatePositionIds(position_ids, batch_beam_size, total_length, new_kv_length, stream); }
template <>
void Launch_UpdateAttentionMask<int32_t>(int32_t* mask_data, const int32_t* old_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, cudaStream_t stream) { GetCudaInterface()->Launch_UpdateAttentionMask(mask_data, old_data, batch_beam_size, new_kv_length, total_length, max_length, update_only, stream); }
template <>
void Launch_UpdateAttentionMask<int64_t>(int64_t* mask_data, const int64_t* old_data, int batch_beam_size, int new_kv_length, int total_length, int max_length, bool update_only, cudaStream_t stream) { GetCudaInterface()->Launch_UpdateAttentionMask(mask_data, old_data, batch_beam_size, new_kv_length, total_length, max_length, update_only, stream); }
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
    case DeviceType::QNN:
      return "QnnWithSharedMemory";
  }
  throw std::runtime_error("Unknown device type");
}

DeviceInterface* GetDeviceInterface(DeviceType type) {
  switch (type) {
    default:
    case DeviceType::CPU:
      return GetCpuInterface();
#if USE_CUDA
    case DeviceType::CUDA:
      return GetCudaInterface();
#endif
  }
}

GeneratorParams::GeneratorParams(const Config& config)
    : config{config},
      p_device{GetDeviceInterface(DeviceType::CPU)} {
}

GeneratorParams::GeneratorParams(const Model& model)
    : config{*model.config_.get()},
      p_device{model.p_device_},
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
  if (config.model.type == "gpt2" || config.model.type == "llama" || config.model.type == "gemma" || config.model.type == "gemma2" || config.model.type == "mistral" || config.model.type == "phi" || config.model.type == "phi3" || config.model.type == "phi3small" || config.model.type == "phimoe" || config.model.type == "qwen2" || config.model.type == "decoder-pipeline")
    throw std::runtime_error("Please use generator.AppendTokens for " + config.model.type + ". SetInputs is not supported for this model type.");

  for (const auto& [name, tensor] : named_tensors) {
    if (name == Config::Defaults::InputIdsName) {
      aux_input_ids = cpu_span<int32_t>(tensor->ort_tensor_->GetTensorMutableData<int32_t>(),
                                        tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount());
      if (aux_input_ids.size() / search.batch_size > search.max_length)
        throw std::runtime_error("input_ids size (" + std::to_string(aux_input_ids.size()) + ") exceeds max length (" + std::to_string(search.max_length) + ")");
      else if (aux_input_ids.size() == 0)
        throw std::runtime_error("input_ids is empty");
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
  if (params.search.num_beams > 1)
    return params.p_device->CreateBeam(params);
  return params.p_device->CreateGreedy(params);
}

Generator::Generator(const Model& model, const GeneratorParams& params) : model_{model.shared_from_this()} {
  if (params.search.max_length == 0)
    throw std::runtime_error("search max_length is 0");
  if (params.search.max_length > model.config_->model.context_length)
    throw std::runtime_error("max_length (" + std::to_string(params.search.max_length) + ") cannot be greater than model context_length (" + std::to_string(model.config_->model.context_length) + ")");
  if (params.search.batch_size < 1)
    throw std::runtime_error("batch_size must be 1 or greater, is " + std::to_string(params.search.batch_size));
  if (params.config.model.vocab_size < 1)
    throw std::runtime_error("vocab_size must be 1 or greater, is " + std::to_string(params.config.model.vocab_size));

  search_ = CreateSearch(params);
  state_ = model.CreateState(search_->GetSequenceLengths(), params);  // Search sequence lengths set when creating state

  // Temporary solution for multimodal and whisper models
  if (!params.aux_input_ids.empty() && params.aux_input_ids.data() != nullptr) {
    AuxAppendTokens(params.aux_input_ids);
  }
}

DeviceSpan<int32_t> Generator::AllocateInputIdsOnDevice(cpu_span<const int32_t> input_ids) {
  size_t padded_input_ids_size = input_ids.size();
  if (model_->config_->model.decoder.sliding_window.has_value()) {
    // If the model has a sliding window, pad the input_ids to the next multiple of the window size
    // so that the input_ids can be divided into window size chunks.
    const auto window_size = model_->config_->model.decoder.sliding_window->window_size;
    padded_input_ids_size = ((input_ids.size() + window_size - 1) / window_size) * window_size;
  }
  auto input_ids_device = state_->params_->p_device->Allocate<int32_t>(padded_input_ids_size);
  auto cpu_span = input_ids_device.CpuSpan();
  std::fill_n(cpu_span.begin(), padded_input_ids_size - input_ids.size(), model_->config_->model.pad_token_id);
  std::copy_backward(input_ids.begin(), input_ids.end(), cpu_span.end());
  input_ids_device.CopyCpuToDevice();
  return input_ids_device;
}

// TODO(aciddelgado): Remove this function once SetInputs is moved to generator
void Generator::AuxAppendTokens(cpu_span<const int32_t> input_ids) {
  ThrowErrorIfSessionTerminated(state_->session_terminated_);
  if (input_ids.size() == 0)
    throw std::runtime_error("input_ids is empty");
  if (search_->GetSequenceLength() != 0 && state_->params_->search.batch_size > 1)
    throw std::runtime_error("AppendTokens can only be called once for batch_size > 1. To call AppendTokens again, use RewindToLength(0)");

  auto input_ids_device = AllocateInputIdsOnDevice(input_ids);
  search_->AppendTokens(input_ids_device);
  computed_logits_ = false;
  ComputeLogits(input_ids_device);
}

void Generator::AppendTokens(cpu_span<const int32_t> input_ids) {
  ThrowErrorIfSessionTerminated(state_->session_terminated_);
  if (input_ids.size() == 0)
    throw std::runtime_error("input_ids is empty");
  if ((input_ids.size() / state_->params_->search.batch_size) + search_->GetSequenceLength() > state_->params_->search.max_length)
    throw std::runtime_error("input_ids size (" + std::to_string(input_ids.size()) + ") + current sequence length (" + std::to_string(search_->GetSequenceLength()) + ") exceeds max length (" + std::to_string(state_->params_->search.max_length) + ")");
  if (model_->config_->model.type == "whisper" || model_->config_->model.type == "phi3v")
    throw std::runtime_error("Please use params.SetInputs for " + model_->config_->model.type + ". AppendTokens is not supported for this model type.");
  if (search_->GetSequenceLength() != 0 && state_->params_->search.batch_size > 1)
    throw std::runtime_error("AppendTokens can only be called once for batch_size > 1. To call AppendTokens again, use RewindToLength(0)");

  constexpr std::array<DeviceType, 3> devices_supporting_continuous_decoding{DeviceType::CPU, DeviceType::CUDA, DeviceType::WEBGPU};
  if (search_->GetSequenceLength() != 0 &&
      std::none_of(devices_supporting_continuous_decoding.begin(), devices_supporting_continuous_decoding.end(),
                   [this](DeviceType device_type) { return device_type == state_->params_->device_type; }))
    throw std::runtime_error("Continuous decoding is not supported on the selected device type (" + to_string(state_->params_->device_type) +
                             "). Please recreate the generator instance to avoid using continuous decoding.");

  if (last_action_ == Action::generated) {
    ComputeLogits(search_->GetNextTokens());
  }

  auto input_ids_device = AllocateInputIdsOnDevice(input_ids);
  search_->AppendTokens(input_ids_device);
  computed_logits_ = false;
  ComputeLogits(input_ids_device);
}

void Generator::ComputeLogits(DeviceSpan<int32_t> next_tokens) {
  if (computed_logits_)
    throw std::runtime_error("ComputeLogits called again without calling AppendTokens or GenerateNextToken first");

  auto logits = state_->Run(search_->GetSequenceLength(), next_tokens, search_->GetNextIndices());
  if (g_log.enabled && g_log.model_logits) {
    auto& stream = Log("model_logits");
    DumpSpan(stream, logits.CopyDeviceToCpu());
    stream << std::endl;
  }
  SetLogits(logits);
  last_action_ = Action::standard;
  computed_logits_ = true;
}

void Generator::SetRuntimeOption(const char* key, const char* value) {
  // TODO: Need a better way to handle different keys
  // We can create a config manager to host all configurations and do comparison at that point
  if (strcmp(key, "terminate_session") == 0) {
    if (strcmp(value, "0") == 0) {
      state_->UnsetTerminate();
    } else if (strcmp(value, "1") == 0) {
      state_->SetTerminate();
    } else {
      // Value not expected
      throw std::runtime_error(std::string("terminate_session key value unexpected: ") + value);
    }
  } else {
    throw std::runtime_error(std::string("SetRuntimeOption key is not expected: ") + key);
  }
}

bool Generator::IsDone() const {
  ThrowErrorIfSessionTerminated(state_->session_terminated_);
  if (computed_logits_) {
    return false;
  }

  bool is_done = search_->IsDone();
  if (is_done) {
    state_->Finalize();
  }

  return is_done;
}

bool Generator::IsSessionTerminated() const {
  return state_->session_terminated_;
}

void Generator::SetLogits(DeviceSpan<float> logits) {
  search_->SetLogits(logits);
  computed_logits_ = true;
}

void Generator::GenerateNextToken() {
  ThrowErrorIfSessionTerminated(state_->session_terminated_);
  if (search_->GetSequenceLength() == 0 && !computed_logits_)
    throw std::runtime_error("GenerateNextToken called with no prior state. Please call AppendTokens, SetLogits, or params.SetInputs before calling GenerateNextToken.");

  // TODO: Extend the solution to make it work for batch size > 1, num beams > 1, multimodal and DML
  // Phi3 model switches from short factor to long factor at 4097 (original_max_position_embeddings+1) token, needs Recomputation of Position IDs and KV Cache
  // at this stage which is achieved by rewinding to zero and appending the current sequence
  // Scenarios where this solution works: Batch size = 1, Num beams = 1, decoder model, EP is either CPU or CUDA
  // Scenarios where it doesn't work: Batch size > 1 OR Num beams > 1 OR Multimodal model (like phi3 vision) OR EP is DML
  if (search_->params_->BatchBeamSize() == 1) {
    if (((search_->GetSequenceLength() == 4097) && (model_->config_->model.type == "phi3" || model_->config_->model.type == "phimoe")) || ((search_->GetSequenceLength() == 8197) && (model_->config_->model.type == "phi3small"))) {
      auto current_seq = cpu_span<int32_t>(GetSequence(0).CopyDeviceToCpu());
      RewindToLength(0);
      AppendTokens(current_seq);
    }
  }

  if (!computed_logits_) {
    auto next_tokens = search_->GetNextTokens();
    if (last_action_ == Action::rewound)
      search_->AppendTokens(next_tokens);
    ComputeLogits(next_tokens);
  }
  computed_logits_ = false;
  auto& search = search_->params_->search;
  search_->ApplyMinLength(search.min_length);
  search_->ApplyRepetitionPenalty(search.repetition_penalty);

  if (g_log.enabled && g_log.generate_next_token) {
    auto& stream = Log("generate_next_token");
    stream << SGR::Fg_Green << "do_sample: " << SGR::Reset << search.do_sample << ' '
           << SGR::Fg_Green << "top_k: " << SGR::Reset << search.top_k << ' '
           << SGR::Fg_Green << "top_p: " << SGR::Reset << search.top_p << ' '
           << SGR::Fg_Green << "temperature: " << SGR::Reset << search.temperature << ' '
           << SGR::Fg_Cyan << "sequence length: " << SGR::Reset << search_->GetSequenceLength()
           << std::endl;
  }

  last_action_ = Action::generated;
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
  if (search.temperature <= 0.0f)
    throw std::runtime_error("temperature must be greater than 0");

  if (search.top_p > 0.0f && search.top_p < 1.0f && search.top_k > 1) {
    search_->SampleTopKTopP(search.top_k, search.top_p, search.temperature);
  } else if (search.top_k > 1) {
    search_->SampleTopK(search.top_k, search.temperature);
  } else {
    assert(search.top_k == 0);
    search_->SampleTopP(search.top_p, search.temperature);
  }
}

void Generator::RewindToLength(size_t new_length) {
  if (model_->config_->model.type == "whisper" || model_->config_->model.type == "phi3v" || model_->config_->model.type == "decoder-pipeline")
    throw std::runtime_error("RewindTo is currently not supported for " + model_->config_->model.type + ".");
  if (new_length > search_->GetSequenceLength())
    throw std::runtime_error("Cannot rewind to a length greater than the current sequence length");
  if (new_length == search_->GetSequenceLength())
    return;
  size_t batch_size = search_->params_->search.batch_size;
  if (batch_size > 1 && new_length != 0)
    throw std::runtime_error("RewindToLength must be called with new_length=0 when batch_size > 1");
  search_->RewindTo(new_length);
  state_->RewindTo(new_length);
  computed_logits_ = false;
  last_action_ = Action::rewound;
}

DeviceSpan<float> Generator::GetLogits() {
  if (!computed_logits_) {
    ComputeLogits(search_->GetNextTokens());
  }
  return search_->GetLogits();
}

DeviceSpan<int32_t> Generator::GetSequence(size_t index) const {
  return search_->GetSequence(index);
}

}  // namespace Generators

#if USE_CUDA
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) { return Generators::GetCudaInterface()->cudaMemcpyAsync(dst, src, count, kind, stream); }
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) { return Generators::GetCudaInterface()->cudaMemcpy(dst, src, count, kind); }
cudaError_t cudaMemsetAsync(void* ptr, int value, size_t count, cudaStream_t stream) { return Generators::GetCudaInterface()->cudaMemsetAsync(ptr, value, count, stream); }
cudaError_t cudaMemset(void* ptr, int value, size_t count) { return Generators::GetCudaInterface()->cudaMemset(ptr, value, count); }
#endif
