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
#include "dml/interface.h"
#include "qnn/interface.h"
#include "webgpu/interface.h"

#if defined(_WIN32)
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

#include "dll_load_error.h"
#endif

void ThrowErrorIfSessionTerminated(bool is_session_terminated) {
  if (is_session_terminated)
    throw std::runtime_error("Session in Terminated state, exiting!");
}

namespace Generators {

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

  // Init the CPU device (special case because it always exists, and its allocator is special
  GetDeviceInterface(DeviceType::CPU)->InitOrt(*Ort::api, allocator_cpu);
}

// Ensure Shutdown() has been called before process exit
struct EnsureShutdown {
  ~EnsureShutdown() {
    if (GetOrtGlobals()) {
      Shutdown();
    }
  }
};

std::unique_ptr<OrtGlobals>&
GetOrtGlobals() {
  static auto globals = std::make_unique<OrtGlobals>();
  static auto validate = std::make_unique<EnsureShutdown>();  // Must be after the above line so the destructor runs before the above destructor
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
  }

  GetOrtGlobals().reset();  // Delete now because on process exit is too late
}

OrtEnv& GetOrtEnv() {
  return *GetOrtGlobals()->env_;
}

// Fallback to copy between two separate device buffers by going through CPU memory (slow unless we're the CPU device)
void CopyThroughCpu(DeviceBuffer& dest, size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) {
  source.CopyDeviceToCpu();
  auto source_span = std::span<const uint8_t>(source.p_cpu_ + begin_source, size_in_bytes);
  // If we're overwriting the entire destination
  if (dest.size_in_bytes_ == size_in_bytes)
    dest.AllocateCpu();
  else
    dest.CopyDeviceToCpu();  // Overwriting part of destination, so copy over initial contents first
  std::copy(source_span.begin(), source_span.end(), dest.p_cpu_ + begin_dest);
  dest.CopyCpuToDevice();
}

struct GenaiInterfaceImpl : GenaiInterface {
#if _WIN32
  void* HeapAllocate(size_t size) override { return std::malloc(size); }
  void HeapFree(void* p) override { std::free(p); }
#endif

  void CopyThroughCpu(DeviceBuffer& dest, size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    return Generators::CopyThroughCpu(dest, begin_dest, source, begin_source, size_in_bytes);
  }

  Generators::LogItems& GetLogItems() override { return g_log; }
  std::ostream& operator_leftshift(std::ostream& stream, Generators::SGR sgr_code) override { return stream << sgr_code; }
  std::ostream& Log(std::string_view label, std::string_view text = {}) override { return Log(label, text); }

  void DumpSpan(std::ostream& stream, std::span<const float> values) override { return Generators::DumpSpan(stream, values); }
  void DumpSpan(std::ostream& stream, std::span<const int> values) override { return Generators::DumpSpan(stream, values); }

  void Sequences_AfterAppendNextTokens(Sequences* p_this, DeviceSpan<int32_t> next_tokens, size_t batch_beam_size) override { return p_this->AfterAppendNextTokens(next_tokens, batch_beam_size); }
  void Sequences_RewindTo(Sequences* p_this, size_t new_length) override { return p_this->RewindTo(new_length); }
} g_genai;

#if defined(_WIN32)
struct LibraryHandle {
  LibraryHandle(const char* filename) {
    auto path = CurrentModulePath() + filename;
    handle_ = LoadLibrary(path.c_str());
    if (!handle_)
      throw std::runtime_error(std::string("Failed to load library: ") + DetermineLoadLibraryError(filename));
  };

  ~LibraryHandle() { FreeLibrary(handle_); }

  FARPROC __stdcall GetSymbol(const char* name) { return ::GetProcAddress(handle_, name); }

  operator HANDLE() { return handle_; }

 private:
  HMODULE handle_{};
};
#elif defined(__linux__) && !defined(__ANDROID__)
struct LibraryHandle {
  LibraryHandle(const char* filename) {
    auto path = Ort::GetCurrentModuleDir() + "/" + filename;
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_)
      throw std::runtime_error(std::string("Failed to load library: ") + dlerror());  // dlerror() includes the path
  }
  ~LibraryHandle() {
    dlclose(handle_);
  }

  void* GetSymbol(const char* name) { return ::dlsym(handle_, name); }

  operator void*() { return handle_; }

 private:
  void* handle_{};
};
#else
struct LibraryHandle {
  LibraryHandle(const char* filename) {}
  ~LibraryHandle() {}

  void* GetSymbol(const char* name) { return nullptr; }

  operator bool() { return false; }
};
#endif

DeviceInterface* GetCudaInterface() {
  try {
#if defined(_WIN32)
    static LibraryHandle library{"onnxruntime-genai-cuda.dll"};
#elif defined(__linux__) && !defined(__ANDROID__)
    static LibraryHandle library{"libonnxruntime-genai-cuda.so"};
#else
    static LibraryHandle library{""};
#endif
    if (!library)
      throw std::runtime_error("Shared library load failure (see first error)");

    Generators::DeviceInterface* GetInterface(GenaiInterface * p_genai);
    static DeviceInterface* cuda_interface = reinterpret_cast<decltype(&GetInterface)>(library.GetSymbol("GetInterface"))(&g_genai);

    return cuda_interface;
  } catch (const std::exception& e) {
    throw std::runtime_error("Cuda interface not available: " + std::string(e.what()));
  }
}

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
    default:
      throw std::runtime_error("Unknown device type");
  }
}

DeviceInterface* GetDeviceInterface(DeviceType type) {
  switch (type) {
    default:
    case DeviceType::CPU:
      return GetCpuInterface();
    case DeviceType::CUDA:
      return GetCudaInterface();
#if USE_DML
    case DeviceType::DML:
      return GetDmlInterface();
#endif
    case DeviceType::WEBGPU:
      return GetWebGPUInterface();
    case DeviceType::QNN:
      return GetQNNInterface();
  }
}

GeneratorParams::GeneratorParams(const Config& config)
    : config{config},
      p_device{GetDeviceInterface(DeviceType::CPU)} {
}

GeneratorParams::GeneratorParams(const Model& model)
    : config{*model.config_.get()},
      use_graph_capture{IsGraphCaptureEnabled(model.config_->model.decoder.session_options)},
      p_device{model.p_device_inputs_} {
  if (use_graph_capture) {
    max_batch_size = 1;  // set it to 1 by default
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
  auto padding_begin = cpu_span.begin();
  auto data_end = cpu_span.end();
  if (model_->config_->model.decoder.sliding_window.has_value() && model_->config_->model.decoder.sliding_window->alignment == "left") {
    padding_begin = cpu_span.begin() + input_ids.size();
    data_end = padding_begin;
  }
  std::fill_n(padding_begin, padded_input_ids_size - input_ids.size(), model_->config_->model.pad_token_id);
  std::copy_backward(input_ids.begin(), input_ids.end(), data_end);
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
                   [this](DeviceType device_type) { return device_type == state_->params_->p_device->GetType(); }))
    throw std::runtime_error("Continuous decoding is not supported on the selected device type (" + to_string(state_->params_->p_device->GetType()) +
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
    DumpValues(stream, Ort::TypeToTensorType<float>, logits.CopyDeviceToCpu().data(), logits.size());
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
  if (!search.do_sample || search.top_k == 1 || search.temperature == 0) {
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
