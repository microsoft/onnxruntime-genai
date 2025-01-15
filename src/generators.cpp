// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "sequences.h"
#include "models/model.h"
#include "models/decoder_only.h"
#include "search.h"
#include "cpu/interface.h"
#include "cuda/interface.h"
#include "dml/interface.h"

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

DeviceInterface* GetCudaInterface() {
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

  Generators::DeviceInterface* GetInterface(GenaiInterface * p_genai);
  static DeviceInterface* cuda_interface{[] {
#ifdef _WIN32
    auto get_cuda_fn = reinterpret_cast<decltype(&GetInterface)>(GetProcAddress(reinterpret_cast<HMODULE>(cuda_library.get()), "GetInterface"));
#else
    auto get_cuda_fn = reinterpret_cast<decltype(&GetInterface)>(dlsym(cuda_library.get(), "GetInterface"));
#endif
    return get_cuda_fn(&g_genai);
  }()};

  return cuda_interface;
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
  }
  throw std::runtime_error("Unknown device type");
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
    AppendTokens(params.aux_input_ids);
  }
}

DeviceSpan<int32_t> Generator::AllocateInputIdsOnDevice(const cpu_span<int32_t>& input_ids) {
  auto input_ids_device = state_->params_->p_device->Allocate<int32_t>(input_ids.size());
  auto cpu_span = input_ids_device.CpuSpan();
  std::copy(input_ids.begin(), input_ids.end(), cpu_span.begin());
  input_ids_device.CopyCpuToDevice();
  return input_ids_device;
}

void Generator::AppendTokens(const cpu_span<int32_t>& input_ids) {
  ThrowErrorIfSessionTerminated(state_->session_terminated_);
  if (input_ids.size() == 0)
    throw std::runtime_error("input_ids is empty");
  if (model_->config_->model.type == "whisper" || model_->config_->model.type == "phi3v")
    throw std::runtime_error("Please use params.SetInputs for " + model_->config_->model.type + ". AppendTokens is not supported for this model type.");
  if (search_->GetSequenceLength() != 0 && state_->params_->search.batch_size > 1)
    throw std::runtime_error("AppendTokens can only be called once for batch_size > 1. To call AppendTokens again, use RewindToLength(0)");

  auto input_ids_device = AllocateInputIdsOnDevice(input_ids);
  search_->AppendTokens(input_ids_device);

  computed_logits_ = false;
  ComputeLogits(input_ids_device);
}

void Generator::ComputeLogits(DeviceSpan<int32_t>& next_tokens) {
  if (computed_logits_)
    throw std::runtime_error("ComputeLogits called again without calling AppendTokens or GenerateNextToken first");

  auto logits = state_->Run(search_->GetSequenceLength(), next_tokens, search_->GetNextIndices());
  if (g_log.enabled && g_log.model_logits) {
    auto& stream = Log("model_logits");
    DumpSpan(stream, logits.CopyDeviceToCpu());
    stream << std::endl;
  }
  SetLogits(logits);
  just_rewinded_ = false;
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
  if (!computed_logits_) {
    auto next_tokens = search_->GetNextTokens();
    if (just_rewinded_)
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
  just_rewinded_ = true;
}

DeviceSpan<float> Generator::GetLogits() {
  if (!computed_logits_) {
    auto next_tokens = search_->GetNextTokens();
    ComputeLogits(next_tokens);
  }
  return search_->GetLogits();
}

DeviceSpan<int32_t> Generator::GetSequence(size_t index) const {
  return search_->GetSequence(index);
}

}  // namespace Generators
