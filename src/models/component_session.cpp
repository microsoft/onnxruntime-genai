// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "component_session.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <variant>

#include "../config.h"
#include "../json.h"
#include "../ort_genai_c_internal.h"
#include "model.h"
#include "preprocessing/genai_tokenizer.h"
#include "session_options.h"

namespace Generators {
namespace {

struct IgnoreElement : JSON::Element {
  void OnValue(std::string_view, JSON::Value) override {}
  Element& OnObject(std::string_view) override { return *this; }
  Element& OnArray(std::string_view) override { return *this; }
};

struct ComponentEntry : JSON::Element {
  std::string filename;
  IgnoreElement ignored;
  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "filename") filename = std::string(JSON::Get<std::string_view>(value));
    // Other component metadata is intentionally forward-compatible.
  }
  Element& OnObject(std::string_view) override { return ignored; }
  Element& OnArray(std::string_view) override { return ignored; }
};

struct Components : JSON::Element {
  std::unordered_map<std::string, ComponentEntry> values;
  Element& OnObject(std::string_view name) override {
    if (name.empty()) throw std::runtime_error("component name must not be empty");
    return values[std::string(name)];
  }
};

struct Manifest : JSON::Element {
  double schema_version{};
  std::string model_type;
  Components components;
  IgnoreElement ignored;
  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "schema_version")
      schema_version = JSON::Get<double>(value);
    else if (name == "model_type")
      model_type = std::string(JSON::Get<std::string_view>(value));
  }
  Element& OnObject(std::string_view name) override {
    if (name.empty()) return *this;
    if (name == "components") return components;
    return ignored;
  }
  Element& OnArray(std::string_view) override { return ignored; }
};

std::unordered_map<std::string, fs::path> LoadComponents(fs::path root) {
  auto manifest_path = root / "component_manifest.json";
  const auto canonical_root = std::filesystem::weakly_canonical(root.c_str());
  std::unordered_map<std::string, fs::path> result;
  if (std::filesystem::is_regular_file(manifest_path.c_str())) {
    std::ifstream stream(manifest_path.c_str(), std::ios::binary);
    std::stringstream buffer;
    buffer << stream.rdbuf();
    Manifest manifest;
    try {
      JSON::Parse(manifest, buffer.str());
    } catch (...) {
      JSON::TranslateException("component_manifest.json");
    }
    if (manifest.schema_version != 1)
      throw std::runtime_error("component_manifest.json schema_version must be 1");
    if (manifest.model_type.empty())
      throw std::runtime_error("component_manifest.json model_type must be a string");
    if (manifest.components.values.empty())
      throw std::runtime_error("component_manifest.json components must be non-empty");
    for (const auto& [name, entry] : manifest.components.values) {
      if (entry.filename.empty()) throw std::runtime_error("component \"" + name + "\" requires filename");
      const std::filesystem::path relative(entry.filename);
      if (relative.is_absolute() || std::find(relative.begin(), relative.end(), "..") != relative.end())
        throw std::runtime_error("component \"" + name + "\" filename must be package-relative without traversal");
      fs::path resolved = root / fs::path(entry.filename);
      if (!std::filesystem::is_regular_file(resolved.c_str()))
        throw std::runtime_error("component \"" + name + "\" file does not exist: " + entry.filename);
      const auto canonical_resolved = std::filesystem::weakly_canonical(resolved.c_str());
      const auto relative_to_root = canonical_resolved.lexically_relative(canonical_root);
      if (relative_to_root.empty() || relative_to_root.is_absolute() ||
          (!relative_to_root.empty() && *relative_to_root.begin() == ".."))
        throw std::runtime_error("component \"" + name + "\" filename resolves outside the package");
      result.emplace(name, std::move(resolved));
    }
    return result;
  }

  // Compatibility is deliberately restricted to the two released layouts.
  const std::pair<const char*, const char*> known[] = {
      {"encoder", "encoder/model.onnx"}, {"state_head", "state_head/model.onnx"}, {"action_head", "action_head/model.onnx"}, {"scorer", "scorer/model.onnx"}, {"backbone", "backbone/model.onnx"}, {"pointer_head", "pointer_head/model.onnx"}};
  for (const auto& [name, relative] : known) {
    fs::path candidate = root / relative;
    if (!std::filesystem::is_regular_file(candidate.c_str())) continue;
    const auto canonical_candidate =
        std::filesystem::weakly_canonical(candidate.c_str());
    const auto relative_to_root =
        canonical_candidate.lexically_relative(canonical_root);
    if (relative_to_root.empty() || relative_to_root.is_absolute() ||
        *relative_to_root.begin() == "..")
      throw std::runtime_error(
          "legacy component \"" + std::string(name) +
          "\" resolves outside the package");
    result.emplace(name, std::move(candidate));
  }
  const std::set<std::string> names = [&] {
    std::set<std::string> values;
    for (const auto& [name, _] : result) values.insert(name);
    return values;
  }();
  const std::set<std::string> clm_names = {
      "encoder", "state_head", "action_head", "scorer"};
  const std::set<std::string> kev_names = {"backbone", "pointer_head"};
  if (names != clm_names && names != kev_names)
    throw std::runtime_error("directory has no component_manifest.json and is not a recognized CLM/KEV layout");
  return result;
}

bool IsCudaProvider(std::string_view provider) {
  return provider == "cuda" || provider == "CUDAExecutionProvider";
}

bool KevCudaGraphEnabled() {
  const char* value = std::getenv("ORT_GENAI_KEV_CUDA_GRAPH");
  if (!value || !*value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument(
      "ORT_GENAI_KEV_CUDA_GRAPH must be 0 or 1");
}

std::unique_ptr<OrtSession> CreateComponentSession(
    const fs::path& model_path, const std::vector<std::string>& providers,
    bool capture, const Config* component_config,
    const std::map<std::string, int64_t>& dimension_overrides = {}) {
  auto options = OrtSessionOptions::Create();
  Config config = component_config ? *component_config : Config{};
  auto& configured_session_options =
      config.model.decoder.session_options;
  if (component_config) {
    if (configured_session_options.intra_op_num_threads)
      options->SetIntraOpNumThreads(
          *configured_session_options.intra_op_num_threads);
    if (configured_session_options.inter_op_num_threads)
      options->SetInterOpNumThreads(
          *configured_session_options.inter_op_num_threads);
    for (const auto& [name, value] :
         configured_session_options.config_entries)
      options->AddConfigEntry(name.c_str(), value.c_str());
  }
  configured_session_options.providers.clear();
  configured_session_options.provider_options.clear();
  for (const auto& provider : providers) {
    if (provider.empty())
      throw std::runtime_error("provider name must not be empty");
    SetProviderOption(config, provider, {}, {});
    if (capture && IsCudaProvider(provider))
      SetProviderOption(config, provider, "enable_cuda_graph", "1");
  }
  if (!providers.empty())
    SetProviderSessionOptions(
        *options, config.model.decoder.session_options.providers,
        config.model.decoder.session_options.provider_options, true, config);
  for (const auto& [name, value] : dimension_overrides)
    options->AddFreeDimensionOverrideByName(name.c_str(), value);
  return OrtSession::Create(GetOrtEnv(), model_path.c_str(), options.get());
}

std::string TensorSignature(const std::vector<OgaComponentInput>& inputs,
                            const std::vector<std::string>& outputs) {
  std::ostringstream stream;
  for (const auto& input : inputs) {
    stream << input.name.size() << ':' << input.name << ':'
           << static_cast<int>(input.type) << ':' << input.byte_count << ':';
    for (const auto dimension : input.shape) stream << dimension << ',';
    stream << ';';
  }
  stream << "->";
  for (const auto& output : outputs)
    stream << output.size() << ':' << output << ';';
  return stream.str();
}

size_t TensorBytes(const std::vector<int64_t>& shape, OgaElementType type) {
  size_t count = 1;
  for (const auto dimension : shape) {
    if (dimension < 0)
      throw std::runtime_error(
          "captured component tensor has a negative dimension");
    const auto value = static_cast<size_t>(dimension);
    if (value && count > std::numeric_limits<size_t>::max() / value)
      throw std::runtime_error(
          "captured component tensor element count overflows size_t");
    count *= value;
  }
  const auto element_size =
      Ort::SizeOf(static_cast<ONNXTensorElementDataType>(type));
  if (element_size && count > std::numeric_limits<size_t>::max() / element_size)
    throw std::runtime_error(
        "captured component tensor byte count overflows size_t");
  return count * element_size;
}

}  // namespace

struct ComponentCudaGraphState {
  struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    OgaElementType type{};
    size_t byte_count{};
    DeviceSpan<std::byte> device;
    std::unique_ptr<OrtValue> value;
  };

  struct Run {
    std::string signature;
    int graph_id{1};
    bool captured{};
    std::unique_ptr<OrtRunOptions> options;
    std::unique_ptr<OrtIoBinding> binding;
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
  };

  fs::path model_path;
  std::vector<std::string> providers;
  std::unique_ptr<Config> config;
  DeviceInterface* device{};
  std::unique_ptr<OrtMemoryInfo> device_memory;
  std::unique_ptr<OrtRunOptions> eager_options;
  std::unique_ptr<Run> run;
  bool specialized{};
  bool disabled{};
};

ComponentSession::ComponentSession(const fs::path& package_path, std::string component,
                                   const std::vector<std::string>& providers) {
  auto components = LoadComponents(package_path);
  auto found = components.find(component);
  if (found == components.end()) throw std::runtime_error("component not declared: " + component);
  std::unique_ptr<Config> component_config;
  if (std::filesystem::is_regular_file(
          (package_path / "genai_config.json").c_str()))
    component_config =
        std::make_unique<Config>(package_path, std::string_view{});
  session_ = CreateComponentSession(found->second, providers, false,
                                    component_config.get());
  input_names_ = session_->GetInputNames();
  output_names_ = session_->GetOutputNames();
  for (size_t i = 0; i < input_names_.size(); ++i) {
    auto type = session_->GetInputTypeInfo(i);
    const auto& tensor = type->GetTensorTypeAndShapeInfo();
    OgaComponentInfo info;
    info.name = input_names_[i];
    info.shape = tensor.GetShape();
    info.type = static_cast<OgaElementType>(tensor.GetElementType());
    for (const char* symbol : tensor.GetSymbolicDimensions())
      info.symbolic_dimensions.emplace_back(symbol ? symbol : "");
    inputs_.push_back(std::move(info));
  }
  if (component == "backbone" &&
      std::any_of(providers.begin(), providers.end(), IsCudaProvider) &&
      KevCudaGraphEnabled()) {
    Config config;
    for (const auto& provider : providers)
      SetProviderOption(config, provider, {}, {});
    cuda_graph_ = std::make_unique<ComponentCudaGraphState>();
    cuda_graph_->model_path = found->second;
    cuda_graph_->providers = providers;
    cuda_graph_->config = std::move(component_config);
    cuda_graph_->device = GetDeviceInterface(DeviceType::CUDA);
    EnsureDeviceOrtInit(*cuda_graph_->device, config);
    cuda_graph_->device_memory = cuda_graph_->device->GetMemoryInfo();
    cuda_graph_->eager_options = OrtRunOptions::Create();
    cuda_graph_->eager_options->AddConfigEntry("gpu_graph_id", "-1");
  }
}

ComponentSession::~ComponentSession() {
#if ORT_API_VERSION >= 27
  if (cuda_graph_ && cuda_graph_->run && cuda_graph_->run->captured) {
    try {
      session_->ReleaseCapturedGraph(cuda_graph_->run->graph_id);
    } catch (...) {
      if (g_log.enabled && g_log.ort_lib)
        Log("ort_lib") << "ReleaseCapturedGraph(id="
                       << cuda_graph_->run->graph_id
                       << ") failed during component-session cleanup"
                       << std::endl;
    }
  }
#endif
}

std::vector<OgaComponentTensor> ComponentSession::Run(
    const std::vector<OgaComponentInput>& inputs, const std::vector<std::string>& requested) {
  std::lock_guard lock(mutex_);
  const auto& output_names = requested.empty() ? output_names_ : requested;
  const auto signature = TensorSignature(inputs, output_names);

  // Fixing symbolic dimensions folds host-side shape nodes so the backbone is
  // fully CUDA-resident and eligible for graph capture.
  if (cuda_graph_ && !cuda_graph_->specialized && !cuda_graph_->disabled) {
    std::map<std::string, int64_t> overrides;
    for (const auto& input : inputs) {
      const auto found = std::find_if(
          inputs_.begin(), inputs_.end(),
          [&](const OgaComponentInfo& value) {
            return value.name == input.name;
          });
      if (found == inputs_.end() ||
          found->symbolic_dimensions.size() != input.shape.size())
        throw std::runtime_error(
            "captured component input metadata does not match: " +
            input.name);
      for (size_t i = 0; i < input.shape.size(); ++i) {
        const auto& symbol = found->symbolic_dimensions[i];
        if (symbol.empty()) continue;
        const auto [entry, inserted] =
            overrides.emplace(symbol, input.shape[i]);
        if (!inserted && entry->second != input.shape[i])
          throw std::runtime_error(
              "captured component symbolic dimension has conflicting values: " +
              symbol);
      }
    }
    session_.reset();
    try {
      session_ = CreateComponentSession(
          cuda_graph_->model_path, cuda_graph_->providers, true,
          cuda_graph_->config.get(), overrides);
      cuda_graph_->specialized = true;
    } catch (...) {
      session_ = CreateComponentSession(
          cuda_graph_->model_path, cuda_graph_->providers, false,
          cuda_graph_->config.get());
      cuda_graph_->disabled = true;
      throw;
    }
  }

  if (cuda_graph_ && cuda_graph_->run &&
      cuda_graph_->run->signature != signature) {
    // A captured graph owns fixed launch dimensions and buffer addresses.
    // Restore the generic session rather than recapturing unbounded shapes.
#if ORT_API_VERSION >= 27
    if (cuda_graph_->run->captured)
      session_->ReleaseCapturedGraph(cuda_graph_->run->graph_id);
#endif
    cuda_graph_->run.reset();
    session_.reset();
    session_ = CreateComponentSession(
        cuda_graph_->model_path, cuda_graph_->providers, false,
        cuda_graph_->config.get());
    cuda_graph_->specialized = false;
    cuda_graph_->disabled = true;
  }

  if (cuda_graph_ && cuda_graph_->run) {
    auto& run = *cuda_graph_->run;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i].data && inputs[i].byte_count)
        throw std::runtime_error(
            "component input data must not be null when byte_count is non-zero");
      if (inputs[i].byte_count != run.inputs[i].byte_count)
        throw std::runtime_error(
            "captured component input byte count changed");
      if (inputs[i].byte_count)
        run.inputs[i].device.CopyFromCpu(
            {static_cast<const std::byte*>(inputs[i].data),
             inputs[i].byte_count});
    }
    cuda_graph_->device->Synchronize();
    session_->Run(run.options.get(), *run.binding);
    run.captured = true;
    std::vector<OgaComponentTensor> result;
    result.reserve(run.outputs.size());
    for (auto& output : run.outputs) {
      OgaComponentTensor tensor;
      tensor.name = output.name;
      const auto host = output.device.CopyDeviceToCpu();
      tensor.data.assign(
          host.begin(),
          host.begin() + static_cast<std::ptrdiff_t>(output.byte_count));
      tensor.shape = output.shape;
      tensor.type = output.type;
      result.push_back(std::move(tensor));
    }
    return result;
  }

  auto memory =
      OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::vector<std::unique_ptr<OrtValue>> values;
  std::vector<const OrtValue*> value_ptrs;
  std::vector<const char*> names;
  static std::byte empty_tensor_storage{};
  for (const auto& input : inputs) {
    if (!input.data && input.byte_count)
      throw std::runtime_error("component input data must not be null when byte_count is non-zero");
    void* data = input.data ? const_cast<void*>(input.data) : &empty_tensor_storage;
    names.push_back(input.name.c_str());
    values.push_back(OrtValue::CreateTensor(*memory, data, input.byte_count,
                                            input.shape, static_cast<ONNXTensorElementDataType>(input.type)));
    value_ptrs.push_back(values.back().get());
  }
  std::vector<const char*> output_ptrs;
  for (const auto& name : output_names) output_ptrs.push_back(name.c_str());
  auto ort_outputs = session_->Run(
      cuda_graph_ && cuda_graph_->specialized
          ? cuda_graph_->eager_options.get()
          : nullptr,
      names.data(), value_ptrs.data(), value_ptrs.size(), output_ptrs.data(),
      output_ptrs.size());
  std::vector<OgaComponentTensor> result;
  for (size_t i = 0; i < ort_outputs.size(); ++i) {
    auto info = ort_outputs[i]->GetTensorTypeAndShapeInfo();
    OgaComponentTensor tensor;
    tensor.name = output_names[i];
    tensor.shape = info->GetShape();
    tensor.type = static_cast<OgaElementType>(info->GetElementType());
    tensor.data.resize(info->GetElementCount() * Ort::SizeOf(info->GetElementType()));
    if (!tensor.data.empty())
      std::memcpy(tensor.data.data(), ort_outputs[i]->GetTensorRawData(), tensor.data.size());
    result.push_back(std::move(tensor));
  }
  if (cuda_graph_ && cuda_graph_->specialized) {
    auto run = std::make_unique<ComponentCudaGraphState::Run>();
    run->signature = signature;
    run->options = OrtRunOptions::Create();
    run->options->AddConfigEntry("gpu_graph_id",
                                 std::to_string(run->graph_id).c_str());
    run->inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
      ComponentCudaGraphState::Tensor tensor;
      tensor.name = input.name;
      tensor.shape = input.shape;
      tensor.type = input.type;
      tensor.byte_count = input.byte_count;
      tensor.device = cuda_graph_->device->Allocate<std::byte>(
          std::max<size_t>(input.byte_count, 1));
      tensor.value = OrtValue::CreateTensor(
          *cuda_graph_->device_memory, tensor.device.Span().data(),
          input.byte_count, input.shape,
          static_cast<ONNXTensorElementDataType>(input.type));
      run->inputs.push_back(std::move(tensor));
    }
    run->outputs.reserve(result.size());
    for (const auto& output : result) {
      ComponentCudaGraphState::Tensor tensor;
      tensor.name = output.name;
      tensor.shape = output.shape;
      tensor.type = output.type;
      tensor.byte_count = TensorBytes(output.shape, output.type);
      tensor.device = cuda_graph_->device->Allocate<std::byte>(
          std::max<size_t>(tensor.byte_count, 1));
      tensor.value = OrtValue::CreateTensor(
          *cuda_graph_->device_memory, tensor.device.Span().data(),
          tensor.byte_count, tensor.shape,
          static_cast<ONNXTensorElementDataType>(tensor.type));
      run->outputs.push_back(std::move(tensor));
    }
    run->binding = OrtIoBinding::Create(*session_);
    for (auto& input : run->inputs)
      run->binding->BindInput(input.name.c_str(), *input.value);
    for (auto& output : run->outputs)
      run->binding->BindOutput(output.name.c_str(), *output.value);
    cuda_graph_->run = std::move(run);
  }
  return result;
}

}  // namespace Generators

namespace {

template <class T>
T& Required(T* value, const char* name) {
  if (!value) throw std::invalid_argument(std::string(name) + " must not be null");
  return *value;
}

const char* Required(const char* value, const char* name) {
  if (!value) throw std::invalid_argument(std::string(name) + " must not be null");
  return value;
}

std::vector<std::string> CopyStrings(const char* const* values, size_t count,
                                     const char* name) {
  if (count && !values) throw std::invalid_argument(std::string(name) + " must not be null");
  std::vector<std::string> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (!values[i]) throw std::invalid_argument(std::string(name) + " item must not be null");
    result.emplace_back(values[i]);
  }
  return result;
}

struct TokenValue : JSON::Element {
  std::string content;
  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "content" &&
        std::holds_alternative<std::string_view>(value))
      content = std::string(JSON::Get<std::string_view>(value));
  }
};

struct IgnoredTokenizerMetadata : JSON::Element {
  void OnValue(std::string_view, JSON::Value) override {}
  Element& OnObject(std::string_view) override { return *this; }
  Element& OnArray(std::string_view) override { return *this; }
};

struct TokenizerMetadata : JSON::Element {
  std::optional<int32_t> pad_token_id;
  std::string pad_token;
  TokenValue pad_token_object;
  IgnoredTokenizerMetadata ignored;
  void OnValue(std::string_view name, JSON::Value value) override {
    if (name == "pad_token_id" && std::holds_alternative<double>(value)) {
      const auto number = JSON::Get<double>(value);
      if (number < std::numeric_limits<int32_t>::min() ||
          number > std::numeric_limits<int32_t>::max() ||
          number != std::trunc(number))
        throw std::runtime_error("tokenizer_config.json pad_token_id must be an integer");
      pad_token_id = static_cast<int32_t>(number);
    } else if (name == "pad_token" &&
               std::holds_alternative<std::string_view>(value)) {
      pad_token = std::string(JSON::Get<std::string_view>(value));
    }
  }
  Element& OnObject(std::string_view name) override {
    if (name.empty()) return *this;
    if (name == "pad_token") return pad_token_object;
    return ignored;
  }
  Element& OnArray(std::string_view name) override {
    if (name.empty()) return *this;
    return ignored;
  }
};

Generators::Config DirectoryTokenizerConfig(const char* path) {
  if (!path) throw std::invalid_argument("package_path must not be null");
  Generators::Config config;
  config.config_path = fs::path(path);
  return config;
}

int32_t ResolvePadTokenId(
    const char* path, const Generators::Tokenizer& tokenizer) {
  const auto config_path = fs::path(path) / "tokenizer_config.json";
  std::ifstream stream(config_path.c_str(), std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot open tokenizer_config.json");
  std::stringstream buffer;
  buffer << stream.rdbuf();
  TokenizerMetadata metadata;
  try {
    JSON::Parse(metadata, buffer.str());
  } catch (...) {
    JSON::TranslateException("tokenizer_config.json");
  }
  if (metadata.pad_token_id) return *metadata.pad_token_id;
  const auto& pad_token = metadata.pad_token.empty()
                              ? metadata.pad_token_object.content
                              : metadata.pad_token;
  if (pad_token.empty()) return tokenizer.GetPadTokenId();
  const auto ids = tokenizer.Encode(pad_token.c_str());
  if (ids.size() != 1)
    throw std::runtime_error(
        "tokenizer_config.json pad_token must encode to exactly one token");
  return ids.front();
}

}  // namespace

#if defined(__GNUC__) && !defined(_WIN32)
#define OGA_CAPI_HANDLE __attribute__((visibility("hidden")))
#else
#define OGA_CAPI_HANDLE
#endif

struct OGA_CAPI_HANDLE OgaComponentSession {
  OgaComponentSession(const char* path, const char* component,
                      const char* const* providers, size_t provider_count)
      : value(fs::path(Required(path, "package_path")),
              Required(component, "component"),
              CopyStrings(providers, provider_count, "providers")) {}
  Generators::ComponentSession value;
};

struct OGA_CAPI_HANDLE OgaComponentInputs {
  struct OwnedInput {
    std::string name;
    std::vector<uint8_t> data;
    std::vector<int64_t> shape;
    OgaElementType type;
  };
  std::vector<OwnedInput> owned;

  std::vector<OgaComponentInput> Values() const {
    static constexpr uint8_t kEmptyInput = 0;
    std::vector<OgaComponentInput> values;
    values.reserve(owned.size());
    for (const auto& input : owned) {
      const void* data = input.data.empty() ? &kEmptyInput : input.data.data();
      values.push_back({input.name, data, input.data.size(),
                        input.shape, input.type});
    }
    return values;
  }
};

struct OGA_CAPI_HANDLE OgaComponentTensors {
  std::vector<OgaComponentTensor> values;
};

struct OGA_CAPI_HANDLE OgaDirectoryTokenizer {
  explicit OgaDirectoryTokenizer(const char* path)
      : value(DirectoryTokenizerConfig(path)),
        pad_token_id(ResolvePadTokenId(path, value)) {}
  Generators::Tokenizer value;
  int32_t pad_token_id;
};

struct OGA_CAPI_HANDLE OgaTokenIds {
  std::vector<int32_t> values;
};

extern "C" {

OgaResult* OGA_API_CALL OgaCreateComponentSession(
    const char* package_path, const char* component, const char* const* providers,
    size_t provider_count, OgaComponentSession** out) {
  OGA_CAPI_TRY
  auto& output = Required(out, "out");
  auto result = std::make_unique<OgaComponentSession>(
      package_path, component, providers, provider_count);
  output = result.release();
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyComponentSession(OgaComponentSession* session) { delete session; }

OgaResult* OGA_API_CALL OgaComponentSessionGetInputCount(
    const OgaComponentSession* session, size_t* out) {
  OGA_CAPI_TRY
  Required(out, "out") = Required(session, "session").value.InputNames().size();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentSessionGetOutputCount(
    const OgaComponentSession* session, size_t* out) {
  OGA_CAPI_TRY
  Required(out, "out") = Required(session, "session").value.OutputNames().size();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentSessionGetInputName(
    const OgaComponentSession* session, size_t index, const char** out) {
  OGA_CAPI_TRY
  Required(out, "out") = Required(session, "session").value.InputNames().at(index).c_str();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentSessionGetOutputName(
    const OgaComponentSession* session, size_t index, const char** out) {
  OGA_CAPI_TRY
  Required(out, "out") = Required(session, "session").value.OutputNames().at(index).c_str();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentSessionGetInputType(
    const OgaComponentSession* session, size_t index, OgaElementType* out) {
  OGA_CAPI_TRY
  Required(out, "out") = Required(session, "session").value.Inputs().at(index).type;
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentSessionGetInputShapeRank(
    const OgaComponentSession* session, size_t index, size_t* out) {
  OGA_CAPI_TRY
  Required(out, "out") = Required(session, "session").value.Inputs().at(index).shape.size();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentSessionGetInputShapeDimension(
    const OgaComponentSession* session, size_t index, size_t dimension, int64_t* out) {
  OGA_CAPI_TRY
  Required(out, "out") = Required(session, "session").value.Inputs().at(index).shape.at(dimension);
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentSessionGetInputSymbolicDimension(
    const OgaComponentSession* session, size_t index, size_t dimension, const char** out) {
  OGA_CAPI_TRY
  Required(out, "out") =
      Required(session, "session").value.Inputs().at(index).symbolic_dimensions.at(dimension).c_str();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaCreateComponentInputs(OgaComponentInputs** out) {
  OGA_CAPI_TRY
  auto& output = Required(out, "out");
  auto result = std::make_unique<OgaComponentInputs>();
  output = result.release();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentInputsAdd(
    OgaComponentInputs* inputs, const char* name, const void* data, size_t byte_count,
    const int64_t* shape, size_t shape_rank, OgaElementType type) {
  OGA_CAPI_TRY
  if (!data && byte_count) throw std::invalid_argument("data must not be null");
  if (!shape && shape_rank) throw std::invalid_argument("shape must not be null");
  auto& values = Required(inputs, "inputs").owned;
  OgaComponentInputs::OwnedInput input{
      Required(name, "name"), {}, {}, type};
  if (byte_count) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    input.data.assign(bytes, bytes + byte_count);
  }
  if (shape_rank) input.shape.assign(shape, shape + shape_rank);
  values.push_back(std::move(input));
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyComponentInputs(OgaComponentInputs* inputs) { delete inputs; }

OgaResult* OGA_API_CALL OgaComponentSessionRun(
    OgaComponentSession* session, const OgaComponentInputs* inputs,
    const char* const* output_names, size_t output_count, OgaComponentTensors** out) {
  OGA_CAPI_TRY
  auto& output = Required(out, "out");
  auto result = std::make_unique<OgaComponentTensors>();
  const auto input_values = Required(inputs, "inputs").Values();
  result->values = Required(session, "session").value.Run(input_values, CopyStrings(output_names, output_count, "output_names"));
  output = result.release();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentTensorsGetCount(
    const OgaComponentTensors* tensors, size_t* out) {
  OGA_CAPI_TRY
  Required(out, "out") = Required(tensors, "tensors").values.size();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentTensorsGetName(
    const OgaComponentTensors* tensors, size_t index, const char** out) {
  OGA_CAPI_TRY
  Required(out, "out") = Required(tensors, "tensors").values.at(index).name.c_str();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentTensorsGetType(
    const OgaComponentTensors* tensors, size_t index, OgaElementType* out) {
  OGA_CAPI_TRY
  Required(out, "out") = Required(tensors, "tensors").values.at(index).type;
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentTensorsGetShapeRank(
    const OgaComponentTensors* tensors, size_t index, size_t* out) {
  OGA_CAPI_TRY
  Required(out, "out") = Required(tensors, "tensors").values.at(index).shape.size();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentTensorsGetShapeDimension(
    const OgaComponentTensors* tensors, size_t index, size_t dimension, int64_t* out) {
  OGA_CAPI_TRY
  Required(out, "out") = Required(tensors, "tensors").values.at(index).shape.at(dimension);
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaComponentTensorsGetData(
    const OgaComponentTensors* tensors, size_t index, const void** data, size_t* byte_count) {
  OGA_CAPI_TRY
  const auto& value = Required(tensors, "tensors").values.at(index).data;
  Required(data, "data") = value.data();
  Required(byte_count, "byte_count") = value.size();
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyComponentTensors(OgaComponentTensors* tensors) { delete tensors; }

OgaResult* OGA_API_CALL OgaCreateDirectoryTokenizer(
    const char* package_path, OgaDirectoryTokenizer** out) {
  OGA_CAPI_TRY
  auto& output = Required(out, "out");
  auto result = std::make_unique<OgaDirectoryTokenizer>(package_path);
  output = result.release();
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyDirectoryTokenizer(OgaDirectoryTokenizer* tokenizer) {
  delete tokenizer;
}

OgaResult* OGA_API_CALL OgaDirectoryTokenizerEncode(
    const OgaDirectoryTokenizer* tokenizer, const char* text, OgaTokenIds** out) {
  OGA_CAPI_TRY
  auto& output = Required(out, "out");
  auto result = std::make_unique<OgaTokenIds>();
  result->values = Required(tokenizer, "tokenizer").value.Encode(Required(text, "text"));
  output = result.release();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaDirectoryTokenizerGetPadTokenId(
    const OgaDirectoryTokenizer* tokenizer, int32_t* out) {
  OGA_CAPI_TRY
  Required(out, "out") = Required(tokenizer, "tokenizer").pad_token_id;
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaTokenIdsGetData(
    const OgaTokenIds* token_ids, const int32_t** data, size_t* count) {
  OGA_CAPI_TRY
  const auto& values = Required(token_ids, "token_ids").values;
  Required(data, "data") = values.data();
  Required(count, "count") = values.size();
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyTokenIds(OgaTokenIds* token_ids) { delete token_ids; }

}  // extern "C"
