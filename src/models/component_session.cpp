// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "component_session.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <variant>

#include "../config.h"
#include "../json.h"
#include "../ort_genai_c_internal.h"
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
    if (std::filesystem::is_regular_file(candidate.c_str())) result.emplace(name, std::move(candidate));
  }
  if (result.size() != 4 && result.size() != 2)
    throw std::runtime_error("directory has no component_manifest.json and is not a recognized CLM/KEV layout");
  return result;
}

}  // namespace

ComponentSession::ComponentSession(const fs::path& package_path, std::string component,
                                   const std::vector<std::string>& providers) {
  auto components = LoadComponents(package_path);
  auto found = components.find(component);
  if (found == components.end()) throw std::runtime_error("component not declared: " + component);
  auto options = OrtSessionOptions::Create();
  Config config;
  for (const auto& provider : providers) {
    if (provider.empty()) throw std::runtime_error("provider name must not be empty");
    SetProviderOption(config, provider, {}, {});
  }
  if (!providers.empty())
    SetProviderSessionOptions(*options, config.model.decoder.session_options.providers,
                              config.model.decoder.session_options.provider_options, true, config);
  session_ = OrtSession::Create(GetOrtEnv(), found->second.c_str(), options.get());
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
}

std::vector<OgaComponentTensor> ComponentSession::Run(
    const std::vector<OgaComponentInput>& inputs, const std::vector<std::string>& requested) {
  std::lock_guard lock(mutex_);
  auto memory = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
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
  const auto& output_names = requested.empty() ? output_names_ : requested;
  std::vector<const char*> output_ptrs;
  for (const auto& name : output_names) output_ptrs.push_back(name.c_str());
  auto ort_outputs = session_->Run(nullptr, names.data(), value_ptrs.data(), value_ptrs.size(),
                                   output_ptrs.data(), output_ptrs.size());
  std::vector<OgaComponentTensor> result;
  for (size_t i = 0; i < ort_outputs.size(); ++i) {
    auto info = ort_outputs[i]->GetTensorTypeAndShapeInfo();
    OgaComponentTensor tensor;
    tensor.name = output_names[i];
    tensor.shape = info->GetShape();
    tensor.type = static_cast<OgaElementType>(info->GetElementType());
    tensor.data.resize(info->GetElementCount() * Ort::SizeOf(info->GetElementType()));
    std::memcpy(tensor.data.data(), ort_outputs[i]->GetTensorRawData(), tensor.data.size());
    result.push_back(std::move(tensor));
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
