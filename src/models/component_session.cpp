// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "component_session.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <variant>

#include "../config.h"
#include "../config_utils.h"
#include "../json.h"
#include "io/quantized_kv_cache.h"
#include "model.h"
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
      const fs::path relative_utf8(entry.filename);
      const std::filesystem::path relative(relative_utf8.c_str());
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

Config ResolveComponentConfig(const Config& component_config) {
  Config config = component_config;
  auto& configured_session_options =
      config.model.decoder.session_options;
  const auto effective_providers = configured_session_options.providers;
  for (const auto& provider : effective_providers)
    if (provider == "cuda")
      SetProviderOption(config, provider, "enable_cuda_graph", "0");
  return config;
}

std::unique_ptr<OrtSession> CreateComponentSession(
    const fs::path& model_path, const Config& config) {
  auto options = OrtSessionOptions::Create();
  const auto& configured_session_options =
      config.model.decoder.session_options;
  ApplySessionOptionsFromConfig(config, configured_session_options, *options);
  if (!configured_session_options.providers.empty())
    SetProviderSessionOptions(
        *options, config.model.decoder.session_options.providers,
        config.model.decoder.session_options.provider_options, true, config);
  return OrtSession::Create(GetOrtEnv(), model_path.c_str(), options.get());
}

}  // namespace

ComponentPackage::ComponentPackage(const fs::path& package_path)
    : components_(LoadComponents(package_path)) {}

bool ComponentPackage::Has(std::string_view component) const {
  return components_.count(std::string(component)) != 0;
}

const fs::path& ComponentPackage::Path(std::string_view component) const {
  const auto found = components_.find(std::string(component));
  if (found == components_.end())
    throw std::runtime_error("component not declared: " +
                             std::string(component));
  return found->second;
}

ComponentSession::ComponentSession(
    const Config& config, const ComponentPackage& package,
    std::string component) {
  auto effective_config = ResolveComponentConfig(config);
  uses_cuda_ = std::find(
                   effective_config.model.decoder.session_options.providers.begin(),
                   effective_config.model.decoder.session_options.providers.end(),
                   "cuda") !=
               effective_config.model.decoder.session_options.providers.end();
  try {
    session_ = CreateComponentSession(package.Path(component), effective_config);
  } catch (const std::exception& error) {
    if (std::string_view(error.what()).find("LinearAttentionGate") !=
        std::string_view::npos)
      throw std::runtime_error(
          "this KEV backbone requires ONNX Runtime 1.30 or newer with "
          "com.microsoft::LinearAttentionGate support: " +
          std::string(error.what()));
    throw;
  }
  input_names_ = session_->GetInputNames();
  output_names_ = session_->GetOutputNames();
  for (size_t i = 0; i < input_names_.size(); ++i) {
    auto type = session_->GetInputTypeInfo(i);
    const auto& tensor = type->GetTensorTypeAndShapeInfo();
    ComponentInfo info;
    info.name = input_names_[i];
    info.shape = tensor.GetShape();
    info.type = tensor.GetElementType();
    for (const char* symbol : tensor.GetSymbolicDimensions())
      info.symbolic_dimensions.emplace_back(symbol ? symbol : "");
    for (size_t dimension = 0;
         dimension < info.symbolic_dimensions.size(); ++dimension) {
      if (info.symbolic_dimensions[dimension] != "kv_cache_dim") continue;
      const auto head_size = effective_config.model.decoder.head_size;
      if (head_size <= 0)
        throw std::runtime_error(
            "model.decoder.head_size must resolve kv_cache_dim");
      int quantization_bits{};
      for (const auto& provider :
           effective_config.model.decoder.session_options.providers)
        quantization_bits = std::max(
            quantization_bits,
            GetKvCacheQuantizationBits(
                effective_config.model.decoder.session_options, provider));
      info.shape[dimension] = ComputeQuantizedKvCacheHeadSize(
          head_size, quantization_bits, info.type);
    }
    inputs_.push_back(std::move(info));
  }
}

ComponentSession::~ComponentSession() = default;

NamedTensors ComponentSession::Run(
    const NamedTensors& inputs, const std::vector<std::string>& requested) {
  std::lock_guard lock(mutex_);
  const auto& output_names = requested.empty() ? output_names_ : requested;
  std::vector<const OrtValue*> value_ptrs;
  std::vector<const char*> names;
  names.reserve(inputs.size());
  value_ptrs.reserve(inputs.size());
  for (const auto& [name, tensor] : inputs) {
    if (!tensor || !tensor->GetOrtTensor())
      throw std::invalid_argument("component input tensor must not be null: " +
                                  name);
    names.push_back(name.c_str());
    value_ptrs.push_back(tensor->GetOrtTensor());
  }
  std::vector<const char*> output_ptrs;
  for (const auto& name : output_names) output_ptrs.push_back(name.c_str());
  auto ort_outputs = session_->Run(nullptr, names.data(), value_ptrs.data(),
                                   value_ptrs.size(), output_ptrs.data(),
                                   output_ptrs.size());
  NamedTensors result;
  for (size_t i = 0; i < ort_outputs.size(); ++i) {
    result.emplace(output_names[i],
                   std::make_shared<Tensor>(std::move(ort_outputs[i])));
  }
  return result;
}

}  // namespace Generators
