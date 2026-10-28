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
#include "../config_utils.h"
#include "../json.h"
#include "../ort_genai_c_internal.h"
#include "io/quantized_kv_cache.h"
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
    const fs::path& model_path, const Config& config,
    bool has_component_config) {
  auto options = OrtSessionOptions::Create();
  const auto& configured_session_options =
      config.model.decoder.session_options;
  if (has_component_config) {
    if (configured_session_options.intra_op_num_threads)
      options->SetIntraOpNumThreads(
          *configured_session_options.intra_op_num_threads);
    if (configured_session_options.inter_op_num_threads)
      options->SetInterOpNumThreads(
          *configured_session_options.inter_op_num_threads);
    if (configured_session_options.enable_cpu_mem_arena) {
      if (*configured_session_options.enable_cpu_mem_arena)
        options->EnableCpuMemArena();
      else
        options->DisableCpuMemArena();
    }
    if (configured_session_options.enable_mem_pattern) {
      if (*configured_session_options.enable_mem_pattern)
        options->EnableMemPattern();
      else
        options->DisableMemPattern();
    }
    if (configured_session_options.log_id)
      options->SetLogId(configured_session_options.log_id->c_str());
    if (configured_session_options.log_severity_level)
      options->SetLogSeverityLevel(
          *configured_session_options.log_severity_level);
    if (configured_session_options.log_verbosity_level)
      options->SetLogVerbosityLevel(
          *configured_session_options.log_verbosity_level);
    if (configured_session_options.enable_profiling)
      options->EnableProfiling(
          fs::path(*configured_session_options.enable_profiling).c_str());
    if (configured_session_options.graph_optimization_level)
      options->SetGraphOptimizationLevel(
          *configured_session_options.graph_optimization_level);
    for (const auto& [name, value] :
         configured_session_options.config_entries)
      options->AddConfigEntry(name.c_str(), value.c_str());
  }
  if (!configured_session_options.providers.empty())
    SetProviderSessionOptions(
        *options, config.model.decoder.session_options.providers,
        config.model.decoder.session_options.provider_options, true, config);
  return OrtSession::Create(GetOrtEnv(), model_path.c_str(), options.get());
}

}  // namespace

bool PackageHasComponent(const fs::path& package_path,
                         std::string_view component) {
  return LoadComponents(package_path).count(std::string(component)) != 0;
}

ComponentSession::ComponentSession(const Config& config, std::string component) {
  const auto& package_path = config.config_path;
  auto components = LoadComponents(package_path);
  auto found = components.find(component);
  if (found == components.end()) throw std::runtime_error("component not declared: " + component);
  auto effective_config = ResolveComponentConfig(config);
  uses_cuda_ = std::find(
                   effective_config.model.decoder.session_options.providers.begin(),
                   effective_config.model.decoder.session_options.providers.end(),
                   "cuda") !=
               effective_config.model.decoder.session_options.providers.end();
  try {
    session_ = CreateComponentSession(
        found->second, effective_config, true);
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

namespace {

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

int32_t ResolvePadTokenId(
    const fs::path& path, const Generators::Tokenizer& tokenizer) {
  const auto config_path = path / "tokenizer_config.json";
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

Generators::PackageTokenizer::PackageTokenizer(const Config& config)
    : config_(config),
      tokenizer_(config_),
      pad_token_id_(ResolvePadTokenId(config_.config_path, tokenizer_)),
      context_length_(
          config_.model.context_length > 0
              ? std::min<size_t>(
                    8192, static_cast<size_t>(config_.model.context_length))
              : 8192) {}

std::vector<int32_t> Generators::PackageTokenizer::Encode(
    const std::string& text) const {
  if (text.find('\0') != std::string::npos)
    throw std::invalid_argument(
        "structured text must not contain embedded NUL characters");
  return tokenizer_.Encode(text.c_str());
}
