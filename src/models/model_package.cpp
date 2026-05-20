// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "model_package.h"
#include "onnxruntime_api.h"
#include "../json.h"

namespace Generators {

// --- Package detection ---

bool IsModelPackage(const fs::path& path) {
  return fs::exists(path / "manifest.json");
}

// --- EP defaulting ---

#if ORT_HAS_MODEL_PACKAGE
std::string DefaultEpFromPackage(const OrtModelPackageContext& pkg_ctx,
                                 const std::vector<std::string>& scoped_components) {
  // Determine which components to consider: if scoped_components is non-empty, use only those;
  // otherwise, use all components in the package.
  std::vector<std::string> components_to_check;
  if (!scoped_components.empty()) {
    components_to_check = scoped_components;
  } else {
    auto all_names = pkg_ctx.GetComponentNames();
    for (const auto& n : all_names) {
      components_to_check.push_back(std::string(n));
    }
  }

  if (components_to_check.empty()) {
    throw std::runtime_error("Model package has no components");
  }

  // For each component, collect the union of EP names across all its variants.
  // Then intersect across components.
  std::unordered_set<std::string> intersection;
  bool first_component = true;

  for (const auto& comp_name : components_to_check) {
    std::unordered_set<std::string> comp_eps;
    auto variant_names = pkg_ctx.GetVariantNames(comp_name.c_str());

    for (const auto& var_name : variant_names) {
      size_t ep_count = pkg_ctx.GetVariantEpCompatibilityCount(comp_name.c_str(), var_name.c_str());
      for (size_t i = 0; i < ep_count; ++i) {
        const char* ep_name = nullptr;
        pkg_ctx.GetVariantEpCompatibility(comp_name.c_str(), var_name.c_str(), i,
                                          &ep_name, nullptr, nullptr);
        if (ep_name != nullptr) {
          comp_eps.insert(ep_name);
        }
      }
    }

    if (first_component) {
      intersection = std::move(comp_eps);
      first_component = false;
    } else {
      // Intersect
      std::unordered_set<std::string> new_intersection;
      for (const auto& ep : comp_eps) {
        if (intersection.count(ep)) {
          new_intersection.insert(ep);
        }
      }
      intersection = std::move(new_intersection);
    }
  }

  if (intersection.empty()) {
    std::ostringstream oss;
    oss << "No EP is compatible with all components in the model package. "
        << "Please specify an EP explicitly.";
    throw std::runtime_error(oss.str());
  }

  if (intersection.size() == 1) {
    return *intersection.begin();
  }

  // Multiple EPs are compatible. List them and ask the user to choose.
  std::ostringstream oss;
  oss << "Multiple EPs are compatible with all components: ";
  bool first = true;
  for (const auto& ep : intersection) {
    if (!first) oss << ", ";
    oss << ep;
    first = false;
  }
  oss << ". Please specify an EP explicitly.";
  throw std::runtime_error(oss.str());
}

// --- ModelPackageState ---

ModelPackageState::ModelPackageState(const fs::path& package_root, OrtEnv& env,
                                     const OrtSessionOptions& session_options,
                                     const std::string& resolved_ep_name)
    : package_root_(package_root),
      configs_path_(package_root / "configs"),
      resolved_ep_name_(resolved_ep_name) {
  pkg_ctx_ = OrtModelPackageContext::Create(package_root.c_str());
  pkg_opts_ = OrtModelPackageOptions::Create(env, session_options);
}

OrtModelPackageComponentContext* ModelPackageState::SelectComponent(const std::string& component_name) {
  auto it = component_contexts_.find(component_name);
  if (it != component_contexts_.end()) {
    return it->second.get();
  }

  auto cix = pkg_ctx_->SelectComponent(component_name.c_str(), *pkg_opts_);
  auto* raw = cix.get();
  component_contexts_[component_name] = std::move(cix);
  return raw;
}

OrtModelPackageComponentContext* ModelPackageState::GetComponent(const std::string& component_name) const {
  auto it = component_contexts_.find(component_name);
  return (it != component_contexts_.end()) ? it->second.get() : nullptr;
}

// GetGenAIConfigOverlay is defined after the JSON mini-DOM utilities below.

#endif

// --- RFC 7386 JSON Merge Patch ---
//
// Operates on raw JSON strings. Produces a merged JSON string.
// Uses a minimal DOM representation for the merge operation.

namespace {

// Minimal JSON value type for merge patch
struct JsonValue {
  enum Type { Null, String, Number, Bool, Array, Object };
  Type type = Null;
  std::string str_val;           // for String, Number (as string), Bool ("true"/"false")
  std::string raw;               // raw JSON representation (for arrays and simple values)
  std::vector<std::pair<std::string, JsonValue>> obj_members;  // for Object, preserves order

  bool is_null() const { return type == Null; }
  bool is_object() const { return type == Object; }
};

// Skip whitespace
size_t SkipWs(std::string_view s, size_t pos) {
  while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
    ++pos;
  return pos;
}

// Parse a JSON string (starting at the opening quote). Returns the string content and advances pos past the closing quote.
std::string ParseJsonString(std::string_view s, size_t& pos) {
  if (pos >= s.size() || s[pos] != '"')
    throw std::runtime_error("Expected '\"' in JSON");
  ++pos;
  std::string result;
  while (pos < s.size()) {
    char c = s[pos++];
    if (c == '\\') {
      if (pos >= s.size()) throw std::runtime_error("Unexpected end of JSON string");
      char esc = s[pos++];
      switch (esc) {
        case '"': result += '"'; break;
        case '\\': result += '\\'; break;
        case '/': result += '/'; break;
        case 'b': result += '\b'; break;
        case 'f': result += '\f'; break;
        case 'n': result += '\n'; break;
        case 'r': result += '\r'; break;
        case 't': result += '\t'; break;
        case 'u': {
          // 4 hex digits, just pass through
          if (pos + 4 > s.size()) throw std::runtime_error("Invalid unicode escape");
          result += "\\u";
          result += s.substr(pos, 4);
          pos += 4;
          break;
        }
        default: result += esc; break;
      }
    } else if (c == '"') {
      return result;
    } else {
      result += c;
    }
  }
  throw std::runtime_error("Unterminated JSON string");
}

// Capture a raw JSON value (string, number, bool, null, array, object) as a string and
// also parse it into a JsonValue. Advances pos past the value.
JsonValue ParseJsonValue(std::string_view s, size_t& pos);

JsonValue ParseJsonValue(std::string_view s, size_t& pos) {
  pos = SkipWs(s, pos);
  if (pos >= s.size()) throw std::runtime_error("Unexpected end of JSON");

  JsonValue val;
  char c = s[pos];

  if (c == '"') {
    size_t start = pos;
    val.str_val = ParseJsonString(s, pos);
    val.type = JsonValue::String;
    val.raw = std::string(s.substr(start, pos - start));
    return val;
  }

  if (c == '{') {
    val.type = JsonValue::Object;
    ++pos;
    pos = SkipWs(s, pos);
    if (pos < s.size() && s[pos] == '}') {
      ++pos;
      return val;
    }
    while (true) {
      pos = SkipWs(s, pos);
      std::string key = ParseJsonString(s, pos);
      pos = SkipWs(s, pos);
      if (pos >= s.size() || s[pos] != ':') throw std::runtime_error("Expected ':' in JSON object");
      ++pos;
      JsonValue member_val = ParseJsonValue(s, pos);
      val.obj_members.emplace_back(std::move(key), std::move(member_val));
      pos = SkipWs(s, pos);
      if (pos >= s.size()) throw std::runtime_error("Unterminated JSON object");
      if (s[pos] == '}') { ++pos; return val; }
      if (s[pos] == ',') { ++pos; continue; }
      throw std::runtime_error("Expected ',' or '}' in JSON object");
    }
  }

  if (c == '[') {
    // Capture the entire array as raw JSON (arrays replace wholesale in merge patch)
    size_t start = pos;
    int depth = 0;
    bool in_str = false;
    for (size_t i = pos; i < s.size(); ++i) {
      char ch = s[i];
      if (in_str) {
        if (ch == '\\') ++i;
        else if (ch == '"') in_str = false;
      } else {
        if (ch == '"') in_str = true;
        else if (ch == '[') ++depth;
        else if (ch == ']') {
          --depth;
          if (depth == 0) {
            val.type = JsonValue::Array;
            val.raw = std::string(s.substr(start, i - start + 1));
            pos = i + 1;
            return val;
          }
        }
      }
    }
    throw std::runtime_error("Unterminated JSON array");
  }

  if (c == 'n' && s.substr(pos, 4) == "null") {
    val.type = JsonValue::Null;
    val.raw = "null";
    pos += 4;
    return val;
  }
  if (c == 't' && s.substr(pos, 4) == "true") {
    val.type = JsonValue::Bool;
    val.raw = "true";
    pos += 4;
    return val;
  }
  if (c == 'f' && s.substr(pos, 5) == "false") {
    val.type = JsonValue::Bool;
    val.raw = "false";
    pos += 5;
    return val;
  }

  // Number
  size_t start = pos;
  if (c == '-') ++pos;
  while (pos < s.size() && ((s[pos] >= '0' && s[pos] <= '9') || s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+' || s[pos] == '-')) {
    // Avoid consuming '-' or '+' unless preceded by 'e'/'E'
    if ((s[pos] == '+' || s[pos] == '-') && pos > start && s[pos - 1] != 'e' && s[pos - 1] != 'E') break;
    ++pos;
  }
  if (pos == start || (pos == start + 1 && c == '-')) throw std::runtime_error("Invalid JSON number");
  val.type = JsonValue::Number;
  val.raw = std::string(s.substr(start, pos - start));
  return val;
}

// Serialize a JsonValue to a JSON string
std::string SerializeJson(const JsonValue& val) {
  switch (val.type) {
    case JsonValue::Null: return "null";
    case JsonValue::Bool:
    case JsonValue::Number:
    case JsonValue::Array:
      return val.raw;
    case JsonValue::String: {
      std::string result = "\"";
      for (char c : val.str_val) {
        switch (c) {
          case '"': result += "\\\""; break;
          case '\\': result += "\\\\"; break;
          case '\b': result += "\\b"; break;
          case '\f': result += "\\f"; break;
          case '\n': result += "\\n"; break;
          case '\r': result += "\\r"; break;
          case '\t': result += "\\t"; break;
          default: result += c; break;
        }
      }
      result += "\"";
      return result;
    }
    case JsonValue::Object: {
      std::string result = "{";
      bool first = true;
      for (const auto& [key, member] : val.obj_members) {
        if (!first) result += ",";
        result += "\"";
        result += key;
        result += "\":";
        result += SerializeJson(member);
        first = false;
      }
      result += "}";
      return result;
    }
  }
  return "null";
}

// RFC 7386: merge patch into target. Both must be parsed JsonValues.
// Returns the result.
JsonValue MergePatch(JsonValue target, const JsonValue& patch) {
  if (!patch.is_object()) {
    // If patch is not an object, it replaces the target entirely
    return patch;
  }

  if (!target.is_object()) {
    // If target is not an object, start with an empty object
    target = JsonValue{};
    target.type = JsonValue::Object;
  }

  for (const auto& [key, patch_val] : patch.obj_members) {
    if (patch_val.is_null()) {
      // Delete the key
      auto it = std::find_if(target.obj_members.begin(), target.obj_members.end(),
                             [&key](const auto& p) { return p.first == key; });
      if (it != target.obj_members.end()) {
        target.obj_members.erase(it);
      }
    } else {
      // Find or create the key
      auto it = std::find_if(target.obj_members.begin(), target.obj_members.end(),
                             [&key](const auto& p) { return p.first == key; });
      if (it != target.obj_members.end()) {
        it->second = MergePatch(std::move(it->second), patch_val);
      } else {
        target.obj_members.emplace_back(key, MergePatch(JsonValue{}, patch_val));
      }
    }
  }

  return target;
}

}  // anonymous namespace

#if ORT_HAS_MODEL_PACKAGE
std::string ModelPackageState::GetGenAIConfigOverlay(const std::string& component_name) const {
  auto* cix = GetComponent(component_name);
  if (!cix) {
    return {};
  }

  std::string consumer_metadata = cix->GetSelectedVariantConsumerMetadata();
  if (consumer_metadata.empty()) {
    return {};
  }

  // Parse consumer_metadata as a JSON object and extract the "genai_config_overlay" value.
  // Malformed metadata is a producer error: throw rather than silently ignoring.
  size_t pos = 0;
  JsonValue root;
  try {
    root = ParseJsonValue(consumer_metadata, pos);
  } catch (const std::exception& e) {
    throw std::runtime_error(MakeString("Component '", component_name,
                                        "': consumer_metadata is not valid JSON: ", e.what()));
  }

  if (!root.is_object()) {
    throw std::runtime_error(MakeString("Component '", component_name,
                                        "': consumer_metadata must be a JSON object"));
  }

  for (const auto& [key, value] : root.obj_members) {
    if (key == "genai_config_overlay") {
      if (!value.is_object()) {
        throw std::runtime_error(MakeString("Component '", component_name,
                                            "': genai_config_overlay must be a JSON object"));
      }
      return SerializeJson(value);
    }
  }

  return {};
}

fs::path ModelPackageState::GetVariantDir(const std::string& component_name) const {
  auto* cix = GetComponent(component_name);
  if (!cix) {
    return {};
  }
  auto folder = cix->GetSelectedVariantFolderPath();
  return fs::path(folder);
}

std::unordered_map<std::string, size_t> ModelPackageState::BuildFileIndexMap(
    const std::string& component_name) const {
  auto* cix = GetComponent(component_name);
  if (!cix) {
    throw std::runtime_error("Component '" + component_name +
                             "' was not selected from the package");
  }

  size_t file_count = cix->GetSelectedVariantFileCount();
  std::unordered_map<std::string, size_t> file_map;
  file_map.reserve(file_count);

  for (size_t i = 0; i < file_count; ++i) {
    auto path_str = cix->GetSelectedVariantFilePath(i);
    std::string basename(path_str);
    auto last_sep = basename.find_last_of("/\\");
    if (last_sep != std::string::npos) {
      basename = basename.substr(last_sep + 1);
    }
    if (!file_map.emplace(basename, i).second) {
      throw std::runtime_error(
          "Package variant for component '" + component_name +
          "' has duplicate file basename '" + basename + "'");
    }
  }
  return file_map;
}

size_t ModelPackageState::ResolveFileIndex(const std::string& component_name,
                                           const std::string& filename) const {
  auto* cix = GetComponent(component_name);
  if (!cix) {
    throw std::runtime_error("Component '" + component_name +
                             "' was not selected from the package");
  }

  size_t file_count = cix->GetSelectedVariantFileCount();
  for (size_t i = 0; i < file_count; ++i) {
    auto path_str = cix->GetSelectedVariantFilePath(i);
    std::string basename(path_str);
    auto last_sep = basename.find_last_of("/\\");
    if (last_sep != std::string::npos) {
      basename = basename.substr(last_sep + 1);
    }
    if (basename == filename) {
      return i;
    }
  }
  throw std::runtime_error("File '" + filename + "' not found in package variant for component '" +
                           component_name + "'");
}

// --- Package session option helpers ---

namespace {

bool ParseBoolValue(std::string_view key, std::string_view value) {
  if (value == "true" || value == "1") return true;
  if (value == "false" || value == "0") return false;
  throw std::runtime_error(
      "variant.json: session_options[\"" + std::string(key) +
      "\"] must be a boolean (got '" + std::string(value) + "')");
}

int ParseIntValue(std::string_view key, std::string_view value) {
  try {
    size_t consumed = 0;
    int parsed = std::stoi(std::string(value), &consumed);
    if (consumed != value.size()) throw std::invalid_argument("trailing characters");
    return parsed;
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "variant.json: session_options[\"" + std::string(key) +
        "\"] must be an integer (got '" + std::string(value) + "'): " + e.what());
  }
}

GraphOptimizationLevel ParseGraphOptLevel(std::string_view value) {
  if (value == "ORT_DISABLE_ALL") return ORT_DISABLE_ALL;
  if (value == "ORT_ENABLE_BASIC") return ORT_ENABLE_BASIC;
  if (value == "ORT_ENABLE_EXTENDED") return ORT_ENABLE_EXTENDED;
  if (value == "ORT_ENABLE_ALL") return ORT_ENABLE_ALL;
  throw std::runtime_error(
      "variant.json: session_options[\"graph_optimization_level\"] has unrecognized value '" +
      std::string(value) + "'");
}

}  // namespace

void InjectPackageEp(Config::SessionOptions& session_options, const std::string& resolved_ep) {
  if (resolved_ep.empty() || resolved_ep == "CPUExecutionProvider") return;

  std::string tag = EpNameToGenAIProviderName(resolved_ep);
  if (tag == resolved_ep) return;  // unrecognized EP, no-op

  auto& providers = session_options.providers;
  auto it = std::find(providers.begin(), providers.end(), tag);
  if (it == providers.end()) {
    providers.insert(providers.begin(), tag);
  } else if (it != providers.begin()) {
    // Already present but not first; rotate to position 0
    std::rotate(providers.begin(), it, std::next(it));
  }

  // Ensure a matching ProviderOptions entry exists
  auto& po_list = session_options.provider_options;
  auto po_it = std::find_if(po_list.begin(), po_list.end(),
                            [&](const Config::ProviderOptions& po) { return po.name == tag; });
  if (po_it == po_list.end()) {
    Config::ProviderOptions po;
    po.name = tag;
    po_list.push_back(std::move(po));
  }
}

void ApplyVariantFileDefaults(Config::SessionOptions& so,
                              OrtModelPackageComponentContext* cix,
                              size_t file_index,
                              const std::string& resolved_ep) {
  // Read per-file session_options from ORT package API
  const char* const* so_keys = nullptr;
  const char* const* so_values = nullptr;
  size_t so_count = 0;
  cix->GetSelectedVariantFileSessionOptions(file_index, &so_keys, &so_values, &so_count);

  // Apply as layer-1 defaults: only fill unset fields
  for (size_t i = 0; i < so_count; ++i) {
    std::string_view key(so_keys[i]);
    std::string_view val(so_values[i]);

    if (key == "intra_op_num_threads") {
      if (!so.intra_op_num_threads.has_value()) so.intra_op_num_threads = ParseIntValue(key, val);
    } else if (key == "inter_op_num_threads") {
      if (!so.inter_op_num_threads.has_value()) so.inter_op_num_threads = ParseIntValue(key, val);
    } else if (key == "log_severity_level") {
      if (!so.log_severity_level.has_value()) so.log_severity_level = ParseIntValue(key, val);
    } else if (key == "log_verbosity_level") {
      if (!so.log_verbosity_level.has_value()) so.log_verbosity_level = ParseIntValue(key, val);
    } else if (key == "enable_cpu_mem_arena") {
      if (!so.enable_cpu_mem_arena.has_value()) so.enable_cpu_mem_arena = ParseBoolValue(key, val);
    } else if (key == "enable_mem_pattern") {
      if (!so.enable_mem_pattern.has_value()) so.enable_mem_pattern = ParseBoolValue(key, val);
    } else if (key == "log_id") {
      if (!so.log_id.has_value()) so.log_id = std::string(val);
    } else if (key == "enable_profiling") {
      if (!so.enable_profiling.has_value()) so.enable_profiling = std::string(val);
    } else if (key == "custom_ops_library") {
      if (!so.custom_ops_library.has_value()) so.custom_ops_library = std::string(val);
    } else if (key == "graph_optimization_level") {
      if (!so.graph_optimization_level.has_value()) so.graph_optimization_level = ParseGraphOptLevel(val);
    } else {
      // Unknown keys go to config_entries if not already present
      bool exists = std::any_of(so.config_entries.begin(), so.config_entries.end(),
                                [&](const auto& e) { return e.first == key; });
      if (!exists) {
        so.config_entries.emplace_back(std::string(key), std::string(val));
      }
    }
  }

  // Read per-file provider_options and merge into matching ProviderOptions entry
  if (resolved_ep.empty() || resolved_ep == "CPUExecutionProvider") return;

  const char* const* po_keys = nullptr;
  const char* const* po_values = nullptr;
  size_t po_count = 0;
  cix->GetSelectedVariantFileProviderOptions(file_index, &po_keys, &po_values, &po_count);
  if (po_count == 0) return;

  std::string tag = EpNameToGenAIProviderName(resolved_ep);
  auto& po_list = so.provider_options;
  auto po_it = std::find_if(po_list.begin(), po_list.end(),
                            [&](const Config::ProviderOptions& po) { return po.name == tag; });
  if (po_it == po_list.end()) return;  // no matching entry to merge into

  for (size_t i = 0; i < po_count; ++i) {
    bool exists = std::any_of(po_it->options.begin(), po_it->options.end(),
                              [&](const auto& e) { return e.first == po_keys[i]; });
    if (!exists) {
      po_it->options.emplace_back(std::string(po_keys[i]), std::string(po_values[i]));
    }
  }
}

#endif

std::string JsonMergePatch(std::string_view base_json, std::string_view patch_json) {
  if (patch_json.empty()) {
    return std::string(base_json);
  }
  if (base_json.empty()) {
    return std::string(patch_json);
  }

  size_t pos = 0;
  JsonValue base = ParseJsonValue(base_json, pos);
  pos = 0;
  JsonValue patch = ParseJsonValue(patch_json, pos);

  JsonValue result = MergePatch(std::move(base), patch);
  return SerializeJson(result);
}

// --- EP name normalization and mapping ---

std::string NormalizeEpName(const std::string& ep_name) {
  // Accept short aliases (case-insensitive) and map to canonical ORT EP names.
  std::string lower;
  lower.reserve(ep_name.size());
  for (char c : ep_name) lower.push_back(static_cast<char>(std::tolower(c)));

  static const std::unordered_map<std::string, std::string> alias_map = {
      {"cuda", "CUDAExecutionProvider"},
      {"cudaexecutionprovider", "CUDAExecutionProvider"},
      {"cpu", "CPUExecutionProvider"},
      {"cpuexecutionprovider", "CPUExecutionProvider"},
      {"dml", "DmlExecutionProvider"},
      {"dmlexecutionprovider", "DmlExecutionProvider"},
      {"qnn", "QNNExecutionProvider"},
      {"qnnexecutionprovider", "QNNExecutionProvider"},
      {"openvino", "OpenVINOExecutionProvider"},
      {"openvinoexecutionprovider", "OpenVINOExecutionProvider"},
      {"webgpu", "WebGpuExecutionProvider"},
      {"webgpuexecutionprovider", "WebGpuExecutionProvider"},
      {"nvtensorrtrtx", "NvTensorRtRtxExecutionProvider"},
      {"nvtensorrtrtxexecutionprovider", "NvTensorRtRtxExecutionProvider"},
      {"vitisai", "VitisAIExecutionProvider"},
      {"vitisaiexecutionprovider", "VitisAIExecutionProvider"},
      {"ryzenai", "RyzenAIExecutionProvider"},
      {"ryzenaiexecutionprovider", "RyzenAIExecutionProvider"},
  };

  auto it = alias_map.find(lower);
  if (it != alias_map.end()) {
    return it->second;
  }
  return ep_name;
}

DeviceInterface* DeviceFromEpName(const std::string& ep_name) {
  static const std::unordered_map<std::string, DeviceType> ep_device_map = {
      {"CUDAExecutionProvider", DeviceType::CUDA},
      {"DmlExecutionProvider", DeviceType::DML},
      {"QNNExecutionProvider", DeviceType::QNN},
      {"NvTensorRtRtxExecutionProvider", DeviceType::NvTensorRtRtx},
      {"WebGpuExecutionProvider", DeviceType::WEBGPU},
      {"RyzenAIExecutionProvider", DeviceType::RyzenAI},
      {"CPUExecutionProvider", DeviceType::CPU},
  };

  auto it = ep_device_map.find(ep_name);
  if (it != ep_device_map.end()) {
    return GetDeviceInterface(it->second);
  }
  return GetDeviceInterface(DeviceType::CPU);
}

std::string EpNameToGenAIProviderName(const std::string& ep_name) {
  static const std::unordered_map<std::string, std::string> ep_to_genai = {
      {"CUDAExecutionProvider", "cuda"},
      {"DmlExecutionProvider", "DML"},
      {"QNNExecutionProvider", "QNN"},
      {"NvTensorRtRtxExecutionProvider", "NvTensorRtRtx"},
      {"WebGpuExecutionProvider", "WebGPU"},
      {"RyzenAIExecutionProvider", "RyzenAI"},
      {"OpenVINOExecutionProvider", "OpenVINO"},
      {"VitisAIExecutionProvider", "VitisAI"},
  };

  auto it = ep_to_genai.find(ep_name);
  if (it != ep_to_genai.end()) {
    return it->second;
  }
  return ep_name;
}

}  // namespace Generators
