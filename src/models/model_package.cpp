// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "model_package.h"
#include "onnxruntime_api.h"
#include "../json.h"

namespace Generators {

// --- Package detection ---

bool IsModelPackage(const fs::path& path) {
  // A model package is identified by a top-level manifest.json carrying schema_version,
  // layout, and the components map.
  return fs::exists(path / "manifest.json");
}

// --- EP defaulting ---

#if ORT_HAS_MODEL_PACKAGE
std::string DefaultEpFromPackage(const OrtModelPackageContext& pkg_ctx,
                                 const std::vector<std::string>& scoped_components) {
  if (scoped_components.empty()) {
    throw std::runtime_error(
        "Cannot auto-detect EP for model package: the genai config does not reference any "
        "package components (each role must set a \"component\" field). Specify an EP explicitly.");
  }

  // For each component, collect the union of EP names across all its variants.
  // Then intersect across components.
  std::unordered_set<std::string> intersection;
  bool first_component = true;

  for (const auto& comp_name : scoped_components) {
    std::unordered_set<std::string> comp_eps;
    auto variant_names = pkg_ctx.GetVariantNames(comp_name.c_str());

    for (const auto& var_name : variant_names) {
      auto ep_name = pkg_ctx.GetVariantEpName(comp_name.c_str(), var_name.c_str());
      if (ep_name.has_value()) {
        comp_eps.insert(*ep_name);
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
  enum Type { Null,
              String,
              Number,
              Bool,
              Array,
              Object };
  Type type = Null;
  std::string str_val;                                         // for String, Number (as string), Bool ("true"/"false")
  std::string raw;                                             // raw JSON representation (for arrays and simple values)
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
        case '"':
          result += '"';
          break;
        case '\\':
          result += '\\';
          break;
        case '/':
          result += '/';
          break;
        case 'b':
          result += '\b';
          break;
        case 'f':
          result += '\f';
          break;
        case 'n':
          result += '\n';
          break;
        case 'r':
          result += '\r';
          break;
        case 't':
          result += '\t';
          break;
        case 'u': {
          if (pos + 4 > s.size()) throw std::runtime_error("Invalid unicode escape");
          auto parse_hex4 = [&](size_t p) -> uint32_t {
            uint32_t v = 0;
            for (int i = 0; i < 4; ++i) {
              char ch = s[p + i];
              uint32_t d;
              if (ch >= '0' && ch <= '9')
                d = static_cast<uint32_t>(ch - '0');
              else if (ch >= 'a' && ch <= 'f')
                d = static_cast<uint32_t>(ch - 'a' + 10);
              else if (ch >= 'A' && ch <= 'F')
                d = static_cast<uint32_t>(ch - 'A' + 10);
              else
                throw std::runtime_error("Invalid hex digit in \\u escape");
              v = (v << 4) | d;
            }
            return v;
          };
          uint32_t cp = parse_hex4(pos);
          pos += 4;
          // Handle UTF-16 surrogate pair: a high surrogate must be followed by \uXXXX with a low surrogate.
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (pos + 6 > s.size() || s[pos] != '\\' || s[pos + 1] != 'u')
              throw std::runtime_error("Expected low surrogate after high surrogate in \\u escape");
            uint32_t lo = parse_hex4(pos + 2);
            if (lo < 0xDC00 || lo > 0xDFFF)
              throw std::runtime_error("Invalid low surrogate in \\u escape");
            cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
            pos += 6;
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            throw std::runtime_error("Unexpected low surrogate in \\u escape");
          }
          // Encode the codepoint as UTF-8 so SerializeJson can pass it through unchanged.
          if (cp < 0x80) {
            result += static_cast<char>(cp);
          } else if (cp < 0x800) {
            result += static_cast<char>(0xC0 | (cp >> 6));
            result += static_cast<char>(0x80 | (cp & 0x3F));
          } else if (cp < 0x10000) {
            result += static_cast<char>(0xE0 | (cp >> 12));
            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (cp & 0x3F));
          } else {
            result += static_cast<char>(0xF0 | (cp >> 18));
            result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (cp & 0x3F));
          }
          break;
        }
        default:
          result += esc;
          break;
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
      if (s[pos] == '}') {
        ++pos;
        return val;
      }
      if (s[pos] == ',') {
        ++pos;
        continue;
      }
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
        if (ch == '\\')
          ++i;
        else if (ch == '"')
          in_str = false;
      } else {
        if (ch == '"')
          in_str = true;
        else if (ch == '[')
          ++depth;
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
    case JsonValue::Null:
      return "null";
    case JsonValue::Bool:
    case JsonValue::Number:
    case JsonValue::Array:
      return val.raw;
    case JsonValue::String: {
      std::string result = "\"";
      for (unsigned char c : val.str_val) {
        switch (c) {
          case '"':
            result += "\\\"";
            break;
          case '\\':
            result += "\\\\";
            break;
          case '\b':
            result += "\\b";
            break;
          case '\f':
            result += "\\f";
            break;
          case '\n':
            result += "\\n";
            break;
          case '\r':
            result += "\\r";
            break;
          case '\t':
            result += "\\t";
            break;
          default:
            if (c < 0x20) {
              // Other control characters must be escaped as \u00XX to produce valid JSON.
              static constexpr char kHex[] = "0123456789abcdef";
              result += "\\u00";
              result += kHex[(c >> 4) & 0xF];
              result += kHex[c & 0xF];
            } else {
              // UTF-8 bytes (>= 0x80) and printable ASCII pass through unchanged.
              result += static_cast<char>(c);
            }
            break;
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

  // The overlay lives at <variant_dir>/<kVariantOverlayFilename>. ORT itself doesn't read
  // this file — it's a GenAI consumer convention layered on top of the model_package schema.
  // Use std::filesystem here (not the in-tree fs::path wrapper) so we can pass through the
  // ORTCHAR_T-typed path returned by the ORT package API uniformly on Windows and POSIX.
  std::filesystem::path variant_dir(cix->GetSelectedVariantFolderPath());
  std::filesystem::path overlay_path = variant_dir / std::filesystem::path(kVariantOverlayFilename);

  std::error_code ec;
  if (!std::filesystem::exists(overlay_path, ec) || ec) {
    return {};
  }

  std::ifstream f(overlay_path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error(MakeString("Component '", component_name,
                                        "': failed to open variant overlay file"));
  }
  std::ostringstream buf;
  buf << f.rdbuf();
  std::string contents = buf.str();

  // Light validation: the overlay must be a JSON object (RFC 7386 merge patch). Skip BOM and
  // leading whitespace, then require a '{'. JsonMergePatch will perform full parsing later.
  size_t i = 0;
  if (contents.size() >= 3 &&
      static_cast<unsigned char>(contents[0]) == 0xEF &&
      static_cast<unsigned char>(contents[1]) == 0xBB &&
      static_cast<unsigned char>(contents[2]) == 0xBF) {
    i = 3;
  }
  while (i < contents.size() && std::isspace(static_cast<unsigned char>(contents[i]))) ++i;
  if (i >= contents.size()) {
    return {};
  }
  if (contents[i] != '{') {
    throw std::runtime_error(MakeString("Component '", component_name,
                                        "': ", kVariantOverlayFilename,
                                        " must contain a JSON object"));
  }

  return contents;
}

namespace {

// Collect every component name referenced by the Config's role fields, in declaration order.
std::vector<std::string> ReferencedComponents(const Config& config) {
  std::vector<std::string> components;
  auto add = [&](const std::string& comp) {
    if (!comp.empty() &&
        std::find(components.begin(), components.end(), comp) == components.end()) {
      components.push_back(comp);
    }
  };
  add(config.model.decoder.component);
  add(config.model.encoder.component);
  add(config.model.vision.component);
  add(config.model.speech.component);
  add(config.model.embedding.component);
  add(config.model.joiner.component);
  add(config.model.vad.component);
  return components;
}

}  // namespace

std::shared_ptr<ModelPackageState> OpenAndPrepareModelPackage(
    OrtEnv& env,
    const fs::path& package_root,
    std::string_view base_json,
    const std::string& explicit_ep,
    std::string& out_resolved_ep) {
  if (Ort::runtime_api_version < 28) {
    throw std::runtime_error("Model packages require ONNX Runtime API version 28 or newer at runtime");
  }

  auto pkg_ctx = OrtModelPackageContext::Create(package_root.c_str());

  if (!explicit_ep.empty()) {
    out_resolved_ep = NormalizeEpName(explicit_ep);
  } else {
    // Parse the base config to scope EP auto-detect to the components the model actually uses.
    auto pre_config = std::make_unique<Config>();
    ParseConfigFromString(base_json, *pre_config);
    out_resolved_ep = DefaultEpFromPackage(*pkg_ctx, ReferencedComponents(*pre_config));
  }

  // Build a selection-time OrtSessionOptions with the resolved EP appended. CUDA needs the
  // dedicated V2 path; everything else goes through the generic AppendExecutionProvider.
  // This SO is consumed by CreateModelPackageOptionsFromSessionOptions which snapshots the EP
  // list, so we can let it go out of scope after constructing the state.
  auto temp_so = OrtSessionOptions::Create();
  if (out_resolved_ep != "CPUExecutionProvider") {
    if (out_resolved_ep == "CUDAExecutionProvider") {
      auto cuda_opts = OrtCUDAProviderOptionsV2::Create();
      temp_so->AppendExecutionProvider_CUDA_V2(*cuda_opts);
    } else {
      temp_so->AppendExecutionProvider(out_resolved_ep.c_str(), nullptr, nullptr, 0);
    }
  }

  return std::make_shared<ModelPackageState>(package_root, env, *temp_so, out_resolved_ep);
}

void NormalizePackageIntoConfig(Config& config, ModelPackageState& pkg_state) {
  // For each role that names a package component, record the selected variant's directory on
  // the role's asset_dir. The rest of the codebase (Model::CreateSession, the custom_ops_library
  // resolver, the LoRA loader) uses asset_dir as the primary search root when resolving any
  // relative paths supplied by genai_config.json (and, transitively, by the variant overlay).
  //
  // The variant filename, session_options, provider_options, custom_ops_library, and
  // adapter_filename all flow through the genai_config.json + per-variant
  // genai_config_overlay.json pipeline. We no longer pull those fields from ORT.
  //
  // We also reject any component name that surfaces in the merged Config but wasn't present
  // in the base genai_config.json — overlays (variant or runtime) may not introduce new
  // component references.
  auto ort_path_to_string = [](const std::basic_string<ORTCHAR_T>& s) -> std::string {
    if (s.empty()) return {};
    std::filesystem::path p(s);
    // u8string() returns UTF-8 in both C++17 and C++20.
    auto u8 = p.u8string();
    return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
  };

  auto require_selected = [&](const std::string& component) -> OrtModelPackageComponentContext* {
    if (component.empty()) return nullptr;
    auto* cix = pkg_state.GetComponent(component);
    if (!cix) {
      throw std::runtime_error(
          "Model package: component '" + component +
          "' is referenced in the final config but was not present in the base genai_config.json. "
          "Overlays (variant or runtime) may not introduce new component references.");
    }
    return cix;
  };

  auto bind_asset_dir = [&](const std::string& component, fs::path& asset_dir) {
    auto* cix = require_selected(component);
    if (!cix) return;
    asset_dir = fs::path(ort_path_to_string(cix->GetSelectedVariantFolderPath()));
  };

  bind_asset_dir(config.model.decoder.component, config.model.decoder.asset_dir);
  bind_asset_dir(config.model.encoder.component, config.model.encoder.asset_dir);
  bind_asset_dir(config.model.vision.component, config.model.vision.asset_dir);
  bind_asset_dir(config.model.speech.component, config.model.speech.asset_dir);
  bind_asset_dir(config.model.embedding.component, config.model.embedding.asset_dir);
  bind_asset_dir(config.model.joiner.component, config.model.joiner.asset_dir);
  bind_asset_dir(config.model.vad.component, config.model.vad.asset_dir);
}

#endif  // ORT_HAS_MODEL_PACKAGE

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
  for (char c : ep_name)
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

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
