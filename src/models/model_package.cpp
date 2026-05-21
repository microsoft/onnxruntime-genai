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

  // Parse consumer_metadata as a JSON object and extract the kGenAIConfigOverlayKey value.
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
    if (key == kGenAIConfigOverlayKey) {
      // null means "no overlay" (not an error)
      if (value.is_null()) {
        return {};
      }
      if (!value.is_object()) {
        throw std::runtime_error(MakeString("Component '", component_name,
                                            "': ", kGenAIConfigOverlayKey,
                                            " must be a JSON object or null"));
      }
      return SerializeJson(value);
    }
  }

  return {};
}

// --- Helpers extracted from Model::BuildSessionOptionsForPackageFile ---

namespace {

bool ParseVariantBool(std::string_view key, std::string_view value) {
  if (value == "true" || value == "1") return true;
  if (value == "false" || value == "0") return false;
  throw std::runtime_error(
      "variant session_options[\"" + std::string(key) +
      "\"] must be a boolean (got '" + std::string(value) + "')");
}

int ParseVariantInt(std::string_view key, std::string_view value) {
  try {
    size_t consumed = 0;
    int parsed = std::stoi(std::string(value), &consumed);
    if (consumed != value.size()) throw std::invalid_argument("trailing characters");
    return parsed;
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "variant session_options[\"" + std::string(key) +
        "\"] must be an integer (got '" + std::string(value) + "'): " + e.what());
  }
}

GraphOptimizationLevel ParseVariantGraphOptLevel(std::string_view value) {
  if (value == "ORT_DISABLE_ALL") return ORT_DISABLE_ALL;
  if (value == "ORT_ENABLE_BASIC") return ORT_ENABLE_BASIC;
  if (value == "ORT_ENABLE_EXTENDED") return ORT_ENABLE_EXTENDED;
  if (value == "ORT_ENABLE_ALL") return ORT_ENABLE_ALL;
  throw std::runtime_error(
      "variant session_options[\"graph_optimization_level\"] has unrecognized value '" +
      std::string(value) + "'");
}

// Apply a variant file's per-file session_options and resolved-EP provider_options into
// `target` as layered defaults: target's existing values win on conflicts, the variant fills
// gaps. Used by NormalizePackageIntoConfig to merge variant data into the genai_config role
// SO without losing either side's information.
//
// Session-options handling:
//   - Typed fields (intra_op_num_threads, enable_cpu_mem_arena, graph_optimization_level, ...)
//     are filled only when target's std::optional is empty.
//   - Other keys go into config_entries only when no entry with the same key already exists.
//
// Provider-options handling:
//   - For non-CPU/non-empty ep_for_file, the resolved EP becomes a provider_options entry
//     keyed by its GenAI tag (e.g. "cuda"). If target already has a same-named entry, the
//     variant's options back-fill missing keys (target's existing keys win). Otherwise the
//     entry is appended verbatim.
//   - CPU is implicit in ORT (no provider tag), so CPU/empty ep_for_file skips the provider
//     entry. A variant that declares non-empty provider_options under a CPU-only file is a
//     producer error and throws.
void ApplyVariantFileOptions(Config::SessionOptions& target,
                             OrtModelPackageComponentContext& cix,
                             size_t file_index,
                             const std::string& ep_for_file) {
  // Variant per-file session_options as defaults.
  const char* const* so_keys = nullptr;
  const char* const* so_values = nullptr;
  size_t so_count = 0;
  cix.GetSelectedVariantFileSessionOptions(file_index, &so_keys, &so_values, &so_count);

  for (size_t i = 0; i < so_count; ++i) {
    std::string_view key(so_keys[i]);
    std::string_view val(so_values[i]);

    if (key == "intra_op_num_threads") {
      if (!target.intra_op_num_threads.has_value()) target.intra_op_num_threads = ParseVariantInt(key, val);
    } else if (key == "inter_op_num_threads") {
      if (!target.inter_op_num_threads.has_value()) target.inter_op_num_threads = ParseVariantInt(key, val);
    } else if (key == "log_severity_level") {
      if (!target.log_severity_level.has_value()) target.log_severity_level = ParseVariantInt(key, val);
    } else if (key == "log_verbosity_level") {
      if (!target.log_verbosity_level.has_value()) target.log_verbosity_level = ParseVariantInt(key, val);
    } else if (key == "enable_cpu_mem_arena") {
      if (!target.enable_cpu_mem_arena.has_value()) target.enable_cpu_mem_arena = ParseVariantBool(key, val);
    } else if (key == "enable_mem_pattern") {
      if (!target.enable_mem_pattern.has_value()) target.enable_mem_pattern = ParseVariantBool(key, val);
    } else if (key == "log_id") {
      if (!target.log_id.has_value()) target.log_id = std::string(val);
    } else if (key == "enable_profiling") {
      if (!target.enable_profiling.has_value()) target.enable_profiling = std::string(val);
    } else if (key == "custom_ops_library") {
      if (!target.custom_ops_library.has_value()) target.custom_ops_library = std::string(val);
    } else if (key == "graph_optimization_level") {
      if (!target.graph_optimization_level.has_value()) target.graph_optimization_level = ParseVariantGraphOptLevel(val);
    } else {
      bool exists = std::any_of(target.config_entries.begin(), target.config_entries.end(),
                                [&](const auto& e) { return e.first == key; });
      if (!exists) target.config_entries.emplace_back(std::string(key), std::string(val));
    }
  }

  // Variant per-file provider_options.
  const char* const* po_keys = nullptr;
  const char* const* po_values = nullptr;
  size_t po_count = 0;
  cix.GetSelectedVariantFileProviderOptions(file_index, &po_keys, &po_values, &po_count);

  const bool ep_is_cpu = ep_for_file.empty() || ep_for_file == "CPUExecutionProvider";
  if (ep_is_cpu) {
    if (po_count != 0) {
      throw std::runtime_error(
          "variant declares provider_options under a CPU-only file; CPU has no provider tag "
          "in GenAI's dispatch so these options would be silently dropped");
    }
    return;
  }

  std::string genai_provider_name = EpNameToGenAIProviderName(ep_for_file);
  auto it = std::find_if(target.provider_options.begin(), target.provider_options.end(),
                         [&](const Config::ProviderOptions& p) { return p.name == genai_provider_name; });
  if (it == target.provider_options.end()) {
    // No matching entry in genai_config: append the variant's entry verbatim.
    Config::ProviderOptions po;
    po.name = genai_provider_name;
    for (size_t i = 0; i < po_count; ++i) {
      po.options.emplace_back(std::string(po_keys[i]), std::string(po_values[i]));
    }
    target.provider_options.push_back(std::move(po));
  } else {
    // Back-fill: target's existing keys win; variant supplies anything else.
    for (size_t i = 0; i < po_count; ++i) {
      std::string_view k(po_keys[i]);
      bool exists = std::any_of(it->options.begin(), it->options.end(),
                                [&](const auto& e) { return e.first == k; });
      if (!exists) it->options.emplace_back(std::string(k), std::string(po_values[i]));
    }
  }
}

// Rebuild providers list from provider_options. Keeps the two in sync after we materialize
// variant data without re-running the heavier FinalizeConfig (which would also re-append
// duplicates).
void RebuildProvidersFromProviderOptions(Config::SessionOptions& so) {
  so.providers.clear();
  so.providers.reserve(so.provider_options.size());
  for (const auto& po : so.provider_options) {
    so.providers.push_back(po.name);
  }
}

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
  if (Ort::runtime_api_version < 27) {
    throw std::runtime_error("Model packages require ONNX Runtime API version 27 or newer at runtime");
  }

  auto pkg_ctx = OrtModelPackageContext::Create(package_root.c_str());

  if (!explicit_ep.empty()) {
    out_resolved_ep = NormalizeEpName(explicit_ep);
  } else {
    // Parse the base config to scope EP auto-detect to the components the model actually uses.
    extern void ParseConfigFromString(std::string_view json, Config& config);
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
  const std::string& resolved_ep = pkg_state.GetResolvedEpName();

  auto basename_of = [](std::string_view path) {
    auto sep = path.find_last_of("/\\");
    return std::string(sep == std::string_view::npos ? path : path.substr(sep + 1));
  };

  // Reject components introduced by overlays. The base-config component set was used to drive
  // SelectComponent calls; anything that surfaces in the merged Config but wasn't in the base
  // is a contract violation (overlays may not introduce new components).
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

  // Normalize an optional-SO role. session_options is treated as the genai_config-derived
  // receiver; variant data fills gaps and back-fills missing EP provider_options keys.
  auto normalize_single_optional = [&](const std::string& component,
                                       std::string& filename_slot,
                                       fs::path& asset_dir_slot,
                                       std::optional<Config::SessionOptions>& so_slot) {
    auto* cix = require_selected(component);
    if (!cix) return;
    if (cix->GetSelectedVariantFileCount() == 0) {
      throw std::runtime_error("Component '" + component +
                               "' has no files in the selected variant");
    }
    filename_slot = basename_of(cix->GetSelectedVariantFilePath(0));
    asset_dir_slot = fs::path(cix->GetSelectedVariantFolderPath());

    Config::SessionOptions merged = so_slot.has_value() ? std::move(*so_slot) : Config::SessionOptions{};
    ApplyVariantFileOptions(merged, *cix, 0, resolved_ep);
    RebuildProvidersFromProviderOptions(merged);
    so_slot = std::move(merged);
  };

  // Decoder: SessionOptions is non-optional and may carry a pipeline.
  auto& dec = config.model.decoder;
  if (!dec.component.empty()) {
    auto* cix = require_selected(dec.component);
    auto file_count = cix->GetSelectedVariantFileCount();

    if (dec.pipeline.empty()) {
      // Single-file decoder.
      if (file_count == 0) {
        throw std::runtime_error("Decoder component '" + dec.component +
                                 "' has no files in the selected variant");
      }
      dec.filename = basename_of(cix->GetSelectedVariantFilePath(0));
      dec.asset_dir = fs::path(cix->GetSelectedVariantFolderPath());

      ApplyVariantFileOptions(dec.session_options, *cix, 0, resolved_ep);
      RebuildProvidersFromProviderOptions(dec.session_options);
    } else {
      // Pipeline: each element maps positionally to variant.files[i].
      if (dec.pipeline.size() != file_count) {
        throw std::runtime_error(
            "Decoder pipeline has " + std::to_string(dec.pipeline.size()) +
            " stages but selected variant for component '" + dec.component + "' has " +
            std::to_string(file_count) +
            " files; positional mapping requires equal counts");
      }
      dec.asset_dir = fs::path(cix->GetSelectedVariantFolderPath());
      // Decoder-level SessionOptions in pipeline mode is consulted by
      // Model::CreateSessionOptions() to derive p_device_. Reset to a clean SO carrying just
      // the resolved EP entry — per-file detail belongs on each pipeline stage's own SO.
      dec.session_options = Config::SessionOptions{};
      ApplyVariantFileOptions(dec.session_options, *cix, 0, resolved_ep);
      RebuildProvidersFromProviderOptions(dec.session_options);

      for (size_t i = 0; i < dec.pipeline.size(); ++i) {
        auto& pipe = dec.pipeline[i];
        pipe.filename = basename_of(cix->GetSelectedVariantFilePath(i));
        const std::string& ep = pipe.run_on_cpu ? std::string{} : resolved_ep;

        Config::SessionOptions merged = pipe.session_options.has_value()
                                            ? std::move(*pipe.session_options)
                                            : Config::SessionOptions{};
        ApplyVariantFileOptions(merged, *cix, i, ep);
        RebuildProvidersFromProviderOptions(merged);
        pipe.session_options = std::move(merged);
      }
    }
  }

  normalize_single_optional(config.model.encoder.component, config.model.encoder.filename,
                            config.model.encoder.asset_dir, config.model.encoder.session_options);
  normalize_single_optional(config.model.vision.component, config.model.vision.filename,
                            config.model.vision.asset_dir, config.model.vision.session_options);
  normalize_single_optional(config.model.speech.component, config.model.speech.filename,
                            config.model.speech.asset_dir, config.model.speech.session_options);
  normalize_single_optional(config.model.embedding.component, config.model.embedding.filename,
                            config.model.embedding.asset_dir, config.model.embedding.session_options);
  normalize_single_optional(config.model.joiner.component, config.model.joiner.filename,
                            config.model.joiner.asset_dir, config.model.joiner.session_options);
  normalize_single_optional(config.model.vad.component, config.model.vad.filename,
                            config.model.vad.asset_dir, config.model.vad.session_options);
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
