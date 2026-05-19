// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "model_package.h"
#include "onnxruntime_api.h"
#include "../json.h"

namespace Generators {

// --- Package detection ---

bool IsModelPackage(const fs::path& path) {
  return fs::exists(path / "manifest.json");
}

// --- EP defaulting ---

std::string DefaultEpFromPackage(const OrtModelPackageContext& pkg_ctx) {
  auto component_names = pkg_ctx.GetComponentNames();
  if (component_names.empty()) {
    throw std::runtime_error("Model package has no components");
  }

  // For each component, collect the union of EP names across all its variants.
  // Then intersect across components.
  std::unordered_set<std::string> intersection;
  bool first_component = true;

  for (const auto& comp_name : component_names) {
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
                                     const OrtSessionOptions& session_options)
    : package_root_(package_root),
      configs_path_(package_root / "configs") {
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

std::string ModelPackageState::GetGenAIConfigOverlay(const std::string& component_name) const {
  auto* cix = GetComponent(component_name);
  if (!cix) {
    return {};
  }

  std::string consumer_metadata = cix->GetSelectedVariantConsumerMetadata();
  if (consumer_metadata.empty()) {
    return {};
  }

  // Extract genai_config_overlay from consumer_metadata JSON.
  // consumer_metadata is a JSON object. We need to find the "genai_config_overlay" key
  // and return its value as a JSON string.
  //
  // Simple extraction: find "genai_config_overlay" key and extract the object value.
  // We use a minimal JSON element handler to capture it.
  struct OverlayExtractor : JSON::Element {
    std::string overlay_json;
    bool found = false;

    void OnValue(std::string_view name, JSON::Value value) override {
      // Ignore non-overlay fields
      if (name != "genai_config_overlay") {
        return;
      }
    }

    Element& OnObject(std::string_view name) override {
      if (name == "genai_config_overlay") {
        found = true;
        // We can't easily capture a sub-object as a raw JSON string with the SAX parser.
        // Instead, we'll use a different approach below.
      }
      throw JSON::unknown_value_error{};
    }
  };

  // Since the SAX parser doesn't easily support extracting a sub-object as a raw JSON string,
  // we use a simpler approach: find the key "genai_config_overlay" and extract the balanced JSON.
  const std::string key = "\"genai_config_overlay\"";
  auto pos = consumer_metadata.find(key);
  if (pos == std::string::npos) {
    return {};
  }

  // Skip past the key and the colon
  pos += key.size();
  while (pos < consumer_metadata.size() && (consumer_metadata[pos] == ' ' || consumer_metadata[pos] == ':' ||
                                              consumer_metadata[pos] == '\t' || consumer_metadata[pos] == '\n' ||
                                              consumer_metadata[pos] == '\r')) {
    ++pos;
  }

  if (pos >= consumer_metadata.size()) {
    return {};
  }

  // Extract balanced JSON value starting from pos
  if (consumer_metadata[pos] == '{') {
    int depth = 0;
    bool in_string = false;
    size_t start = pos;
    for (size_t i = pos; i < consumer_metadata.size(); ++i) {
      char c = consumer_metadata[i];
      if (in_string) {
        if (c == '\\') {
          ++i;  // skip escaped char
        } else if (c == '"') {
          in_string = false;
        }
      } else {
        if (c == '"') {
          in_string = true;
        } else if (c == '{') {
          ++depth;
        } else if (c == '}') {
          --depth;
          if (depth == 0) {
            return consumer_metadata.substr(start, i - start + 1);
          }
        }
      }
    }
  }

  return {};
}

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

}  // namespace Generators
