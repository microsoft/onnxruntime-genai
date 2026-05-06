// JSON Parser
//
// Just implement the JSON::Element structure per JSON element you're interested in (note it's per element,
// so a JSON tree structure requires a tree of JSON::Element objects)
// Then call JSON::Parse with the root JSON element object, and a string_view of the JSON data
//
// For the elements inside of an array, the names will be empty strings
// The root element also has no names.
//
#pragma once

#include <exception>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace JSON {
struct unknown_value_error : std::exception {};  // Throw this from any Element callback to throw a std::runtime error reporting the unknown value name
struct type_mismatch {                           // When a file has one type, but we're expecting another type. "seen" & "expected" are indices into the Value std::variant below
  size_t seen, expected;
};

using Value = std::variant<std::string_view, double, bool, std::nullptr_t>;

// To see descriptive errors when types don't match, use this instead of std::get
template <typename T>
T Get(Value& var) {
  try {
    return std::get<T>(var);
  } catch (const std::bad_variant_access&) {
    throw type_mismatch{var.index(), Value{T{}}.index()};
  }
}

struct Element {
  virtual void OnComplete(bool empty) {}  // Called when parsing for this element is finished (empty is true when it's an empty element)

  virtual void OnValue(std::string_view name, Value value) { throw unknown_value_error{}; }

  virtual Element& OnArray(std::string_view name) { throw unknown_value_error{}; }
  virtual Element& OnObject(std::string_view name) { throw unknown_value_error{}; }
};

void Parse(Element& element, std::string_view document);
void TranslateException(std::string_view name);  // Translate JSON exceptions into std::runtime_exception with a useful message

// ---------------------------------------------------------------------------
// Lightweight JSON DOM
//
// Built on top of the streaming parser above. Use this for cases where we need
// to manipulate JSON values (e.g. apply a JSON Merge Patch) before consuming
// them. For simple "read into a struct" config loading, prefer the streaming
// Element interface directly — it is lower overhead.
// ---------------------------------------------------------------------------
struct Document;
using Object = std::map<std::string, Document>;
using Array = std::vector<Document>;
using DocumentValue = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;

struct Document {
  DocumentValue value;

  Document() : value(nullptr) {}
  /* implicit */ Document(std::nullptr_t) : value(nullptr) {}
  /* implicit */ Document(bool v) : value(v) {}
  /* implicit */ Document(double v) : value(v) {}
  /* implicit */ Document(int v) : value(static_cast<double>(v)) {}
  /* implicit */ Document(std::string v) : value(std::move(v)) {}
  /* implicit */ Document(const char* v) : value(std::string(v)) {}
  /* implicit */ Document(Array v) : value(std::move(v)) {}
  /* implicit */ Document(Object v) : value(std::move(v)) {}

  bool IsNull() const { return std::holds_alternative<std::nullptr_t>(value); }
  bool IsBool() const { return std::holds_alternative<bool>(value); }
  bool IsNumber() const { return std::holds_alternative<double>(value); }
  bool IsString() const { return std::holds_alternative<std::string>(value); }
  bool IsArray() const { return std::holds_alternative<Array>(value); }
  bool IsObject() const { return std::holds_alternative<Object>(value); }

  bool AsBool() const { return std::get<bool>(value); }
  double AsNumber() const { return std::get<double>(value); }
  const std::string& AsString() const { return std::get<std::string>(value); }
  const Array& AsArray() const { return std::get<Array>(value); }
  Array& AsArray() { return std::get<Array>(value); }
  const Object& AsObject() const { return std::get<Object>(value); }
  Object& AsObject() { return std::get<Object>(value); }
};

// Parse a JSON document into a DOM. Throws std::runtime_error on parse error.
Document ParseDocument(std::string_view text);

// Serialize a DOM back to a JSON string. Object keys are serialized in
// std::map-defined (lexicographic) order. Numbers that are integral within the
// double range are emitted without a decimal point.
std::string SerializeDocument(const Document& doc);

// Apply RFC 7386 (https://datatracker.ietf.org/doc/html/rfc7386) JSON Merge
// Patch in-place to |target|:
//   - If |patch| is not an object, |target| is replaced wholesale.
//   - For each key in |patch|:
//       * If the value is null, the key is removed from |target|.
//       * Else, MergePatch is applied recursively to target[key] and
//         patch[key]. If target[key] is missing, the patch value is inserted.
//   - Keys in |target| not mentioned in |patch| are left unchanged.
//   - Arrays and scalars in |patch| always replace wholesale; there is no
//     element-level array merge.
void MergePatch(Document& target, const Document& patch);
}  // namespace JSON
