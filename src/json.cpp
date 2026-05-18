#include "generators.h"
#include "json.h"

#include <cmath>
#include <charconv>
#include <cstdio>
#include <limits>
#include <list>
#include <sstream>
#include <type_traits>

namespace JSON {
static constexpr const char* value_names[] = {"string", "number", "bool", "null"};
static_assert(std::size(value_names) == std::variant_size_v<Value>);

struct JSON {
  JSON(Element& element, std::string_view document);

 private:
  void Parse_Whitespace();
  void Parse_Object(Element& element);
  void Parse_Array(Element& element);
  void Parse_Value(Element& element, std::string_view name);

  double Parse_Number();
  std::string Parse_String();

  bool Skip(char c);  // If *current_ is 'c' skip over it and return true
  template <size_t TCount>
  bool Skip(const char (&sz)[TCount]);  // If current_ matches the given string, skip over it and return true
  unsigned char GetChar();

  const char* begin_;
  const char* current_{begin_};
  const char* end_;
};

void Parse(Element& element, std::string_view document) {
  JSON{element, document};
}

void TranslateException(std::string_view name) {
  try {
    throw;
  } catch (const unknown_value_error&) {
    throw std::runtime_error(" Unknown value \"" + std::string(name) + "\"");
  } catch (const type_mismatch& e) {
    throw std::runtime_error(std::string(name) + " - Expected a " + std::string(value_names[e.expected]) + " but saw a " + std::string(value_names[e.seen]));
  } catch (...) {
    throw;
  }
}

JSON::JSON(Element& element, std::string_view document) : begin_{document.data()}, end_{document.data() + document.size()} {
  try {
    Parse_Value(element, {});
    element.OnComplete(false);
  } catch (const std::exception& message) {
    // Figure out line number of error by counting carriage returns seen from start to error location
    int line = 1;
    const auto* last_cr = begin_;
    for (const auto* p = begin_; p < current_; p++) {
      if (*p == '\n') {
        line++;
        last_cr = p;
      }
    }

    std::ostringstream oss;
    oss << "JSON Error: " << message.what() << " at line " << line << " index " << static_cast<int>(current_ - last_cr);
    throw std::runtime_error(oss.str());
  }
}

bool JSON::Skip(char c) {
  if (current_ == end_ || *current_ != c) {
    return false;
  }

  current_++;
  return true;
}

template <size_t TCount>
bool JSON::Skip(const char (&sz)[TCount]) {
  size_t const count = TCount - 1;  // Remove the null terminator from the string literal
  if (current_ + count > end_ || std::strncmp(current_, sz, count) != 0) {
    return false;
  }

  current_ += count;
  return true;
}

unsigned char JSON::GetChar() {
  if (current_ == end_) {
    throw std::runtime_error("Unexpected end of JSON data");
  }

  return *current_++;
}

void JSON::Parse_Whitespace() {
  while (current_ != end_) {
    char const c = *current_;
    if (c != '\x20' && c != '\x9' && c != '\xD' && c != '\xA') {  // Space, tab, cr, lf
      return;
    }
    current_++;
  }
}

void JSON::Parse_Object(Element& element) {
  Parse_Whitespace();
  if (Skip('}')) {
    element.OnComplete(true);
    return;
  }

  while (true) {
    if (!Skip('\"')) {
      throw std::runtime_error("Expecting \" to start next object name, possibly due to an extra trailing ',' before this");
    }

    auto name = Parse_String();

    Parse_Whitespace();
    if (GetChar() != ':') {
      throw std::runtime_error("Expecting :");
    }

    Parse_Value(element, name);

    char const c = GetChar();
    if (c == ',') {
      Parse_Whitespace();
      continue;
    }

    if (c == '}') {
      element.OnComplete(false);
      return;
    }

    throw std::runtime_error("Expecting } or ,");
  }
}

void JSON::Parse_Value(Element& element, std::string_view name) {
  Parse_Whitespace();
  try {
    switch (char const c = GetChar()) {
      case '{': {
        auto& element_object = element.OnObject(name);
        Parse_Object(element_object);
      } break;
      case '[': {
        auto& element_array = element.OnArray(name);
        Parse_Array(element_array);
      } break;
      case '"': {
        element.OnValue(name, Parse_String());
      } break;
      case 't':
        if (Skip("rue")) {
          element.OnValue(name, true);
        }
        break;
      case 'f':
        if (Skip("alse")) {
          element.OnValue(name, false);
        }
        break;
      case 'n':
        if (Skip("ull")) {
          element.OnValue(name, nullptr);
        }
        break;
      default:
        if (c >= '0' && c <= '9' || c == '-') {
          --current_;
          element.OnValue(name, Parse_Number());
        } else
          throw unknown_value_error{};
        break;
    }
  } catch (const std::runtime_error& e) {
    if (!name.empty())
      throw std::runtime_error(std::string(name) + ":" + e.what());
    throw;
  } catch (...) {
    TranslateException(name);
  }

  Parse_Whitespace();
}

void JSON::Parse_Array(Element& element) {
  Parse_Whitespace();
  if (Skip(']')) {
    element.OnComplete(true);
    return;
  }

  while (true) {
    Parse_Value(element, {});
    char const c = GetChar();
    if (c == ',') {
      continue;
    }
    if (c == ']') {
      element.OnComplete(false);
      return;
    }

    throw std::runtime_error("Expecting ] or ,");
  }
}

double JSON::Parse_Number() {
  double value = NAN;
#if !defined(USE_CXX17) && !defined(__APPLE__)
  auto result = std::from_chars(current_, end_, value);
  if (result.ec != std::errc{}) {
    throw std::runtime_error("Expecting number");
  }
  current_ = result.ptr;
#else
  auto end = const_cast<char*>(end_);
  value = std::strtod(current_, &end);
  if (current_ == end) {
    throw std::runtime_error("Expecting number");
  }
  current_ = end;
  // strtod returns ±HUGE_VAL on overflow without otherwise signalling failure;
  // surface that as a parse error rather than admitting Inf into the DOM
  // (the serializer rejects non-finite doubles, so this would otherwise be
  // a deferred crash on a later round-trip).
  if (!std::isfinite(value)) {
    throw std::runtime_error("Number out of range");
  }
#endif
  return value;
}

std::string JSON::Parse_String() {
  std::string string;
  while (char c = GetChar()) {
    if (c == '"') {
      break;
    }

    if (c == '\\') {
      switch (c = GetChar()) {
        case '"':
        case '\\':
        case '/':
          break;
        case 'b':
          c = '\b';
          break;
        case 'n':
          c = '\n';
          break;
        case 'f':
          c = '\f';
          break;
        case 'r':
          c = '\r';
          break;
        case 't':
          c = '\t';
          break;
        case 'u':  // 16-bit unicode escape code
        {
          if (current_ + 4 > end_) {
            throw std::runtime_error("End of file parsing string uXXXX code");
          }

          unsigned value = 0;
          auto result = std::from_chars(current_, current_ + 4, value, 16);
          if (result.ec != std::errc{} || result.ptr != current_ + 4) {
            throw std::runtime_error("Error parsing uXXXX code");
          }
          current_ = result.ptr;
          throw std::runtime_error("Unsupported uXXXX code used");  // TODO: Current we ignore these, as strings are already utf8
          continue;                                                 // We don't push_back the char in this case
        }
      }
    }
    string.push_back(c);
  }
  return string;
}

// ---------------------------------------------------------------------------
// JSON DOM (Document) — built on top of the streaming Parse() above.
// ---------------------------------------------------------------------------
namespace {

Document FromValue(Value v) {
  if (auto* sv = std::get_if<std::string_view>(&v)) {
    return Document(std::string(*sv));
  }
  if (auto* d = std::get_if<double>(&v)) {
    return Document(*d);
  }
  if (auto* b = std::get_if<bool>(&v)) {
    return Document(*b);
  }
  return Document(nullptr);
}

// Builder used for any nested Object / Array. The streaming parser hands us
// reference-stable Element pointers to drive each nested level; we own the
// per-level builders in std::list so emplace_back doesn't invalidate the
// references we've already returned to the parser.
struct DocumentBuilder : Element {
  Document& target_;
  std::list<DocumentBuilder> children_;

  explicit DocumentBuilder(Document& t) : target_(t) {}

  Document& InsertChild(std::string_view name) {
    if (auto* obj = std::get_if<Object>(&target_.value)) {
      // operator[] default-constructs a Document if the key is absent,
      // which is what the caller will fill in next. If the key is already
      // present (parser saw a duplicate key), the new value wins — same
      // behaviour as the streaming Element parser.
      return (*obj)[std::string(name)];
    }
    if (auto* arr = std::get_if<Array>(&target_.value)) {
      arr->emplace_back();
      return arr->back();
    }
    // Should not happen — parser only nests inside containers — but if it
    // does, just clobber the slot.
    return target_;
  }

  void OnValue(std::string_view name, Value v) override {
    InsertChild(name) = FromValue(v);
  }

  Element& OnObject(std::string_view name) override {
    Document& slot = InsertChild(name);
    slot = Document(Object{});
    children_.emplace_back(slot);
    return children_.back();
  }

  Element& OnArray(std::string_view name) override {
    Document& slot = InsertChild(name);
    slot = Document(Array{});
    children_.emplace_back(slot);
    return children_.back();
  }
};

// Top-level builder. The streaming parser invokes one of OnValue / OnObject /
// OnArray with an empty name to seed the root value.
struct RootDocumentBuilder : Element {
  Document& root_;
  std::list<DocumentBuilder> children_;

  explicit RootDocumentBuilder(Document& root) : root_(root) {}

  void OnValue(std::string_view, Value v) override {
    root_ = FromValue(v);
  }

  Element& OnObject(std::string_view) override {
    root_ = Document(Object{});
    children_.emplace_back(root_);
    return children_.back();
  }

  Element& OnArray(std::string_view) override {
    root_ = Document(Array{});
    children_.emplace_back(root_);
    return children_.back();
  }
};

void EscapeString(const std::string& s, std::ostringstream& oss) {
  oss << '"';
  for (char c : s) {
    switch (c) {
      case '"':
        oss << "\\\"";
        break;
      case '\\':
        oss << "\\\\";
        break;
      case '\b':
        oss << "\\b";
        break;
      case '\f':
        oss << "\\f";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
          oss << buf;
        } else {
          oss << c;
        }
    }
  }
  oss << '"';
}

void SerializeImpl(const Document& doc, std::ostringstream& oss) {
  std::visit(
      [&](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::nullptr_t>) {
          oss << "null";
        } else if constexpr (std::is_same_v<T, bool>) {
          oss << (v ? "true" : "false");
        } else if constexpr (std::is_same_v<T, double>) {
          // JSON has no representation for non-finite numbers (RFC 8259 §6
          // restricts the grammar to the finite reals). Refuse to emit them
          // rather than producing nan/inf token soup.
          if (!std::isfinite(v)) {
            throw std::runtime_error("JSON serialize: non-finite number");
          }
          // Emit integral values without a decimal point so a base config
          // round-trips byte-for-byte through a no-op merge in the common
          // case (e.g. token ids, head_size, hidden_size). Bounds-check
          // before the cast — converting an out-of-range double to long long
          // is undefined behavior — and then use modf to test integrality
          // without going through the cast at all. The upper bound uses a
          // strict `<` because static_cast<double>(LLONG_MAX) rounds up to
          // 2^63 (LLONG_MAX = 2^63-1 is not exactly representable as
          // double); admitting v = 2^63 would still overflow the cast.
          double integral_part = 0.0;
          if (v >= static_cast<double>(std::numeric_limits<long long>::min()) &&
              v < static_cast<double>(std::numeric_limits<long long>::max()) &&
              std::modf(v, &integral_part) == 0.0) {
            oss << static_cast<long long>(v);
          } else {
            // Use enough precision to round-trip a double exactly.
            std::ostringstream tmp;
            tmp.precision(17);
            tmp << v;
            oss << tmp.str();
          }
        } else if constexpr (std::is_same_v<T, std::string>) {
          EscapeString(v, oss);
        } else if constexpr (std::is_same_v<T, Array>) {
          oss << '[';
          bool first = true;
          for (const auto& e : v) {
            if (!first) oss << ',';
            first = false;
            SerializeImpl(e, oss);
          }
          oss << ']';
        } else if constexpr (std::is_same_v<T, Object>) {
          oss << '{';
          bool first = true;
          for (const auto& [k, val] : v) {
            if (!first) oss << ',';
            first = false;
            EscapeString(k, oss);
            oss << ':';
            SerializeImpl(val, oss);
          }
          oss << '}';
        }
      },
      doc.value);
}

}  // namespace

Document ParseDocument(std::string_view text) {
  Document root;
  RootDocumentBuilder builder(root);
  Parse(builder, text);
  return root;
}

std::string SerializeDocument(const Document& doc) {
  std::ostringstream oss;
  SerializeImpl(doc, oss);
  return oss.str();
}

void MergePatch(Document& target, const Document& patch) {
  // RFC 7386: if the patch is not an object, the target is replaced wholesale.
  if (!patch.IsObject()) {
    target = patch;
    return;
  }
  // If the target is anything other than an object, RFC 7386 says treat it as
  // an empty object before applying the patch.
  if (!target.IsObject()) {
    target = Document(Object{});
  }
  Object& target_obj = target.AsObject();
  const Object& patch_obj = patch.AsObject();
  for (const auto& [key, patch_val] : patch_obj) {
    if (patch_val.IsNull()) {
      target_obj.erase(key);
    } else {
      auto it = target_obj.find(key);
      if (it == target_obj.end()) {
        // Insert a deep copy.
        target_obj.emplace(key, patch_val);
      } else {
        MergePatch(it->second, patch_val);
      }
    }
  }
}

}  // namespace JSON
