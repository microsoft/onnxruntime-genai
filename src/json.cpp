#include "generators.h"
#include "json.h"

#include <cmath>
#include <charconv>
#include <sstream>

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
  if (current_ + count >= end_ || std::strncmp(current_, sz, count) != 0) {
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

}  // namespace JSON
