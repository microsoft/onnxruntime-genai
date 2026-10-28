// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../ort_genai.h"
#include "../ort_genai_c_internal.h"
#include "../json.h"
#include "component_session.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <list>
#include <limits>
#include <locale>
#include <mutex>
#include <numeric>
#include <regex>
#include <sstream>
#include <string_view>
#include <unordered_map>

namespace {

struct OgaStructuredValue {
  using Array = std::vector<OgaStructuredValue>;
  using Object = std::vector<std::pair<std::string, OgaStructuredValue>>;
  using Value = std::variant<std::monostate, bool, int64_t, double, std::string,
                             Array, Object>;
  Value value;

  OgaStructuredValue() = default;
  OgaStructuredValue(std::nullptr_t) {}
  OgaStructuredValue(bool v) : value(v) {}
  OgaStructuredValue(int64_t v) : value(v) {}
  OgaStructuredValue(double v) : value(v) {}
  OgaStructuredValue(std::string v) : value(std::move(v)) {}
  OgaStructuredValue(Array v) : value(std::move(v)) {}
  OgaStructuredValue(Object v) : value(std::move(v)) {}
};

struct OgaQuestion {
  std::string type;
  OgaStructuredValue instructions;
  OgaStructuredValue criteria;
};

struct OgaStructuredRequest {
  OgaStructuredValue state;
  std::vector<std::pair<std::string, OgaQuestion>> questions;
  float temperature{1.0f};
};

struct OgaAnswer {
  std::string type;
  std::optional<double> noul;
  std::optional<std::string> choice;
  std::optional<double> score;
  std::optional<double> confidence;
  std::vector<std::pair<std::string, double>> probabilities;
  std::vector<std::pair<std::string, std::string>> legend;
};

struct OgaModelResult {
  std::string model;
  std::vector<std::pair<std::string, OgaAnswer>> answers;
};

struct OgaFreeFormRankRequest {
  OgaStructuredValue state;
  OgaStructuredValue instructions;
  std::vector<std::pair<std::string, OgaStructuredValue>> candidates;
  float temperature{1.0f};
};

struct OgaRankedItem {
  size_t rank{};
  std::string key;
  double probability{};
};

struct OgaRankingResult {
  std::string model;
  std::vector<OgaRankedItem> ranked;
};

constexpr size_t kDefaultClmCacheEntries = 256;
constexpr size_t kDefaultClmCacheBytes = 64 * 1024 * 1024;
constexpr size_t kDefaultKevCacheEntries = 512;
constexpr size_t kDefaultKevCacheBytes = 16 * 1024 * 1024;
constexpr size_t kDefaultKevPrefixCacheEntries = 32;
constexpr size_t kDefaultKevPrefixCacheBytes = 512 * 1024 * 1024;
std::string PackageIdentity(const std::string& package_path,
                            const std::vector<std::string>& providers) {
  namespace stdfs = std::filesystem;
  std::error_code error;
#ifdef _WIN32
  const fs::path utf8_source(package_path);
  const stdfs::path source(utf8_source.c_str());
#else
  const stdfs::path source(package_path);
#endif
  auto canonical = stdfs::weakly_canonical(source, error);
  if (error) canonical = stdfs::absolute(source, error);
  std::ostringstream result;
  const auto canonical_utf8 = canonical.generic_u8string();
  result.write(reinterpret_cast<const char*>(canonical_utf8.data()),
               static_cast<std::streamsize>(canonical_utf8.size()));
  for (const auto& provider : providers) result << "|provider=" << provider;
  return result.str();
}

template <typename Value>
class SessionLruCache {
 public:
  using SizeFunction = size_t (*)(const Value&);

  SessionLruCache(size_t entries, size_t bytes, SizeFunction size)
      : entry_capacity_(entries), byte_capacity_(bytes), size_(size) {}

  bool Get(const std::string& key, Value& value) {
    std::lock_guard lock(mutex_);
    const auto found = index_.find(key);
    if (found == index_.end()) return false;
    entries_.splice(entries_.begin(), entries_, found->second);
    value = found->second->value;
    return true;
  }

  void Put(std::string key, Value value) {
    std::lock_guard lock(mutex_);
    if (!entry_capacity_ || !byte_capacity_) return;
    const auto value_bytes = size_(value);
    if (key.size() >
        ((std::numeric_limits<size_t>::max)() - value_bytes) / 2)
      throw std::overflow_error("cache entry byte count overflows size_t");
    const auto bytes = 2 * key.size() + value_bytes;
    if (bytes > byte_capacity_) return;
    if (const auto found = index_.find(key); found != index_.end()) {
      byte_size_ -= found->second->bytes;
      entries_.erase(found->second);
      index_.erase(found);
    }
    entries_.push_front({std::move(key), std::move(value), bytes});
    try {
      index_.emplace(entries_.front().key, entries_.begin());
    } catch (...) {
      entries_.pop_front();
      throw;
    }
    byte_size_ += bytes;
    Evict();
  }

 private:
  struct Entry {
    std::string key;
    Value value;
    size_t bytes;
  };
  void Evict() {
    while ((!entry_capacity_ || !byte_capacity_ ||
            entries_.size() > entry_capacity_ || byte_size_ > byte_capacity_) &&
           !entries_.empty()) {
      byte_size_ -= entries_.back().bytes;
      index_.erase(entries_.back().key);
      entries_.pop_back();
    }
  }
  std::mutex mutex_;
  std::list<Entry> entries_;
  std::unordered_map<std::string, typename std::list<Entry>::iterator> index_;
  size_t entry_capacity_{};
  size_t byte_capacity_{};
  size_t byte_size_{};
  SizeFunction size_;
};

constexpr int64_t kStateToken = 248060;
constexpr int64_t kQuestionToken = 248061;
constexpr int64_t kOptionStartToken = 248049;
constexpr int64_t kOptionEndToken = 248050;
constexpr int64_t kDecideToken = 248062;

class StructuredJsonParser {
 public:
  explicit StructuredJsonParser(std::string_view source) : source_(source) {}

  OgaStructuredValue Parse() {
    auto value = ParseValue();
    SkipWhitespace();
    if (position_ != source_.size())
      throw std::invalid_argument("unexpected trailing JSON data");
    return value;
  }

 private:
  void SkipWhitespace() {
    while (position_ < source_.size() &&
           (source_[position_] == ' ' || source_[position_] == '\t' ||
            source_[position_] == '\r' || source_[position_] == '\n'))
      ++position_;
  }

  char Take() {
    if (position_ >= source_.size())
      throw std::invalid_argument("unexpected end of JSON");
    return source_[position_++];
  }

  void Expect(char expected) {
    if (Take() != expected)
      throw std::invalid_argument("invalid JSON syntax");
  }

  void ExpectLiteral(std::string_view literal) {
    if (source_.substr(position_, literal.size()) != literal)
      throw std::invalid_argument("invalid JSON literal");
    position_ += literal.size();
  }

  static void AppendUtf8(std::string& output, uint32_t codepoint) {
    if (codepoint <= 0x7f) {
      output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
      output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
      output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
      output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
      output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
      output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0x10ffff) {
      output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
      output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
      output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
      output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
      throw std::invalid_argument("invalid JSON Unicode codepoint");
    }
  }

  uint32_t ParseHex4() {
    uint32_t value{};
    for (int i = 0; i < 4; ++i) {
      const char c = Take();
      value <<= 4;
      if (c >= '0' && c <= '9')
        value |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f')
        value |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        value |= static_cast<uint32_t>(c - 'A' + 10);
      else
        throw std::invalid_argument("invalid JSON Unicode escape");
    }
    return value;
  }

  std::string ParseString() {
    Expect('"');
    std::string result;
    while (true) {
      const char c = Take();
      if (c == '"') return result;
      if (static_cast<unsigned char>(c) < 0x20)
        throw std::invalid_argument("unescaped JSON control character");
      if (c != '\\') {
        result.push_back(c);
        continue;
      }
      switch (Take()) {
        case '"':
          result.push_back('"');
          break;
        case '\\':
          result.push_back('\\');
          break;
        case '/':
          result.push_back('/');
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        case 'u': {
          uint32_t codepoint = ParseHex4();
          if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
            Expect('\\');
            Expect('u');
            const uint32_t low = ParseHex4();
            if (low < 0xdc00 || low > 0xdfff)
              throw std::invalid_argument("invalid JSON surrogate pair");
            codepoint =
                0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
          } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
            throw std::invalid_argument("invalid JSON surrogate pair");
          }
          AppendUtf8(result, codepoint);
          break;
        }
        default:
          throw std::invalid_argument("invalid JSON escape");
      }
    }
  }

  OgaStructuredValue ParseNumber() {
    const size_t start = position_;
    if (source_[position_] == '-') ++position_;
    if (position_ >= source_.size())
      throw std::invalid_argument("invalid JSON number");
    if (source_[position_] == '0') {
      ++position_;
    } else {
      if (source_[position_] < '1' || source_[position_] > '9')
        throw std::invalid_argument("invalid JSON number");
      while (position_ < source_.size() && source_[position_] >= '0' &&
             source_[position_] <= '9')
        ++position_;
    }
    bool floating = false;
    if (position_ < source_.size() && source_[position_] == '.') {
      floating = true;
      ++position_;
      const size_t fraction_start = position_;
      while (position_ < source_.size() && source_[position_] >= '0' &&
             source_[position_] <= '9')
        ++position_;
      if (fraction_start == position_)
        throw std::invalid_argument("invalid JSON number");
    }
    if (position_ < source_.size() &&
        (source_[position_] == 'e' || source_[position_] == 'E')) {
      floating = true;
      ++position_;
      if (position_ < source_.size() &&
          (source_[position_] == '+' || source_[position_] == '-'))
        ++position_;
      const size_t exponent_start = position_;
      while (position_ < source_.size() && source_[position_] >= '0' &&
             source_[position_] <= '9')
        ++position_;
      if (exponent_start == position_)
        throw std::invalid_argument("invalid JSON number");
    }
    const auto token = source_.substr(start, position_ - start);
    if (!floating) {
      int64_t integer{};
      const auto [end, error] =
          std::from_chars(token.data(), token.data() + token.size(), integer);
      if (error != std::errc{} || end != token.data() + token.size())
        throw std::invalid_argument("JSON integer is outside int64 range");
      return integer;
    }
    std::string owned(token);
    char* end{};
    const double number = std::strtod(owned.c_str(), &end);
    if (end != owned.c_str() + owned.size() || !std::isfinite(number))
      throw std::invalid_argument("invalid JSON floating-point number");
    return number;
  }

  OgaStructuredValue ParseArray() {
    Expect('[');
    OgaStructuredValue::Array result;
    SkipWhitespace();
    if (position_ < source_.size() && source_[position_] == ']') {
      ++position_;
      return result;
    }
    while (true) {
      result.push_back(ParseValue());
      SkipWhitespace();
      const char delimiter = Take();
      if (delimiter == ']') return result;
      if (delimiter != ',')
        throw std::invalid_argument("invalid JSON array");
    }
  }

  OgaStructuredValue ParseObject() {
    Expect('{');
    OgaStructuredValue::Object result;
    SkipWhitespace();
    if (position_ < source_.size() && source_[position_] == '}') {
      ++position_;
      return result;
    }
    while (true) {
      SkipWhitespace();
      auto key = ParseString();
      SkipWhitespace();
      Expect(':');
      auto value = ParseValue();
      if (std::any_of(result.begin(), result.end(),
                      [&](const auto& item) { return item.first == key; }))
        throw std::invalid_argument("duplicate JSON object key: " + key);
      result.emplace_back(std::move(key), std::move(value));
      SkipWhitespace();
      const char delimiter = Take();
      if (delimiter == '}') return result;
      if (delimiter != ',')
        throw std::invalid_argument("invalid JSON object");
    }
  }

  OgaStructuredValue ParseValue() {
    SkipWhitespace();
    if (position_ >= source_.size())
      throw std::invalid_argument("missing JSON value");
    switch (source_[position_]) {
      case 'n':
        ExpectLiteral("null");
        return nullptr;
      case 't':
        ExpectLiteral("true");
        return true;
      case 'f':
        ExpectLiteral("false");
        return false;
      case '"':
        return ParseString();
      case '[':
        return ParseArray();
      case '{':
        return ParseObject();
      default:
        return ParseNumber();
    }
  }

  std::string_view source_;
  size_t position_{};
};

template <class T>
const T* Get(const OgaStructuredValue& value) {
  return std::get_if<T>(&value.value);
}

bool Empty(const OgaStructuredValue& value) {
  if (std::holds_alternative<std::monostate>(value.value)) return true;
  if (const auto* text = Get<std::string>(value)) return text->empty();
  if (const auto* array = Get<OgaStructuredValue::Array>(value)) return array->empty();
  if (const auto* object = Get<OgaStructuredValue::Object>(value)) return object->empty();
  return false;
}

bool SameDouble(std::string_view text, double expected) {
  std::string owned(text);
  char* end{};
  const auto parsed = std::strtod(owned.c_str(), &end);
  if (end != owned.c_str() + owned.size()) return false;
  uint64_t parsed_bits{};
  uint64_t expected_bits{};
  std::memcpy(&parsed_bits, &parsed, sizeof(parsed_bits));
  std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
  return parsed_bits == expected_bits;
}

std::string FormatPythonFloat(double value) {
  if (!std::isfinite(value))
    throw std::invalid_argument(
        "structured floating-point value cannot be rendered");
  if (value == 0) return std::signbit(value) ? "-0.0" : "0.0";

  const auto exponent = static_cast<int>(
      std::floor(std::log10(std::abs(value))));
  const bool scientific = exponent < -4 || exponent >= 16;
  for (int significant = 1;
       significant <= std::numeric_limits<double>::max_digits10;
       ++significant) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    if (scientific) {
      stream << std::scientific << std::setprecision(significant - 1)
             << value;
    } else {
      stream << std::fixed
             << std::setprecision(std::max(0, significant - 1 - exponent))
             << value;
    }
    auto result = stream.str();
    const auto exponent_position = result.find('e');
    auto mantissa_end =
        exponent_position == std::string::npos ? result.size()
                                               : exponent_position;
    if (const auto dot = result.find('.');
        dot != std::string::npos && dot < mantissa_end) {
      while (mantissa_end > dot + 1 && result[mantissa_end - 1] == '0') {
        result.erase(mantissa_end - 1, 1);
        --mantissa_end;
      }
      if (!scientific && mantissa_end == dot + 1)
        result.insert(mantissa_end, "0");
      else if (scientific && mantissa_end == dot + 1)
        result.erase(dot, 1);
    } else if (!scientific) {
      result.insert(mantissa_end, ".0");
    }
    if (SameDouble(result, value)) return result;
  }
  throw std::invalid_argument(
      "structured floating-point value cannot be rendered");
}

std::string Scalar(const OgaStructuredValue& value, bool kev) {
  if (std::holds_alternative<std::monostate>(value.value)) return {};
  if (const auto* text = Get<std::string>(value)) return *text;
  if (const auto* boolean = Get<bool>(value))
    return *boolean ? (kev ? "True" : "true") : (kev ? "False" : "false");
  if (const auto* integer = Get<int64_t>(value)) return std::to_string(*integer);
  if (const auto* number = Get<double>(value)) return FormatPythonFloat(*number);
  throw std::invalid_argument("structured value is not scalar");
}

std::string RenderClm(const OgaStructuredValue& value, int indent = 0) {
  if (!Get<OgaStructuredValue::Array>(value) && !Get<OgaStructuredValue::Object>(value))
    return Scalar(value, false);
  const std::string pad(static_cast<size_t>(indent), ' ');
  std::string result;
  if (const auto* object = Get<OgaStructuredValue::Object>(value)) {
    for (const auto& [key, item] : *object) {
      if (!result.empty()) result += indent == 0 ? "\n\n" : "\n";
      if ((Get<OgaStructuredValue::Array>(item) || Get<OgaStructuredValue::Object>(item)) &&
          !Empty(item))
        result += pad + key + ":\n" + RenderClm(item, indent + 2);
      else
        result += pad + key + ": " + RenderClm(item);
    }
  } else {
    for (const auto& item : *Get<OgaStructuredValue::Array>(value)) {
      if (!result.empty()) result += "\n";
      if ((Get<OgaStructuredValue::Array>(item) || Get<OgaStructuredValue::Object>(item)) &&
          !Empty(item))
        result += pad + "-\n" + RenderClm(item, indent + 2);
      else
        result += pad + "- " + RenderClm(item);
    }
  }
  return result;
}

std::string RenderKev(const OgaStructuredValue& value, int indent = 0) {
  if (!Get<OgaStructuredValue::Array>(value) && !Get<OgaStructuredValue::Object>(value))
    return Scalar(value, true);
  const std::string pad(static_cast<size_t>(indent) * 2, ' ');
  std::string result;
  if (const auto* array = Get<OgaStructuredValue::Array>(value)) {
    for (const auto& item : *array) {
      if (!result.empty()) result += "\n";
      auto rendered = RenderKev(item, indent + 1);
      const auto first = rendered.find_first_not_of(" \t\r\n");
      if (first != std::string::npos) rendered.erase(0, first);
      result += pad + "- " + rendered;
    }
  } else {
    for (const auto& [key, item] : *Get<OgaStructuredValue::Object>(value)) {
      if (!result.empty()) result += "\n";
      if (Get<OgaStructuredValue::Array>(item) || Get<OgaStructuredValue::Object>(item))
        result += pad + key + ":\n" + RenderKev(item, indent + 1);
      else
        result += pad + key + ": " + RenderKev(item);
    }
  }
  return result;
}

std::string Trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}

const OgaStructuredValue* Find(const OgaStructuredValue& value, std::string_view key) {
  const auto* object = Get<OgaStructuredValue::Object>(value);
  if (!object) return nullptr;
  const auto found = std::find_if(object->begin(), object->end(),
                                  [&](const auto& item) { return item.first == key; });
  return found == object->end() ? nullptr : &found->second;
}

const OgaStructuredValue& RequiredField(const OgaStructuredValue& object,
                                        std::string_view key) {
  const auto* value = Find(object, key);
  if (!value)
    throw std::invalid_argument("request requires field: " +
                                std::string(key));
  return *value;
}

float OptionalTemperature(const OgaStructuredValue& object) {
  const auto* value = Find(object, "temperature");
  if (!value) return 1.0f;
  if (const auto* integer = Get<int64_t>(*value))
    return static_cast<float>(*integer);
  if (const auto* number = Get<double>(*value))
    return static_cast<float>(*number);
  throw std::invalid_argument("temperature must be numeric");
}

OgaStructuredRequest ParseDecisionRequestJson(std::string_view source) {
  auto root = StructuredJsonParser(source).Parse();
  if (!Get<OgaStructuredValue::Object>(root))
    throw std::invalid_argument("request must be a JSON object");
  OgaStructuredRequest request;
  if (const auto* state = Find(root, "state")) request.state = *state;
  request.temperature = OptionalTemperature(root);
  const auto& questions_value = RequiredField(root, "questions");
  const auto* questions = Get<OgaStructuredValue::Object>(questions_value);
  if (!questions || questions->empty())
    throw std::invalid_argument("questions must be a non-empty object");
  for (const auto& [id, value] : *questions) {
    if (id.find('\0') != std::string::npos)
      throw std::invalid_argument(
          "question id must not contain embedded NUL characters");
    const auto* object = Get<OgaStructuredValue::Object>(value);
    if (!object)
      throw std::invalid_argument("question must be an object: " + id);
    const auto* type = Get<std::string>(RequiredField(value, "type"));
    if (!type)
      throw std::invalid_argument("question type must be a string: " + id);
    OgaQuestion question;
    question.type = *type;
    if (const auto* instructions = Find(value, "instructions"))
      question.instructions = *instructions;
    if (const auto* criteria = Find(value, "criteria"))
      question.criteria = *criteria;
    request.questions.emplace_back(id, std::move(question));
  }
  return request;
}

OgaFreeFormRankRequest ParseRankingRequestJson(std::string_view source) {
  auto root = StructuredJsonParser(source).Parse();
  if (!Get<OgaStructuredValue::Object>(root))
    throw std::invalid_argument("request must be a JSON object");
  OgaFreeFormRankRequest request;
  if (const auto* state = Find(root, "state")) request.state = *state;
  if (const auto* instructions = Find(root, "instructions"))
    request.instructions = *instructions;
  request.temperature = OptionalTemperature(root);
  const auto& candidates_value = RequiredField(root, "candidates");
  const auto* candidates = Get<OgaStructuredValue::Object>(candidates_value);
  if (!candidates || candidates->empty())
    throw std::invalid_argument("candidates must be a non-empty object");
  for (const auto& [key, value] : *candidates) {
    if (key.find('\0') != std::string::npos)
      throw std::invalid_argument(
          "candidate key must not contain embedded NUL characters");
    request.candidates.emplace_back(key, value);
  }
  return request;
}

const Generators::Tensor& FindTensor(
    const Generators::NamedTensors& tensors, const std::string& name) {
  const auto found = tensors.find(name);
  if (found == tensors.end() || !found->second)
    throw std::runtime_error("component did not return output: " + name);
  return *found->second;
}

template <typename T>
std::vector<T> CopyTensor(const Generators::Tensor& tensor,
                          ONNXTensorElementDataType expected) {
  if (tensor.GetType() != expected)
    throw std::runtime_error("component output has unexpected element type");
  std::vector<T> result(tensor.GetElementCount());
  if (result.empty()) return result;
  const auto& value = *tensor.ort_tensor_;
  if (value.GetTensorMemoryInfo().GetDeviceType() ==
      OrtMemoryInfoDeviceType_CPU) {
    std::memcpy(result.data(), value.GetTensorRawData(),
                result.size() * sizeof(T));
  } else {
    auto* device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
    auto memory = device->WrapMemory(std::span<T>(
        const_cast<OrtValue&>(value).GetTensorMutableData<T>(),
        result.size()));
    const auto host = memory.CopyDeviceToCpu();
    std::copy(host.begin(), host.end(), result.begin());
  }
  return result;
}

std::vector<float> FloatTensor(const Generators::Tensor& tensor) {
  if (tensor.GetType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    return CopyTensor<float>(tensor, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  auto bits = CopyTensor<uint16_t>(
      tensor, tensor.GetType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                  ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                  : ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  std::vector<float> result(bits.size());
  if (tensor.GetType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
    for (size_t i = 0; i < bits.size(); ++i) {
      uint32_t expanded = static_cast<uint32_t>(bits[i]) << 16;
      std::memcpy(&result[i], &expanded, sizeof(float));
    }
    return result;
  }

  if (tensor.GetType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    throw std::runtime_error("component output must be float32, float16, or bfloat16");
  for (size_t i = 0; i < bits.size(); ++i) {
    const uint32_t sign = static_cast<uint32_t>(bits[i] & 0x8000) << 16;
    uint32_t exponent = (bits[i] >> 10) & 0x1f;
    uint32_t mantissa = bits[i] & 0x03ff;
    uint32_t expanded;
    if (!exponent) {
      if (!mantissa)
        expanded = sign;
      else {
        exponent = 113;
        while (!(mantissa & 0x0400)) {
          mantissa <<= 1;
          --exponent;
        }
        expanded = sign | (exponent << 23) | ((mantissa & 0x03ff) << 13);
      }
    } else if (exponent == 0x1f) {
      expanded = sign | 0x7f800000 | (mantissa << 13);
    } else {
      expanded = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    std::memcpy(&result[i], &expanded, sizeof(float));
  }
  return result;
}

size_t TensorElementCount(const Generators::Tensor& tensor,
                          std::string_view name) {
  size_t count = 1;
  for (const auto dimension : tensor.GetShape()) {
    if (dimension < 0)
      throw std::runtime_error(std::string(name) + " has a negative output dimension");
    const auto value = static_cast<size_t>(dimension);
    if (value && count > std::numeric_limits<size_t>::max() / value)
      throw std::runtime_error(std::string(name) + " element count overflows size_t");
    count *= value;
  }
  return count;
}

void RequireShape(const Generators::Tensor& tensor,
                  std::initializer_list<size_t> dimensions,
                  std::string_view name) {
  const auto shape = tensor.GetShape();
  if (shape.size() != dimensions.size())
    throw std::runtime_error(std::string(name) + " must have rank " +
                             std::to_string(dimensions.size()));
  size_t index = 0;
  for (const auto expected : dimensions) {
    if (shape[index] < 0 ||
        static_cast<size_t>(shape[index]) != expected)
      throw std::runtime_error(std::string(name) + " dimension " +
                               std::to_string(index) + " must be " +
                               std::to_string(expected));
    ++index;
  }
  if (TensorElementCount(tensor, name) !=
      std::accumulate(dimensions.begin(), dimensions.end(), size_t{1},
                      std::multiplies<size_t>()))
    throw std::runtime_error(std::string(name) + " has an invalid element count");
}

size_t RequireHiddenShape(const Generators::Tensor& tensor, size_t rows,
                          size_t width, std::string_view name) {
  const auto shape = tensor.GetShape();
  if (shape.size() != 3)
    throw std::runtime_error(std::string(name) + " must have rank 3");
  if (shape[0] < 0 || static_cast<size_t>(shape[0]) != rows)
    throw std::runtime_error(std::string(name) + " row count does not match the token batch");
  if (shape[1] < 0 || static_cast<size_t>(shape[1]) != width)
    throw std::runtime_error(std::string(name) + " sequence width does not match the token batch");
  if (shape[2] <= 0)
    throw std::runtime_error(std::string(name) + " hidden dimension must be positive");
  const auto hidden_size = static_cast<size_t>(shape[2]);
  if (rows && width > std::numeric_limits<size_t>::max() / rows)
    throw std::runtime_error(std::string(name) + " element count overflows size_t");
  const auto tokens = rows * width;
  if (hidden_size > std::numeric_limits<size_t>::max() / std::max<size_t>(tokens, 1))
    throw std::runtime_error(std::string(name) + " element count overflows size_t");
  if (TensorElementCount(tensor, name) != tokens * hidden_size)
    throw std::runtime_error(std::string(name) + " has an invalid element count");
  return hidden_size;
}

void RequireFloatCount(const Generators::Tensor& tensor,
                       const std::vector<float>& values,
                       std::string_view name) {
  if (values.size() != TensorElementCount(tensor, name))
    throw std::runtime_error(std::string(name) +
                             " data size does not match its shape");
}

struct FeedStorage {
  std::vector<std::vector<std::byte>> bytes;
  Generators::NamedTensors inputs;
  template <typename T>
  void Add(std::string name, const std::vector<T>& value, std::vector<int64_t> shape,
           OgaElementType type) {
    bytes.emplace_back(value.size() * sizeof(T));
    if (!value.empty()) std::memcpy(bytes.back().data(), value.data(), bytes.back().size());
    auto memory = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto tensor = OrtValue::CreateTensor(
        *memory, bytes.back().data(), bytes.back().size(), shape,
        static_cast<ONNXTensorElementDataType>(type));
    inputs.emplace(std::move(name),
                   std::make_shared<Generators::Tensor>(std::move(tensor)));
  }
  void AddZeros(std::string name, std::vector<int64_t> shape,
                ONNXTensorElementDataType type) {
    const auto count = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                       [](size_t left, int64_t right) { return left * static_cast<size_t>(right); });
    const auto size = count * Ort::SizeOf(type);
    bytes.emplace_back(std::max<size_t>(size, 1), std::byte{});
    auto memory = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto tensor = OrtValue::CreateTensor(
        *memory, bytes.back().data(), size, shape,
        type);
    inputs.emplace(std::move(name),
                   std::make_shared<Generators::Tensor>(std::move(tensor)));
  }
};

struct TokenBatch {
  std::vector<int64_t> ids, mask;
  size_t rows{}, width{};
};

void AddPositionIds(FeedStorage& feeds, const Generators::ComponentSession& session,
                    const std::vector<int64_t>& positions, size_t rows,
                    size_t width) {
  const auto found = std::find_if(
      session.Inputs().begin(), session.Inputs().end(),
      [](const Generators::ComponentInfo& info) { return info.name == "position_ids"; });
  if (found == session.Inputs().end()) return;
  if (found->type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    throw std::runtime_error("position_ids must use int64 elements");
  if (positions.size() != rows * width)
    throw std::runtime_error("position_ids data size does not match the token batch");

  const auto require_dimension = [&](size_t index, size_t value) {
    if (found->shape[index] >= 0 &&
        found->shape[index] != static_cast<int64_t>(value))
      throw std::runtime_error("position_ids input shape does not match the token batch");
  };
  if (found->shape.size() == 2) {
    require_dimension(0, rows);
    require_dimension(1, width);
    feeds.Add("position_ids", positions,
              {static_cast<int64_t>(rows), static_cast<int64_t>(width)},
              OgaElementType_int64);
    return;
  }
  if (found->shape.size() == 3 && found->shape[0] > 0) {
    const auto axes = static_cast<size_t>(found->shape[0]);
    require_dimension(1, rows);
    require_dimension(2, width);
    if (positions.size() > std::numeric_limits<size_t>::max() / axes)
      throw std::runtime_error("position_ids element count overflows size_t");
    std::vector<int64_t> expanded;
    expanded.reserve(positions.size() * axes);
    for (size_t axis = 0; axis < axes; ++axis)
      expanded.insert(expanded.end(), positions.begin(), positions.end());
    feeds.Add("position_ids", expanded,
              {static_cast<int64_t>(axes), static_cast<int64_t>(rows),
               static_cast<int64_t>(width)},
              OgaElementType_int64);
    return;
  }
  throw std::runtime_error(
      "position_ids must have shape [batch, sequence] or "
      "[axes, batch, sequence] with a fixed axis count");
}

TokenBatch Tokenize(Generators::PackageTokenizer& tokenizer, const std::vector<std::string>& texts) {
  if (texts.empty()) throw std::invalid_argument("questions must be non-empty");
  std::vector<std::vector<int32_t>> rows;
  size_t width = 0;
  for (const auto& text : texts) {
    auto row = tokenizer.Encode(text.empty() ? " " : text);
    if (row.size() > 2048) row.resize(2048);
    if (row.empty()) throw std::invalid_argument("tokenization produced an empty row");
    width = std::max(width, row.size());
    rows.push_back(std::move(row));
  }
  TokenBatch result;
  result.rows = rows.size();
  result.width = width;
  result.ids.assign(result.rows * width, tokenizer.PadTokenId());
  result.mask.assign(result.ids.size(), 0);
  for (size_t row = 0; row < rows.size(); ++row)
    for (size_t column = 0; column < rows[row].size(); ++column) {
      result.ids[row * width + column] = rows[row][column];
      result.mask[row * width + column] = 1;
    }
  return result;
}

TokenBatch BucketClmBatch(TokenBatch batch) {
  static constexpr size_t buckets[] = {96, 120, 144, 168, 256, 512, 1024, 2048};
  const auto found = std::lower_bound(
      std::begin(buckets), std::end(buckets), batch.width);
  if (found == std::end(buckets) || *found == batch.width) return batch;
  TokenBatch padded;
  padded.rows = batch.rows;
  padded.width = *found;
  padded.ids.assign(padded.rows * padded.width, batch.ids.front());
  padded.mask.assign(padded.ids.size(), 0);
  for (size_t row = 0; row < batch.rows; ++row) {
    std::copy_n(batch.ids.begin() + row * batch.width, batch.width,
                padded.ids.begin() + row * padded.width);
    std::copy_n(batch.mask.begin() + row * batch.width, batch.width,
                padded.mask.begin() + row * padded.width);
  }
  return padded;
}

FeedStorage BackboneFeeds(const Generators::ComponentSession& session, const TokenBatch& batch) {
  FeedStorage feeds;
  feeds.bytes.reserve(session.Inputs().size());
  feeds.inputs.reserve(session.Inputs().size());
  const std::vector<int64_t> shape{static_cast<int64_t>(batch.rows),
                                   static_cast<int64_t>(batch.width)};
  feeds.Add("input_ids", batch.ids, shape, OgaElementType_int64);
  feeds.Add("attention_mask", batch.mask, shape, OgaElementType_int64);
  if (std::find(session.InputNames().begin(), session.InputNames().end(), "position_ids") !=
      session.InputNames().end()) {
    std::vector<int64_t> positions(batch.mask.size());
    for (size_t row = 0; row < batch.rows; ++row) {
      int64_t position = 0;
      for (size_t column = 0; column < batch.width; ++column) {
        const auto index = row * batch.width + column;
        positions[index] = batch.mask[index] ? position++ : 0;
      }
    }
    AddPositionIds(feeds, session, positions, batch.rows, batch.width);
  }
  for (const auto& info : session.Inputs()) {
    if (info.name.rfind("past_key_values.", 0) != 0 &&
        info.name.rfind("past.", 0) != 0)
      continue;
    auto dimensions = info.shape;
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (dimensions[i] >= 0) continue;
      const auto symbol = i < info.symbolic_dimensions.size() ? info.symbolic_dimensions[i] : "";
      if (symbol.find("batch") != std::string::npos)
        dimensions[i] = static_cast<int64_t>(batch.rows);
      else if (symbol.find("past") != std::string::npos ||
               symbol.find("sequence") != std::string::npos)
        dimensions[i] = 0;
      else
        throw std::runtime_error("cannot initialize symbolic state dimension: " + symbol);
    }
    feeds.AddZeros(info.name, std::move(dimensions), info.type);
  }
  return feeds;
}

std::string HiddenOutputName(const Generators::ComponentSession& session) {
  return std::find(session.OutputNames().begin(), session.OutputNames().end(),
                   "token_hidden_states") != session.OutputNames().end()
             ? "token_hidden_states"
             : "hidden_states";
}

struct Candidates {
  std::vector<std::string> keys, texts;
};

Candidates ClmCandidates(const OgaQuestion& question) {
  Candidates result;
  const auto instructions = Trim(RenderClm(question.instructions));
  if (question.type == "choice") {
    const auto* criteria = Get<OgaStructuredValue::Object>(question.criteria);
    if (!criteria || criteria->empty())
      throw std::invalid_argument("choice question needs a non-empty criteria object");
    for (const auto& [key, value] : *criteria) {
      result.keys.push_back(key);
      result.texts.push_back(Empty(value) ? key : RenderClm(value));
    }
  } else if (question.type == "score") {
    const auto* criteria = Get<OgaStructuredValue::Array>(question.criteria);
    if (!criteria || criteria->size() < 2)
      throw std::invalid_argument("score question needs at least two levels");
    for (size_t i = 0; i < criteria->size(); ++i) {
      result.keys.push_back(std::to_string(i));
      result.texts.push_back(RenderClm((*criteria)[i]));
    }
  } else if (question.type == "noul") {
    for (const auto* key : {"false", "true"}) {
      const auto* value = Find(question.criteria, key);
      std::string description;
      if (!value || Empty(*value))
        description = instructions.empty() ? key : std::string(key == std::string_view("true") ? "Yes. This is true: " : "No. This is false: ") + instructions;
      else
        description = RenderClm(*value);
      result.keys.emplace_back(key);
      result.texts.push_back(std::string(key) + ": " + description);
    }
  } else
    throw std::invalid_argument("unknown question type: " + question.type);
  return result;
}

Candidates KevCandidates(const OgaQuestion& question) {
  Candidates result;
  if (question.type == "noul") {
    for (const auto& pair : {std::pair{"false", "no"}, std::pair{"true", "yes"}}) {
      result.keys.emplace_back(pair.first);
      const auto* value = Find(question.criteria, pair.first);
      result.texts.push_back(!value || Empty(*value) ? pair.second : std::string(pair.second) + ": " + RenderKev(*value));
    }
  } else if (question.type == "choice") {
    const auto* criteria = Get<OgaStructuredValue::Object>(question.criteria);
    if (!criteria || criteria->empty() || criteria->size() > 255)
      throw std::invalid_argument("invalid choice KEV question criteria");
    for (const auto& [key, value] : *criteria) {
      result.keys.push_back(key);
      result.texts.push_back(Empty(value) ? key : key + ": " + RenderKev(value));
    }
  } else if (question.type == "score") {
    const auto* criteria = Get<OgaStructuredValue::Array>(question.criteria);
    if (!criteria || criteria->empty() || criteria->size() > 255)
      throw std::invalid_argument("invalid score KEV question criteria");
    for (size_t i = 0; i < criteria->size(); ++i) {
      result.keys.push_back(std::to_string(i));
      result.texts.push_back(RenderKev((*criteria)[i]));
    }
  } else
    throw std::invalid_argument("invalid " + question.type + " KEV question criteria");
  return result;
}

double Round4(double value) {
  // KEV's Python reference uses bankers rounding for exact halfway values.
  return std::nearbyint(value * 10000.0) / 10000.0;
}

OgaAnswer Answer(const OgaQuestion& question, const Candidates& candidates,
                 const std::vector<float>& probabilities, bool kev) {
  OgaAnswer result;
  result.type = question.type;
  for (size_t i = 0; i < probabilities.size(); ++i)
    result.probabilities.emplace_back(candidates.keys[i],
                                      kev ? Round4(probabilities[i]) : probabilities[i]);
  if (question.type == "noul") {
    result.noul = kev ? Round4(probabilities[1]) : probabilities[1];
    result.probabilities.clear();
    return result;
  }
  const auto selected = std::distance(probabilities.begin(),
                                      std::max_element(probabilities.begin(), probabilities.end()));
  const auto total =
      std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
  std::vector<double> normalized(probabilities.size());
  if (total == 0) {
    std::fill(normalized.begin(), normalized.end(),
              1.0 / probabilities.size());
  } else {
    std::transform(probabilities.begin(), probabilities.end(),
                   normalized.begin(),
                   [total](float value) { return value / total; });
  }
  double confidence;
  if (kev && question.type == "choice") {
    confidence = probabilities.size() == 1 ? 1 : (normalized[selected] - 1.0 / probabilities.size()) / (1.0 - 1.0 / probabilities.size());
  } else if (kev) {
    double distance = 0;
    for (size_t i = 0; i < probabilities.size(); ++i)
      distance += normalized[i] * std::abs(static_cast<double>(i) - selected);
    const auto count = probabilities.size();
    const auto midpoint = static_cast<double>(count - 1) / 2.0;
    double denominator = 0;
    for (size_t i = 0; i < count; ++i)
      denominator += std::abs(static_cast<double>(i) - midpoint);
    denominator /= count;
    confidence = count == 1 ? 1 : std::max(0.0, 1.0 - distance / denominator);
  } else {
    const auto others = std::accumulate(probabilities.begin(), probabilities.end(), 0.0) -
                        probabilities[selected];
    confidence = probabilities.size() < 2 ? 1 : std::clamp(probabilities[selected] - others / (probabilities.size() - 1), 0.0, 1.0);
  }
  result.confidence = kev ? Round4(confidence) : confidence;
  if (question.type == "choice")
    result.choice = candidates.keys[selected];
  else {
    double score = 0;
    const auto& criteria = *Get<OgaStructuredValue::Array>(question.criteria);
    for (size_t i = 0; i < probabilities.size(); ++i) {
      score += static_cast<double>(i) *
               static_cast<double>(probabilities[i]);
      result.legend.emplace_back(std::to_string(i),
                                 kev ? RenderKev(criteria[i]) : RenderClm(criteria[i]));
    }
    result.score = kev ? Round4(score) : score;
  }
  return result;
}

void AppendJsonString(std::string& output, std::string_view value) {
  static constexpr char hex[] = "0123456789abcdef";
  output.push_back('"');
  for (const unsigned char c : value) {
    switch (c) {
      case '"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (c < 0x20) {
          output += "\\u00";
          output.push_back(hex[c >> 4]);
          output.push_back(hex[c & 0xf]);
        } else {
          output.push_back(static_cast<char>(c));
        }
    }
  }
  output.push_back('"');
}

void AppendJsonMemberPrefix(std::string& output, std::string_view key,
                            bool& first) {
  if (!first) output.push_back(',');
  first = false;
  AppendJsonString(output, key);
  output.push_back(':');
}

void AppendAnswerJson(std::string& output, const OgaAnswer& answer) {
  output.push_back('{');
  bool first = true;
  AppendJsonMemberPrefix(output, "type", first);
  AppendJsonString(output, answer.type);
  if (answer.noul) {
    AppendJsonMemberPrefix(output, "noul", first);
    output += FormatPythonFloat(*answer.noul);
  }
  if (answer.choice) {
    AppendJsonMemberPrefix(output, "choice", first);
    AppendJsonString(output, *answer.choice);
  }
  if (answer.score) {
    AppendJsonMemberPrefix(output, "score", first);
    output += FormatPythonFloat(*answer.score);
  }
  if (answer.confidence) {
    AppendJsonMemberPrefix(output, "confidence", first);
    output += FormatPythonFloat(*answer.confidence);
  }
  if (!answer.probabilities.empty()) {
    AppendJsonMemberPrefix(output, "probabilities", first);
    output.push_back('{');
    bool first_probability = true;
    for (const auto& [key, probability] : answer.probabilities) {
      AppendJsonMemberPrefix(output, key, first_probability);
      output += FormatPythonFloat(probability);
    }
    output.push_back('}');
  }
  if (!answer.legend.empty()) {
    AppendJsonMemberPrefix(output, "legend", first);
    output.push_back('{');
    bool first_legend = true;
    for (const auto& [key, value] : answer.legend) {
      AppendJsonMemberPrefix(output, key, first_legend);
      AppendJsonString(output, value);
    }
    output.push_back('}');
  }
  output.push_back('}');
}

std::string SerializeJson(const OgaModelResult& result) {
  std::string output{"{\"model\":"};
  AppendJsonString(output, result.model);
  output += ",\"answers\":{";
  bool first = true;
  for (const auto& [id, answer] : result.answers) {
    AppendJsonMemberPrefix(output, id, first);
    AppendAnswerJson(output, answer);
  }
  output += "}}";
  return output;
}

std::string SerializeJson(const OgaRankingResult& result) {
  std::string output{"{\"model\":"};
  AppendJsonString(output, result.model);
  output += ",\"ranked\":[";
  for (size_t i = 0; i < result.ranked.size(); ++i) {
    if (i) output.push_back(',');
    const auto& item = result.ranked[i];
    output += "{\"rank\":" + std::to_string(item.rank) + ",\"key\":";
    AppendJsonString(output, item.key);
    output += ",\"probability\":" + FormatPythonFloat(item.probability) + "}";
  }
  output += "]}";
  return output;
}

}  // namespace

namespace {

size_t FloatVectorBytes(const std::vector<float>& value) {
  return value.size() * sizeof(float);
}

struct NativeRankingSession {
  NativeRankingSession(std::string path, std::vector<std::string> configured_providers)
      : package_path(std::move(path)), providers(std::move(configured_providers)), identity(PackageIdentity(package_path, providers)), cache(kDefaultClmCacheEntries, kDefaultClmCacheBytes, FloatVectorBytes), tokenizer(fs::path(package_path)), encoder(Component(Generators::PackageHasComponent(fs::path(package_path), "encoder") ? "encoder" : "backbone")) {
    if (Generators::PackageHasComponent(fs::path(package_path), "clm_heads")) {
      combined = std::make_unique<Generators::ComponentSession>(
          package_path, "clm_heads", providers);
    } else {
      state = std::make_unique<Generators::ComponentSession>(
          package_path, "state_head", providers);
      action = std::make_unique<Generators::ComponentSession>(
          package_path, "action_head", providers);
      scorer = std::make_unique<Generators::ComponentSession>(
          package_path, "scorer", providers);
    }
    if (Generators::PackageHasComponent(fs::path(package_path), "fused_state_ranking"))
      fused = std::make_unique<Generators::ComponentSession>(
          package_path, "fused_state_ranking", providers);
  }
  Generators::ComponentSession Component(const std::string& name) const {
    return Generators::ComponentSession(fs::path(package_path), name,
                                        providers);
  }
  std::string package_path;
  std::vector<std::string> providers;
  std::string identity;
  mutable std::mutex operation_mutex;
  SessionLruCache<std::vector<float>> cache;
  Generators::PackageTokenizer tokenizer;
  Generators::ComponentSession encoder;
  std::unique_ptr<Generators::ComponentSession> combined, state, action, scorer, fused;
  OgaModelResult Run(const OgaStructuredRequest& request);
  OgaRankingResult Rank(const OgaFreeFormRankRequest& request);
};

std::pair<std::vector<float>, size_t> EncodeAndPool(
    Generators::PackageTokenizer& tokenizer, Generators::ComponentSession& encoder,
    const std::vector<std::string>& texts) {
  if (texts.empty()) return {{}, 0};
  auto batch = Tokenize(tokenizer, texts);
  auto feeds = BackboneFeeds(encoder, batch);
  const auto hidden_name = HiddenOutputName(encoder);
  const auto hidden_tensors = encoder.Run(feeds.inputs, {hidden_name});
  const auto& hidden_tensor = FindTensor(hidden_tensors, hidden_name);
  const auto hidden_size =
      RequireHiddenShape(hidden_tensor, batch.rows, batch.width, "encoder hidden states");
  const auto hidden = FloatTensor(hidden_tensor);
  RequireFloatCount(hidden_tensor, hidden, "encoder hidden states");
  std::vector<float> pooled(batch.rows * hidden_size);
  for (size_t row = 0; row < batch.rows; ++row) {
    const auto attended = std::accumulate(batch.mask.begin() + row * batch.width,
                                          batch.mask.begin() + (row + 1) * batch.width,
                                          int64_t{});
    if (attended <= 0 || static_cast<size_t>(attended) > batch.width)
      throw std::runtime_error("encoder attention mask has no valid token or exceeds sequence width");
    const auto source =
        (row * batch.width + static_cast<size_t>(attended) - 1) * hidden_size;
    if (source > hidden.size() || hidden_size > hidden.size() - source)
      throw std::runtime_error("encoder pooled-token index exceeds hidden-state data");
    double norm = 0;
    for (size_t i = 0; i < hidden_size; ++i)
      norm += static_cast<double>(hidden[source + i]) *
              static_cast<double>(hidden[source + i]);
    norm = std::max(std::sqrt(norm), 1e-12);
    for (size_t i = 0; i < hidden_size; ++i)
      pooled[row * hidden_size + i] =
          static_cast<float>(hidden[source + i] / norm);
  }
  return {std::move(pooled), hidden_size};
}

OgaModelResult NativeRankingSession::Run(const OgaStructuredRequest& request) {
  std::lock_guard operation_lock(operation_mutex);
  if (request.questions.empty()) throw std::invalid_argument("questions must be non-empty");
  if (!(request.temperature > 0 && request.temperature <= 100))
    throw std::invalid_argument("temperature must be in (0, 100]");
  std::vector<std::string> texts;
  std::vector<Candidates> candidates;
  const auto state_text = Trim(RenderClm(request.state));
  for (const auto& [id, question] : request.questions) {
    const auto instructions = Trim(RenderClm(question.instructions));
    texts.push_back(!state_text.empty() && !instructions.empty()
                        ? state_text + "\n\n" + instructions
                        : state_text + instructions);
    candidates.push_back(ClmCandidates(question));
  }
  const auto question_count = texts.size();
  std::vector<std::string> action_texts;
  for (const auto& values : candidates)
    action_texts.insert(action_texts.end(), values.texts.begin(), values.texts.end());
  size_t expected_candidate_count = 0;
  for (const auto& value : candidates) {
    if (value.keys.size() >
        std::numeric_limits<size_t>::max() - expected_candidate_count)
      throw std::runtime_error("candidate count overflows size_t");
    expected_candidate_count += value.keys.size();
  }
  if (!expected_candidate_count)
    throw std::runtime_error("ranking request must produce at least one candidate");
  const auto candidate_count = action_texts.size();
  if (candidate_count != expected_candidate_count)
    throw std::runtime_error("candidate count does not match rendered actions");

  std::vector<std::string> state_cache_keys(question_count);
  std::vector<std::vector<float>> cached_states(question_count);
  bool all_states_cached = !combined;
  if (!combined) {
    for (size_t i = 0; i < texts.size(); ++i) {
      std::ostringstream key;
      key << identity << "|clm-state|split|float32|" << texts[i].size()
          << ':' << texts[i];
      state_cache_keys[i] = key.str();
      if (!cache.Get(state_cache_keys[i], cached_states[i]))
        all_states_cached = false;
    }
  }

  std::vector<std::string> cache_keys(candidate_count);
  std::vector<std::vector<float>> cached(candidate_count);
  std::vector<size_t> projection_source(candidate_count);
  std::vector<size_t> missing_indices;
  std::vector<std::string> missing_texts;
  std::unordered_map<std::string, size_t> pending;
  for (size_t i = 0; i < action_texts.size(); ++i) {
    projection_source[i] = i;
    std::ostringstream key;
    key << identity << "|clm-action|" << (combined ? "combined" : "split")
        << "|float32|" << action_texts[i].size() << ':' << action_texts[i];
    cache_keys[i] = key.str();
    if (!cache.Get(cache_keys[i], cached[i])) {
      if (const auto duplicate = pending.find(cache_keys[i]); duplicate != pending.end()) {
        projection_source[i] = duplicate->second;
      } else {
        pending.emplace(cache_keys[i], i);
        missing_indices.push_back(i);
        missing_texts.push_back(action_texts[i]);
      }
    }
  }
  std::vector<int64_t> owners;
  for (size_t owner = 0; owner < candidates.size(); ++owner)
    owners.insert(owners.end(), candidates[owner].keys.size(), static_cast<int64_t>(owner));
  if (owners.size() != candidate_count)
    throw std::runtime_error("candidate owner count does not match candidates");
  if (std::any_of(owners.begin(), owners.end(), [&](int64_t owner) {
        return owner < 0 || static_cast<size_t>(owner) >= question_count;
      }))
    throw std::runtime_error("candidate owner index is out of range");
  std::vector<float> probabilities;
  bool fused_scored = false;
  if (fused && !all_states_cached && missing_indices.empty()) {
    const auto projection_size = cached.front().size();
    if (!projection_size)
      throw std::runtime_error(
          "cached CLM action projection must not be empty");
    std::vector<float> action_projection(candidate_count * projection_size);
    for (size_t row = 0; row < candidate_count; ++row) {
      if (cached[row].size() != projection_size)
        throw std::runtime_error(
            "cached CLM action projection layout mismatch");
      std::copy(cached[row].begin(), cached[row].end(),
                action_projection.begin() + row * projection_size);
    }
    auto batch = BucketClmBatch(Tokenize(tokenizer, texts));
    auto feeds = BackboneFeeds(*fused, batch);
    const auto selector_info = std::find_if(
        fused->Inputs().begin(), fused->Inputs().end(),
        [](const Generators::ComponentInfo& value) {
          return value.name == "last_token_selector";
        });
    if (selector_info == fused->Inputs().end())
      throw std::runtime_error(
          "fused CLM component is missing last_token_selector metadata");
    std::vector<size_t> selected_indices(batch.rows);
    for (size_t row = 0; row < batch.rows; ++row) {
      const auto attended = std::accumulate(
          batch.mask.begin() + row * batch.width,
          batch.mask.begin() + (row + 1) * batch.width, int64_t{});
      if (attended <= 0 || static_cast<size_t>(attended) > batch.width)
        throw std::runtime_error(
            "fused CLM attention mask has no valid token");
      selected_indices[row] =
          row * batch.width + static_cast<size_t>(attended) - 1;
    }
    const std::vector<int64_t> selector_shape{
        static_cast<int64_t>(batch.rows),
        static_cast<int64_t>(batch.width), 1};
    if (selector_info->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      std::vector<uint16_t> selector(
          batch.rows * batch.width, uint16_t{});
      for (const auto index : selected_indices)
        selector[index] = uint16_t{0x3c00};
      feeds.Add("last_token_selector", selector, selector_shape,
                OgaElementType_float16);
    } else if (selector_info->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::vector<float> selector(batch.rows * batch.width, 0.0f);
      for (const auto index : selected_indices) selector[index] = 1.0f;
      feeds.Add("last_token_selector", selector, selector_shape,
                OgaElementType_float32);
    } else {
      throw std::runtime_error(
          "fused CLM last_token_selector must be float16 or float32");
    }
    feeds.Add(
        "action_projections", action_projection,
        {static_cast<int64_t>(candidate_count),
         static_cast<int64_t>(projection_size)},
        OgaElementType_float32);
    feeds.Add("temperature", std::vector<float>{request.temperature},
              {}, OgaElementType_float32);
    feeds.Add(
        "candidate_owners", owners,
        {static_cast<int64_t>(owners.size())}, OgaElementType_int64);
    const auto outputs = fused->Run(
        feeds.inputs, {"probabilities", "state/projections"});
    const auto& probability_tensor =
        FindTensor(outputs, "probabilities");
    RequireShape(probability_tensor, {candidate_count},
                 "fused CLM probabilities");
    probabilities = FloatTensor(probability_tensor);
    RequireFloatCount(probability_tensor, probabilities,
                      "fused CLM probabilities");
    const auto& state_tensor = FindTensor(outputs, "state/projections");
    RequireShape(state_tensor, {question_count, projection_size},
                 "fused CLM state projections");
    const auto state_projection = FloatTensor(state_tensor);
    RequireFloatCount(state_tensor, state_projection,
                      "fused CLM state projections");
    for (size_t row = 0; row < question_count; ++row) {
      cache.Put(
          state_cache_keys[row],
          std::vector<float>(
              state_projection.begin() + row * projection_size,
              state_projection.begin() + (row + 1) * projection_size));
    }
    fused_scored = true;
  }
  const bool fully_cached =
      !combined && all_states_cached && missing_indices.empty();
  std::vector<float> pooled;
  size_t hidden_size{};
  if (!fully_cached && !fused_scored) {
    std::tie(pooled, hidden_size) = EncodeAndPool(tokenizer, encoder, texts);
    if (!hidden_size || pooled.size() != texts.size() * hidden_size)
      throw std::runtime_error(
          "CLM pooled encoder output has invalid dimensions");
    if (!missing_texts.empty()) {
      auto [missing_pooled, missing_hidden_size] =
          EncodeAndPool(tokenizer, encoder, missing_texts);
      if (missing_hidden_size != hidden_size)
        throw std::runtime_error(
            "CLM state/action encoder layouts do not match");
      pooled.insert(pooled.end(), missing_pooled.begin(),
                    missing_pooled.end());
    }
  }
  if (fused_scored) {
    // The fused component already produced probabilities and cached state rows.
  } else if (fully_cached) {
    const auto projection_size = cached_states.front().size();
    if (!projection_size)
      throw std::runtime_error(
          "cached CLM state projection must not be empty");
    std::vector<float> state_projection(question_count * projection_size);
    for (size_t row = 0; row < question_count; ++row) {
      if (cached_states[row].size() != projection_size)
        throw std::runtime_error(
            "cached CLM state projection layout mismatch");
      std::copy(cached_states[row].begin(), cached_states[row].end(),
                state_projection.begin() + row * projection_size);
    }
    std::vector<float> action_projection(candidate_count * projection_size);
    for (size_t row = 0; row < candidate_count; ++row) {
      if (cached[row].size() != projection_size)
        throw std::runtime_error(
            "cached CLM action projection layout mismatch");
      std::copy(cached[row].begin(), cached[row].end(),
                action_projection.begin() + row * projection_size);
    }
    FeedStorage scorer_feeds;
    scorer_feeds.bytes.reserve(4);
    scorer_feeds.inputs.reserve(4);
    scorer_feeds.Add(
        "state_projections", state_projection,
        {static_cast<int64_t>(question_count),
         static_cast<int64_t>(projection_size)},
        OgaElementType_float32);
    scorer_feeds.Add(
        "action_projections", action_projection,
        {static_cast<int64_t>(candidate_count),
         static_cast<int64_t>(projection_size)},
        OgaElementType_float32);
    scorer_feeds.Add("temperature", std::vector<float>{request.temperature},
                     {}, OgaElementType_float32);
    scorer_feeds.Add(
        "candidate_owners", owners,
        {static_cast<int64_t>(owners.size())}, OgaElementType_int64);
    const auto scorer_outputs =
        scorer->Run(scorer_feeds.inputs, {"probabilities"});
    const auto& probability_tensor =
        FindTensor(scorer_outputs, "probabilities");
    RequireShape(probability_tensor, {candidate_count},
                 "scorer probabilities");
    probabilities = FloatTensor(probability_tensor);
    RequireFloatCount(probability_tensor, probabilities,
                      "scorer probabilities");
  } else if (combined) {
    FeedStorage head;
    head.bytes.reserve(2);
    head.inputs.reserve(2);
    head.Add("state_hidden_states",
             std::vector<float>(pooled.begin(), pooled.begin() + question_count * hidden_size),
             {static_cast<int64_t>(question_count), static_cast<int64_t>(hidden_size)},
             OgaElementType_float32);
    head.Add("action_hidden_states",
             std::vector<float>(pooled.begin() + question_count * hidden_size, pooled.end()),
             {static_cast<int64_t>(missing_indices.size()), static_cast<int64_t>(hidden_size)},
             OgaElementType_float32);
    auto projected = combined->Run(head.inputs);
    const auto& state_tensor = FindTensor(projected, "state_embedding");
    const auto& action_tensor = FindTensor(projected, "action_embedding");
    const auto& scale_tensor = FindTensor(projected, "effective_logit_scale");
    const auto action_shape = action_tensor.GetShape();
    if (action_shape.size() != 2 || action_shape[1] <= 0)
      throw std::runtime_error("action embedding must be a rank-2 tensor with a positive projection dimension");
    const auto projection_size = static_cast<size_t>(action_shape[1]);
    RequireShape(action_tensor, {missing_indices.size(), projection_size},
                 "action embedding");
    RequireShape(state_tensor, {question_count, projection_size},
                 "state embedding");
    RequireShape(scale_tensor, {}, "effective logit scale");
    const auto state_projection = FloatTensor(state_tensor);
    const auto missing_projection = FloatTensor(action_tensor);
    const auto scale = FloatTensor(scale_tensor);
    RequireFloatCount(state_tensor, state_projection, "state embedding");
    RequireFloatCount(action_tensor, missing_projection, "action embedding");
    RequireFloatCount(scale_tensor, scale, "effective logit scale");
    for (size_t row = 0; row < missing_indices.size(); ++row) {
      auto value = std::vector<float>(
          missing_projection.begin() + row * projection_size,
          missing_projection.begin() + (row + 1) * projection_size);
      cache.Put(cache_keys[missing_indices[row]], value);
      cached[missing_indices[row]] = std::move(value);
    }
    for (size_t row = 0; row < candidate_count; ++row)
      if (projection_source[row] != row &&
          !cache.Get(cache_keys[row], cached[row]))
        cached[row] = cached[projection_source[row]];
    std::vector<float> action_projection(candidate_count * projection_size);
    for (size_t row = 0; row < candidate_count; ++row) {
      if (cached[row].size() != projection_size)
        throw std::runtime_error("cached CLM action projection layout mismatch");
      std::copy(cached[row].begin(), cached[row].end(),
                action_projection.begin() + row * projection_size);
    }
    std::vector<float> logits(candidate_count);
    for (size_t i = 0; i < candidate_count; ++i) {
      double dot = 0;
      for (size_t j = 0; j < projection_size; ++j)
        dot +=
            static_cast<double>(
                state_projection[owners[i] * projection_size + j]) *
            static_cast<double>(action_projection[i * projection_size + j]);
      logits[i] = static_cast<float>(scale.at(0) * dot / request.temperature);
    }
    probabilities.resize(candidate_count);
    for (size_t owner = 0; owner < question_count; ++owner) {
      float maximum = -std::numeric_limits<float>::infinity();
      for (size_t i = 0; i < logits.size(); ++i)
        if (owners[i] == static_cast<int64_t>(owner)) maximum = std::max(maximum, logits[i]);
      double sum = 0;
      for (size_t i = 0; i < logits.size(); ++i)
        if (owners[i] == static_cast<int64_t>(owner)) sum += std::exp(logits[i] - maximum);
      for (size_t i = 0; i < logits.size(); ++i)
        if (owners[i] == static_cast<int64_t>(owner))
          probabilities[i] = static_cast<float>(std::exp(logits[i] - maximum) / sum);
    }
  } else {
    auto project = [](Generators::ComponentSession& session, std::vector<float> values,
                      size_t rows, size_t columns, std::string_view name) {
      FeedStorage feeds;
      feeds.bytes.reserve(1);
      feeds.inputs.reserve(1);
      feeds.Add("embeddings", values, {static_cast<int64_t>(rows), static_cast<int64_t>(columns)}, OgaElementType_float32);
      const auto tensors = session.Run(feeds.inputs, {"projections"});
      const auto& tensor = FindTensor(tensors, "projections");
      const auto tensor_shape = tensor.GetShape();
      if (tensor_shape.size() != 2 || tensor_shape[1] <= 0)
        throw std::runtime_error(std::string(name) +
                                 " must be rank 2 with a positive projection dimension");
      const auto projection_size = static_cast<size_t>(tensor_shape[1]);
      RequireShape(tensor, {rows, projection_size}, name);
      auto result = FloatTensor(tensor);
      RequireFloatCount(tensor, result, name);
      return std::pair{std::move(result), projection_size};
    };
    auto [state_projection, projection_size] = project(
        *state, {pooled.begin(), pooled.begin() + question_count * hidden_size},
        question_count, hidden_size, "state projections");
    for (size_t row = 0; row < question_count; ++row) {
      auto value = std::vector<float>(
          state_projection.begin() + row * projection_size,
          state_projection.begin() + (row + 1) * projection_size);
      cache.Put(state_cache_keys[row], value);
    }
    if (!missing_indices.empty()) {
      auto [missing_projection, action_projection_size] = project(
          *action, {pooled.begin() + question_count * hidden_size, pooled.end()},
          missing_indices.size(), hidden_size, "action projections");
      if (action_projection_size != projection_size)
        throw std::runtime_error("state and action projection dimensions do not match");
      for (size_t row = 0; row < missing_indices.size(); ++row) {
        auto value = std::vector<float>(
            missing_projection.begin() + row * projection_size,
            missing_projection.begin() + (row + 1) * projection_size);
        cache.Put(cache_keys[missing_indices[row]], value);
        cached[missing_indices[row]] = std::move(value);
      }
    }
    for (size_t row = 0; row < candidate_count; ++row)
      if (projection_source[row] != row &&
          !cache.Get(cache_keys[row], cached[row]))
        cached[row] = cached[projection_source[row]];
    std::vector<float> action_projection(candidate_count * projection_size);
    for (size_t row = 0; row < candidate_count; ++row) {
      if (cached[row].size() != projection_size)
        throw std::runtime_error("cached CLM action projection layout mismatch");
      std::copy(cached[row].begin(), cached[row].end(),
                action_projection.begin() + row * projection_size);
    }
    FeedStorage scorer_feeds;
    scorer_feeds.bytes.reserve(4);
    scorer_feeds.inputs.reserve(4);
    scorer_feeds.Add(
        "state_projections", state_projection,
        {static_cast<int64_t>(question_count),
         static_cast<int64_t>(projection_size)},
        OgaElementType_float32);
    scorer_feeds.Add(
        "action_projections", action_projection,
        {static_cast<int64_t>(candidate_count),
         static_cast<int64_t>(projection_size)},
        OgaElementType_float32);
    scorer_feeds.Add("temperature", std::vector<float>{request.temperature},
                     {}, OgaElementType_float32);
    scorer_feeds.Add(
        "candidate_owners", owners,
        {static_cast<int64_t>(owners.size())}, OgaElementType_int64);
    const auto scorer_outputs =
        scorer->Run(scorer_feeds.inputs, {"probabilities"});
    const auto& probability_tensor = FindTensor(scorer_outputs, "probabilities");
    RequireShape(probability_tensor, {candidate_count}, "scorer probabilities");
    probabilities = FloatTensor(probability_tensor);
    RequireFloatCount(probability_tensor, probabilities, "scorer probabilities");
  }
  if (probabilities.size() != candidate_count)
    throw std::runtime_error("probability count does not match candidate count");
  OgaModelResult result{"clm", {}};
  size_t offset = 0;
  for (size_t i = 0; i < request.questions.size(); ++i) {
    const auto size = candidates[i].keys.size();
    result.answers.emplace_back(request.questions[i].first,
                                Answer(request.questions[i].second, candidates[i],
                                       {probabilities.begin() + offset, probabilities.begin() + offset + size}, false));
    offset += size;
  }
  if (offset != probabilities.size())
    throw std::runtime_error("consumed probability count does not match scorer output");
  return result;
}

OgaRankingResult NativeRankingSession::Rank(const OgaFreeFormRankRequest& request) {
  OgaStructuredRequest translated;
  translated.state = request.state;
  translated.temperature = request.temperature;
  OgaQuestion question{"choice", request.instructions, OgaStructuredValue::Object{}};
  auto& criteria = std::get<OgaStructuredValue::Object>(question.criteria.value);
  for (const auto& [key, value] : request.candidates) criteria.emplace_back(key, value);
  translated.questions.emplace_back("rank", std::move(question));
  const auto answer = Run(translated).answers.front().second;
  if (answer.probabilities.size() != request.candidates.size())
    throw std::runtime_error("ranking probability count does not match candidates");
  std::vector<size_t> order(request.candidates.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
    return answer.probabilities[left].second > answer.probabilities[right].second;
  });
  OgaRankingResult result{"clm", {}};
  for (size_t rank = 0; rank < order.size(); ++rank) {
    const auto index = order[rank];
    result.ranked.push_back({rank + 1, request.candidates[index].first,
                             answer.probabilities[index].second});
  }
  return result;
}

struct CachedKevTokens {
  std::vector<int64_t> tokens;
  std::vector<int64_t> option_indices;
};

size_t KevTokenBytes(const CachedKevTokens& value) {
  return (value.tokens.size() + value.option_indices.size()) * sizeof(int64_t);
}

struct KevStateBinding {
  const Generators::ComponentInfo* input{};
  std::string output;
  size_t batch_axis{};
  std::optional<size_t> growing_axis;
};

struct CachedKevPrefix {
  size_t length{};
  std::vector<std::shared_ptr<Generators::Tensor>> states;
};

size_t KevPrefixBytes(const CachedKevPrefix& value) {
  size_t bytes = sizeof(value.length);
  for (const auto& state : value.states) {
    if (!state) continue;
    bytes += state->GetElementCount() * Ort::SizeOf(state->GetType()) +
             state->GetShape().size() * sizeof(int64_t);
  }
  return bytes;
}

std::string StateSuffix(std::string_view name) {
  static constexpr std::string_view prefixes[] = {
      "past_key_values.", "present_key_values.", "past.", "present."};
  for (const auto prefix : prefixes)
    if (name.rfind(prefix, 0) == 0) return std::string(name.substr(prefix.size()));
  return {};
}

std::vector<KevStateBinding> DiscoverKevStateBindings(
    const Generators::ComponentSession& backbone, std::string& reason) {
  std::vector<KevStateBinding> result;
  for (const auto& input : backbone.Inputs()) {
    if (input.name.rfind("past_key_values.", 0) != 0 &&
        input.name.rfind("past.", 0) != 0)
      continue;
    const auto suffix = StateSuffix(input.name);
    std::vector<std::string> matches;
    for (const auto& output : backbone.OutputNames())
      if (StateSuffix(output) == suffix) matches.push_back(output);
    if (matches.size() != 1) {
      reason = matches.empty() ? "missing present output for " + input.name
                               : "ambiguous present outputs for " + input.name;
      return {};
    }
    if (input.shape.empty() || input.symbolic_dimensions.size() != input.shape.size()) {
      reason = "state input lacks complete shape metadata: " + input.name;
      return {};
    }
    std::vector<size_t> batch_axes;
    std::vector<size_t> growing_axes;
    for (size_t axis = 0; axis < input.symbolic_dimensions.size(); ++axis) {
      const auto& symbol = input.symbolic_dimensions[axis];
      if (symbol.find("batch") != std::string::npos)
        batch_axes.push_back(axis);
      if (symbol.find("past") != std::string::npos ||
          symbol.find("sequence") != std::string::npos)
        growing_axes.push_back(axis);
    }
    if (batch_axes.size() != 1) {
      reason = "state input must declare exactly one batch axis: " + input.name;
      return {};
    }
    const auto tail = suffix.substr(suffix.rfind('.') + 1);
    const bool growing = tail == "key" || tail == "value";
    if (growing && growing_axes.size() != 1) {
      reason = "growing state input must declare exactly one sequence axis: " +
               input.name;
      return {};
    }
    if (!growing && !growing_axes.empty()) {
      reason = "fixed state input unexpectedly declares a sequence axis: " +
               input.name;
      return {};
    }
    result.push_back(
        {&input, std::move(matches.front()), batch_axes.front(),
         growing ? std::optional<size_t>(growing_axes.front()) : std::nullopt});
  }
  if (result.empty()) {
    reason = "backbone has no explicit past_key_values inputs";
    return {};
  }
  if (std::find(backbone.InputNames().begin(), backbone.InputNames().end(),
                "position_ids") == backbone.InputNames().end()) {
    reason = "stateful backbone has no position_ids input";
    return {};
  }
  reason = "compatible explicit state I/O";
  return result;
}

void ValidatePrefixState(const KevStateBinding& binding,
                         const Generators::Tensor& state) {
  const auto& input = *binding.input;
  const auto shape = state.GetShape();
  if (state.GetType() != input.type)
    throw std::runtime_error("state dtype mismatch for " + input.name);
  if (shape.size() != input.shape.size() || shape.empty())
    throw std::runtime_error("state rank mismatch for " + input.name);
  if (binding.batch_axis >= shape.size() ||
      shape[binding.batch_axis] != 1)
    throw std::runtime_error(
        "prefix state output must have batch dimension 1: " + binding.output);
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i == binding.batch_axis) continue;
    if (shape[i] < 0)
      throw std::runtime_error("prefix state output has a negative dimension: " +
                               binding.output);
    if (input.shape[i] >= 0 &&
        (!binding.growing_axis || i != *binding.growing_axis) &&
        input.shape[i] != shape[i])
      throw std::runtime_error("state dimension mismatch between " + input.name +
                               " and " + binding.output);
  }
  if (TensorElementCount(state, binding.output) != state.GetElementCount())
    throw std::runtime_error("state element count does not match shape: " +
                             binding.output);
}

void AddRepeatedState(FeedStorage& feeds, const KevStateBinding& binding,
                      const std::shared_ptr<Generators::Tensor>& state,
                      size_t batch) {
  if (!state) throw std::runtime_error("cached state tensor is null");
  ValidatePrefixState(binding, *state);
  if (batch != 1)
    throw std::runtime_error(
        "cached state reuse currently requires branch batch size one");
  feeds.inputs.emplace(binding.input->name, state);
}

struct NativeDecisionSession {
  NativeDecisionSession(std::string path, std::vector<std::string> configured_providers)
      : package_path(std::move(path)), providers(std::move(configured_providers)), identity(PackageIdentity(package_path, providers)), cache(kDefaultKevCacheEntries, kDefaultKevCacheBytes, KevTokenBytes), prefix_cache(kDefaultKevPrefixCacheEntries, kDefaultKevPrefixCacheBytes, KevPrefixBytes), tokenizer(fs::path(package_path)), backbone(Component("backbone")) {
    const auto pointer_name =
        Generators::PackageHasComponent(fs::path(package_path), "pointer_head")
            ? "pointer_head"
            : "kev_head";
    pointer = std::make_unique<Generators::ComponentSession>(
        package_path, pointer_name, providers);
    flat = pointer_name == std::string_view("kev_head");
    std::string compatibility_reason;
    state_bindings =
        DiscoverKevStateBindings(backbone, compatibility_reason);
  }
  Generators::ComponentSession Component(const std::string& name) const {
    return Generators::ComponentSession(fs::path(package_path), name,
                                        providers);
  }
  std::string package_path;
  std::vector<std::string> providers;
  std::string identity;
  mutable std::mutex operation_mutex;
  SessionLruCache<CachedKevTokens> cache;
  SessionLruCache<CachedKevPrefix> prefix_cache;
  Generators::PackageTokenizer tokenizer;
  Generators::ComponentSession backbone;
  std::unique_ptr<Generators::ComponentSession> pointer;
  std::vector<KevStateBinding> state_bindings;
  bool flat{};
  OgaModelResult Decide(const OgaStructuredRequest& request);
};

OgaModelResult NativeDecisionSession::Decide(const OgaStructuredRequest& request) {
  std::lock_guard operation_lock(operation_mutex);
  if (request.questions.empty()) throw std::invalid_argument("questions must be non-empty");
  if (request.temperature != 1.0f)
    throw std::invalid_argument(
        "temperature is supported only by ranking sessions");
  auto encode = [&](std::string text) {
    static const std::regex delimiter(R"(<\|([A-Za-z0-9_]+)\|>)");
    text = std::regex_replace(text, delimiter, "<¦$1¦>");
    const auto tokens = tokenizer.Encode(text);
    return std::vector<int64_t>(tokens.begin(), tokens.end());
  };
  const auto max_context = tokenizer.ContextLength();
  if (max_context < 2)
    throw std::runtime_error("KEV context length must be at least 2");
  const auto rendered_state = RenderKev(request.state);
  const auto state_key = identity + "|kev-prefix|" + rendered_state;
  CachedKevTokens state_value;
  if (!cache.Get(state_key, state_value)) {
    state_value.tokens = {kStateToken};
    auto state_tokens = encode(rendered_state);
    if (state_tokens.size() > max_context - 1)
      state_tokens.resize(max_context - 1);
    state_value.tokens.insert(state_value.tokens.end(), state_tokens.begin(),
                              state_tokens.end());
    cache.Put(state_key, state_value);
  }
  std::vector<std::vector<int64_t>> rows, option_indices;
  std::vector<Candidates> candidates;
  for (const auto& [id, question] : request.questions) {
    candidates.push_back(KevCandidates(question));
    const auto rendered_instruction = RenderKev(question.instructions);
    std::ostringstream row_key_builder;
    row_key_builder << identity << "|kev-row|" << rendered_state.size() << ':'
                    << rendered_state << '|' << question.type << '|'
                    << rendered_instruction.size() << ':' << rendered_instruction;
    for (const auto& option : candidates.back().texts)
      row_key_builder << '|' << option.size() << ':' << option;
    const auto row_key = row_key_builder.str();
    CachedKevTokens row_value;
    if (cache.Get(row_key, row_value)) {
      if (row_value.tokens.size() > max_context ||
          state_value.tokens.size() >
              max_context - row_value.tokens.size())
        throw std::invalid_argument(
            "state+question row exceeds configured context length " +
            std::to_string(max_context) + ": " +
            std::to_string(state_value.tokens.size() +
                           row_value.tokens.size()));
      rows.push_back(std::move(row_value.tokens));
      option_indices.push_back(std::move(row_value.option_indices));
      continue;
    }
    std::vector<int64_t> row{kQuestionToken};
    auto instruction_tokens = encode(rendered_instruction);
    row.insert(row.end(), instruction_tokens.begin(), instruction_tokens.end());
    std::vector<int64_t> indices;
    for (const auto& option : candidates.back().texts) {
      row.push_back(kOptionStartToken);
      auto option_tokens = encode(option);
      row.insert(row.end(), option_tokens.begin(), option_tokens.end());
      row.push_back(kOptionEndToken);
      indices.push_back(static_cast<int64_t>(row.size() - 1));
    }
    row.push_back(kDecideToken);
    if (row.size() > max_context)
      throw std::invalid_argument(
          "state+question row exceeds configured context length " +
          std::to_string(max_context) + ": " + std::to_string(row.size()));
    if (state_value.tokens.size() > max_context - row.size())
      throw std::invalid_argument(
          "state+question row exceeds configured context length " +
          std::to_string(max_context) + ": " +
          std::to_string(state_value.tokens.size() + row.size()));
    cache.Put(row_key, {row, indices});
    rows.push_back(std::move(row));
    option_indices.push_back(std::move(indices));
  }
  // Component outputs are currently host-owned, so copying and repeating CUDA
  // state tensors costs more than recomputing a short prefix.
  const bool use_prefix_reuse =
      !state_bindings.empty() && !backbone.UsesCuda() && rows.size() == 1;
  if (!use_prefix_reuse) {
    for (size_t i = 0; i < rows.size(); ++i) {
      auto& row = rows[i];
      row.insert(row.begin(), state_value.tokens.begin(), state_value.tokens.end());
      for (auto& index : option_indices[i])
        index += static_cast<int64_t>(state_value.tokens.size());
    }
  }
  TokenBatch batch;
  batch.rows = rows.size();
  if (batch.rows != request.questions.size() || batch.rows != option_indices.size())
    throw std::runtime_error("KEV row count does not match questions");
  batch.width = std::max_element(rows.begin(), rows.end(),
                                 [](const auto& left, const auto& right) { return left.size() < right.size(); })
                    ->size();
  if (!batch.width)
    throw std::runtime_error("KEV token batch width must be positive");
  if (batch.rows > std::numeric_limits<size_t>::max() / batch.width)
    throw std::runtime_error("KEV token batch element count overflows size_t");
  batch.ids.assign(batch.rows * batch.width, tokenizer.PadTokenId());
  batch.mask.assign(batch.ids.size(), 0);
  for (size_t row = 0; row < rows.size(); ++row) {
    std::copy(rows[row].begin(), rows[row].end(), batch.ids.begin() + row * batch.width);
    std::fill_n(batch.mask.begin() + row * batch.width, rows[row].size(), 1);
  }
  const auto hidden_name = HiddenOutputName(backbone);
  std::shared_ptr<Generators::Tensor> hidden_tensor;
  if (use_prefix_reuse) {
    std::string prefix_key = identity + "|kev-model-prefix|" +
                             std::to_string(state_value.tokens.size()) + ":";
    prefix_key.append(
        reinterpret_cast<const char*>(state_value.tokens.data()),
        state_value.tokens.size() * sizeof(state_value.tokens.front()));
    CachedKevPrefix prefix;
    if (!prefix_cache.Get(prefix_key, prefix)) {
      TokenBatch prefix_batch;
      prefix_batch.rows = 1;
      prefix_batch.width = state_value.tokens.size();
      prefix_batch.ids = state_value.tokens;
      prefix_batch.mask.assign(prefix_batch.width, 1);
      auto prefix_feeds = BackboneFeeds(backbone, prefix_batch);
      std::vector<std::string> outputs;
      outputs.reserve(state_bindings.size() + 1);
      outputs.push_back(hidden_name);
      for (const auto& binding : state_bindings) outputs.push_back(binding.output);
      const auto tensors = backbone.Run(prefix_feeds.inputs, outputs);
      prefix.length = state_value.tokens.size();
      prefix.states.reserve(state_bindings.size());
      for (const auto& binding : state_bindings) {
        const auto state = tensors.at(binding.output);
        ValidatePrefixState(binding, *state);
        prefix.states.push_back(state);
      }
      prefix_cache.Put(prefix_key, prefix);
    }
    if (prefix.length != state_value.tokens.size() ||
        prefix.states.size() != state_bindings.size())
      throw std::runtime_error("cached KEV prefix state layout mismatch");

    FeedStorage branch_feeds;
    branch_feeds.bytes.reserve(3 + state_bindings.size());
    branch_feeds.inputs.reserve(3 + state_bindings.size());
    const std::vector<int64_t> token_shape{static_cast<int64_t>(batch.rows),
                                           static_cast<int64_t>(batch.width)};
    branch_feeds.Add("input_ids", batch.ids, token_shape, OgaElementType_int64);
    std::vector<int64_t> attention(
        batch.rows * (prefix.length + batch.width), 0);
    std::vector<int64_t> positions(batch.rows * batch.width, 0);
    for (size_t row = 0; row < batch.rows; ++row) {
      std::fill_n(attention.begin() + row * (prefix.length + batch.width),
                  prefix.length, 1);
      for (size_t column = 0; column < batch.width; ++column) {
        const auto token_index = row * batch.width + column;
        if (!batch.mask[token_index]) continue;
        attention[row * (prefix.length + batch.width) + prefix.length + column] = 1;
        positions[token_index] = static_cast<int64_t>(prefix.length + column);
      }
    }
    branch_feeds.Add(
        "attention_mask", attention,
        {static_cast<int64_t>(batch.rows),
         static_cast<int64_t>(prefix.length + batch.width)},
        OgaElementType_int64);
    AddPositionIds(branch_feeds, backbone, positions, batch.rows, batch.width);
    for (size_t i = 0; i < state_bindings.size(); ++i)
      AddRepeatedState(branch_feeds, state_bindings[i], prefix.states[i], batch.rows);
    auto hidden_outputs =
        backbone.Run(branch_feeds.inputs, {hidden_name});
    hidden_tensor = hidden_outputs.at(hidden_name);
  } else {
    auto feeds = BackboneFeeds(backbone, batch);
    auto hidden_outputs = backbone.Run(feeds.inputs, {hidden_name});
    hidden_tensor = hidden_outputs.at(hidden_name);
  }
  const auto hidden_size =
      RequireHiddenShape(*hidden_tensor, batch.rows, batch.width,
                         "backbone hidden states");
  const auto hidden = FloatTensor(*hidden_tensor);
  RequireFloatCount(*hidden_tensor, hidden, "backbone hidden states");
  std::vector<int64_t> owners;
  for (size_t i = 0; i < option_indices.size(); ++i)
    owners.insert(owners.end(), option_indices[i].size(), static_cast<int64_t>(i));
  const auto expected_probability_count = owners.size();
  if (!expected_probability_count)
    throw std::runtime_error("KEV request must produce at least one option");
  if (std::any_of(owners.begin(), owners.end(), [&](int64_t owner) {
        return owner < 0 || static_cast<size_t>(owner) >= rows.size();
      }))
    throw std::runtime_error("KEV option owner index is out of range");
  for (size_t row = 0; row < option_indices.size(); ++row) {
    if (option_indices[row].size() != candidates[row].keys.size())
      throw std::runtime_error("KEV option index count does not match candidates");
    for (const auto index : option_indices[row])
      if (index < 0 || static_cast<size_t>(index) >= rows[row].size() - 1)
        throw std::runtime_error("KEV option index is out of range");
  }
  FeedStorage head;
  head.bytes.reserve(5);
  head.inputs.reserve(5);
  std::vector<float> probabilities;
  if (flat) {
    const auto hidden_input = std::find_if(
        pointer->Inputs().begin(), pointer->Inputs().end(),
        [](const Generators::ComponentInfo& info) { return info.name == "hidden_states"; });
    if (hidden_input == pointer->Inputs().end() ||
        (hidden_input->shape.size() != 2 && hidden_input->shape.size() != 3))
      throw std::runtime_error(
          "kev_head hidden_states must have rank 2 or rank 3");
    const bool flattened_hidden = hidden_input->shape.size() == 2;
    std::vector<int64_t> decide_indices;
    size_t max_options = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
      const auto row_offset = flattened_hidden ? i * batch.width : 0;
      decide_indices.push_back(
          static_cast<int64_t>(row_offset + rows[i].size() - 1));
      max_options = std::max(max_options, option_indices[i].size());
    }
    if (!max_options)
      throw std::runtime_error("KEV maximum option count must be positive");
    std::vector<int64_t> padded(rows.size() * max_options);
    std::vector<uint8_t> mask(rows.size() * max_options);
    for (size_t i = 0; i < rows.size(); ++i)
      for (size_t j = 0; j < option_indices[i].size(); ++j) {
        const auto row_offset = flattened_hidden ? i * batch.width : 0;
        padded[i * max_options + j] =
            option_indices[i][j] + static_cast<int64_t>(row_offset);
        mask[i * max_options + j] = 1;
      }
    auto hidden_shape = flattened_hidden
                            ? std::vector<int64_t>{
                                  static_cast<int64_t>(batch.rows * batch.width),
                                  static_cast<int64_t>(hidden_size)}
                            : std::vector<int64_t>{static_cast<int64_t>(batch.rows), static_cast<int64_t>(batch.width), static_cast<int64_t>(hidden_size)};
    head.Add("hidden_states", hidden, std::move(hidden_shape),
             OgaElementType_float32);
    head.Add("decide_indices", decide_indices, {static_cast<int64_t>(rows.size())},
             OgaElementType_int64);
    head.Add("option_indices", padded,
             {static_cast<int64_t>(rows.size()), static_cast<int64_t>(max_options)},
             OgaElementType_int64);
    head.Add("option_mask", mask,
             {static_cast<int64_t>(rows.size()), static_cast<int64_t>(max_options)},
             OgaElementType_bool);
    const auto pointer_outputs = pointer->Run(head.inputs, {"probabilities"});
    const auto& probability_tensor = FindTensor(pointer_outputs, "probabilities");
    RequireShape(probability_tensor, {rows.size(), max_options},
                 "KEV probability matrix");
    const auto matrix = FloatTensor(probability_tensor);
    RequireFloatCount(probability_tensor, matrix, "KEV probability matrix");
    for (size_t i = 0; i < rows.size(); ++i)
      for (size_t j = 0; j < option_indices[i].size(); ++j)
        probabilities.push_back(matrix[i * max_options + j]);
  } else {
    std::vector<int64_t> decide_indices, indices;
    for (size_t i = 0; i < rows.size(); ++i) {
      decide_indices.push_back(static_cast<int64_t>(rows[i].size() - 1));
      indices.insert(indices.end(), option_indices[i].begin(), option_indices[i].end());
    }
    head.Add("hidden_states", hidden,
             {static_cast<int64_t>(batch.rows), static_cast<int64_t>(batch.width),
              static_cast<int64_t>(hidden_size)},
             OgaElementType_float32);
    head.Add("decide_indices", decide_indices, {static_cast<int64_t>(rows.size())},
             OgaElementType_int64);
    head.Add("option_indices", indices, {static_cast<int64_t>(indices.size())},
             OgaElementType_int64);
    head.Add("option_owners", owners, {static_cast<int64_t>(owners.size())},
             OgaElementType_int64);
    const auto pointer_outputs = pointer->Run(head.inputs, {"probabilities"});
    const auto& probability_tensor = FindTensor(pointer_outputs, "probabilities");
    RequireShape(probability_tensor, {expected_probability_count},
                 "KEV probabilities");
    probabilities = FloatTensor(probability_tensor);
    RequireFloatCount(probability_tensor, probabilities, "KEV probabilities");
  }
  if (probabilities.size() != expected_probability_count)
    throw std::runtime_error("KEV probability count does not match options");
  OgaModelResult result{"kev", {}};
  size_t offset = 0;
  for (size_t i = 0; i < request.questions.size(); ++i) {
    const auto size = candidates[i].keys.size();
    result.answers.emplace_back(request.questions[i].first,
                                Answer(request.questions[i].second, candidates[i],
                                       {probabilities.begin() + offset, probabilities.begin() + offset + size}, true));
    offset += size;
  }
  if (offset != probabilities.size())
    throw std::runtime_error("consumed KEV probability count does not match pointer output");
  return result;
}

}  // namespace

namespace {

template <class T>
T& CapiRequired(T* value, const char* name) {
  if (!value) throw std::invalid_argument(std::string(name) + " must not be null");
  return *value;
}

const char* CapiRequired(const char* value, const char* name) {
  if (!value) throw std::invalid_argument(std::string(name) + " must not be null");
  return value;
}

std::vector<std::string> CapiStrings(const char* const* values, size_t count,
                                     const char* name) {
  if (count && !values) throw std::invalid_argument(std::string(name) + " must not be null");
  std::vector<std::string> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i)
    result.emplace_back(CapiRequired(values[i], name));
  return result;
}

}  // namespace

struct OGA_CPP_ONLY OgaNonGenerativeSession {
  using Runtime = std::variant<std::unique_ptr<NativeRankingSession>,
                               std::unique_ptr<NativeDecisionSession>>;

  OgaNonGenerativeSession(std::string path,
                          std::vector<std::string> configured_providers) {
    const fs::path package(path);
    const bool decision =
        Generators::PackageHasComponent(package, "pointer_head") ||
        Generators::PackageHasComponent(package, "kev_head");
    const bool ranking =
        Generators::PackageHasComponent(package, "clm_heads") ||
        Generators::PackageHasComponent(package, "state_head");
    if (decision == ranking)
      throw std::invalid_argument(
          "package must declare exactly one ranking or decision workflow");
    if (ranking)
      runtime = std::make_unique<NativeRankingSession>(
          std::move(path), std::move(configured_providers));
    else
      runtime = std::make_unique<NativeDecisionSession>(
          std::move(path), std::move(configured_providers));
  }

  Runtime runtime;
};

extern "C" {

OgaResult* OGA_API_CALL OgaCreateNonGenerativeSession(
    const char* package_path, const char* const* providers, size_t provider_count,
    OgaNonGenerativeSession** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  auto provider_values = CapiStrings(providers, provider_count, "providers");
  auto result = std::make_unique<OgaNonGenerativeSession>(
      CapiRequired(package_path, "package_path"), std::move(provider_values));
  output = result.release();
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyNonGenerativeSession(OgaNonGenerativeSession* session) {
  delete session;
}

OgaResult* OGA_API_CALL OgaNonGenerativeSessionRunJson(
    OgaNonGenerativeSession* session, const char* request,
    size_t request_length, const char** response) {
  OGA_CAPI_TRY
  auto& value = CapiRequired(session, "session");
  const auto json = std::string_view(CapiRequired(request, "request"),
                                     request_length);
  std::string result;
  if (auto* ranking =
          std::get_if<std::unique_ptr<NativeRankingSession>>(&value.runtime)) {
    result = SerializeJson((*ranking)->Rank(ParseRankingRequestJson(json)));
  } else {
    result = SerializeJson(
        std::get<std::unique_ptr<NativeDecisionSession>>(value.runtime)
            ->Decide(ParseDecisionRequestJson(json)));
  }
  auto output = std::make_unique<char[]>(result.size() + 1);
  std::memcpy(output.get(), result.c_str(), result.size() + 1);
  CapiRequired(response, "response") = output.release();
  return nullptr;
  OGA_CAPI_CATCH
}

}  // extern "C"
