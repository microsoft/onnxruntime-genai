// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../ort_genai.h"
#include "../ort_genai_c_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

constexpr size_t kDefaultClmCacheEntries = 256;
constexpr size_t kDefaultClmCacheBytes = 64 * 1024 * 1024;
constexpr size_t kDefaultKevCacheEntries = 512;
constexpr size_t kDefaultKevCacheBytes = 16 * 1024 * 1024;
constexpr size_t kDefaultKevPrefixCacheEntries = 32;
constexpr size_t kDefaultKevPrefixCacheBytes = 512 * 1024 * 1024;
constexpr size_t kMinCudaKevPrefixTokens = 128;

bool UsesCuda(const std::vector<std::string>& providers) {
  return std::find(providers.begin(), providers.end(), "cuda") != providers.end() ||
         std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") !=
             providers.end();
}

std::string PackageIdentity(const std::string& package_path,
                            const std::vector<std::string>& providers) {
  namespace stdfs = std::filesystem;
  std::error_code error;
  auto canonical = stdfs::weakly_canonical(stdfs::path(package_path), error);
  if (error) canonical = stdfs::absolute(stdfs::path(package_path), error);
  std::ostringstream result;
  result << canonical.generic_string();
  for (const auto& provider : providers) result << "|provider=" << provider;
  static constexpr std::string_view files[] = {
      "component_manifest.json", "tokenizer.json", "tokenizer_config.json",
      "encoder/model.onnx", "backbone/model.onnx", "state_head/model.onnx",
      "action_head/model.onnx", "scorer/model.onnx", "clm_heads/model.onnx",
      "pointer_head/model.onnx", "kev_head/model.onnx",
      "encoder/model.onnx.data", "backbone/model.onnx.data",
      "state_head/model.onnx.data", "action_head/model.onnx.data",
      "scorer/model.onnx.data", "clm_heads/model.onnx.data",
      "pointer_head/model.onnx.data", "kev_head/model.onnx.data"};
  for (const auto file : files) {
    const auto path = canonical / file;
    if (!stdfs::exists(path, error) || error) {
      error.clear();
      continue;
    }
    const auto size = stdfs::file_size(path, error);
    if (error) {
      error.clear();
      continue;
    }
    const auto modified = stdfs::last_write_time(path, error);
    if (error) {
      error.clear();
      continue;
    }
    const auto modified_milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            modified.time_since_epoch())
            .count();
    result << '|' << file << ':' << size << ':'
           << static_cast<long long>(modified_milliseconds);
  }
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
    if (found == index_.end()) {
      ++misses_;
      return false;
    }
    entries_.splice(entries_.begin(), entries_, found->second);
    value = found->second->value;
    ++hits_;
    return true;
  }

  void Put(std::string key, Value value) {
    std::lock_guard lock(mutex_);
    if (!entry_capacity_ || !byte_capacity_) return;
    const auto bytes = key.size() + size_(value);
    if (bytes > byte_capacity_) return;
    if (const auto found = index_.find(key); found != index_.end()) {
      byte_size_ -= found->second->bytes;
      entries_.erase(found->second);
      index_.erase(found);
    }
    entries_.push_front({std::move(key), std::move(value), bytes});
    index_[entries_.front().key] = entries_.begin();
    byte_size_ += bytes;
    Evict();
  }

  void SetCapacity(size_t entries, size_t bytes) {
    std::lock_guard lock(mutex_);
    entry_capacity_ = entries;
    byte_capacity_ = bytes;
    Evict();
  }

  void Clear() {
    std::lock_guard lock(mutex_);
    entries_.clear();
    index_.clear();
    byte_size_ = 0;
  }

  OgaNonGenerativeCacheStats Stats() const {
    std::lock_guard lock(mutex_);
    return {hits_, misses_, evictions_, entries_.size(), byte_size_,
            entry_capacity_, byte_capacity_};
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
      ++evictions_;
    }
  }
  mutable std::mutex mutex_;
  std::list<Entry> entries_;
  std::unordered_map<std::string, typename std::list<Entry>::iterator> index_;
  size_t entry_capacity_{};
  size_t byte_capacity_{};
  size_t byte_size_{};
  uint64_t hits_{};
  uint64_t misses_{};
  uint64_t evictions_{};
  SizeFunction size_;
};

constexpr int64_t kStateToken = 248060;
constexpr int64_t kQuestionToken = 248061;
constexpr int64_t kOptionStartToken = 248049;
constexpr int64_t kOptionEndToken = 248050;
constexpr int64_t kDecideToken = 248062;

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

std::string Scalar(const OgaStructuredValue& value, bool kev) {
  if (std::holds_alternative<std::monostate>(value.value)) return {};
  if (const auto* text = Get<std::string>(value)) return *text;
  if (const auto* boolean = Get<bool>(value))
    return *boolean ? (kev ? "True" : "true") : (kev ? "False" : "false");
  if (const auto* integer = Get<int64_t>(value)) return std::to_string(*integer);
  if (const auto* number = Get<double>(value)) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<double>::max_digits10)
           << *number;
    if (!stream)
      throw std::invalid_argument(
          "structured floating-point value cannot be rendered");
    return stream.str();
  }
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

size_t ElementSize(OgaElementType type) {
  switch (type) {
    case OgaElementType_bool:
    case OgaElementType_int8:
    case OgaElementType_uint8:
      return 1;
    case OgaElementType_float16:
    case OgaElementType_bfloat16:
    case OgaElementType_int16:
    case OgaElementType_uint16:
      return 2;
    case OgaElementType_float32:
    case OgaElementType_int32:
    case OgaElementType_uint32:
      return 4;
    case OgaElementType_float64:
    case OgaElementType_int64:
    case OgaElementType_uint64:
      return 8;
    default:
      throw std::invalid_argument("unsupported component tensor element type");
  }
}

const OgaComponentTensor& FindTensor(const std::vector<OgaComponentTensor>& tensors,
                                     const std::string& name) {
  const auto found = std::find_if(tensors.begin(), tensors.end(),
                                  [&](const auto& tensor) { return tensor.name == name; });
  if (found == tensors.end()) throw std::runtime_error("component did not return output: " + name);
  return *found;
}

template <typename T>
std::vector<T> CopyTensor(const OgaComponentTensor& tensor, OgaElementType expected) {
  if (tensor.type != expected || tensor.data.size() % sizeof(T))
    throw std::runtime_error("component output has unexpected element type");
  std::vector<T> result(tensor.data.size() / sizeof(T));
  if (!tensor.data.empty())
    std::memcpy(result.data(), tensor.data.data(), tensor.data.size());
  return result;
}

std::vector<float> FloatTensor(const OgaComponentTensor& tensor) {
  if (tensor.type == OgaElementType_float32)
    return CopyTensor<float>(tensor, OgaElementType_float32);
  auto bits = CopyTensor<uint16_t>(
      tensor, tensor.type == OgaElementType_float16 ? OgaElementType_float16
                                                    : OgaElementType_bfloat16);
  std::vector<float> result(bits.size());
  if (tensor.type == OgaElementType_bfloat16) {
    for (size_t i = 0; i < bits.size(); ++i) {
      uint32_t expanded = static_cast<uint32_t>(bits[i]) << 16;
      std::memcpy(&result[i], &expanded, sizeof(float));
    }
    return result;
  }

  if (tensor.type != OgaElementType_float16)
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

size_t TensorElementCount(const OgaComponentTensor& tensor, std::string_view name) {
  size_t count = 1;
  for (const auto dimension : tensor.shape) {
    if (dimension < 0)
      throw std::runtime_error(std::string(name) + " has a negative output dimension");
    const auto value = static_cast<size_t>(dimension);
    if (value && count > std::numeric_limits<size_t>::max() / value)
      throw std::runtime_error(std::string(name) + " element count overflows size_t");
    count *= value;
  }
  return count;
}

void RequireShape(const OgaComponentTensor& tensor,
                  std::initializer_list<size_t> dimensions,
                  std::string_view name) {
  if (tensor.shape.size() != dimensions.size())
    throw std::runtime_error(std::string(name) + " must have rank " +
                             std::to_string(dimensions.size()));
  size_t index = 0;
  for (const auto expected : dimensions) {
    if (tensor.shape[index] < 0 ||
        static_cast<size_t>(tensor.shape[index]) != expected)
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

size_t RequireHiddenShape(const OgaComponentTensor& tensor, size_t rows,
                          size_t width, std::string_view name) {
  if (tensor.shape.size() != 3)
    throw std::runtime_error(std::string(name) + " must have rank 3");
  if (tensor.shape[0] < 0 || static_cast<size_t>(tensor.shape[0]) != rows)
    throw std::runtime_error(std::string(name) + " row count does not match the token batch");
  if (tensor.shape[1] < 0 || static_cast<size_t>(tensor.shape[1]) != width)
    throw std::runtime_error(std::string(name) + " sequence width does not match the token batch");
  if (tensor.shape[2] <= 0)
    throw std::runtime_error(std::string(name) + " hidden dimension must be positive");
  const auto hidden_size = static_cast<size_t>(tensor.shape[2]);
  if (rows && width > std::numeric_limits<size_t>::max() / rows)
    throw std::runtime_error(std::string(name) + " element count overflows size_t");
  const auto tokens = rows * width;
  if (hidden_size > std::numeric_limits<size_t>::max() / std::max<size_t>(tokens, 1))
    throw std::runtime_error(std::string(name) + " element count overflows size_t");
  if (TensorElementCount(tensor, name) != tokens * hidden_size)
    throw std::runtime_error(std::string(name) + " has an invalid element count");
  return hidden_size;
}

void RequireFloatCount(const OgaComponentTensor& tensor,
                       const std::vector<float>& values,
                       std::string_view name) {
  if (values.size() != TensorElementCount(tensor, name))
    throw std::runtime_error(std::string(name) +
                             " data size does not match its shape");
}

struct FeedStorage {
  std::vector<std::vector<std::byte>> bytes;
  std::vector<OgaComponentInput> inputs;
  template <typename T>
  void Add(std::string name, const std::vector<T>& value, std::vector<int64_t> shape,
           OgaElementType type) {
    bytes.emplace_back(value.size() * sizeof(T));
    if (!value.empty()) std::memcpy(bytes.back().data(), value.data(), bytes.back().size());
    inputs.push_back({std::move(name), bytes.back().data(), bytes.back().size(),
                      std::move(shape), type});
  }
  void AddZeros(std::string name, std::vector<int64_t> shape, OgaElementType type) {
    const auto count = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                       [](size_t left, int64_t right) { return left * static_cast<size_t>(right); });
    const auto size = count * ElementSize(type);
    bytes.emplace_back(std::max<size_t>(size, 1), std::byte{});
    inputs.push_back({std::move(name), bytes.back().data(), size, std::move(shape), type});
  }
};

struct TokenBatch {
  std::vector<int64_t> ids, mask;
  size_t rows{}, width{};
};

void AddPositionIds(FeedStorage& feeds, const NamedComponentSession& session,
                    const std::vector<int64_t>& positions, size_t rows,
                    size_t width) {
  const auto found = std::find_if(
      session.Inputs().begin(), session.Inputs().end(),
      [](const OgaComponentInfo& info) { return info.name == "position_ids"; });
  if (found == session.Inputs().end()) return;
  if (found->type != OgaElementType_int64)
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

TokenBatch Tokenize(DirectoryTokenizer& tokenizer, const std::vector<std::string>& texts) {
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

FeedStorage BackboneFeeds(const NamedComponentSession& session, const TokenBatch& batch) {
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
    if (info.name.rfind("past_key_values.", 0) != 0) continue;
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

std::string HiddenOutputName(const NamedComponentSession& session) {
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
  double confidence;
  if (kev && question.type == "choice") {
    confidence = probabilities.size() == 1 ? 1 : (probabilities[selected] - 1.0 / probabilities.size()) / (1.0 - 1.0 / probabilities.size());
  } else if (kev) {
    double distance = 0;
    for (size_t i = 0; i < probabilities.size(); ++i)
      distance += probabilities[i] * std::abs(static_cast<double>(i) - selected);
    confidence = probabilities.size() == 1 ? 1 : std::max(0.0, 1.0 - distance / (probabilities.size() - 1));
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

template <class Package>
NamedComponentSession TryComponent(const Package& package, const std::string& preferred,
                                   const std::string& fallback) {
  try {
    return package.Component(preferred);
  } catch (const std::exception& error) {
    if (std::string_view(error.what()).find("component not declared:") == std::string_view::npos)
      throw;
    return package.Component(fallback);
  }
}

}  // namespace

namespace {

size_t FloatVectorBytes(const std::vector<float>& value) {
  return value.size() * sizeof(float);
}

struct NativeRankingSession {
  NativeRankingSession(std::string path, std::vector<std::string> configured_providers)
      : package_path(std::move(path)), providers(std::move(configured_providers)), identity(PackageIdentity(package_path, providers)), cache(kDefaultClmCacheEntries, kDefaultClmCacheBytes, FloatVectorBytes), tokenizer(package_path), encoder(TryComponent(*this, "encoder", "backbone")) {
    try {
      combined = std::make_unique<NamedComponentSession>(Component("clm_heads"));
    } catch (const std::exception& error) {
      if (std::string_view(error.what()).find("component not declared:") == std::string_view::npos)
        throw;
      state = std::make_unique<NamedComponentSession>(Component("state_head"));
      action = std::make_unique<NamedComponentSession>(Component("action_head"));
      scorer = std::make_unique<NamedComponentSession>(Component("scorer"));
    }
  }
  NamedComponentSession Component(const std::string& name) const {
    return NamedComponentSession(package_path, name, providers);
  }
  std::string package_path;
  std::vector<std::string> providers;
  std::string identity;
  mutable std::mutex operation_mutex;
  SessionLruCache<std::vector<float>> cache;
  DirectoryTokenizer tokenizer;
  NamedComponentSession encoder;
  std::unique_ptr<NamedComponentSession> combined, state, action, scorer;
  OgaModelResult Run(const OgaStructuredRequest& request);
  OgaRankingResult Rank(const OgaFreeFormRankRequest& request);
  void SetCacheCapacity(size_t entries, size_t bytes) {
    std::lock_guard lock(operation_mutex);
    cache.SetCapacity(entries, bytes);
  }
  OgaNonGenerativeCacheStats CacheStats() const {
    std::lock_guard lock(operation_mutex);
    return cache.Stats();
  }
  void ClearCache() {
    std::lock_guard lock(operation_mutex);
    cache.Clear();
  }
  void InvalidateCache() {
    std::lock_guard lock(operation_mutex);
    identity = PackageIdentity(package_path, providers);
    cache.Clear();
  }
};

std::pair<std::vector<float>, size_t> EncodeAndPool(
    DirectoryTokenizer& tokenizer, NamedComponentSession& encoder,
    const std::vector<std::string>& texts) {
  if (texts.empty()) return {{}, 0};
  auto batch = Tokenize(tokenizer, texts);
  auto feeds = BackboneFeeds(encoder, batch);
  const auto hidden_name = HiddenOutputName(encoder);
  auto hidden_tensor = FindTensor(encoder.Run(feeds.inputs, {hidden_name}), hidden_name);
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
  auto [pooled, hidden_size] = EncodeAndPool(tokenizer, encoder, texts);
  if (!hidden_size || pooled.size() != texts.size() * hidden_size)
    throw std::runtime_error("CLM pooled encoder output has invalid dimensions");
  if (!missing_texts.empty()) {
    auto [missing_pooled, missing_hidden_size] =
        EncodeAndPool(tokenizer, encoder, missing_texts);
    if (missing_hidden_size != hidden_size)
      throw std::runtime_error("CLM state/action encoder layouts do not match");
    pooled.insert(pooled.end(), missing_pooled.begin(), missing_pooled.end());
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
  if (combined) {
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
    if (action_tensor.shape.size() != 2 || action_tensor.shape[1] <= 0)
      throw std::runtime_error("action embedding must be a rank-2 tensor with a positive projection dimension");
    const auto projection_size = static_cast<size_t>(action_tensor.shape[1]);
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
    auto project = [](NamedComponentSession& session, std::vector<float> values,
                      size_t rows, size_t columns, std::string_view name) {
      FeedStorage feeds;
      feeds.bytes.reserve(1);
      feeds.inputs.reserve(1);
      feeds.Add("embeddings", values, {static_cast<int64_t>(rows), static_cast<int64_t>(columns)}, OgaElementType_float32);
      const auto tensors = session.Run(feeds.inputs, {"projections"});
      const auto& tensor = FindTensor(tensors, "projections");
      if (tensor.shape.size() != 2 || tensor.shape[1] <= 0)
        throw std::runtime_error(std::string(name) +
                                 " must be rank 2 with a positive projection dimension");
      const auto projection_size = static_cast<size_t>(tensor.shape[1]);
      RequireShape(tensor, {rows, projection_size}, name);
      auto result = FloatTensor(tensor);
      RequireFloatCount(tensor, result, name);
      return std::pair{std::move(result), projection_size};
    };
    auto [state_projection, projection_size] = project(
        *state, {pooled.begin(), pooled.begin() + question_count * hidden_size},
        question_count, hidden_size, "state projections");
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
                             request.candidates[index].second,
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
  const OgaComponentInfo* input{};
  std::string output;
  bool growing{};
};

struct CachedKevPrefix {
  size_t length{};
  std::vector<OgaComponentTensor> states;
};

size_t KevPrefixBytes(const CachedKevPrefix& value) {
  size_t bytes = sizeof(value.length);
  for (const auto& state : value.states)
    bytes += state.name.size() + state.data.size() +
             state.shape.size() * sizeof(int64_t);
  return bytes;
}

std::string StateSuffix(std::string_view name) {
  static constexpr std::string_view prefixes[] = {
      "past_key_values.", "present_key_values.", "present."};
  for (const auto prefix : prefixes)
    if (name.rfind(prefix, 0) == 0) return std::string(name.substr(prefix.size()));
  return {};
}

std::vector<KevStateBinding> DiscoverKevStateBindings(
    const NamedComponentSession& backbone, std::string& reason) {
  std::vector<KevStateBinding> result;
  for (const auto& input : backbone.Inputs()) {
    if (input.name.rfind("past_key_values.", 0) != 0) continue;
    const auto suffix = StateSuffix(input.name);
    std::vector<std::string> matches;
    for (const auto& output : backbone.OutputNames())
      if (StateSuffix(output) == suffix) matches.push_back(output);
    if (matches.size() != 1) {
      reason = matches.empty() ? "missing present output for " + input.name
                               : "ambiguous present outputs for " + input.name;
      return {};
    }
    const auto tail = suffix.substr(suffix.rfind('.') + 1);
    const bool growing = tail == "key" || tail == "value";
    result.push_back({&input, std::move(matches.front()), growing});
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
                         const OgaComponentTensor& state) {
  const auto& input = *binding.input;
  if (state.type != input.type)
    throw std::runtime_error("state dtype mismatch between " + input.name +
                             " and " + state.name);
  if (state.shape.size() != input.shape.size() || state.shape.empty())
    throw std::runtime_error("state rank mismatch between " + input.name +
                             " and " + state.name);
  if (state.shape[0] != 1)
    throw std::runtime_error("prefix state output must have batch dimension 1: " +
                             state.name);
  for (size_t i = 1; i < state.shape.size(); ++i) {
    if (state.shape[i] < 0)
      throw std::runtime_error("prefix state output has a negative dimension: " +
                               state.name);
    if (input.shape[i] >= 0 &&
        (!binding.growing || i != 2) &&
        input.shape[i] != state.shape[i])
      throw std::runtime_error("state dimension mismatch between " + input.name +
                               " and " + state.name);
  }
  const auto expected = TensorElementCount(state, state.name) * ElementSize(state.type);
  if (expected != state.data.size())
    throw std::runtime_error("state byte count does not match shape: " + state.name);
}

void AddRepeatedState(FeedStorage& feeds, const KevStateBinding& binding,
                      const OgaComponentTensor& state, size_t batch) {
  ValidatePrefixState(binding, state);
  auto shape = state.shape;
  shape[0] = static_cast<int64_t>(batch);
  if (batch && state.data.size() > std::numeric_limits<size_t>::max() / batch)
    throw std::runtime_error("repeated state byte count overflows size_t");
  feeds.bytes.emplace_back(state.data.size() * batch);
  for (size_t i = 0; i < batch; ++i)
    std::memcpy(feeds.bytes.back().data() + i * state.data.size(),
                state.data.data(), state.data.size());
  feeds.inputs.push_back({binding.input->name, feeds.bytes.back().data(),
                          feeds.bytes.back().size(), std::move(shape), state.type});
}

struct NativeDecisionSession {
  NativeDecisionSession(std::string path, std::vector<std::string> configured_providers)
      : package_path(std::move(path)), providers(std::move(configured_providers)), identity(PackageIdentity(package_path, providers)), cache(kDefaultKevCacheEntries, kDefaultKevCacheBytes, KevTokenBytes), prefix_cache(kDefaultKevPrefixCacheEntries, kDefaultKevPrefixCacheBytes, KevPrefixBytes), tokenizer(package_path), backbone(Component("backbone")) {
    try {
      pointer = std::make_unique<NamedComponentSession>(Component("pointer_head"));
    } catch (const std::exception& error) {
      if (std::string_view(error.what()).find("component not declared:") == std::string_view::npos)
        throw;
      pointer = std::make_unique<NamedComponentSession>(Component("kev_head"));
      flat = true;
    }
    state_bindings = DiscoverKevStateBindings(backbone, compatibility_status);
    prefix_status = compatibility_status;
  }
  NamedComponentSession Component(const std::string& name) const {
    return NamedComponentSession(package_path, name, providers);
  }
  std::string package_path;
  std::vector<std::string> providers;
  std::string identity;
  mutable std::mutex operation_mutex;
  SessionLruCache<CachedKevTokens> cache;
  SessionLruCache<CachedKevPrefix> prefix_cache;
  DirectoryTokenizer tokenizer;
  NamedComponentSession backbone;
  std::unique_ptr<NamedComponentSession> pointer;
  std::vector<KevStateBinding> state_bindings;
  std::string compatibility_status;
  std::string prefix_status;
  bool prefix_reuse_enabled{true};
  uint64_t prefix_runs{};
  uint64_t branch_runs{};
  uint64_t fallback_runs{};
  bool flat{};
  OgaModelResult Run(const OgaStructuredRequest& request) { return Decide(request); }
  OgaModelResult Decide(const OgaStructuredRequest& request);
  void SetCacheCapacity(size_t entries, size_t bytes) {
    std::lock_guard lock(operation_mutex);
    cache.SetCapacity(entries, bytes);
  }
  OgaNonGenerativeCacheStats CacheStats() const {
    std::lock_guard lock(operation_mutex);
    return cache.Stats();
  }
  OgaNonGenerativeCacheStats PrefixCacheStats() const {
    std::lock_guard lock(operation_mutex);
    return prefix_cache.Stats();
  }
  OgaKevPrefixReuseStats PrefixReuseStats() const {
    std::lock_guard lock(operation_mutex);
    return {prefix_runs, branch_runs, fallback_runs};
  }
  void SetPrefixReuseEnabled(bool enabled) {
    std::lock_guard lock(operation_mutex);
    prefix_reuse_enabled = enabled;
    prefix_status = enabled ? compatibility_status : "disabled by policy";
  }
  bool PrefixReuseEnabled() const {
    std::lock_guard lock(operation_mutex);
    return prefix_reuse_enabled;
  }
  std::string PrefixReuseStatus() const {
    std::lock_guard lock(operation_mutex);
    return prefix_status;
  }
  void SetPrefixCacheCapacity(size_t entries, size_t bytes) {
    std::lock_guard lock(operation_mutex);
    prefix_cache.SetCapacity(entries, bytes);
  }
  void ClearCache() {
    std::lock_guard lock(operation_mutex);
    cache.Clear();
    prefix_cache.Clear();
  }
  void InvalidateCache() {
    std::lock_guard lock(operation_mutex);
    identity = PackageIdentity(package_path, providers);
    cache.Clear();
    prefix_cache.Clear();
  }
};

OgaModelResult NativeDecisionSession::Decide(const OgaStructuredRequest& request) {
  std::lock_guard operation_lock(operation_mutex);
  if (request.questions.empty()) throw std::invalid_argument("questions must be non-empty");
  auto encode = [&](std::string text) {
    static const std::regex delimiter(R"(<\|([A-Za-z0-9_]+)\|>)");
    text = std::regex_replace(text, delimiter, "<¦$1¦>");
    const auto tokens = tokenizer.Encode(text);
    return std::vector<int64_t>(tokens.begin(), tokens.end());
  };
  const auto rendered_state = RenderKev(request.state);
  const auto state_key = identity + "|kev-prefix|" + rendered_state;
  CachedKevTokens state_value;
  if (!cache.Get(state_key, state_value)) {
    state_value.tokens = {kStateToken};
    auto state_tokens = encode(rendered_state);
    if (state_tokens.size() > 8191) state_tokens.resize(8191);
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
      if (row_value.tokens.size() > 8192 ||
          state_value.tokens.size() > 8192 - row_value.tokens.size())
        throw std::invalid_argument(
            "state+question row exceeds 8192 tokens: " +
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
    if (row.size() > 8192)
      throw std::invalid_argument("state+question row exceeds 8192 tokens: " +
                                  std::to_string(row.size()));
    if (state_value.tokens.size() > 8192 - row.size())
      throw std::invalid_argument(
          "state+question row exceeds 8192 tokens: " +
          std::to_string(state_value.tokens.size() + row.size()));
    cache.Put(row_key, {row, indices});
    rows.push_back(std::move(row));
    option_indices.push_back(std::move(indices));
  }
  // Component outputs are currently host-owned, so copying and repeating CUDA
  // state tensors costs more than recomputing a short prefix.
  const bool use_prefix_reuse =
      prefix_reuse_enabled && !state_bindings.empty() &&
      (!UsesCuda(providers) ||
       state_value.tokens.size() >= kMinCudaKevPrefixTokens);
  if (!use_prefix_reuse) {
    ++fallback_runs;
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
  OgaComponentTensor hidden_tensor;
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
        const auto& state = FindTensor(tensors, binding.output);
        ValidatePrefixState(binding, state);
        prefix.states.push_back(state);
      }
      prefix_cache.Put(prefix_key, prefix);
      ++prefix_runs;
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
    hidden_tensor =
        FindTensor(backbone.Run(branch_feeds.inputs, {hidden_name}), hidden_name);
    ++branch_runs;
  } else {
    auto feeds = BackboneFeeds(backbone, batch);
    hidden_tensor = FindTensor(backbone.Run(feeds.inputs, {hidden_name}), hidden_name);
  }
  const auto hidden_size =
      RequireHiddenShape(hidden_tensor, batch.rows, batch.width,
                         "backbone hidden states");
  const auto hidden = FloatTensor(hidden_tensor);
  RequireFloatCount(hidden_tensor, hidden, "backbone hidden states");
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
        [](const OgaComponentInfo& info) { return info.name == "hidden_states"; });
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

OgaStructuredValue& Value(OgaStructuredValueHandle* value) {
  return CapiRequired(reinterpret_cast<OgaStructuredValue*>(value), "value");
}
const OgaStructuredValue& Value(const OgaStructuredValueHandle* value) {
  return CapiRequired(reinterpret_cast<const OgaStructuredValue*>(value), "value");
}
OgaQuestion& Question(OgaQuestionHandle* value) {
  return CapiRequired(reinterpret_cast<OgaQuestion*>(value), "question");
}
const OgaQuestion& Question(const OgaQuestionHandle* value) {
  return CapiRequired(reinterpret_cast<const OgaQuestion*>(value), "question");
}
OgaStructuredRequest& Request(OgaStructuredRequestHandle* value) {
  return CapiRequired(reinterpret_cast<OgaStructuredRequest*>(value), "request");
}
const OgaStructuredRequest& Request(const OgaStructuredRequestHandle* value) {
  return CapiRequired(reinterpret_cast<const OgaStructuredRequest*>(value), "request");
}
OgaFreeFormRankRequest& RankRequest(OgaFreeFormRankRequestHandle* value) {
  return CapiRequired(reinterpret_cast<OgaFreeFormRankRequest*>(value), "request");
}
const OgaFreeFormRankRequest& RankRequest(const OgaFreeFormRankRequestHandle* value) {
  return CapiRequired(reinterpret_cast<const OgaFreeFormRankRequest*>(value), "request");
}
const OgaModelResult& ModelResult(const OgaModelResultHandle* value) {
  return CapiRequired(reinterpret_cast<const OgaModelResult*>(value), "result");
}
const OgaRankingResult& RankingResult(const OgaRankingResultHandle* value) {
  return CapiRequired(reinterpret_cast<const OgaRankingResult*>(value), "result");
}
const OgaAnswer& AnswerAt(const OgaModelResultHandle* value, size_t answer) {
  return ModelResult(value).answers.at(answer).second;
}

template <class T>
void OptionalNumber(const std::optional<T>& source, T* value, bool* present) {
  CapiRequired(present, "present") = source.has_value();
  if (source) CapiRequired(value, "value") = *source;
}

}  // namespace

struct OGA_CPP_ONLY OgaRankingSessionHandle {
  OgaRankingSessionHandle(std::string path, std::vector<std::string> providers)
      : package_path(std::move(path)), providers(std::move(providers)), value(package_path, this->providers) {}
  std::string package_path;
  std::vector<std::string> providers;
  NativeRankingSession value;
};

struct OGA_CPP_ONLY OgaDecisionSessionHandle {
  OgaDecisionSessionHandle(std::string path, std::vector<std::string> providers)
      : package_path(std::move(path)), providers(std::move(providers)), value(package_path, this->providers) {}
  std::string package_path;
  std::vector<std::string> providers;
  NativeDecisionSession value;
};

extern "C" {

#define OGA_CREATE_STRUCTURED_VALUE(name, expression)                       \
  OgaResult* OGA_API_CALL name(OgaStructuredValueHandle** out) {            \
    OGA_CAPI_TRY                                                            \
    auto& output = CapiRequired(out, "out");                                \
    auto result = std::make_unique<OgaStructuredValue>(expression);         \
    output = reinterpret_cast<OgaStructuredValueHandle*>(result.release()); \
    return nullptr;                                                         \
    OGA_CAPI_CATCH                                                          \
  }

OGA_CREATE_STRUCTURED_VALUE(OgaCreateStructuredValueNull, nullptr)

OgaResult* OGA_API_CALL OgaCreateStructuredValueBool(
    bool value, OgaStructuredValueHandle** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  auto result = std::make_unique<OgaStructuredValue>(value);
  output = reinterpret_cast<OgaStructuredValueHandle*>(result.release());
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaCreateStructuredValueInt64(
    int64_t value, OgaStructuredValueHandle** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  auto result = std::make_unique<OgaStructuredValue>(value);
  output = reinterpret_cast<OgaStructuredValueHandle*>(result.release());
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaCreateStructuredValueDouble(
    double value, OgaStructuredValueHandle** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  auto result = std::make_unique<OgaStructuredValue>(value);
  output = reinterpret_cast<OgaStructuredValueHandle*>(result.release());
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaCreateStructuredValueString(
    const char* value, OgaStructuredValueHandle** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  auto result = std::make_unique<OgaStructuredValue>(
      std::string(CapiRequired(value, "value")));
  output = reinterpret_cast<OgaStructuredValueHandle*>(result.release());
  return nullptr;
  OGA_CAPI_CATCH
}

OGA_CREATE_STRUCTURED_VALUE(OgaCreateStructuredValueArray, OgaStructuredValue::Array{})
OGA_CREATE_STRUCTURED_VALUE(OgaCreateStructuredValueObject, OgaStructuredValue::Object{})

#undef OGA_CREATE_STRUCTURED_VALUE

OgaResult* OGA_API_CALL OgaStructuredValueArrayAppend(
    OgaStructuredValueHandle* array, const OgaStructuredValueHandle* value) {
  OGA_CAPI_TRY
  auto* values = std::get_if<OgaStructuredValue::Array>(&Value(array).value);
  if (!values) throw std::invalid_argument("value is not an array");
  values->push_back(Value(value));
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaStructuredValueObjectAppend(
    OgaStructuredValueHandle* object, const char* key,
    const OgaStructuredValueHandle* value) {
  OGA_CAPI_TRY
  auto* values = std::get_if<OgaStructuredValue::Object>(&Value(object).value);
  if (!values) throw std::invalid_argument("value is not an object");
  values->emplace_back(CapiRequired(key, "key"), Value(value));
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaStructuredValueGetType(
    const OgaStructuredValueHandle* value, OgaStructuredValueType* out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") =
      static_cast<OgaStructuredValueType>(Value(value).value.index());
  return nullptr;
  OGA_CAPI_CATCH
}

#define OGA_GET_STRUCTURED_SCALAR(name, type, alternative)                     \
  OgaResult* OGA_API_CALL name(const OgaStructuredValueHandle* value,          \
                               type* out) {                                    \
    OGA_CAPI_TRY                                                               \
    const auto* item = std::get_if<alternative>(&Value(value).value);          \
    if (!item) throw std::invalid_argument("structured value has wrong type"); \
    CapiRequired(out, "out") = *item;                                          \
    return nullptr;                                                            \
    OGA_CAPI_CATCH                                                             \
  }

OGA_GET_STRUCTURED_SCALAR(OgaStructuredValueGetBool, bool, bool)
OGA_GET_STRUCTURED_SCALAR(OgaStructuredValueGetInt64, int64_t, int64_t)
OGA_GET_STRUCTURED_SCALAR(OgaStructuredValueGetDouble, double, double)

#undef OGA_GET_STRUCTURED_SCALAR

OgaResult* OGA_API_CALL OgaStructuredValueGetString(
    const OgaStructuredValueHandle* value, const char** out) {
  OGA_CAPI_TRY
  const auto* item = std::get_if<std::string>(&Value(value).value);
  if (!item) throw std::invalid_argument("structured value is not a string");
  CapiRequired(out, "out") = item->c_str();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaStructuredValueGetCount(
    const OgaStructuredValueHandle* value, size_t* out) {
  OGA_CAPI_TRY
  if (const auto* array = std::get_if<OgaStructuredValue::Array>(&Value(value).value))
    CapiRequired(out, "out") = array->size();
  else if (const auto* object = std::get_if<OgaStructuredValue::Object>(&Value(value).value))
    CapiRequired(out, "out") = object->size();
  else
    throw std::invalid_argument("structured value is not an array or object");
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaStructuredValueGetArrayItem(
    const OgaStructuredValueHandle* value, size_t index,
    const OgaStructuredValueHandle** out) {
  OGA_CAPI_TRY
  const auto* array = std::get_if<OgaStructuredValue::Array>(&Value(value).value);
  if (!array) throw std::invalid_argument("structured value is not an array");
  CapiRequired(out, "out") =
      reinterpret_cast<const OgaStructuredValueHandle*>(&array->at(index));
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaStructuredValueGetObjectItem(
    const OgaStructuredValueHandle* value, size_t index, const char** key,
    const OgaStructuredValueHandle** out) {
  OGA_CAPI_TRY
  const auto* object = std::get_if<OgaStructuredValue::Object>(&Value(value).value);
  if (!object) throw std::invalid_argument("structured value is not an object");
  const auto& item = object->at(index);
  CapiRequired(key, "key") = item.first.c_str();
  CapiRequired(out, "out") =
      reinterpret_cast<const OgaStructuredValueHandle*>(&item.second);
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyStructuredValue(OgaStructuredValueHandle* value) {
  delete reinterpret_cast<OgaStructuredValue*>(value);
}

OgaResult* OGA_API_CALL OgaCreateQuestion(
    const char* type, const OgaStructuredValueHandle* instructions,
    const OgaStructuredValueHandle* criteria, OgaQuestionHandle** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  auto question = std::make_unique<OgaQuestion>();
  question->type = CapiRequired(type, "type");
  question->instructions = Value(instructions);
  question->criteria = Value(criteria);
  output = reinterpret_cast<OgaQuestionHandle*>(question.release());
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyQuestion(OgaQuestionHandle* question) {
  delete reinterpret_cast<OgaQuestion*>(question);
}

OgaResult* OGA_API_CALL OgaCreateStructuredRequest(OgaStructuredRequestHandle** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  auto result = std::make_unique<OgaStructuredRequest>();
  output = reinterpret_cast<OgaStructuredRequestHandle*>(result.release());
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaStructuredRequestSetState(
    OgaStructuredRequestHandle* request, const OgaStructuredValueHandle* state) {
  OGA_CAPI_TRY
  Request(request).state = Value(state);
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaStructuredRequestAddQuestion(
    OgaStructuredRequestHandle* request, const char* id,
    const OgaQuestionHandle* question) {
  OGA_CAPI_TRY
  Request(request).questions.emplace_back(CapiRequired(id, "id"), Question(question));
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaStructuredRequestSetTemperature(
    OgaStructuredRequestHandle* request, float temperature) {
  OGA_CAPI_TRY
  Request(request).temperature = temperature;
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyStructuredRequest(OgaStructuredRequestHandle* request) {
  delete reinterpret_cast<OgaStructuredRequest*>(request);
}

OgaResult* OGA_API_CALL OgaCreateFreeFormRankRequest(
    OgaFreeFormRankRequestHandle** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  auto result = std::make_unique<OgaFreeFormRankRequest>();
  output = reinterpret_cast<OgaFreeFormRankRequestHandle*>(result.release());
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaFreeFormRankRequestSetState(
    OgaFreeFormRankRequestHandle* request, const OgaStructuredValueHandle* state) {
  OGA_CAPI_TRY
  RankRequest(request).state = Value(state);
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaFreeFormRankRequestSetInstructions(
    OgaFreeFormRankRequestHandle* request,
    const OgaStructuredValueHandle* instructions) {
  OGA_CAPI_TRY
  RankRequest(request).instructions = Value(instructions);
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaFreeFormRankRequestAddCandidate(
    OgaFreeFormRankRequestHandle* request, const char* key,
    const OgaStructuredValueHandle* value) {
  OGA_CAPI_TRY
  RankRequest(request).candidates.emplace_back(CapiRequired(key, "key"), Value(value));
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaFreeFormRankRequestSetTemperature(
    OgaFreeFormRankRequestHandle* request, float temperature) {
  OGA_CAPI_TRY
  RankRequest(request).temperature = temperature;
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyFreeFormRankRequest(OgaFreeFormRankRequestHandle* request) {
  delete reinterpret_cast<OgaFreeFormRankRequest*>(request);
}

OgaResult* OGA_API_CALL OgaCreateRankingSession(
    const char* package_path, const char* const* providers, size_t provider_count,
    OgaRankingSessionHandle** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  auto provider_values = CapiStrings(providers, provider_count, "providers");
  auto result = std::make_unique<OgaRankingSessionHandle>(
      CapiRequired(package_path, "package_path"), std::move(provider_values));
  output = result.release();
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyRankingSession(OgaRankingSessionHandle* session) {
  delete session;
}

OgaResult* OGA_API_CALL OgaRankingSessionCreateComponent(
    const OgaRankingSessionHandle* session, const char* name,
    OgaComponentSession** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  const auto& value = CapiRequired(session, "session");
  std::vector<const char*> providers;
  for (const auto& provider : value.providers) providers.push_back(provider.c_str());
  return OgaCreateComponentSession(value.package_path.c_str(), name, providers.data(),
                                   providers.size(), &output);
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaRankingSessionRun(
    OgaRankingSessionHandle* session, const OgaStructuredRequestHandle* request,
    OgaModelResultHandle** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  auto result = std::make_unique<OgaModelResult>(
      CapiRequired(session, "session").value.Run(Request(request)));
  output = reinterpret_cast<OgaModelResultHandle*>(result.release());
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaRankingSessionRank(
    OgaRankingSessionHandle* session, const OgaFreeFormRankRequestHandle* request,
    OgaRankingResultHandle** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  auto result = std::make_unique<OgaRankingResult>(
      CapiRequired(session, "session").value.Rank(RankRequest(request)));
  output = reinterpret_cast<OgaRankingResultHandle*>(result.release());
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaRankingSessionSetCacheCapacity(
    OgaRankingSessionHandle* session, size_t entry_capacity, size_t byte_capacity) {
  OGA_CAPI_TRY
  CapiRequired(session, "session").value.SetCacheCapacity(entry_capacity, byte_capacity);
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaRankingSessionGetCacheStats(
    const OgaRankingSessionHandle* session, OgaNonGenerativeCacheStats* out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  output = CapiRequired(session, "session").value.CacheStats();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaRankingSessionClearCache(OgaRankingSessionHandle* session) {
  OGA_CAPI_TRY
  CapiRequired(session, "session").value.ClearCache();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaRankingSessionInvalidateCache(OgaRankingSessionHandle* session) {
  OGA_CAPI_TRY
  CapiRequired(session, "session").value.InvalidateCache();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaCreateDecisionSession(
    const char* package_path, const char* const* providers, size_t provider_count,
    OgaDecisionSessionHandle** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  auto provider_values = CapiStrings(providers, provider_count, "providers");
  auto result = std::make_unique<OgaDecisionSessionHandle>(
      CapiRequired(package_path, "package_path"), std::move(provider_values));
  output = result.release();
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyDecisionSession(OgaDecisionSessionHandle* session) {
  delete session;
}

OgaResult* OGA_API_CALL OgaDecisionSessionCreateComponent(
    const OgaDecisionSessionHandle* session, const char* name,
    OgaComponentSession** out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  const auto& value = CapiRequired(session, "session");
  std::vector<const char*> providers;
  for (const auto& provider : value.providers) providers.push_back(provider.c_str());
  return OgaCreateComponentSession(value.package_path.c_str(), name, providers.data(),
                                   providers.size(), &output);
  OGA_CAPI_CATCH
}

#define OGA_DECISION_RUN(name, method)                                              \
  OgaResult* OGA_API_CALL name(                                                     \
      OgaDecisionSessionHandle* session, const OgaStructuredRequestHandle* request, \
      OgaModelResultHandle** out) {                                                 \
    OGA_CAPI_TRY                                                                    \
    auto& output = CapiRequired(out, "out");                                        \
    auto result = std::make_unique<OgaModelResult>(                                 \
        CapiRequired(session, "session").value.method(Request(request)));           \
    output = reinterpret_cast<OgaModelResultHandle*>(result.release());             \
    return nullptr;                                                                 \
    OGA_CAPI_CATCH                                                                  \
  }

OGA_DECISION_RUN(OgaDecisionSessionRun, Run)
OGA_DECISION_RUN(OgaDecisionSessionDecide, Decide)
#undef OGA_DECISION_RUN

OgaResult* OGA_API_CALL OgaDecisionSessionSetCacheCapacity(
    OgaDecisionSessionHandle* session, size_t entry_capacity, size_t byte_capacity) {
  OGA_CAPI_TRY
  CapiRequired(session, "session").value.SetCacheCapacity(entry_capacity, byte_capacity);
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaDecisionSessionGetCacheStats(
    const OgaDecisionSessionHandle* session, OgaNonGenerativeCacheStats* out) {
  OGA_CAPI_TRY
  auto& output = CapiRequired(out, "out");
  output = CapiRequired(session, "session").value.CacheStats();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaDecisionSessionClearCache(OgaDecisionSessionHandle* session) {
  OGA_CAPI_TRY
  CapiRequired(session, "session").value.ClearCache();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaDecisionSessionInvalidateCache(OgaDecisionSessionHandle* session) {
  OGA_CAPI_TRY
  CapiRequired(session, "session").value.InvalidateCache();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaDecisionSessionSetPrefixReuseEnabled(
    OgaDecisionSessionHandle* session, bool enabled) {
  OGA_CAPI_TRY
  CapiRequired(session, "session").value.SetPrefixReuseEnabled(enabled);
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaDecisionSessionGetPrefixReuseEnabled(
    const OgaDecisionSessionHandle* session, bool* out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") =
      CapiRequired(session, "session").value.PrefixReuseEnabled();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaDecisionSessionGetPrefixReuseStatus(
    const OgaDecisionSessionHandle* session, const char** out) {
  OGA_CAPI_TRY
  const auto& value = CapiRequired(session, "session").value;
  std::lock_guard lock(value.operation_mutex);
  // compatibility_status is immutable after construction and the disabled
  // literal has static storage, so this legacy borrowed pointer is stable.
  CapiRequired(out, "out") = value.prefix_reuse_enabled
                                 ? value.compatibility_status.c_str()
                                 : "disabled by policy";
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaDecisionSessionCopyPrefixReuseStatus(
    const OgaDecisionSessionHandle* session, char* buffer, size_t buffer_capacity,
    size_t* required_size) {
  OGA_CAPI_TRY
  auto& required_output = CapiRequired(required_size, "out");
  const std::string status =
      CapiRequired(session, "session").value.PrefixReuseStatus();
  const size_t required = status.size() + 1;
  required_output = required;
  if (!buffer) {
    if (buffer_capacity != 0)
      throw std::invalid_argument("buffer must not be null when buffer_capacity is non-zero");
    return nullptr;
  }
  if (buffer_capacity < required)
    throw std::invalid_argument("buffer_capacity is too small");
  std::memcpy(buffer, status.c_str(), required);
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaDecisionSessionSetPrefixCacheCapacity(
    OgaDecisionSessionHandle* session, size_t entry_capacity, size_t byte_capacity) {
  OGA_CAPI_TRY
  CapiRequired(session, "session").value.SetPrefixCacheCapacity(entry_capacity, byte_capacity);
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaDecisionSessionGetPrefixCacheStats(
    const OgaDecisionSessionHandle* session, OgaNonGenerativeCacheStats* out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") =
      CapiRequired(session, "session").value.PrefixCacheStats();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaDecisionSessionGetPrefixReuseStats(
    const OgaDecisionSessionHandle* session, OgaKevPrefixReuseStats* out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") =
      CapiRequired(session, "session").value.PrefixReuseStats();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaModelResultGetModel(
    const OgaModelResultHandle* result, const char** out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") = ModelResult(result).model.c_str();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaModelResultGetAnswerCount(
    const OgaModelResultHandle* result, size_t* out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") = ModelResult(result).answers.size();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaModelResultGetAnswerId(
    const OgaModelResultHandle* result, size_t answer, const char** out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") = ModelResult(result).answers.at(answer).first.c_str();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaModelResultGetAnswerType(
    const OgaModelResultHandle* result, size_t answer, const char** out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") = AnswerAt(result, answer).type.c_str();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaModelResultGetAnswerNoul(
    const OgaModelResultHandle* result, size_t answer, double* value, bool* present) {
  OGA_CAPI_TRY
  OptionalNumber(AnswerAt(result, answer).noul, value, present);
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaModelResultGetAnswerChoice(
    const OgaModelResultHandle* result, size_t answer, const char** value, bool* present) {
  OGA_CAPI_TRY
  const auto& choice = AnswerAt(result, answer).choice;
  CapiRequired(present, "present") = choice.has_value();
  if (choice) CapiRequired(value, "value") = choice->c_str();
  return nullptr;
  OGA_CAPI_CATCH
}

#define OGA_MODEL_OPTIONAL(name, member)                                \
  OgaResult* OGA_API_CALL name(                                         \
      const OgaModelResultHandle* result, size_t answer, double* value, \
      bool* present) {                                                  \
    OGA_CAPI_TRY                                                        \
    OptionalNumber(AnswerAt(result, answer).member, value, present);    \
    return nullptr;                                                     \
    OGA_CAPI_CATCH                                                      \
  }
OGA_MODEL_OPTIONAL(OgaModelResultGetAnswerScore, score)
OGA_MODEL_OPTIONAL(OgaModelResultGetAnswerConfidence, confidence)
#undef OGA_MODEL_OPTIONAL

OgaResult* OGA_API_CALL OgaModelResultGetProbabilityCount(
    const OgaModelResultHandle* result, size_t answer, size_t* out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") = AnswerAt(result, answer).probabilities.size();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaModelResultGetProbability(
    const OgaModelResultHandle* result, size_t answer, size_t index,
    const char** key, double* value) {
  OGA_CAPI_TRY
  const auto& item = AnswerAt(result, answer).probabilities.at(index);
  CapiRequired(key, "key") = item.first.c_str();
  CapiRequired(value, "value") = item.second;
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaModelResultGetLegendCount(
    const OgaModelResultHandle* result, size_t answer, size_t* out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") = AnswerAt(result, answer).legend.size();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaModelResultGetLegend(
    const OgaModelResultHandle* result, size_t answer, size_t index,
    const char** key, const char** value) {
  OGA_CAPI_TRY
  const auto& item = AnswerAt(result, answer).legend.at(index);
  CapiRequired(key, "key") = item.first.c_str();
  CapiRequired(value, "value") = item.second.c_str();
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyModelResult(OgaModelResultHandle* result) {
  delete reinterpret_cast<OgaModelResult*>(result);
}

OgaResult* OGA_API_CALL OgaRankingResultGetModel(
    const OgaRankingResultHandle* result, const char** out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") = RankingResult(result).model.c_str();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaRankingResultGetCount(
    const OgaRankingResultHandle* result, size_t* out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") = RankingResult(result).ranked.size();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaRankingResultGetRank(
    const OgaRankingResultHandle* result, size_t index, size_t* out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") = RankingResult(result).ranked.at(index).rank;
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaRankingResultGetKey(
    const OgaRankingResultHandle* result, size_t index, const char** out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") = RankingResult(result).ranked.at(index).key.c_str();
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaRankingResultGetValue(
    const OgaRankingResultHandle* result, size_t index,
    const OgaStructuredValueHandle** out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") = reinterpret_cast<const OgaStructuredValueHandle*>(
      &RankingResult(result).ranked.at(index).value);
  return nullptr;
  OGA_CAPI_CATCH
}

OgaResult* OGA_API_CALL OgaRankingResultGetProbability(
    const OgaRankingResultHandle* result, size_t index, double* out) {
  OGA_CAPI_TRY
  CapiRequired(out, "out") = RankingResult(result).ranked.at(index).probability;
  return nullptr;
  OGA_CAPI_CATCH
}

void OGA_API_CALL OgaDestroyRankingResult(OgaRankingResultHandle* result) {
  delete reinterpret_cast<OgaRankingResult*>(result);
}

}  // extern "C"
