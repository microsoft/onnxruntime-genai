// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#if __cplusplus >= 202002L
#include <span>
#define OGA_USE_SPAN 1
#endif

#include "ort_genai_c.h"

#if defined(__GNUC__) && !defined(_WIN32) && defined(BUILDING_ORT_GENAI_C)
#define OGA_CPP_ONLY __attribute__((visibility("hidden")))
#else
#define OGA_CPP_ONLY
#endif

// GenAI C++ API
//
// This is a zero cost wrapper around the C API, and provides for a set of C++ classes with automatic resource management

/* A simple end to end example of how to generate an answer from a prompt:
 *
 * auto model = OgaModel::Create("phi-2");
 * auto tokenizer = OgaTokenizer::Create(*model);
 *
 * auto sequences = OgaSequences::Create();
 * tokenizer->Encode("A great recipe for Kung Pao chicken is ", *sequences);
 *
 * auto params = OgaGeneratorParams::Create(*model);
 * params->SetSearchOption("max_length", 200);
 * params->SetSearchOption("batch_size", 1);
 *
 * auto generator = OgaGenerator::Create(*model, *params);
 * generator->AppendTokenSequences(*sequences);
 * while (!generator->IsDone()) {
 *  generator->GenerateNextToken();
 * }
 * auto output_sequence = generator->GetSequenceData(0);
 * auto output_string = tokenizer->Decode(output_sequence, generator->GetSequenceCount(0));
 *
 * std::cout << "Output: " << std::endl << output_string << std::endl;
 */

// The types defined in this file are to give us zero overhead C++ style interfaces around an opaque C pointer.
// For example, there is no actual 'OgaModel' type defined anywhere, so we create a fake definition here
// that lets users have a C++ style OgaModel type that can be held in a std::unique_ptr.
//
// This OgaAbstract struct is to prevent accidentally trying to use them by value.
struct OgaAbstract {
  OgaAbstract() = delete;
  OgaAbstract(const OgaAbstract&) = delete;
  void operator=(const OgaAbstract&) = delete;
};

// Uncached, C++-only execution API for independently exported ONNX components.
struct OgaComponentInput {
  std::string name;
  const void* data{};
  size_t byte_count{};
  std::vector<int64_t> shape;
  OgaElementType type{};
};

struct OgaComponentTensor {
  std::string name;
  std::vector<std::byte> data;
  std::vector<int64_t> shape;
  OgaElementType type{};
};
struct OgaComponentInfo {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<std::string> symbolic_dimensions;
  OgaElementType type{};
};

class OGA_CPP_ONLY NamedComponentSession {
 public:
  NamedComponentSession(const std::string& package_path, const std::string& component,
                        const std::vector<std::string>& providers = {});
  ~NamedComponentSession() { OgaDestroyComponentSession(handle_); }
  NamedComponentSession(NamedComponentSession&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)),
        input_names_(std::move(other.input_names_)),
        output_names_(std::move(other.output_names_)),
        inputs_(std::move(other.inputs_)) {}
  NamedComponentSession& operator=(NamedComponentSession&& other) noexcept {
    if (this != &other) {
      OgaDestroyComponentSession(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
      input_names_ = std::move(other.input_names_);
      output_names_ = std::move(other.output_names_);
      inputs_ = std::move(other.inputs_);
    }
    return *this;
  }
  NamedComponentSession(const NamedComponentSession&) = delete;
  NamedComponentSession& operator=(const NamedComponentSession&) = delete;
  std::vector<OgaComponentTensor> Run(const std::vector<OgaComponentInput>& inputs,
                                      const std::vector<std::string>& outputs = {});
  const std::vector<std::string>& InputNames() const;
  const std::vector<std::string>& OutputNames() const;
  const std::vector<OgaComponentInfo>& Inputs() const;

 private:
  friend class RankingSession;
  friend class DecisionSession;
  explicit NamedComponentSession(OgaComponentSession* handle);
  void LoadMetadata();
  OgaComponentSession* handle_{};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<OgaComponentInfo> inputs_;
};

using ComponentSession = NamedComponentSession;

// C++-only tokenizer for component packages that contain tokenizer files but
// are not generative OgaModel packages.
class OGA_CPP_ONLY DirectoryTokenizer {
 public:
  explicit DirectoryTokenizer(const std::string& package_path);
  ~DirectoryTokenizer() { OgaDestroyDirectoryTokenizer(handle_); }
  DirectoryTokenizer(DirectoryTokenizer&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}
  DirectoryTokenizer& operator=(DirectoryTokenizer&& other) noexcept {
    if (this != &other) {
      OgaDestroyDirectoryTokenizer(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  DirectoryTokenizer(const DirectoryTokenizer&) = delete;
  DirectoryTokenizer& operator=(const DirectoryTokenizer&) = delete;
  std::vector<int32_t> Encode(const std::string& text) const;
  int32_t PadTokenId() const;

 private:
  OgaDirectoryTokenizer* handle_{};
};

// JSON-independent structured values used by the non-generative model APIs.
// Objects are vectors rather than maps so request ordering is preserved.
struct OgaStructuredValue {
  using Array = std::vector<OgaStructuredValue>;
  using Object = std::vector<std::pair<std::string, OgaStructuredValue>>;
  using Value = std::variant<std::monostate, bool, int64_t, double, std::string, Array, Object>;
  Value value;

  OgaStructuredValue() = default;
  OgaStructuredValue(std::nullptr_t) {}
  OgaStructuredValue(bool v) : value(v) {}
  OgaStructuredValue(int v) : value(static_cast<int64_t>(v)) {}
  OgaStructuredValue(int64_t v) : value(v) {}
  OgaStructuredValue(double v) : value(v) {}
  OgaStructuredValue(const char* v) : value(std::string(v)) {}
  OgaStructuredValue(std::string v) : value(std::move(v)) {}
  OgaStructuredValue(Array v) : value(std::move(v)) {}
  OgaStructuredValue(Object v) : value(std::move(v)) {}
};

struct OgaQuestion {
  std::string type;  // "noul", "choice", or "score"
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
  OgaStructuredValue value;
  double probability{};
};

struct OgaRankingResult {
  std::string model;
  std::vector<OgaRankedItem> ranked;
};

// Typed package entry points. Component() remains available for callers that
// need direct access to a named ONNX graph.
class OGA_CPP_ONLY RankingSession {
 public:
  RankingSession(std::string package_path, std::vector<std::string> providers = {});
  ~RankingSession() { OgaDestroyRankingSession(handle_); }
  RankingSession(RankingSession&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}
  RankingSession& operator=(RankingSession&& other) noexcept {
    if (this != &other) {
      OgaDestroyRankingSession(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  RankingSession(const RankingSession&) = delete;
  RankingSession& operator=(const RankingSession&) = delete;
  NamedComponentSession Component(const std::string& name) const;
  OgaModelResult Run(const OgaStructuredRequest& request);
  OgaRankingResult Rank(const OgaFreeFormRankRequest& request);
  void SetCacheCapacity(size_t entry_capacity, size_t byte_capacity);
  OgaNonGenerativeCacheStats CacheStats() const;
  void ClearCache();
  void InvalidateCache();

 private:
  OgaRankingSessionHandle* handle_{};
};

class OGA_CPP_ONLY DecisionSession {
 public:
  DecisionSession(std::string package_path, std::vector<std::string> providers = {});
  ~DecisionSession() { OgaDestroyDecisionSession(handle_); }
  DecisionSession(DecisionSession&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}
  DecisionSession& operator=(DecisionSession&& other) noexcept {
    if (this != &other) {
      OgaDestroyDecisionSession(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  DecisionSession(const DecisionSession&) = delete;
  DecisionSession& operator=(const DecisionSession&) = delete;
  NamedComponentSession Component(const std::string& name) const;
  OgaModelResult Run(const OgaStructuredRequest& request);
  OgaModelResult Decide(const OgaStructuredRequest& request);
  void SetCacheCapacity(size_t entry_capacity, size_t byte_capacity);
  OgaNonGenerativeCacheStats CacheStats() const;
  void SetPrefixReuseEnabled(bool enabled);
  bool PrefixReuseEnabled() const;
  std::string PrefixReuseStatus() const;
  void SetPrefixCacheCapacity(size_t entry_capacity, size_t byte_capacity);
  OgaNonGenerativeCacheStats PrefixCacheStats() const;
  OgaKevPrefixReuseStats PrefixReuseStats() const;
  void ClearCache();
  void InvalidateCache();

 private:
  OgaDecisionSessionHandle* handle_{};
};

struct OgaResult : OgaAbstract {
  const char* GetError() const { return OgaResultGetError(this); }
  static void operator delete(void* p) { OgaDestroyResult(reinterpret_cast<OgaResult*>(p)); }
};

// This is used to turn OgaResult return values from the C API into std::runtime_error exceptions
inline void OgaCheckResult(OgaResult* result) {
  if (result) {
    std::unique_ptr<OgaResult> p_result{result};  // Take ownership so it's destroyed properly
    throw std::runtime_error(p_result->GetError());
  }
}

namespace OgaDetail {

template <class T, void (*Destroy)(T*)>
using Handle = std::unique_ptr<T, decltype(Destroy)>;

using StructuredValueHandle =
    Handle<OgaStructuredValueHandle, OgaDestroyStructuredValue>;
using QuestionHandle = Handle<OgaQuestionHandle, OgaDestroyQuestion>;
using StructuredRequestHandle =
    Handle<OgaStructuredRequestHandle, OgaDestroyStructuredRequest>;
using RankRequestHandle =
    Handle<OgaFreeFormRankRequestHandle, OgaDestroyFreeFormRankRequest>;
using ModelResultHandle = Handle<OgaModelResultHandle, OgaDestroyModelResult>;
using RankingResultHandle = Handle<OgaRankingResultHandle, OgaDestroyRankingResult>;

inline std::vector<const char*> StringPointers(const std::vector<std::string>& values) {
  std::vector<const char*> result;
  result.reserve(values.size());
  for (const auto& value : values) result.push_back(value.c_str());
  return result;
}

inline StructuredValueHandle MakeValue(const OgaStructuredValue& source) {
  OgaStructuredValueHandle* raw{};
  switch (source.value.index()) {
    case 0:
      OgaCheckResult(OgaCreateStructuredValueNull(&raw));
      break;
    case 1:
      OgaCheckResult(OgaCreateStructuredValueBool(std::get<bool>(source.value), &raw));
      break;
    case 2:
      OgaCheckResult(OgaCreateStructuredValueInt64(std::get<int64_t>(source.value), &raw));
      break;
    case 3:
      OgaCheckResult(OgaCreateStructuredValueDouble(std::get<double>(source.value), &raw));
      break;
    case 4:
      OgaCheckResult(OgaCreateStructuredValueString(
          std::get<std::string>(source.value).c_str(), &raw));
      break;
    case 5: {
      OgaCheckResult(OgaCreateStructuredValueArray(&raw));
      StructuredValueHandle result(raw, OgaDestroyStructuredValue);
      for (const auto& item : std::get<OgaStructuredValue::Array>(source.value)) {
        auto child = MakeValue(item);
        OgaCheckResult(OgaStructuredValueArrayAppend(result.get(), child.get()));
      }
      return result;
    }
    case 6: {
      OgaCheckResult(OgaCreateStructuredValueObject(&raw));
      StructuredValueHandle result(raw, OgaDestroyStructuredValue);
      for (const auto& [key, item] : std::get<OgaStructuredValue::Object>(source.value)) {
        auto child = MakeValue(item);
        OgaCheckResult(
            OgaStructuredValueObjectAppend(result.get(), key.c_str(), child.get()));
      }
      return result;
    }
    default:
      throw std::runtime_error("unsupported structured value");
  }
  return StructuredValueHandle(raw, OgaDestroyStructuredValue);
}

inline OgaStructuredValue ReadValue(const OgaStructuredValueHandle* source) {
  OgaStructuredValueType type{};
  OgaCheckResult(OgaStructuredValueGetType(source, &type));
  switch (type) {
    case OgaStructuredValueType_Null:
      return {};
    case OgaStructuredValueType_Bool: {
      bool value{};
      OgaCheckResult(OgaStructuredValueGetBool(source, &value));
      return value;
    }
    case OgaStructuredValueType_Int64: {
      int64_t value{};
      OgaCheckResult(OgaStructuredValueGetInt64(source, &value));
      return value;
    }
    case OgaStructuredValueType_Double: {
      double value{};
      OgaCheckResult(OgaStructuredValueGetDouble(source, &value));
      return value;
    }
    case OgaStructuredValueType_String: {
      const char* value{};
      OgaCheckResult(OgaStructuredValueGetString(source, &value));
      return value;
    }
    case OgaStructuredValueType_Array: {
      size_t count{};
      OgaCheckResult(OgaStructuredValueGetCount(source, &count));
      OgaStructuredValue::Array result;
      result.reserve(count);
      for (size_t i = 0; i < count; ++i) {
        const OgaStructuredValueHandle* child{};
        OgaCheckResult(OgaStructuredValueGetArrayItem(source, i, &child));
        result.push_back(ReadValue(child));
      }
      return result;
    }
    case OgaStructuredValueType_Object: {
      size_t count{};
      OgaCheckResult(OgaStructuredValueGetCount(source, &count));
      OgaStructuredValue::Object result;
      result.reserve(count);
      for (size_t i = 0; i < count; ++i) {
        const char* key{};
        const OgaStructuredValueHandle* child{};
        OgaCheckResult(OgaStructuredValueGetObjectItem(source, i, &key, &child));
        result.emplace_back(key, ReadValue(child));
      }
      return result;
    }
  }
  throw std::runtime_error("unsupported structured value type");
}

inline StructuredRequestHandle MakeRequest(const OgaStructuredRequest& source) {
  OgaStructuredRequestHandle* raw{};
  OgaCheckResult(OgaCreateStructuredRequest(&raw));
  StructuredRequestHandle result(raw, OgaDestroyStructuredRequest);
  auto state = MakeValue(source.state);
  OgaCheckResult(OgaStructuredRequestSetState(result.get(), state.get()));
  OgaCheckResult(OgaStructuredRequestSetTemperature(result.get(), source.temperature));
  for (const auto& [id, question] : source.questions) {
    auto instructions = MakeValue(question.instructions);
    auto criteria = MakeValue(question.criteria);
    OgaQuestionHandle* question_raw{};
    OgaCheckResult(OgaCreateQuestion(question.type.c_str(), instructions.get(),
                                     criteria.get(), &question_raw));
    QuestionHandle question_handle(question_raw, OgaDestroyQuestion);
    OgaCheckResult(
        OgaStructuredRequestAddQuestion(result.get(), id.c_str(), question_handle.get()));
  }
  return result;
}

inline RankRequestHandle MakeRequest(const OgaFreeFormRankRequest& source) {
  OgaFreeFormRankRequestHandle* raw{};
  OgaCheckResult(OgaCreateFreeFormRankRequest(&raw));
  RankRequestHandle result(raw, OgaDestroyFreeFormRankRequest);
  auto state = MakeValue(source.state);
  auto instructions = MakeValue(source.instructions);
  OgaCheckResult(OgaFreeFormRankRequestSetState(result.get(), state.get()));
  OgaCheckResult(
      OgaFreeFormRankRequestSetInstructions(result.get(), instructions.get()));
  OgaCheckResult(OgaFreeFormRankRequestSetTemperature(result.get(), source.temperature));
  for (const auto& [key, value] : source.candidates) {
    auto native_value = MakeValue(value);
    OgaCheckResult(OgaFreeFormRankRequestAddCandidate(
        result.get(), key.c_str(), native_value.get()));
  }
  return result;
}

inline OgaModelResult ReadModelResult(const OgaModelResultHandle* source) {
  const char* model{};
  OgaCheckResult(OgaModelResultGetModel(source, &model));
  OgaModelResult result;
  result.model = model;
  size_t count{};
  OgaCheckResult(OgaModelResultGetAnswerCount(source, &count));
  result.answers.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const char* id{};
    const char* type{};
    OgaCheckResult(OgaModelResultGetAnswerId(source, i, &id));
    OgaCheckResult(OgaModelResultGetAnswerType(source, i, &type));
    OgaAnswer answer;
    answer.type = type;
    bool present{};
    double number{};
    OgaCheckResult(OgaModelResultGetAnswerNoul(source, i, &number, &present));
    if (present) answer.noul = number;
    const char* choice{};
    OgaCheckResult(OgaModelResultGetAnswerChoice(source, i, &choice, &present));
    if (present) answer.choice = choice;
    OgaCheckResult(OgaModelResultGetAnswerScore(source, i, &number, &present));
    if (present) answer.score = number;
    OgaCheckResult(OgaModelResultGetAnswerConfidence(source, i, &number, &present));
    if (present) answer.confidence = number;
    size_t item_count{};
    OgaCheckResult(OgaModelResultGetProbabilityCount(source, i, &item_count));
    for (size_t j = 0; j < item_count; ++j) {
      const char* key{};
      OgaCheckResult(OgaModelResultGetProbability(source, i, j, &key, &number));
      answer.probabilities.emplace_back(key, number);
    }
    OgaCheckResult(OgaModelResultGetLegendCount(source, i, &item_count));
    for (size_t j = 0; j < item_count; ++j) {
      const char* key{};
      const char* value{};
      OgaCheckResult(OgaModelResultGetLegend(source, i, j, &key, &value));
      answer.legend.emplace_back(key, value);
    }
    result.answers.emplace_back(id, std::move(answer));
  }
  return result;
}

inline OgaRankingResult ReadRankingResult(const OgaRankingResultHandle* source) {
  const char* model{};
  OgaCheckResult(OgaRankingResultGetModel(source, &model));
  OgaRankingResult result;
  result.model = model;
  size_t count{};
  OgaCheckResult(OgaRankingResultGetCount(source, &count));
  result.ranked.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    OgaRankedItem item;
    const char* key{};
    const OgaStructuredValueHandle* value{};
    OgaCheckResult(OgaRankingResultGetRank(source, i, &item.rank));
    OgaCheckResult(OgaRankingResultGetKey(source, i, &key));
    OgaCheckResult(OgaRankingResultGetValue(source, i, &value));
    OgaCheckResult(OgaRankingResultGetProbability(source, i, &item.probability));
    item.key = key;
    item.value = ReadValue(value);
    result.ranked.push_back(std::move(item));
  }
  return result;
}

}  // namespace OgaDetail

inline NamedComponentSession::NamedComponentSession(
    const std::string& package_path, const std::string& component,
    const std::vector<std::string>& providers) {
  const auto provider_ptrs = OgaDetail::StringPointers(providers);
  OgaCheckResult(OgaCreateComponentSession(
      package_path.c_str(), component.c_str(), provider_ptrs.data(),
      provider_ptrs.size(), &handle_));
  try {
    LoadMetadata();
  } catch (...) {
    OgaDestroyComponentSession(std::exchange(handle_, nullptr));
    throw;
  }
}

inline NamedComponentSession::NamedComponentSession(OgaComponentSession* handle)
    : handle_(handle) {
  if (!handle_) throw std::invalid_argument("component session handle must not be null");
  try {
    LoadMetadata();
  } catch (...) {
    OgaDestroyComponentSession(std::exchange(handle_, nullptr));
    throw;
  }
}

inline void NamedComponentSession::LoadMetadata() {
  size_t count{};
  OgaCheckResult(OgaComponentSessionGetInputCount(handle_, &count));
  input_names_.reserve(count);
  inputs_.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const char* name{};
    OgaComponentInfo info;
    OgaCheckResult(OgaComponentSessionGetInputName(handle_, i, &name));
    OgaCheckResult(OgaComponentSessionGetInputType(handle_, i, &info.type));
    info.name = name;
    size_t rank{};
    OgaCheckResult(OgaComponentSessionGetInputShapeRank(handle_, i, &rank));
    info.shape.resize(rank);
    info.symbolic_dimensions.resize(rank);
    for (size_t dimension = 0; dimension < rank; ++dimension) {
      const char* symbol{};
      OgaCheckResult(OgaComponentSessionGetInputShapeDimension(
          handle_, i, dimension, &info.shape[dimension]));
      OgaCheckResult(OgaComponentSessionGetInputSymbolicDimension(
          handle_, i, dimension, &symbol));
      info.symbolic_dimensions[dimension] = symbol;
    }
    input_names_.emplace_back(name);
    inputs_.push_back(std::move(info));
  }
  OgaCheckResult(OgaComponentSessionGetOutputCount(handle_, &count));
  output_names_.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const char* name{};
    OgaCheckResult(OgaComponentSessionGetOutputName(handle_, i, &name));
    output_names_.emplace_back(name);
  }
}

inline std::vector<OgaComponentTensor> NamedComponentSession::Run(
    const std::vector<OgaComponentInput>& inputs,
    const std::vector<std::string>& outputs) {
  OgaComponentInputs* input_raw{};
  OgaCheckResult(OgaCreateComponentInputs(&input_raw));
  OgaDetail::Handle<OgaComponentInputs, OgaDestroyComponentInputs>
      native_inputs(input_raw, OgaDestroyComponentInputs);
  for (const auto& input : inputs)
    OgaCheckResult(OgaComponentInputsAdd(
        native_inputs.get(), input.name.c_str(), input.data, input.byte_count,
        input.shape.data(), input.shape.size(), input.type));
  const auto output_ptrs = OgaDetail::StringPointers(outputs);
  OgaComponentTensors* tensor_raw{};
  OgaCheckResult(OgaComponentSessionRun(
      handle_, native_inputs.get(), output_ptrs.data(), output_ptrs.size(), &tensor_raw));
  OgaDetail::Handle<OgaComponentTensors, OgaDestroyComponentTensors>
      tensors(tensor_raw, OgaDestroyComponentTensors);
  size_t count{};
  OgaCheckResult(OgaComponentTensorsGetCount(tensors.get(), &count));
  std::vector<OgaComponentTensor> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    OgaComponentTensor tensor;
    const char* name{};
    OgaCheckResult(OgaComponentTensorsGetName(tensors.get(), i, &name));
    OgaCheckResult(OgaComponentTensorsGetType(tensors.get(), i, &tensor.type));
    tensor.name = name;
    size_t rank{};
    OgaCheckResult(OgaComponentTensorsGetShapeRank(tensors.get(), i, &rank));
    tensor.shape.resize(rank);
    for (size_t dimension = 0; dimension < rank; ++dimension)
      OgaCheckResult(OgaComponentTensorsGetShapeDimension(
          tensors.get(), i, dimension, &tensor.shape[dimension]));
    const void* data{};
    size_t byte_count{};
    OgaCheckResult(
        OgaComponentTensorsGetData(tensors.get(), i, &data, &byte_count));
    tensor.data.resize(byte_count);
    if (byte_count) std::memcpy(tensor.data.data(), data, byte_count);
    result.push_back(std::move(tensor));
  }
  return result;
}

inline const std::vector<std::string>& NamedComponentSession::InputNames() const {
  return input_names_;
}
inline const std::vector<std::string>& NamedComponentSession::OutputNames() const {
  return output_names_;
}
inline const std::vector<OgaComponentInfo>& NamedComponentSession::Inputs() const {
  return inputs_;
}

inline DirectoryTokenizer::DirectoryTokenizer(const std::string& package_path) {
  OgaCheckResult(OgaCreateDirectoryTokenizer(package_path.c_str(), &handle_));
}
inline std::vector<int32_t> DirectoryTokenizer::Encode(const std::string& text) const {
  OgaTokenIds* raw{};
  OgaCheckResult(OgaDirectoryTokenizerEncode(handle_, text.c_str(), &raw));
  OgaDetail::Handle<OgaTokenIds, OgaDestroyTokenIds> values(raw, OgaDestroyTokenIds);
  const int32_t* data{};
  size_t count{};
  OgaCheckResult(OgaTokenIdsGetData(values.get(), &data, &count));
  if (!count) return {};
  return {data, data + count};
}
inline int32_t DirectoryTokenizer::PadTokenId() const {
  int32_t result{};
  OgaCheckResult(OgaDirectoryTokenizerGetPadTokenId(handle_, &result));
  return result;
}

inline RankingSession::RankingSession(
    std::string package_path, std::vector<std::string> providers) {
  const auto provider_ptrs = OgaDetail::StringPointers(providers);
  OgaCheckResult(OgaCreateRankingSession(
      package_path.c_str(), provider_ptrs.data(), provider_ptrs.size(), &handle_));
}
inline NamedComponentSession RankingSession::Component(const std::string& name) const {
  OgaComponentSession* result{};
  OgaCheckResult(OgaRankingSessionCreateComponent(handle_, name.c_str(), &result));
  return NamedComponentSession(result);
}
inline OgaModelResult RankingSession::Run(const OgaStructuredRequest& request) {
  auto native_request = OgaDetail::MakeRequest(request);
  OgaModelResultHandle* raw{};
  OgaCheckResult(OgaRankingSessionRun(handle_, native_request.get(), &raw));
  OgaDetail::ModelResultHandle result(raw, OgaDestroyModelResult);
  return OgaDetail::ReadModelResult(result.get());
}
inline OgaRankingResult RankingSession::Rank(
    const OgaFreeFormRankRequest& request) {
  auto native_request = OgaDetail::MakeRequest(request);
  OgaRankingResultHandle* raw{};
  OgaCheckResult(OgaRankingSessionRank(handle_, native_request.get(), &raw));
  OgaDetail::RankingResultHandle result(raw, OgaDestroyRankingResult);
  return OgaDetail::ReadRankingResult(result.get());
}

inline void RankingSession::SetCacheCapacity(size_t entries, size_t bytes) {
  OgaCheckResult(OgaRankingSessionSetCacheCapacity(handle_, entries, bytes));
}
inline OgaNonGenerativeCacheStats RankingSession::CacheStats() const {
  OgaNonGenerativeCacheStats result{};
  OgaCheckResult(OgaRankingSessionGetCacheStats(handle_, &result));
  return result;
}
inline void RankingSession::ClearCache() {
  OgaCheckResult(OgaRankingSessionClearCache(handle_));
}
inline void RankingSession::InvalidateCache() {
  OgaCheckResult(OgaRankingSessionInvalidateCache(handle_));
}

inline DecisionSession::DecisionSession(
    std::string package_path, std::vector<std::string> providers) {
  const auto provider_ptrs = OgaDetail::StringPointers(providers);
  OgaCheckResult(OgaCreateDecisionSession(
      package_path.c_str(), provider_ptrs.data(), provider_ptrs.size(), &handle_));
}
inline NamedComponentSession DecisionSession::Component(const std::string& name) const {
  OgaComponentSession* result{};
  OgaCheckResult(OgaDecisionSessionCreateComponent(handle_, name.c_str(), &result));
  return NamedComponentSession(result);
}
inline OgaModelResult DecisionSession::Run(const OgaStructuredRequest& request) {
  auto native_request = OgaDetail::MakeRequest(request);
  OgaModelResultHandle* raw{};
  OgaCheckResult(OgaDecisionSessionRun(handle_, native_request.get(), &raw));
  OgaDetail::ModelResultHandle result(raw, OgaDestroyModelResult);
  return OgaDetail::ReadModelResult(result.get());
}
inline OgaModelResult DecisionSession::Decide(const OgaStructuredRequest& request) {
  auto native_request = OgaDetail::MakeRequest(request);
  OgaModelResultHandle* raw{};
  OgaCheckResult(OgaDecisionSessionDecide(handle_, native_request.get(), &raw));
  OgaDetail::ModelResultHandle result(raw, OgaDestroyModelResult);
  return OgaDetail::ReadModelResult(result.get());
}
inline void DecisionSession::SetCacheCapacity(size_t entries, size_t bytes) {
  OgaCheckResult(OgaDecisionSessionSetCacheCapacity(handle_, entries, bytes));
}
inline OgaNonGenerativeCacheStats DecisionSession::CacheStats() const {
  OgaNonGenerativeCacheStats result{};
  OgaCheckResult(OgaDecisionSessionGetCacheStats(handle_, &result));
  return result;
}
inline void DecisionSession::SetPrefixReuseEnabled(bool enabled) {
  OgaCheckResult(OgaDecisionSessionSetPrefixReuseEnabled(handle_, enabled));
}
inline bool DecisionSession::PrefixReuseEnabled() const {
  bool result{};
  OgaCheckResult(OgaDecisionSessionGetPrefixReuseEnabled(handle_, &result));
  return result;
}
inline std::string DecisionSession::PrefixReuseStatus() const {
  size_t required{};
  OgaCheckResult(OgaDecisionSessionCopyPrefixReuseStatus(
      handle_, nullptr, 0, &required));
  std::string result(required, '\0');
  if (required) {
    OgaCheckResult(OgaDecisionSessionCopyPrefixReuseStatus(
        handle_, result.data(), required, &required));
    result.resize(required - 1);
  }
  return result;
}
inline void DecisionSession::SetPrefixCacheCapacity(size_t entries, size_t bytes) {
  OgaCheckResult(OgaDecisionSessionSetPrefixCacheCapacity(handle_, entries, bytes));
}
inline OgaNonGenerativeCacheStats DecisionSession::PrefixCacheStats() const {
  OgaNonGenerativeCacheStats result{};
  OgaCheckResult(OgaDecisionSessionGetPrefixCacheStats(handle_, &result));
  return result;
}
inline OgaKevPrefixReuseStats DecisionSession::PrefixReuseStats() const {
  OgaKevPrefixReuseStats result{};
  OgaCheckResult(OgaDecisionSessionGetPrefixReuseStats(handle_, &result));
  return result;
}
inline void DecisionSession::ClearCache() {
  OgaCheckResult(OgaDecisionSessionClearCache(handle_));
}
inline void DecisionSession::InvalidateCache() {
  OgaCheckResult(OgaDecisionSessionInvalidateCache(handle_));
}

struct OgaSpeculativeStats : OgaAbstract {
  uint64_t GetCount(const char* name) const {
    uint64_t value;
    OgaCheckResult(OgaSpeculativeStatsGetCount(this, name, &value));
    return value;
  }

  uint64_t GetAcceptanceLengthCount(size_t accepted_length) const {
    uint64_t value{};
    OgaCheckResult(OgaSpeculativeStatsGetAcceptanceLengthCount(
        this, accepted_length, &value));
    return value;
  }

  size_t GetAcceptanceLengthHistogramSize() const {
    size_t value{};
    OgaCheckResult(OgaSpeculativeStatsGetAcceptanceLengthHistogramSize(this, &value));
    return value;
  }

  double GetNumber(const char* name) const {
    double value;
    OgaCheckResult(OgaSpeculativeStatsGetNumber(this, name, &value));
    return value;
  }

  bool GetBool(const char* name) const {
    bool value;
    OgaCheckResult(OgaSpeculativeStatsGetBool(this, name, &value));
    return value;
  }

  static void operator delete(void* p) {
    OgaDestroySpeculativeStats(reinterpret_cast<OgaSpeculativeStats*>(p));
  }
};

struct OgaFloat16_t;
struct OgaBFloat16_t;

// Variable templates to convert a C++ type into it's OgaElementType
template <typename T>
inline constexpr OgaElementType OgaTypeToElementType = T::Unsupported_Type;  // Force a compile error if hit, please add specialized version if type is valid
template <>
inline constexpr OgaElementType OgaTypeToElementType<bool> = OgaElementType_bool;
template <>
inline constexpr OgaElementType OgaTypeToElementType<int8_t> = OgaElementType_int8;
template <>
inline constexpr OgaElementType OgaTypeToElementType<uint8_t> = OgaElementType_uint8;
template <>
inline constexpr OgaElementType OgaTypeToElementType<int16_t> = OgaElementType_int16;
template <>
inline constexpr OgaElementType OgaTypeToElementType<uint16_t> = OgaElementType_uint16;
template <>
inline constexpr OgaElementType OgaTypeToElementType<int32_t> = OgaElementType_int32;
template <>
inline constexpr OgaElementType OgaTypeToElementType<uint32_t> = OgaElementType_uint32;
template <>
inline constexpr OgaElementType OgaTypeToElementType<int64_t> = OgaElementType_int64;
template <>
inline constexpr OgaElementType OgaTypeToElementType<uint64_t> = OgaElementType_uint64;
template <>
inline constexpr OgaElementType OgaTypeToElementType<float> = OgaElementType_float32;
template <>
inline constexpr OgaElementType OgaTypeToElementType<double> = OgaElementType_float64;
template <>
inline constexpr OgaElementType OgaTypeToElementType<OgaFloat16_t> = OgaElementType_float16;
template <>
inline constexpr OgaElementType OgaTypeToElementType<OgaBFloat16_t> = OgaElementType_bfloat16;

struct OgaString {
  OgaString(const char* p) : p_{p} {}
  ~OgaString() { OgaDestroyString(p_); }

  operator const char*() const { return p_; }

  const char* p_;
};

struct OgaStringArray {
  static std::unique_ptr<OgaStringArray> Create() {
    OgaStringArray* p;
    OgaCheckResult(OgaCreateStringArray(&p));
    return std::unique_ptr<OgaStringArray>(p);
  }

  static std::unique_ptr<OgaStringArray> Create(const char* const* strings, size_t count) {
    OgaStringArray* p;
    OgaCheckResult(OgaCreateStringArrayFromStrings(strings, count, &p));
    return std::unique_ptr<OgaStringArray>(p);
  }

  void Add(const char* str) {
    OgaCheckResult(OgaStringArrayAddString(this, str));
  }

  const char* Get(size_t index) const {
    const char* p;
    OgaCheckResult(OgaStringArrayGetString(this, index, &p));
    return p;
  }

  size_t Count() const {
    size_t count;
    OgaCheckResult(OgaStringArrayGetCount(this, &count));
    return count;
  }

  static void operator delete(void* p) { OgaDestroyStringArray(reinterpret_cast<OgaStringArray*>(p)); }
};

struct OgaRuntimeSettings : OgaAbstract {
  static std::unique_ptr<OgaRuntimeSettings> Create() {
    OgaRuntimeSettings* p;
    OgaCheckResult(OgaCreateRuntimeSettings(&p));
    return std::unique_ptr<OgaRuntimeSettings>(p);
  }

  void SetHandle(const char* name, void* handle) {
    OgaCheckResult(OgaRuntimeSettingsSetHandle(this, name, handle));
  }
  void SetHandle(const std::string& name, void* handle) {
    SetHandle(name.c_str(), handle);
  }

  static void operator delete(void* p) { OgaDestroyRuntimeSettings(reinterpret_cast<OgaRuntimeSettings*>(p)); }
};

struct OgaConfig : OgaAbstract {
  static std::unique_ptr<OgaConfig> Create(const char* config_path) {
    OgaConfig* p;
    OgaCheckResult(OgaCreateConfig(config_path, &p));
    return std::unique_ptr<OgaConfig>(p);
  }
  static std::unique_ptr<OgaConfig> CreateFromPackageEp(const char* config_path, const char* ep) {
    OgaConfig* p;
    OgaCheckResult(OgaCreateConfigFromPackageEp(config_path, ep, &p));
    return std::unique_ptr<OgaConfig>(p);
  }

  void ClearProviders() {
    OgaCheckResult(OgaConfigClearProviders(this));
  }

  void AppendProvider(const char* provider) {
    OgaCheckResult(OgaConfigAppendProvider(this, provider));
  }

  void SetProviderOption(const char* provider, const char* name, const char* value) {
    OgaCheckResult(OgaConfigSetProviderOption(this, provider, name, value));
  }

  void Overlay(const char* json) {
    OgaCheckResult(OgaConfigOverlay(this, json));
  }

  void AddModelData(const std::string& model_filename, const void* model_data, size_t model_data_length) {
    OgaCheckResult(OgaConfigAddModelData(this, model_filename.c_str(), model_data, model_data_length));
  }

  void AddModelData(const std::string& model_filename, const std::vector<std::byte>& model_data) {
    OgaCheckResult(OgaConfigAddModelData(this, model_filename.c_str(), model_data.data(), model_data.size()));
  }

#if OGA_USE_SPAN
  void AddModelData(const std::string& model_filename, std::span<const std::byte> model_data) {
    OgaCheckResult(OgaConfigAddModelData(this, model_filename.c_str(), model_data.data(), model_data.size()));
  }
#endif

  void RemoveModelData(const std::string& model_filename) {
    OgaCheckResult(OgaConfigRemoveModelData(this, model_filename.c_str()));
  }

  void SetDecoderProviderOptionsHardwareDeviceType(const char* provider, const char* hardware_device_type) {
    OgaCheckResult(OgaConfigSetDecoderProviderOptionsHardwareDeviceType(this, provider, hardware_device_type));
  }

  void SetDecoderProviderOptionsHardwareDeviceId(const char* provider, uint32_t hardware_device_id) {
    OgaCheckResult(OgaConfigSetDecoderProviderOptionsHardwareDeviceId(this, provider, hardware_device_id));
  }

  void SetDecoderProviderOptionsHardwareVendorId(const char* provider, uint32_t hardware_vendor_id) {
    OgaCheckResult(OgaConfigSetDecoderProviderOptionsHardwareVendorId(this, provider, hardware_vendor_id));
  }

  void ClearDecoderProviderOptionsHardwareDeviceType(const char* provider) {
    OgaCheckResult(OgaConfigClearDecoderProviderOptionsHardwareDeviceType(this, provider));
  }

  void ClearDecoderProviderOptionsHardwareDeviceId(const char* provider) {
    OgaCheckResult(OgaConfigClearDecoderProviderOptionsHardwareDeviceId(this, provider));
  }

  void ClearDecoderProviderOptionsHardwareVendorId(const char* provider) {
    OgaCheckResult(OgaConfigClearDecoderProviderOptionsHardwareVendorId(this, provider));
  }

  static void operator delete(void* p) { OgaDestroyConfig(reinterpret_cast<OgaConfig*>(p)); }
};

struct OgaModel : OgaAbstract {
  static std::unique_ptr<OgaModel> Create(const char* config_path) {
    OgaModel* p;
    OgaCheckResult(OgaCreateModel(config_path, &p));
    return std::unique_ptr<OgaModel>(p);
  }
  static std::unique_ptr<OgaModel> Create(const char* config_path, const OgaRuntimeSettings& settings) {
    OgaModel* p;
    OgaCheckResult(OgaCreateModelWithRuntimeSettings(config_path, &settings, &p));
    return std::unique_ptr<OgaModel>(p);
  }
  static std::unique_ptr<OgaModel> Create(const OgaConfig& config) {
    OgaModel* p;
    OgaCheckResult(OgaCreateModelFromConfig(&config, &p));
    return std::unique_ptr<OgaModel>(p);
  }

  OgaString GetType() const {
    const char* p;
    OgaCheckResult(OgaModelGetType(this, &p));
    return p;
  }

  OgaString GetDeviceType() const {
    const char* p;
    OgaCheckResult(OgaModelGetDeviceType(this, &p));
    return p;
  }

  static void operator delete(void* p) { OgaDestroyModel(reinterpret_cast<OgaModel*>(p)); }
};

struct OgaSequences : OgaAbstract {
  static std::unique_ptr<OgaSequences> Create() {
    OgaSequences* p;
    OgaCheckResult(OgaCreateSequences(&p));
    return std::unique_ptr<OgaSequences>(p);
  }

  size_t Count() const {
    return OgaSequencesCount(this);
  }

  size_t SequenceCount(size_t index) const {
    return OgaSequencesGetSequenceCount(this, index);
  }

  const int32_t* SequenceData(size_t index) const {
    return OgaSequencesGetSequenceData(this, index);
  }

  void Append(const int32_t* tokens, size_t token_cnt) {
    OgaCheckResult(OgaAppendTokenSequence(tokens, token_cnt, this));
  }

  void Append(int32_t token, size_t sequence_index) {
    OgaCheckResult(OgaAppendTokenToSequence(token, this, sequence_index));
  }

#if OGA_USE_SPAN
  std::span<const int32_t> Get(size_t index) const {
    return {SequenceData(index), SequenceCount(index)};
  }
  void Append(std::span<const int32_t> sequence) {
    OgaCheckResult(OgaAppendTokenSequence(sequence.data(), sequence.size(), this));
  }
  void Append(const std::vector<int32_t>& sequence) {
    OgaCheckResult(OgaAppendTokenSequence(sequence.data(), sequence.size(), this));
  }
#endif

  static void operator delete(void* p) { OgaDestroySequences(reinterpret_cast<OgaSequences*>(p)); }
};

struct OgaTokenizer : OgaAbstract {
  static std::unique_ptr<OgaTokenizer> Create(const OgaModel& model) {
    OgaTokenizer* p;
    OgaCheckResult(OgaCreateTokenizer(&model, &p));
    return std::unique_ptr<OgaTokenizer>(p);
  }

  static std::unique_ptr<OgaTokenizer> Create(const OgaConfig& config) {
    OgaTokenizer* p;
    OgaCheckResult(OgaCreateTokenizerFromConfig(&config, &p));
    return std::unique_ptr<OgaTokenizer>(p);
  }

  static std::unique_ptr<OgaTokenizer> Create(const char* config_path) {
    OgaTokenizer* p;
    OgaCheckResult(OgaCreateTokenizerFromPath(config_path, &p));
    return std::unique_ptr<OgaTokenizer>(p);
  }

  void UpdateOptions(const char* const* keys, const char* const* values, size_t num_options) {
    OgaCheckResult(OgaUpdateTokenizerOptions(this, keys, values, num_options));
  }

  int32_t GetBosTokenId() const {
    int32_t token_id;
    OgaCheckResult(OgaTokenizerGetBosTokenId(this, &token_id));
    return token_id;
  }

#if OGA_USE_SPAN
  std::span<const int32_t> GetEosTokenIds() const {
    const int32_t* eos_ids;
    size_t count;
    OgaCheckResult(OgaTokenizerGetEosTokenIds(this, &eos_ids, &count));
    return {eos_ids, count};
  }
#else
  std::vector<int32_t> GetEosTokenIds() const {
    const int32_t* eos_ids_ptr;
    size_t count;
    OgaCheckResult(OgaTokenizerGetEosTokenIds(this, &eos_ids_ptr, &count));
    return std::vector<int32_t>(eos_ids_ptr, eos_ids_ptr + count);
  }
#endif

  int32_t GetPadTokenId() const {
    int32_t token_id;
    OgaCheckResult(OgaTokenizerGetPadTokenId(this, &token_id));
    return token_id;
  }

  // Tool-calling and reasoning token IDs (bot/eot/bor/eor).
  // Throws if the model does not define the token.
  int32_t GetBotTokenId() const {
    int32_t token_id;
    OgaCheckResult(OgaTokenizerGetBotTokenId(this, &token_id));
    return token_id;
  }

  int32_t GetEotTokenId() const {
    int32_t token_id;
    OgaCheckResult(OgaTokenizerGetEotTokenId(this, &token_id));
    return token_id;
  }

  int32_t GetBorTokenId() const {
    int32_t token_id;
    OgaCheckResult(OgaTokenizerGetBorTokenId(this, &token_id));
    return token_id;
  }

  int32_t GetEorTokenId() const {
    int32_t token_id;
    OgaCheckResult(OgaTokenizerGetEorTokenId(this, &token_id));
    return token_id;
  }

  void Encode(const char* str, OgaSequences& sequences) const {
    OgaCheckResult(OgaTokenizerEncode(this, str, &sequences));
  }

  std::unique_ptr<OgaTensor> EncodeBatch(const char** strings, size_t count) const {
    OgaTensor* out;
    OgaCheckResult(OgaTokenizerEncodeBatch(this, strings, count, &out));
    return std::unique_ptr<OgaTensor>(out);
  }

  int32_t ToTokenId(const char* str) const {
    int32_t token_id;
    OgaCheckResult(OgaTokenizerToTokenId(this, str, &token_id));
    return token_id;
  }

  OgaString Decode(const int32_t* tokens_data, size_t tokens_length) const {
    const char* p;
    OgaCheckResult(OgaTokenizerDecode(this, tokens_data, tokens_length, &p));
    return p;
  }

  OgaString ApplyChatTemplate(const char* template_str, const char* messages, const char* tools, bool add_generation_prompt) const {
    const char* p{};
    OgaCheckResult(OgaTokenizerApplyChatTemplate(this, template_str, messages, tools, add_generation_prompt, &p));
    return p;
  }

#if OGA_USE_SPAN
  OgaString Decode(std::span<const int32_t> tokens) const {
    const char* p;
    OgaCheckResult(OgaTokenizerDecode(this, tokens.data(), tokens.size(), &p));
    return p;
  }
#endif

  std::unique_ptr<OgaStringArray> DecodeBatch(const OgaTensor& tensor) const {
    OgaStringArray* p;
    OgaCheckResult(OgaTokenizerDecodeBatch(this, &tensor, &p));
    return std::unique_ptr<OgaStringArray>(p);
  }

  static void operator delete(void* p) { OgaDestroyTokenizer(reinterpret_cast<OgaTokenizer*>(p)); }
};

struct OgaTokenizerStream : OgaAbstract {
  static std::unique_ptr<OgaTokenizerStream> Create(const OgaTokenizer& tokenizer) {
    OgaTokenizerStream* p;
    OgaCheckResult(OgaCreateTokenizerStream(&tokenizer, &p));
    return std::unique_ptr<OgaTokenizerStream>(p);
  }

  static std::unique_ptr<OgaTokenizerStream> Create(const OgaMultiModalProcessor& processor) {
    OgaTokenizerStream* p;
    OgaCheckResult(OgaCreateTokenizerStreamFromProcessor(&processor, &p));
    return std::unique_ptr<OgaTokenizerStream>(p);
  }

  /*
   * Decode a single token in the stream. If this results in a word being generated, it will be returned in 'out'.
   * The caller is responsible for concatenating each chunk together to generate the complete result.
   * 'out' is valid until the next call to OgaTokenizerStreamDecode or when the OgaTokenizerStream is destroyed
   */
  const char* Decode(int32_t token) {
    const char* out;
    OgaCheckResult(OgaTokenizerStreamDecode(this, token, &out));
    return out;
  }

  static void operator delete(void* p) { OgaDestroyTokenizerStream(reinterpret_cast<OgaTokenizerStream*>(p)); }
};

struct OgaGeneratorParams : OgaAbstract {
  static std::unique_ptr<OgaGeneratorParams> Create(const OgaModel& model) {
    OgaGeneratorParams* p;
    OgaCheckResult(OgaCreateGeneratorParams(&model, &p));
    return std::unique_ptr<OgaGeneratorParams>(p);
  }

  void SetSearchOption(const char* name, double value) {
    OgaCheckResult(OgaGeneratorParamsSetSearchNumber(this, name, value));
  }

  void SetSearchOptionBool(const char* name, bool value) {
    OgaCheckResult(OgaGeneratorParamsSetSearchBool(this, name, value));
  }

  void SetGuidance(const char* type, const char* data, bool enable_ff_tokens = false) {
    OgaCheckResult(OgaGeneratorParamsSetGuidance(this, type, data, enable_ff_tokens));
  }

  double GetSearchNumber(const char* name) const {
    double value;
    OgaCheckResult(OgaGeneratorParamsGetSearchNumber(this, name, &value));
    return value;
  }

  bool GetSearchBool(const char* name) const {
    bool value;
    OgaCheckResult(OgaGeneratorParamsGetSearchBool(this, name, &value));
    return value;
  }

  void SetSpeculativeNumber(const char* name, double value) {
    OgaCheckResult(OgaGeneratorParamsSetSpeculativeNumber(this, name, value));
  }

  double GetSpeculativeNumber(const char* name) const {
    double value;
    OgaCheckResult(OgaGeneratorParamsGetSpeculativeNumber(this, name, &value));
    return value;
  }

  void SetSpeculativeBool(const char* name, bool value) {
    OgaCheckResult(OgaGeneratorParamsSetSpeculativeBool(this, name, value));
  }

  bool GetSpeculativeBool(const char* name) const {
    bool value;
    OgaCheckResult(OgaGeneratorParamsGetSpeculativeBool(this, name, &value));
    return value;
  }

  static void operator delete(void* p) { OgaDestroyGeneratorParams(reinterpret_cast<OgaGeneratorParams*>(p)); }
};

struct OgaGenerator : OgaAbstract {
  static std::unique_ptr<OgaGenerator> Create(const OgaModel& model, OgaGeneratorParams& params) {
    OgaGenerator* p;
    OgaCheckResult(OgaCreateGenerator(&model, &params, &p));
    return std::unique_ptr<OgaGenerator>(p);
  }

  bool IsDone() {
    return OgaGenerator_IsDone(this);
  }

  bool IsSessionTerminated() const {
    return OgaGenerator_IsSessionTerminated(this);
  }

  void SetModelInput(const char* name, OgaTensor& tensor) {
    OgaCheckResult(OgaGenerator_SetModelInput(this, name, &tensor));
  }

  void SetInputs(OgaNamedTensors& named_tensors) {
    OgaCheckResult(OgaGenerator_SetInputs(this, &named_tensors));
  }

  void AppendTokenSequences(const OgaSequences& sequences) {
    OgaCheckResult(OgaGenerator_AppendTokenSequences(this, &sequences));
  }

  void AppendTokens(const int32_t* input_ids, size_t input_ids_count) {
    OgaCheckResult(OgaGenerator_AppendTokens(this, input_ids, input_ids_count));
  }

#if OGA_USE_SPAN
  void AppendTokens(std::span<const int32_t> input_ids) {
    OgaCheckResult(OgaGenerator_AppendTokens(this, input_ids.data(), input_ids.size()));
  }
#endif

  size_t TokenCount() const {
    return OgaGenerator_TokenCount(this);
  }

  void GenerateNextToken() {
    OgaCheckResult(OgaGenerator_GenerateNextToken(this));
  }

#if OGA_USE_SPAN
  std::span<const int32_t> GetNextTokens() {
    const int32_t* out;
    size_t out_count;
    OgaCheckResult(OgaGenerator_GetNextTokens(this, &out, &out_count));
    return {out, out_count};
  }
#else
  std::vector<int32_t> GetNextTokens() {
    const int32_t* out;
    size_t out_count;
    OgaCheckResult(OgaGenerator_GetNextTokens(this, &out, &out_count));
    return std::vector<int32_t>(out, out + out_count);
  }
#endif

  void RewindTo(size_t new_length) {
    OgaCheckResult(OgaGenerator_RewindTo(this, new_length));
  }

  void SnapshotState() {
    OgaCheckResult(OgaGenerator_SnapshotState(this));
  }

  void SetHiddenStates(OgaTensor& hidden_states) {
    OgaCheckResult(OgaGenerator_SetHiddenStates(this, &hidden_states));
  }

  void SetRuntimeOption(const char* key, const char* value) {
    OgaCheckResult(OgaGenerator_SetRuntimeOption(this, key, value));
  }

  size_t GetSequenceCount(size_t index) const {
    return OgaGenerator_GetSequenceCount(this, index);
  }

  const int32_t* GetSequenceData(size_t index) const {
    return OgaGenerator_GetSequenceData(this, index);
  }

  std::unique_ptr<OgaSpeculativeStats> GetSpeculativeStats() const {
    OgaSpeculativeStats* stats;
    OgaCheckResult(OgaGenerator_GetSpeculativeStats(this, &stats));
    return std::unique_ptr<OgaSpeculativeStats>(stats);
  }

  std::unique_ptr<OgaTensor> GetInput(const char* name) {
    OgaTensor* out;
    OgaCheckResult(OgaGenerator_GetInput(this, name, &out));
    return std::unique_ptr<OgaTensor>(out);
  }

  std::unique_ptr<OgaTensor> GetOutput(const char* name) {
    OgaTensor* out;
    OgaCheckResult(OgaGenerator_GetOutput(this, name, &out));
    return std::unique_ptr<OgaTensor>(out);
  }

  std::unique_ptr<OgaTensor> GetLogits() {
    OgaTensor* out;
    OgaCheckResult(OgaGenerator_GetLogits(this, &out));
    return std::unique_ptr<OgaTensor>(out);
  }

  void SetLogits(OgaTensor& tensor) {
    OgaCheckResult(OgaGenerator_SetLogits(this, &tensor));
  }

#if OGA_USE_SPAN
  std::span<const int32_t> GetSequence(size_t index) const {
    return {GetSequenceData(index), GetSequenceCount(index)};
  }
#endif

  void SetActiveAdapter(OgaAdapters& adapters, const char* adapter_name) {
    OgaCheckResult(OgaSetActiveAdapter(this, &adapters, adapter_name));
  }

  static void operator delete(void* p) { OgaDestroyGenerator(reinterpret_cast<OgaGenerator*>(p)); }
};

struct OgaMtpGenerator : OgaAbstract {
  static std::unique_ptr<OgaMtpGenerator> Create(const OgaModel& main_model, const OgaModel& mtp_model, OgaGeneratorParams& params) {
    OgaMtpGenerator* p;
    OgaCheckResult(OgaCreateMtpGenerator(&main_model, &mtp_model, &params, &p));
    return std::unique_ptr<OgaMtpGenerator>(p);
  }

  void AppendTokens(const int32_t* input_ids, size_t input_ids_count) {
    OgaCheckResult(OgaMtpGenerator_AppendTokens(this, input_ids, input_ids_count));
  }

  void GenerateNextToken() {
    OgaCheckResult(OgaMtpGenerator_GenerateNextToken(this));
  }

  void Reset() {
    OgaCheckResult(OgaMtpGenerator_Reset(this));
  }

  bool IsDone() const {
    return OgaMtpGenerator_IsDone(this);
  }

  size_t GetSequenceCount() const {
    return OgaMtpGenerator_GetSequenceCount(this);
  }

  const int32_t* GetSequenceData() const {
    return OgaMtpGenerator_GetSequenceData(this);
  }

  size_t GetForwardCount() const { return OgaMtpGenerator_GetForwardCount(this); }
  size_t GetAcceptCount() const { return OgaMtpGenerator_GetAcceptCount(this); }
  size_t GetTrialCount() const { return OgaMtpGenerator_GetTrialCount(this); }
  std::unique_ptr<OgaSpeculativeStats> GetSpeculativeStats() const {
    OgaSpeculativeStats* stats;
    OgaCheckResult(OgaMtpGenerator_GetSpeculativeStats(this, &stats));
    return std::unique_ptr<OgaSpeculativeStats>(stats);
  }

  static void operator delete(void* p) { OgaDestroyMtpGenerator(reinterpret_cast<OgaMtpGenerator*>(p)); }
};

struct OgaTensor : OgaAbstract {
#if OGA_USE_SPAN
  template <typename T>
  static std::unique_ptr<OgaTensor> Create(T* data, std::span<const int64_t> shape) {
    OgaTensor* p;
    OgaCheckResult(OgaCreateTensorFromBuffer(data, shape.data(), shape.size(), OgaTypeToElementType<T>, &p));
    return std::unique_ptr<OgaTensor>(p);
  }

  static std::unique_ptr<OgaTensor> Create(void* data, std::span<const int64_t> shape, OgaElementType type) {
    OgaTensor* p;
    OgaCheckResult(OgaCreateTensorFromBuffer(data, shape.data(), shape.size(), type, &p));
    return std::unique_ptr<OgaTensor>(p);
  }
#endif

  static std::unique_ptr<OgaTensor> Create(void* data, const int64_t* shape_dims, size_t shape_dims_count, OgaElementType element_type) {
    OgaTensor* p;
    OgaCheckResult(OgaCreateTensorFromBuffer(data, shape_dims, shape_dims_count, element_type, &p));
    return std::unique_ptr<OgaTensor>(p);
  }

  OgaElementType Type() {
    OgaElementType type;
    OgaCheckResult(OgaTensorGetType(this, &type));
    return type;
  }

  std::vector<int64_t> Shape() {
    size_t size;
    OgaCheckResult(OgaTensorGetShapeRank(this, &size));
    std::vector<int64_t> shape(size);
    OgaCheckResult(OgaTensorGetShape(this, shape.data(), shape.size()));
    return shape;
  }

  void* Data() {
    void* data;
    OgaCheckResult(OgaTensorGetData(this, &data));
    return data;
  }

  static void operator delete(void* p) { OgaDestroyTensor(reinterpret_cast<OgaTensor*>(p)); }
};

struct OgaImages : OgaAbstract {
  static std::unique_ptr<OgaImages> Load(const std::vector<const char*>& image_paths) {
    OgaImages* p;
    auto strs = OgaStringArray::Create(image_paths.data(), image_paths.size());
    OgaCheckResult(OgaLoadImages(strs.get(), &p));
    return std::unique_ptr<OgaImages>(p);
  }

#if OGA_USE_SPAN
  static std::unique_ptr<OgaImages> Load(std::span<const char* const> image_paths) {
    OgaImages* p;
    auto strs = OgaStringArray::Create(image_paths.data(), image_paths.size());
    OgaCheckResult(OgaLoadImages(strs.get(), &p));
    return std::unique_ptr<OgaImages>(p);
  }
#endif

  static std::unique_ptr<OgaImages> Load(const void** image_data, const size_t* image_data_sizes, size_t count) {
    OgaImages* p;
    OgaCheckResult(OgaLoadImagesFromBuffers(image_data, image_data_sizes, count, &p));
    return std::unique_ptr<OgaImages>(p);
  }

  static void operator delete(void* p) { OgaDestroyImages(reinterpret_cast<OgaImages*>(p)); }
};

struct OgaAudios : OgaAbstract {
  static std::unique_ptr<OgaAudios> Load(const std::vector<const char*>& audio_paths) {
    OgaAudios* p;
    auto strs = OgaStringArray::Create(audio_paths.data(), audio_paths.size());
    OgaCheckResult(OgaLoadAudios(strs.get(), &p));
    return std::unique_ptr<OgaAudios>(p);
  }

#if OGA_USE_SPAN
  static std::unique_ptr<OgaAudios> Load(std::span<const char* const> audio_paths) {
    OgaAudios* p;
    auto strs = OgaStringArray::Create(audio_paths.data(), audio_paths.size());
    OgaCheckResult(OgaLoadAudios(strs.get(), &p));
    return std::unique_ptr<OgaAudios>(p);
  }
#endif

  static std::unique_ptr<OgaAudios> Load(const void** audio_data, const size_t* audio_data_sizes, size_t count) {
    OgaAudios* p;
    OgaCheckResult(OgaLoadAudiosFromBuffers(audio_data, audio_data_sizes, count, &p));
    return std::unique_ptr<OgaAudios>(p);
  }

  static void operator delete(void* p) { OgaDestroyAudios(reinterpret_cast<OgaAudios*>(p)); }
};

struct OgaNamedTensors : OgaAbstract {
  static std::unique_ptr<OgaNamedTensors> Create() {
    OgaNamedTensors* p;
    OgaCheckResult(OgaCreateNamedTensors(&p));
    return std::unique_ptr<OgaNamedTensors>(p);
  }

  std::unique_ptr<OgaTensor> Get(const char* name) {
    OgaTensor* p;
    OgaCheckResult(OgaNamedTensorsGet(this, name, &p));
    return std::unique_ptr<OgaTensor>(p);
  }

  void Set(const char* name, OgaTensor& tensor) {
    OgaCheckResult(OgaNamedTensorsSet(this, name, &tensor));
  }

  void Delete(const char* name) {
    OgaCheckResult(OgaNamedTensorsDelete(this, name));
  }

  size_t Count() const {
    size_t count;
    OgaCheckResult(OgaNamedTensorsCount(this, &count));
    return count;
  }

  std::unique_ptr<OgaStringArray> GetNames() const {
    OgaStringArray* p;
    OgaCheckResult(OgaNamedTensorsGetNames(this, &p));
    return std::unique_ptr<OgaStringArray>(p);
  }

  static void operator delete(void* p) { OgaDestroyNamedTensors(reinterpret_cast<OgaNamedTensors*>(p)); }
};

struct OgaMultiModalProcessor : OgaAbstract {
  static std::unique_ptr<OgaMultiModalProcessor> Create(const OgaModel& model) {
    OgaMultiModalProcessor* p;
    OgaCheckResult(OgaCreateMultiModalProcessor(&model, &p));
    return std::unique_ptr<OgaMultiModalProcessor>(p);
  }

  std::unique_ptr<OgaNamedTensors> ProcessImages(const char* prompt, const OgaImages* images = nullptr) const {
    OgaNamedTensors* p;
    OgaCheckResult(OgaProcessorProcessImages(this, prompt, images, &p));
    return std::unique_ptr<OgaNamedTensors>(p);
  }

  std::unique_ptr<OgaNamedTensors> ProcessImages(const std::vector<const char*>& prompts, const OgaImages* images = nullptr) const {
    OgaNamedTensors* p;
    auto strs = OgaStringArray::Create(prompts.data(), prompts.size());
    OgaCheckResult(OgaProcessorProcessImagesAndPrompts(this, strs.get(), images, &p));
    return std::unique_ptr<OgaNamedTensors>(p);
  }

  std::unique_ptr<OgaNamedTensors> ProcessAudios(const char* prompt, const OgaAudios* audios = nullptr) const {
    OgaNamedTensors* p;
    OgaCheckResult(OgaProcessorProcessAudios(this, prompt, audios, &p));
    return std::unique_ptr<OgaNamedTensors>(p);
  }

  std::unique_ptr<OgaNamedTensors> ProcessAudios(const std::vector<const char*>& prompts, const OgaAudios* audios = nullptr) const {
    OgaNamedTensors* p;
    auto strs = OgaStringArray::Create(prompts.data(), prompts.size());
    OgaCheckResult(OgaProcessorProcessAudiosAndPrompts(this, strs.get(), audios, &p));
    return std::unique_ptr<OgaNamedTensors>(p);
  }

  std::unique_ptr<OgaNamedTensors> ProcessImagesAndAudios(const char* prompt, const OgaImages* images = nullptr, const OgaAudios* audios = nullptr) const {
    OgaNamedTensors* p;
    OgaCheckResult(OgaProcessorProcessImagesAndAudios(this, prompt, images, audios, &p));
    return std::unique_ptr<OgaNamedTensors>(p);
  }

  std::unique_ptr<OgaNamedTensors> ProcessImagesAndAudios(const std::vector<const char*>& prompts, const OgaImages* images = nullptr, const OgaAudios* audios = nullptr) const {
    OgaNamedTensors* p;
    auto strs = OgaStringArray::Create(prompts.data(), prompts.size());
    OgaCheckResult(OgaProcessorProcessImagesAndAudiosAndPrompts(this, strs.get(), images, audios, &p));
    return std::unique_ptr<OgaNamedTensors>(p);
  }

  OgaString Decode(const int32_t* tokens_data, size_t tokens_length) const {
    const char* p;
    OgaCheckResult(OgaProcessorDecode(this, tokens_data, tokens_length, &p));
    return p;
  }

#if OGA_USE_SPAN
  OgaString Decode(std::span<const int32_t> tokens) const {
    const char* p;
    OgaCheckResult(OgaProcessorDecode(this, tokens.data(), tokens.size(), &p));
    return p;
  }
#endif

  static void operator delete(void* p) { OgaDestroyMultiModalProcessor(reinterpret_cast<OgaMultiModalProcessor*>(p)); }
};

struct OgaAdapters : OgaAbstract {
  static std::unique_ptr<OgaAdapters> Create(const OgaModel& model) {
    OgaAdapters* p;
    OgaCheckResult(OgaCreateAdapters(&model, &p));
    return std::unique_ptr<OgaAdapters>(p);
  }

  void LoadAdapter(const char* adapter_file_path,
                   const char* adapter_name) {
    OgaCheckResult(OgaLoadAdapter(this, adapter_file_path, adapter_name));
  }

  void UnloadAdapter(const char* adapter_name) {
    OgaCheckResult(OgaUnloadAdapter(this, adapter_name));
  }

  static void operator delete(void* p) { OgaDestroyAdapters(reinterpret_cast<OgaAdapters*>(p)); }
};

struct OgaTurnUsage : OgaAbstract {
  uint64_t PromptTokens() const {
    uint64_t value{};
    OgaCheckResult(OgaTurnUsageGetPromptTokens(this, &value));
    return value;
  }

  uint64_t GeneratedTokens() const {
    uint64_t value{};
    OgaCheckResult(OgaTurnUsageGetGeneratedTokens(this, &value));
    return value;
  }

  uint64_t CachedPromptTokens() const {
    uint64_t value{};
    OgaCheckResult(OgaTurnUsageGetCachedPromptTokens(this, &value));
    return value;
  }
};

struct OgaEngineCapabilities : OgaAbstract {
  size_t ConfiguredMaxBatchSize() const {
    return OgaEngineCapabilitiesGetConfiguredMaxBatchSize(this);
  }

  size_t MaxScheduledTokens() const {
    return OgaEngineCapabilitiesGetMaxScheduledTokens(this);
  }

  uint64_t MaxRequestLength() const {
    return OgaEngineCapabilitiesGetMaxRequestLength(this);
  }

  static void operator delete(void* p) {
    OgaDestroyEngineCapabilities(reinterpret_cast<OgaEngineCapabilities*>(p));
  }
};

struct OgaEngineEvent : OgaAbstract {
  OgaEngineEventFlags Flags() const {
    OgaEngineEventFlags value{};
    OgaCheckResult(OgaEngineEventGetFlags(this, &value));
    return value;
  }

  std::optional<std::reference_wrapper<const OgaRequest>> Request() const {
    const OgaRequest* value{};
    OgaCheckResult(OgaEngineEventGetRequest(this, &value));
    if (!value) {
      return std::nullopt;
    }
    return *value;
  }

  uint64_t TurnId() const {
    uint64_t value{};
    OgaCheckResult(OgaEngineEventGetTurnId(this, &value));
    return value;
  }

  int32_t Token() const {
    int32_t value{};
    OgaCheckResult(OgaEngineEventGetToken(this, &value));
    return value;
  }

  OgaFinishReason FinishReason() const {
    OgaFinishReason value{};
    OgaCheckResult(OgaEngineEventGetFinishReason(this, &value));
    return value;
  }

  /** Index into the turn's stop-string list that completed a match, or std::nullopt unless
   *  FinishReason() == OgaFinishReason_StopString. */
  std::optional<int32_t> MatchedStopStringIndex() const {
    int32_t value{};
    OgaCheckResult(OgaEngineEventGetMatchedStopStringIndex(this, &value));
    if (value < 0) {
      return std::nullopt;
    }
    return value;
  }

  OgaErrorCode ErrorCode() const {
    OgaErrorCode value{};
    OgaCheckResult(OgaEngineEventGetErrorCode(this, &value));
    return value;
  }

  const OgaTurnUsage& Usage() const {
    const OgaTurnUsage* value{};
    OgaCheckResult(OgaEngineEventGetUsage(this, &value));
    return *value;
  }
};

struct OgaEngineEventBuffer : OgaAbstract {
  static std::unique_ptr<OgaEngineEventBuffer> Create(
      OgaEngine& engine, size_t capacity) {
    OgaEngineEventBuffer* buffer{};
    OgaCheckResult(OgaCreateEngineEventBuffer(&engine, capacity, &buffer));
    return std::unique_ptr<OgaEngineEventBuffer>(buffer);
  }

  size_t Count() const {
    return OgaEngineEventBufferGetCount(this);
  }

  const OgaEngineEvent* Get(size_t index) const {
    return OgaEngineEventBufferGet(this, index);
  }

  static void operator delete(void* p) {
    OgaDestroyEngineEventBuffer(
        reinterpret_cast<OgaEngineEventBuffer*>(p));
  }
};

struct OgaRequestOptions : OgaAbstract {
  static std::unique_ptr<OgaRequestOptions> Create() {
    OgaRequestOptions* options{};
    OgaCheckResult(OgaCreateRequestOptions(&options));
    return std::unique_ptr<OgaRequestOptions>(options);
  }

  /** Total tokens (prompt plus generated, across every Turn) the Request may reach. Zero uses the
   *  configured default capped by a nonzero EngineCapabilities.max_request_length. */
  void SetMaxSessionTokens(uint64_t value) {
    OgaCheckResult(OgaRequestOptionsSetMaxSessionTokens(this, value));
  }

  static void operator delete(void* p) {
    OgaDestroyRequestOptions(reinterpret_cast<OgaRequestOptions*>(p));
  }
};

struct OgaTurnOptions : OgaAbstract {
  static std::unique_ptr<OgaTurnOptions> Create(OgaRequest& request) {
    OgaTurnOptions* options{};
    OgaCheckResult(OgaRequestCreateTurnOptions(&request, &options));
    return std::unique_ptr<OgaTurnOptions>(options);
  }

  /** Caps the tokens this Turn generates. Zero unsets the cap. */
  void SetMaxGeneratedTokens(uint64_t value) {
    OgaCheckResult(OgaTurnOptionsSetMaxGeneratedTokens(this, value));
  }
  /** Masks end-of-sequence until this Turn has generated `value` tokens. Zero unsets it. */
  void SetMinGeneratedTokens(uint64_t value) {
    OgaCheckResult(OgaTurnOptionsSetMinGeneratedTokens(this, value));
  }
  /** Selects random sampling (true) or the top logit (false) for this Turn. */
  void SetDoSample(bool value) {
    OgaCheckResult(OgaTurnOptionsSetDoSample(this, value));
  }
  /** Sampling temperature; zero requests top-logit selection. */
  void SetTemperature(float value) {
    OgaCheckResult(OgaTurnOptionsSetTemperature(this, value));
  }
  /** Nucleus bound between 0.0 and 1.0. */
  void SetTopP(float value) {
    OgaCheckResult(OgaTurnOptionsSetTopP(this, value));
  }
  /** Top-k bound; one requests top-logit selection and zero disables top-k. */
  void SetTopK(int32_t value) {
    OgaCheckResult(OgaTurnOptionsSetTopK(this, value));
  }
  /** Repetition penalty; must be finite and greater than zero. */
  void SetRepetitionPenalty(float value) {
    OgaCheckResult(OgaTurnOptionsSetRepetitionPenalty(this, value));
  }
  /** Forbids repeating any n-gram of this size. Zero disables it. */
  void SetNoRepeatNgramSize(int32_t value) {
    OgaCheckResult(OgaTurnOptionsSetNoRepeatNgramSize(this, value));
  }
  /** Reseeds the Request's random streams at the start of this Turn. Zero is a valid seed. */
  void SetSeed(uint64_t value) {
    OgaCheckResult(OgaTurnOptionsSetSeed(this, value));
  }
  /** Removes a pending reseed, continuing the Request's existing random streams. */
  void ClearSeed() {
    OgaCheckResult(OgaTurnOptionsClearSeed(this));
  }
  /** Copies stop_strings for this turn immediately; reusing or destroying it afterward cannot
   *  affect these options. An empty array (zero entries) clears/disables stop strings; this is
   *  distinct from a nonempty array containing an empty string member, which is invalid. Every
   *  entry in a nonempty array must itself be a nonempty, valid UTF-8 string, and the configuration
   *  as a whole may contain at most 16 entries totaling at most 16 KiB. Matching is exact (no
   *  normalization/trimming/case folding) against only the text this Engine Request generates
   *  during the turn. */
  void SetStopStrings(const OgaStringArray& values) {
    OgaCheckResult(OgaTurnOptionsSetStopStrings(this, &values));
  }
  /** Constrains this Turn's output to a grammar ("json_schema", "regex", or "lark_grammar"),
   *  copying both strings immediately. Guidance is strictly Turn-scoped and cannot be combined
   *  with delimited guidance. */
  void SetGuidance(const char* type, const char* data) {
    OgaCheckResult(OgaTurnOptionsSetGuidance(this, type, data));
  }
  /** Copies a Lark grammar for the body between distinct opening and closing token IDs.
   *  Cannot be combined with whole-turn guidance; ClearGuidance or Reset switches modes. */
  void SetDelimitedGuidance(int32_t opening_token, int32_t closing_token, const char* grammar) {
    OgaCheckResult(OgaTurnOptionsSetDelimitedGuidance(this, opening_token, closing_token, grammar));
  }
  /** Removes whole-turn or delimited guidance, so the Turn is unguided. */
  void ClearGuidance() {
    OgaCheckResult(OgaTurnOptionsClearGuidance(this));
  }
  /** Restores every Turn option to its unset state. */
  void Reset() {
    OgaCheckResult(OgaTurnOptionsReset(this));
  }

  static void operator delete(void* p) {
    OgaDestroyTurnOptions(reinterpret_cast<OgaTurnOptions*>(p));
  }
};

struct OgaRequest : OgaAbstract {
  uint64_t BeginTurn(const int32_t* input_ids, size_t input_ids_count,
                     const OgaTurnOptions* options = nullptr) {
    uint64_t turn_id{};
    OgaCheckResult(OgaRequestBeginTurn(
        this, options, input_ids, static_cast<uint64_t>(input_ids_count),
        &turn_id));
    return turn_id;
  }

#if OGA_USE_SPAN
  uint64_t BeginTurn(std::span<const int32_t> input_ids,
                     const OgaTurnOptions* options = nullptr) {
    return BeginTurn(input_ids.data(), input_ids.size(), options);
  }
#endif

  std::unique_ptr<OgaTurnOptions> CreateTurnOptions() {
    return OgaTurnOptions::Create(*this);
  }

  bool CancelTurn(uint64_t turn_id) {
    bool cancelled{};
    OgaCheckResult(OgaRequestCancelTurn(this, turn_id, &cancelled));
    return cancelled;
  }

  void RewindToStartOfTurn(uint64_t turn_id) {
    OgaCheckResult(OgaRequestRewindToStartOfTurn(this, turn_id));
  }

  /**
   * \brief Proposes speculative draft tokens for the next decode operation.
   *
   * Seeded sampled output is reproducible only with the same proposal and scheduling path.
   */
  void SetDraftTokens(const OgaSequences& tokens) {
    OgaCheckResult(OgaRequestSetDraftTokens(this, &tokens));
  }

  void Close() {
    OgaCheckResult(OgaRequestClose(this));
  }

  static void operator delete(void* p) { OgaDestroyRequest(reinterpret_cast<OgaRequest*>(p)); }
};

struct OgaEngine : OgaAbstract {
  static std::unique_ptr<OgaEngine> Create(OgaModel& model) {
    OgaEngine* p;
    OgaCheckResult(OgaCreateEngine(&model, &p));
    return std::unique_ptr<OgaEngine>(p);
  }

  bool HasPendingRequests() {
    bool f;
    OgaCheckResult(OgaEngineHasPendingRequests(this, &f));
    return f;
  }

  /**
   * \brief Speculative draft tokens a request may attach to one proposal; zero when unsupported.
   */
  size_t MaxDraftTokensPerProposal() const {
    size_t count{};
    OgaCheckResult(OgaEngineMaxDraftTokensPerProposal(this, &count));
    return count;
  }

  std::unique_ptr<OgaEngineCapabilities> GetCapabilities() const {
    OgaEngineCapabilities* capabilities{};
    OgaCheckResult(OgaEngineGetCapabilities(this, &capabilities));
    return std::unique_ptr<OgaEngineCapabilities>(capabilities);
  }

  /**
   * \brief Cumulative target/head work and committed draft acceptance statistics.
   */
  std::unique_ptr<OgaSpeculativeStats> GetSpeculativeStats() const {
    OgaSpeculativeStats* stats;
    OgaCheckResult(OgaEngineGetSpeculativeStats(this, &stats));
    return std::unique_ptr<OgaSpeculativeStats>(stats);
  }

  std::unique_ptr<OgaRequest> CreateRequest(
      const OgaRequestOptions* options = nullptr) {
    OgaRequest* request{};
    OgaCheckResult(OgaEngineCreateRequest(this, options, &request));
    return std::unique_ptr<OgaRequest>(request);
  }

  std::unique_ptr<OgaEngineEventBuffer> CreateEventBuffer(
      size_t capacity) {
    return OgaEngineEventBuffer::Create(*this, capacity);
  }

  size_t Run(OgaEngineEventBuffer& buffer) {
    OgaCheckResult(OgaEngineRun(this, &buffer));
    return buffer.Count();
  }

  static void operator delete(void* p) { OgaDestroyEngine(reinterpret_cast<OgaEngine*>(p)); }
};

/**
 * \brief RAII wrapper that calls OgaShutdown() on destruction.
 *
 * \warning Without explicit shutdown, GenAI's globals are destroyed at static-destruction time in undefined order,
 *          which may crash.
 *
 * \note Typical usage is to construct an instance early in the program so its destructor runs before process exit.
 *
 * \note Only one OgaHandle should be live in the process, and its scope must encompass all GenAI use. GenAI's globals
 *       are not re-creatable after OgaShutdown(); a second OgaHandle whose lifetime starts after the first one's
 *       destruction would leave subsequent GenAI calls broken.
 */
struct OgaHandle {
  OgaHandle() = default;
  ~OgaHandle() noexcept {
    OgaShutdown();
  }
};

// Global Oga functions
namespace Oga {

inline void SetLogBool(const char* name, bool value) {
  OgaCheckResult(OgaSetLogBool(name, value));
}

inline void SetLogString(const char* name, const char* value) {
  OgaCheckResult(OgaSetLogString(name, value));
}

inline void SetLogCallback(void (*callback)(const char* string, size_t length)) {
  OgaCheckResult(OgaSetLogCallback(callback));
}

inline void SetCurrentGpuDeviceId(int device_id) {
  OgaCheckResult(OgaSetCurrentGpuDeviceId(device_id));
}

inline int GetCurrentGpuDeviceId() {
  int device_id;
  OgaCheckResult(OgaGetCurrentGpuDeviceId(&device_id));
  return device_id;
}

inline void SetTelemetryEnabled(bool enabled) {
  OgaSetTelemetryEnabled(enabled);
}

}  // namespace Oga

struct OgaStreamingProcessor : OgaAbstract {
  static std::unique_ptr<OgaStreamingProcessor> Create(OgaModel& model) {
    OgaStreamingProcessor* p;
    OgaCheckResult(OgaCreateStreamingProcessor(&model, &p));
    return std::unique_ptr<OgaStreamingProcessor>(p);
  }

  std::unique_ptr<OgaNamedTensors> Process(const float* audio_data, size_t num_samples) {
    OgaNamedTensors* out;
    OgaCheckResult(OgaStreamingProcessorProcess(this, audio_data, num_samples, &out));
    return std::unique_ptr<OgaNamedTensors>(out);  // May be nullptr if not enough audio
  }

  std::unique_ptr<OgaNamedTensors> Flush() {
    OgaNamedTensors* out;
    OgaCheckResult(OgaStreamingProcessorFlush(this, &out));
    return std::unique_ptr<OgaNamedTensors>(out);
  }

  void SetOption(const char* key, const char* value) {
    OgaCheckResult(OgaStreamingProcessorSetOption(this, key, value));
  }

  OgaString GetOption(const char* key) const {
    const char* value;
    OgaCheckResult(OgaStreamingProcessorGetOption(this, key, &value));
    return value;
  }

  static void operator delete(void* p) { OgaDestroyStreamingProcessor(reinterpret_cast<OgaStreamingProcessor*>(p)); }
};
