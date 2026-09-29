// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <napi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "ort_genai_c.h"

namespace {

constexpr int64_t kMaxSafeInteger = 9007199254740991LL;
constexpr size_t kMaxStructuredDepth = 128;

void Check(Napi::Env env, OgaResult* result) {
  if (!result) return;
  std::string message = OgaResultGetError(result);
  OgaDestroyResult(result);
  throw Napi::Error::New(env, message);
}

std::string StringArgument(Napi::Env env, const Napi::Value& value, const char* name) {
  if (!value.IsString()) {
    throw Napi::TypeError::New(env, std::string(name) + " must be a string");
  }
  return value.As<Napi::String>().Utf8Value();
}

size_t SizeArgument(Napi::Env env, const Napi::Value& value, const char* name) {
  uint64_t result{};
  if (value.IsBigInt()) {
    bool lossless{};
    result = value.As<Napi::BigInt>().Uint64Value(&lossless);
    if (!lossless) {
      throw Napi::RangeError::New(env, std::string(name) + " must fit an unsigned size_t");
    }
  } else if (value.IsNumber()) {
    const double number = value.As<Napi::Number>().DoubleValue();
    if (!std::isfinite(number) || number < 0 || std::floor(number) != number ||
        number > static_cast<double>(kMaxSafeInteger)) {
      throw Napi::RangeError::New(
          env, std::string(name) + " must be a non-negative safe integer or bigint");
    }
    result = static_cast<uint64_t>(number);
  } else {
    throw Napi::TypeError::New(env, std::string(name) + " must be a number or bigint");
  }
  if (result > std::numeric_limits<size_t>::max()) {
    throw Napi::RangeError::New(env, std::string(name) + " exceeds size_t");
  }
  return static_cast<size_t>(result);
}

Napi::Value SizeValue(Napi::Env env, size_t value) {
  if (value <= static_cast<size_t>(kMaxSafeInteger)) {
    return Napi::Number::New(env, static_cast<double>(value));
  }
  return Napi::BigInt::New(env, static_cast<uint64_t>(value));
}

struct StructuredOwner {
  OgaStructuredValueHandle* value{};
  ~StructuredOwner() { OgaDestroyStructuredValue(value); }
  StructuredOwner() = default;
  StructuredOwner(const StructuredOwner&) = delete;
  StructuredOwner& operator=(const StructuredOwner&) = delete;
};

struct RequestOwner {
  OgaStructuredRequestHandle* value{};
  ~RequestOwner() { OgaDestroyStructuredRequest(value); }
};

struct QuestionOwner {
  OgaQuestionHandle* value{};
  ~QuestionOwner() { OgaDestroyQuestion(value); }
};

struct FreeFormOwner {
  OgaFreeFormRankRequestHandle* value{};
  ~FreeFormOwner() { OgaDestroyFreeFormRankRequest(value); }
};

struct ModelResultOwner {
  OgaModelResultHandle* value{};
  ~ModelResultOwner() { OgaDestroyModelResult(value); }
};

struct RankingResultOwner {
  OgaRankingResultHandle* value{};
  ~RankingResultOwner() { OgaDestroyRankingResult(value); }
};

struct BuildContext {
  size_t depth{};
  std::vector<napi_value> active;
};

class ActiveValue {
 public:
  ActiveValue(Napi::Env env, BuildContext& context, const Napi::Value& value)
      : context_(context) {
    if (context_.depth >= kMaxStructuredDepth) {
      throw Napi::TypeError::New(
          env, "Structured value exceeds the maximum nesting depth of 128");
    }
    for (napi_value active : context_.active) {
      bool equal{};
      const napi_status status = napi_strict_equals(env, active, value, &equal);
      if (status != napi_ok) {
        throw Napi::Error::New(env, "Failed to compare structured object identity");
      }
      if (equal) {
        throw Napi::TypeError::New(env, "Structured values must not contain cycles");
      }
    }
    ++context_.depth;
    context_.active.push_back(value);
  }

  ~ActiveValue() {
    context_.active.pop_back();
    --context_.depth;
  }

 private:
  BuildContext& context_;
};

OgaStructuredValueHandle* BuildValue(
    Napi::Env env, const Napi::Value& input, BuildContext& context) {
  StructuredOwner output;
  if (input.IsNull() || input.IsUndefined()) {
    Check(env, OgaCreateStructuredValueNull(&output.value));
  } else if (input.IsBoolean()) {
    Check(env, OgaCreateStructuredValueBool(input.As<Napi::Boolean>().Value(), &output.value));
  } else if (input.IsBigInt()) {
    bool lossless{};
    const int64_t value = input.As<Napi::BigInt>().Int64Value(&lossless);
    if (!lossless) {
      throw Napi::RangeError::New(env, "Structured bigint must fit signed 64-bit");
    }
    Check(env, OgaCreateStructuredValueInt64(value, &output.value));
  } else if (input.IsNumber()) {
    const double value = input.As<Napi::Number>().DoubleValue();
    if (!std::isfinite(value)) {
      throw Napi::RangeError::New(env, "Structured numbers must be finite");
    }
    if (std::floor(value) == value) {
      if (std::abs(value) > static_cast<double>(kMaxSafeInteger)) {
        throw Napi::RangeError::New(
            env, "Structured integral numbers must be safe integers; use bigint");
      }
      Check(env, OgaCreateStructuredValueInt64(static_cast<int64_t>(value), &output.value));
    } else {
      Check(env, OgaCreateStructuredValueDouble(value, &output.value));
    }
  } else if (input.IsString()) {
    const std::string value = input.As<Napi::String>().Utf8Value();
    Check(env, OgaCreateStructuredValueString(value.c_str(), &output.value));
  } else if (input.IsArray()) {
    ActiveValue active{env, context, input};
    Check(env, OgaCreateStructuredValueArray(&output.value));
    const Napi::Array array = input.As<Napi::Array>();
    for (uint32_t i = 0; i < array.Length(); ++i) {
      StructuredOwner child;
      child.value = BuildValue(env, array.Get(i), context);
      Check(env, OgaStructuredValueArrayAppend(output.value, child.value));
    }
  } else if (input.IsObject() && !input.IsFunction() && !input.IsTypedArray() &&
             !input.IsArrayBuffer() && !input.IsDate()) {
    ActiveValue active{env, context, input};
    Check(env, OgaCreateStructuredValueObject(&output.value));
    const Napi::Object object = input.As<Napi::Object>();
    const Napi::Array keys = object.GetPropertyNames();
    for (uint32_t i = 0; i < keys.Length(); ++i) {
      const Napi::Value key_value = keys.Get(i);
      if (!key_value.IsString()) {
        throw Napi::TypeError::New(env, "Structured object keys must be strings");
      }
      const std::string key = key_value.As<Napi::String>().Utf8Value();
      StructuredOwner child;
      child.value = BuildValue(env, object.Get(key_value), context);
      Check(env, OgaStructuredValueObjectAppend(output.value, key.c_str(), child.value));
    }
  } else {
    throw Napi::TypeError::New(
        env, "Unsupported structured value; expected null, boolean, number, bigint, string, array, or object");
  }
  return std::exchange(output.value, nullptr);
}

OgaStructuredValueHandle* BuildValue(Napi::Env env, const Napi::Value& input) {
  BuildContext context;
  return BuildValue(env, input, context);
}

Napi::Object NewMap(Napi::Env env) {
  return Napi::Object::New(env);
}

void SetDataProperty(
    Napi::Object& object, const std::string& key, const Napi::Value& value) {
  object.DefineProperty(Napi::PropertyDescriptor::Value(
      key, value, static_cast<napi_property_attributes>(
                      napi_writable | napi_enumerable | napi_configurable)));
}

Napi::Value ReadValue(Napi::Env env, const OgaStructuredValueHandle* value) {
  OgaStructuredValueType type{};
  Check(env, OgaStructuredValueGetType(value, &type));
  switch (type) {
    case OgaStructuredValueType_Null:
      return env.Null();
    case OgaStructuredValueType_Bool: {
      bool output{};
      Check(env, OgaStructuredValueGetBool(value, &output));
      return Napi::Boolean::New(env, output);
    }
    case OgaStructuredValueType_Int64: {
      int64_t output{};
      Check(env, OgaStructuredValueGetInt64(value, &output));
      if (output >= -kMaxSafeInteger && output <= kMaxSafeInteger) {
        return Napi::Number::New(env, static_cast<double>(output));
      }
      return Napi::BigInt::New(env, output);
    }
    case OgaStructuredValueType_Double: {
      double output{};
      Check(env, OgaStructuredValueGetDouble(value, &output));
      return Napi::Number::New(env, output);
    }
    case OgaStructuredValueType_String: {
      const char* output{};
      Check(env, OgaStructuredValueGetString(value, &output));
      return Napi::String::New(env, output);
    }
    case OgaStructuredValueType_Array: {
      size_t count{};
      Check(env, OgaStructuredValueGetCount(value, &count));
      if (count > std::numeric_limits<uint32_t>::max()) {
        throw Napi::RangeError::New(env, "Structured array is too large for JavaScript");
      }
      Napi::Array output = Napi::Array::New(env, count);
      for (size_t i = 0; i < count; ++i) {
        const OgaStructuredValueHandle* child{};
        Check(env, OgaStructuredValueGetArrayItem(value, i, &child));
        output.Set(static_cast<uint32_t>(i), ReadValue(env, child));
      }
      return output;
    }
    case OgaStructuredValueType_Object: {
      size_t count{};
      Check(env, OgaStructuredValueGetCount(value, &count));
      Napi::Object output = NewMap(env);
      for (size_t i = 0; i < count; ++i) {
        const char* key{};
        const OgaStructuredValueHandle* child{};
        Check(env, OgaStructuredValueGetObjectItem(value, i, &key, &child));
        SetDataProperty(output, key, ReadValue(env, child));
      }
      return output;
    }
  }
  throw Napi::Error::New(env, "Native runtime returned an unknown structured value type");
}

std::vector<std::string> ReadProviders(
    Napi::Env env, const Napi::CallbackInfo& info, size_t index) {
  std::vector<std::string> providers;
  if (info.Length() <= index || info[index].IsUndefined()) return providers;
  if (!info[index].IsArray()) {
    throw Napi::TypeError::New(env, "providers must be an array of strings");
  }
  const Napi::Array array = info[index].As<Napi::Array>();
  providers.reserve(array.Length());
  for (uint32_t i = 0; i < array.Length(); ++i) {
    if (!array.Get(i).IsString()) {
      throw Napi::TypeError::New(env, "Every provider must be a string");
    }
    providers.push_back(array.Get(i).As<Napi::String>().Utf8Value());
  }
  return providers;
}

std::vector<const char*> ProviderPointers(const std::vector<std::string>& providers) {
  std::vector<const char*> result;
  result.reserve(providers.size());
  for (const std::string& provider : providers) result.push_back(provider.c_str());
  return result;
}

OgaStructuredRequestHandle* BuildRequest(Napi::Env env, const Napi::Value& input) {
  if (!input.IsObject() || input.IsArray()) {
    throw Napi::TypeError::New(env, "request must be an object");
  }
  const Napi::Object object = input.As<Napi::Object>();
  if (!object.HasOwnProperty("state")) {
    throw Napi::TypeError::New(env, "request.state is required");
  }
  if (!object.HasOwnProperty("questions") || !object.Get("questions").IsObject() ||
      object.Get("questions").IsArray()) {
    throw Napi::TypeError::New(env, "request.questions must be an object");
  }

  RequestOwner request;
  Check(env, OgaCreateStructuredRequest(&request.value));
  StructuredOwner state;
  state.value = BuildValue(env, object.Get("state"));
  Check(env, OgaStructuredRequestSetState(request.value, state.value));

  const Napi::Object questions = object.Get("questions").As<Napi::Object>();
  const Napi::Array ids = questions.GetPropertyNames();
  for (uint32_t i = 0; i < ids.Length(); ++i) {
    const Napi::Value id_value = ids.Get(i);
    const std::string id = StringArgument(env, id_value, "question id");
    const Napi::Value item_value = questions.Get(id_value);
    if (!item_value.IsObject() || item_value.IsArray()) {
      throw Napi::TypeError::New(env, "Each question must be an object");
    }
    const Napi::Object item = item_value.As<Napi::Object>();
    const std::string type = StringArgument(env, item.Get("type"), "question.type");
    if (type.empty()) throw Napi::TypeError::New(env, "question.type must not be empty");
    if (!item.HasOwnProperty("instructions")) {
      throw Napi::TypeError::New(env, "question.instructions is required");
    }
    StructuredOwner instructions;
    StructuredOwner criteria;
    instructions.value = BuildValue(env, item.Get("instructions"));
    criteria.value = BuildValue(
        env, item.HasOwnProperty("criteria") ? item.Get("criteria") : env.Null());
    QuestionOwner question;
    Check(env, OgaCreateQuestion(
                   type.c_str(), instructions.value, criteria.value, &question.value));
    Check(env, OgaStructuredRequestAddQuestion(request.value, id.c_str(), question.value));
  }

  if (object.HasOwnProperty("temperature") && !object.Get("temperature").IsUndefined()) {
    const Napi::Value value = object.Get("temperature");
    if (!value.IsNumber() || !std::isfinite(value.As<Napi::Number>().DoubleValue())) {
      throw Napi::TypeError::New(env, "request.temperature must be a finite number");
    }
    Check(env, OgaStructuredRequestSetTemperature(
                   request.value, value.As<Napi::Number>().FloatValue()));
  }
  return std::exchange(request.value, nullptr);
}

OgaFreeFormRankRequestHandle* BuildFreeFormRequest(
    Napi::Env env, const Napi::Value& input) {
  if (!input.IsObject() || input.IsArray()) {
    throw Napi::TypeError::New(env, "request must be an object");
  }
  const Napi::Object object = input.As<Napi::Object>();
  if (!object.HasOwnProperty("state") || !object.HasOwnProperty("instructions")) {
    throw Napi::TypeError::New(env, "request.state and request.instructions are required");
  }
  if (!object.HasOwnProperty("candidates") || !object.Get("candidates").IsObject() ||
      object.Get("candidates").IsArray()) {
    throw Napi::TypeError::New(env, "request.candidates must be an object");
  }

  FreeFormOwner request;
  Check(env, OgaCreateFreeFormRankRequest(&request.value));
  StructuredOwner state;
  StructuredOwner instructions;
  state.value = BuildValue(env, object.Get("state"));
  instructions.value = BuildValue(env, object.Get("instructions"));
  Check(env, OgaFreeFormRankRequestSetState(request.value, state.value));
  Check(env, OgaFreeFormRankRequestSetInstructions(request.value, instructions.value));

  const Napi::Object candidates = object.Get("candidates").As<Napi::Object>();
  const Napi::Array keys = candidates.GetPropertyNames();
  for (uint32_t i = 0; i < keys.Length(); ++i) {
    const Napi::Value key_value = keys.Get(i);
    const std::string key = StringArgument(env, key_value, "candidate key");
    StructuredOwner candidate;
    candidate.value = BuildValue(env, candidates.Get(key_value));
    Check(env, OgaFreeFormRankRequestAddCandidate(
                   request.value, key.c_str(), candidate.value));
  }
  if (object.HasOwnProperty("temperature") && !object.Get("temperature").IsUndefined()) {
    const Napi::Value value = object.Get("temperature");
    if (!value.IsNumber() || !std::isfinite(value.As<Napi::Number>().DoubleValue())) {
      throw Napi::TypeError::New(env, "request.temperature must be a finite number");
    }
    Check(env, OgaFreeFormRankRequestSetTemperature(
                   request.value, value.As<Napi::Number>().FloatValue()));
  }
  return std::exchange(request.value, nullptr);
}

Napi::Object ReadCacheStats(Napi::Env env, const OgaNonGenerativeCacheStats& stats) {
  Napi::Object output = NewMap(env);
  SetDataProperty(output, "hits", Napi::BigInt::New(env, stats.hits));
  SetDataProperty(output, "misses", Napi::BigInt::New(env, stats.misses));
  SetDataProperty(output, "evictions", Napi::BigInt::New(env, stats.evictions));
  SetDataProperty(output, "entries", SizeValue(env, stats.entries));
  SetDataProperty(output, "bytes", SizeValue(env, stats.bytes));
  SetDataProperty(output, "entryCapacity", SizeValue(env, stats.entry_capacity));
  SetDataProperty(output, "byteCapacity", SizeValue(env, stats.byte_capacity));
  return output;
}

Napi::Object ReadModelResult(Napi::Env env, const OgaModelResultHandle* result) {
  const char* model{};
  size_t count{};
  Check(env, OgaModelResultGetModel(result, &model));
  Check(env, OgaModelResultGetAnswerCount(result, &count));
  if (count > std::numeric_limits<uint32_t>::max()) {
    throw Napi::RangeError::New(env, "Native answer count is too large for JavaScript");
  }
  Napi::Object output = NewMap(env);
  SetDataProperty(output, "model", Napi::String::New(env, model));
  Napi::Array answers = Napi::Array::New(env, count);
  for (size_t i = 0; i < count; ++i) {
    const char* id{};
    const char* type{};
    const char* text{};
    double number{};
    bool present{};
    Napi::Object answer = NewMap(env);
    Check(env, OgaModelResultGetAnswerId(result, i, &id));
    Check(env, OgaModelResultGetAnswerType(result, i, &type));
    SetDataProperty(answer, "id", Napi::String::New(env, id));
    SetDataProperty(answer, "type", Napi::String::New(env, type));
    Check(env, OgaModelResultGetAnswerNoul(result, i, &number, &present));
    SetDataProperty(answer, "noul", present ? Napi::Number::New(env, number) : env.Null());
    Check(env, OgaModelResultGetAnswerChoice(result, i, &text, &present));
    SetDataProperty(answer, "choice", present ? Napi::String::New(env, text) : env.Null());
    Check(env, OgaModelResultGetAnswerScore(result, i, &number, &present));
    SetDataProperty(answer, "score", present ? Napi::Number::New(env, number) : env.Null());
    Check(env, OgaModelResultGetAnswerConfidence(result, i, &number, &present));
    SetDataProperty(answer, "confidence", present ? Napi::Number::New(env, number) : env.Null());

    size_t values{};
    Check(env, OgaModelResultGetProbabilityCount(result, i, &values));
    Napi::Object probabilities = NewMap(env);
    for (size_t j = 0; j < values; ++j) {
      const char* key{};
      Check(env, OgaModelResultGetProbability(result, i, j, &key, &number));
      SetDataProperty(probabilities, key, Napi::Number::New(env, number));
    }
    SetDataProperty(answer, "probabilities", probabilities);

    Check(env, OgaModelResultGetLegendCount(result, i, &values));
    Napi::Object legend = NewMap(env);
    for (size_t j = 0; j < values; ++j) {
      const char* key{};
      const char* value{};
      Check(env, OgaModelResultGetLegend(result, i, j, &key, &value));
      SetDataProperty(legend, key, Napi::String::New(env, value));
    }
    SetDataProperty(answer, "legend", legend);
    answers.Set(static_cast<uint32_t>(i), answer);
  }
  SetDataProperty(output, "answers", answers);
  return output;
}

Napi::Object ReadRankingResult(Napi::Env env, const OgaRankingResultHandle* result) {
  const char* model{};
  size_t count{};
  Check(env, OgaRankingResultGetModel(result, &model));
  Check(env, OgaRankingResultGetCount(result, &count));
  if (count > std::numeric_limits<uint32_t>::max()) {
    throw Napi::RangeError::New(env, "Native ranking count is too large for JavaScript");
  }
  Napi::Object output = NewMap(env);
  SetDataProperty(output, "model", Napi::String::New(env, model));
  Napi::Array items = Napi::Array::New(env, count);
  for (size_t i = 0; i < count; ++i) {
    size_t rank{};
    const char* key{};
    const OgaStructuredValueHandle* value{};
    double probability{};
    Check(env, OgaRankingResultGetRank(result, i, &rank));
    Check(env, OgaRankingResultGetKey(result, i, &key));
    Check(env, OgaRankingResultGetValue(result, i, &value));
    Check(env, OgaRankingResultGetProbability(result, i, &probability));
    Napi::Object item = NewMap(env);
    SetDataProperty(item, "rank", SizeValue(env, rank));
    SetDataProperty(item, "key", Napi::String::New(env, key));
    SetDataProperty(item, "value", ReadValue(env, value));
    SetDataProperty(item, "probability", Napi::Number::New(env, probability));
    items.Set(static_cast<uint32_t>(i), item);
  }
  SetDataProperty(output, "items", items);
  return output;
}

class DirectoryTokenizer : public Napi::ObjectWrap<DirectoryTokenizer> {
 public:
  static Napi::Function Define(Napi::Env env) {
    return DefineClass(env, "DirectoryTokenizer", {
      InstanceMethod("encode", &DirectoryTokenizer::Encode),
      InstanceAccessor("padTokenId", &DirectoryTokenizer::PadTokenId, nullptr),
      InstanceMethod("close", &DirectoryTokenizer::Close),
    });
  }

  explicit DirectoryTokenizer(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<DirectoryTokenizer>(info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) {
      throw Napi::TypeError::New(env, "packagePath must be a string");
    }
    const std::string path = StringArgument(env, info[0], "packagePath");
    Check(env, OgaCreateDirectoryTokenizer(path.c_str(), &handle_));
  }

  ~DirectoryTokenizer() override {
    std::lock_guard<std::mutex> lock(mutex_);
    OgaDestroyDirectoryTokenizer(std::exchange(handle_, nullptr));
  }

 private:
  OgaDirectoryTokenizer* RequireOpen(Napi::Env env) {
    if (!handle_) throw Napi::Error::New(env, "DirectoryTokenizer is closed");
    return handle_;
  }

  Napi::Value Encode(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    Napi::Env env = info.Env();
    if (info.Length() < 1) throw Napi::TypeError::New(env, "text must be a string");
    const std::string text = StringArgument(env, info[0], "text");
    OgaTokenIds* ids{};
    Check(env, OgaDirectoryTokenizerEncode(RequireOpen(env), text.c_str(), &ids));
    struct Owner {
      OgaTokenIds* value;
      ~Owner() { OgaDestroyTokenIds(value); }
    } owner{ids};
    const int32_t* data{};
    size_t count{};
    Check(env, OgaTokenIdsGetData(ids, &data, &count));
    if (count > std::numeric_limits<uint32_t>::max()) {
      throw Napi::RangeError::New(env, "Token array is too large for JavaScript");
    }
    Napi::Int32Array output = Napi::Int32Array::New(env, count);
    std::copy(data, data + count, output.Data());
    return output;
  }

  Napi::Value PadTokenId(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    int32_t value{};
    Check(info.Env(), OgaDirectoryTokenizerGetPadTokenId(RequireOpen(info.Env()), &value));
    return Napi::Number::New(info.Env(), value);
  }

  void Close(const Napi::CallbackInfo&) {
    std::lock_guard<std::mutex> lock(mutex_);
    OgaDestroyDirectoryTokenizer(std::exchange(handle_, nullptr));
  }

  std::mutex mutex_;
  OgaDirectoryTokenizer* handle_{};
};

class RankingSession : public Napi::ObjectWrap<RankingSession> {
 public:
  static Napi::Function Define(Napi::Env env) {
    return DefineClass(env, "RankingSession", {
      InstanceMethod("run", &RankingSession::Run),
      InstanceMethod("rank", &RankingSession::Rank),
      InstanceMethod("setCacheCapacity", &RankingSession::SetCacheCapacity),
      InstanceMethod("getCacheStats", &RankingSession::GetCacheStats),
      InstanceAccessor("cacheStats", &RankingSession::GetCacheStats, nullptr),
      InstanceMethod("clearCache", &RankingSession::ClearCache),
      InstanceMethod("invalidateCache", &RankingSession::InvalidateCache),
      InstanceMethod("close", &RankingSession::Close),
    });
  }

  explicit RankingSession(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<RankingSession>(info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) throw Napi::TypeError::New(env, "packagePath must be a string");
    const std::string path = StringArgument(env, info[0], "packagePath");
    const auto providers = ReadProviders(env, info, 1);
    const auto pointers = ProviderPointers(providers);
    Check(env, OgaCreateRankingSession(
                   path.c_str(), pointers.data(), pointers.size(), &handle_));
  }

  ~RankingSession() override {
    std::lock_guard<std::mutex> lock(mutex_);
    OgaDestroyRankingSession(std::exchange(handle_, nullptr));
  }

 private:
  OgaRankingSessionHandle* RequireOpen(Napi::Env env) {
    if (!handle_) throw Napi::Error::New(env, "RankingSession is closed");
    return handle_;
  }

  Napi::Value Run(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (info.Length() < 1) throw Napi::TypeError::New(info.Env(), "request is required");
    RequestOwner request;
    request.value = BuildRequest(info.Env(), info[0]);
    ModelResultOwner result;
    Check(info.Env(), OgaRankingSessionRun(RequireOpen(info.Env()), request.value, &result.value));
    return ReadModelResult(info.Env(), result.value);
  }

  Napi::Value Rank(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (info.Length() < 1) throw Napi::TypeError::New(info.Env(), "request is required");
    FreeFormOwner request;
    request.value = BuildFreeFormRequest(info.Env(), info[0]);
    RankingResultOwner result;
    Check(info.Env(), OgaRankingSessionRank(RequireOpen(info.Env()), request.value, &result.value));
    return ReadRankingResult(info.Env(), result.value);
  }

  void SetCacheCapacity(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (info.Length() < 2) {
      throw Napi::TypeError::New(info.Env(), "entries and bytes are required");
    }
    Check(info.Env(), OgaRankingSessionSetCacheCapacity(
                          RequireOpen(info.Env()),
                          SizeArgument(info.Env(), info[0], "entries"),
                          SizeArgument(info.Env(), info[1], "bytes")));
  }

  Napi::Value GetCacheStats(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    OgaNonGenerativeCacheStats stats{};
    Check(info.Env(), OgaRankingSessionGetCacheStats(RequireOpen(info.Env()), &stats));
    return ReadCacheStats(info.Env(), stats);
  }

  void ClearCache(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    Check(info.Env(), OgaRankingSessionClearCache(RequireOpen(info.Env())));
  }

  void InvalidateCache(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    Check(info.Env(), OgaRankingSessionInvalidateCache(RequireOpen(info.Env())));
  }

  void Close(const Napi::CallbackInfo&) {
    std::lock_guard<std::mutex> lock(mutex_);
    OgaDestroyRankingSession(std::exchange(handle_, nullptr));
  }

  std::mutex mutex_;
  OgaRankingSessionHandle* handle_{};
};

class DecisionSession : public Napi::ObjectWrap<DecisionSession> {
 public:
  static Napi::Function Define(Napi::Env env) {
    return DefineClass(env, "DecisionSession", {
      InstanceMethod("run", &DecisionSession::Run),
      InstanceMethod("decide", &DecisionSession::Decide),
      InstanceMethod("setCacheCapacity", &DecisionSession::SetCacheCapacity),
      InstanceMethod("getCacheStats", &DecisionSession::GetCacheStats),
      InstanceAccessor("cacheStats", &DecisionSession::GetCacheStats, nullptr),
      InstanceMethod("clearCache", &DecisionSession::ClearCache),
      InstanceMethod("invalidateCache", &DecisionSession::InvalidateCache),
      InstanceMethod("setPrefixCacheCapacity", &DecisionSession::SetPrefixCacheCapacity),
      InstanceAccessor(
          "prefixReuseEnabled",
          &DecisionSession::GetPrefixReuseEnabled,
          &DecisionSession::SetPrefixReuseEnabled),
      InstanceAccessor("prefixReuseStatus", &DecisionSession::GetPrefixReuseStatus, nullptr),
      InstanceAccessor("prefixCacheStats", &DecisionSession::GetPrefixCacheStats, nullptr),
      InstanceAccessor("prefixReuseStats", &DecisionSession::GetPrefixReuseStats, nullptr),
      InstanceMethod("close", &DecisionSession::Close),
    });
  }

  explicit DecisionSession(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<DecisionSession>(info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1) throw Napi::TypeError::New(env, "packagePath must be a string");
    const std::string path = StringArgument(env, info[0], "packagePath");
    const auto providers = ReadProviders(env, info, 1);
    const auto pointers = ProviderPointers(providers);
    Check(env, OgaCreateDecisionSession(
                   path.c_str(), pointers.data(), pointers.size(), &handle_));
  }

  ~DecisionSession() override {
    std::lock_guard<std::mutex> lock(mutex_);
    OgaDestroyDecisionSession(std::exchange(handle_, nullptr));
  }

 private:
  OgaDecisionSessionHandle* RequireOpen(Napi::Env env) {
    if (!handle_) throw Napi::Error::New(env, "DecisionSession is closed");
    return handle_;
  }

  Napi::Value Execute(const Napi::CallbackInfo& info, bool decide) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (info.Length() < 1) throw Napi::TypeError::New(info.Env(), "request is required");
    RequestOwner request;
    request.value = BuildRequest(info.Env(), info[0]);
    ModelResultOwner result;
    Check(info.Env(), decide
                          ? OgaDecisionSessionDecide(
                                RequireOpen(info.Env()), request.value, &result.value)
                          : OgaDecisionSessionRun(
                                RequireOpen(info.Env()), request.value, &result.value));
    return ReadModelResult(info.Env(), result.value);
  }

  Napi::Value Run(const Napi::CallbackInfo& info) { return Execute(info, false); }
  Napi::Value Decide(const Napi::CallbackInfo& info) { return Execute(info, true); }

  void SetCacheCapacity(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (info.Length() < 2) {
      throw Napi::TypeError::New(info.Env(), "entries and bytes are required");
    }
    Check(info.Env(), OgaDecisionSessionSetCacheCapacity(
                          RequireOpen(info.Env()),
                          SizeArgument(info.Env(), info[0], "entries"),
                          SizeArgument(info.Env(), info[1], "bytes")));
  }

  Napi::Value GetCacheStats(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    OgaNonGenerativeCacheStats stats{};
    Check(info.Env(), OgaDecisionSessionGetCacheStats(RequireOpen(info.Env()), &stats));
    return ReadCacheStats(info.Env(), stats);
  }

  void ClearCache(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    Check(info.Env(), OgaDecisionSessionClearCache(RequireOpen(info.Env())));
  }

  void InvalidateCache(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    Check(info.Env(), OgaDecisionSessionInvalidateCache(RequireOpen(info.Env())));
  }

  void SetPrefixCacheCapacity(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (info.Length() < 2) {
      throw Napi::TypeError::New(info.Env(), "entries and bytes are required");
    }
    Check(info.Env(), OgaDecisionSessionSetPrefixCacheCapacity(
                          RequireOpen(info.Env()),
                          SizeArgument(info.Env(), info[0], "entries"),
                          SizeArgument(info.Env(), info[1], "bytes")));
  }

  Napi::Value GetPrefixReuseEnabled(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool value{};
    Check(info.Env(), OgaDecisionSessionGetPrefixReuseEnabled(
                          RequireOpen(info.Env()), &value));
    return Napi::Boolean::New(info.Env(), value);
  }

  void SetPrefixReuseEnabled(
      const Napi::CallbackInfo& info, const Napi::Value& input) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!input.IsBoolean()) {
      throw Napi::TypeError::New(info.Env(), "prefixReuseEnabled must be a boolean");
    }
    Check(info.Env(), OgaDecisionSessionSetPrefixReuseEnabled(
                          RequireOpen(info.Env()), input.As<Napi::Boolean>().Value()));
  }

  Napi::Value GetPrefixReuseStatus(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t size{};
    Check(info.Env(), OgaDecisionSessionCopyPrefixReuseStatus(
                          RequireOpen(info.Env()), nullptr, 0, &size));
    std::vector<char> buffer(size);
    Check(info.Env(), OgaDecisionSessionCopyPrefixReuseStatus(
                          handle_, buffer.data(), buffer.size(), &size));
    return Napi::String::New(info.Env(), buffer.empty() ? "" : buffer.data());
  }

  Napi::Value GetPrefixCacheStats(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    OgaNonGenerativeCacheStats stats{};
    Check(info.Env(), OgaDecisionSessionGetPrefixCacheStats(RequireOpen(info.Env()), &stats));
    return ReadCacheStats(info.Env(), stats);
  }

  Napi::Value GetPrefixReuseStats(const Napi::CallbackInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    OgaKevPrefixReuseStats stats{};
    Check(info.Env(), OgaDecisionSessionGetPrefixReuseStats(RequireOpen(info.Env()), &stats));
    Napi::Object output = NewMap(info.Env());
    SetDataProperty(output, "prefixRuns", Napi::BigInt::New(info.Env(), stats.prefix_runs));
    SetDataProperty(output, "branchRuns", Napi::BigInt::New(info.Env(), stats.branch_runs));
    SetDataProperty(output, "fallbackRuns", Napi::BigInt::New(info.Env(), stats.fallback_runs));
    return output;
  }

  void Close(const Napi::CallbackInfo&) {
    std::lock_guard<std::mutex> lock(mutex_);
    OgaDestroyDecisionSession(std::exchange(handle_, nullptr));
  }

  std::mutex mutex_;
  OgaDecisionSessionHandle* handle_{};
};

Napi::Value TestRoundTrip(const Napi::CallbackInfo& info) {
  if (info.Length() < 1) throw Napi::TypeError::New(info.Env(), "value is required");
  StructuredOwner value;
  value.value = BuildValue(info.Env(), info[0]);
  return ReadValue(info.Env(), value.value);
}

Napi::Object Initialize(Napi::Env env, Napi::Object exports) {
  exports.Set("DirectoryTokenizer", DirectoryTokenizer::Define(env));
  exports.Set("RankingSession", RankingSession::Define(env));
  exports.Set("DecisionSession", DecisionSession::Define(env));
  exports.Set("__testRoundTrip", Napi::Function::New(env, TestRoundTrip));
  return exports;
}

}  // namespace

NODE_API_MODULE(onnxruntime_genai_node, Initialize)
