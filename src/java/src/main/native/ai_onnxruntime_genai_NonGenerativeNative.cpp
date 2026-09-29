/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_NonGenerativeNative.h"

#include <string>
#include <vector>

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

namespace {
struct ValueOwner {
  OgaStructuredValueHandle* value{};
  ~ValueOwner() { OgaDestroyStructuredValue(value); }
};

jobject BoxLong(JNIEnv* env, jlong value) {
  jclass cls = env->FindClass("java/lang/Long");
  return env->NewObject(cls, env->GetMethodID(cls, "<init>", "(J)V"), value);
}
jobject BoxDouble(JNIEnv* env, jdouble value) {
  jclass cls = env->FindClass("java/lang/Double");
  return env->NewObject(cls, env->GetMethodID(cls, "<init>", "(D)V"), value);
}
jobject BoxBoolean(JNIEnv* env, jboolean value) {
  jclass cls = env->FindClass("java/lang/Boolean");
  return env->NewObject(cls, env->GetMethodID(cls, "<init>", "(Z)V"), value);
}
jobject NewMap(JNIEnv* env) {
  jclass cls = env->FindClass("java/util/LinkedHashMap");
  return env->NewObject(cls, env->GetMethodID(cls, "<init>", "()V"));
}
jobject NewList(JNIEnv* env) {
  jclass cls = env->FindClass("java/util/ArrayList");
  return env->NewObject(cls, env->GetMethodID(cls, "<init>", "()V"));
}
void MapPut(JNIEnv* env, jobject map, const char* key, jobject value) {
  jclass cls = env->FindClass("java/util/Map");
  jmethodID put = env->GetMethodID(cls, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
  jstring jkey = env->NewStringUTF(key);
  env->CallObjectMethod(map, put, jkey, value);
  env->DeleteLocalRef(jkey);
}
void ListAdd(JNIEnv* env, jobject list, jobject value) {
  jclass cls = env->FindClass("java/util/List");
  env->CallBooleanMethod(list, env->GetMethodID(cls, "add", "(Ljava/lang/Object;)Z"), value);
}
jobject MapGet(JNIEnv* env, jobject map, const char* key) {
  jclass cls = env->FindClass("java/util/Map");
  jstring jkey = env->NewStringUTF(key);
  jobject value = env->CallObjectMethod(
      map, env->GetMethodID(cls, "get", "(Ljava/lang/Object;)Ljava/lang/Object;"), jkey);
  env->DeleteLocalRef(jkey);
  return value;
}

constexpr size_t kMaxStructuredDepth = 128;

struct ActiveValueGuard {
  explicit ActiveValueGuard(std::vector<jobject>& active) : active_(active) {}
  ~ActiveValueGuard() { active_.pop_back(); }
  std::vector<jobject>& active_;
};

bool EnterContainer(JNIEnv* env, jobject input, std::vector<jobject>& active, size_t depth) {
  if (depth >= kMaxStructuredDepth) {
    ThrowException(env, "Structured value exceeds the maximum nesting depth of 128");
    return true;
  }
  for (jobject ancestor : active) {
    if (env->IsSameObject(input, ancestor)) {
      ThrowException(env, "Structured value contains a reference cycle");
      return true;
    }
  }
  active.push_back(input);
  return false;
}

bool BuildValueWithContext(
    JNIEnv* env, jobject input, OgaStructuredValueHandle** out,
    std::vector<jobject>& active, size_t depth);

bool BuildValueImpl(
    JNIEnv* env, jobject input, OgaStructuredValueHandle** out,
    std::vector<jobject>& active, size_t depth) {
  if (input == nullptr) return ThrowIfError(env, OgaCreateStructuredValueNull(out));
  jclass string_cls = env->FindClass("java/lang/String");
  jclass boolean_cls = env->FindClass("java/lang/Boolean");
  jclass number_cls = env->FindClass("java/lang/Number");
  jclass byte_cls = env->FindClass("java/lang/Byte");
  jclass short_cls = env->FindClass("java/lang/Short");
  jclass integer_cls = env->FindClass("java/lang/Integer");
  jclass long_cls = env->FindClass("java/lang/Long");
  jclass float_cls = env->FindClass("java/lang/Float");
  jclass double_cls = env->FindClass("java/lang/Double");
  jclass map_cls = env->FindClass("java/util/Map");
  jclass iterable_cls = env->FindClass("java/lang/Iterable");
  if (env->IsInstanceOf(input, string_cls)) {
    CString text{env, static_cast<jstring>(input)};
    return ThrowIfError(env, OgaCreateStructuredValueString(text, out));
  }
  if (env->IsInstanceOf(input, boolean_cls)) {
    jboolean value = env->CallBooleanMethod(input, env->GetMethodID(boolean_cls, "booleanValue", "()Z"));
    return ThrowIfError(env, OgaCreateStructuredValueBool(value == JNI_TRUE, out));
  }
  if (env->IsInstanceOf(input, byte_cls) || env->IsInstanceOf(input, short_cls) ||
      env->IsInstanceOf(input, integer_cls) || env->IsInstanceOf(input, long_cls)) {
    jlong value = env->CallLongMethod(input, env->GetMethodID(number_cls, "longValue", "()J"));
    return ThrowIfError(env, OgaCreateStructuredValueInt64(value, out));
  }
  if (env->IsInstanceOf(input, float_cls) || env->IsInstanceOf(input, double_cls)) {
    jdouble value = env->CallDoubleMethod(input, env->GetMethodID(number_cls, "doubleValue", "()D"));
    return ThrowIfError(env, OgaCreateStructuredValueDouble(value, out));
  }
  if (env->IsInstanceOf(input, number_cls)) {
    ThrowException(env, "Structured numbers must be Byte, Short, Integer, Long, Float, or Double");
    return true;
  }
  if (env->IsInstanceOf(input, map_cls)) {
    if (EnterContainer(env, input, active, depth)) return true;
    ActiveValueGuard active_guard(active);
    if (ThrowIfError(env, OgaCreateStructuredValueObject(out))) return true;
    jobject entries = env->CallObjectMethod(
        input, env->GetMethodID(map_cls, "entrySet", "()Ljava/util/Set;"));
    jclass set_cls = env->FindClass("java/util/Set");
    jobject iterator = env->CallObjectMethod(
        entries, env->GetMethodID(set_cls, "iterator", "()Ljava/util/Iterator;"));
    jclass iterator_cls = env->FindClass("java/util/Iterator");
    jmethodID has_next = env->GetMethodID(iterator_cls, "hasNext", "()Z");
    jmethodID next = env->GetMethodID(iterator_cls, "next", "()Ljava/lang/Object;");
    jclass entry_cls = env->FindClass("java/util/Map$Entry");
    while (env->CallBooleanMethod(iterator, has_next)) {
      jobject entry = env->CallObjectMethod(iterator, next);
      jobject key = env->CallObjectMethod(
          entry, env->GetMethodID(entry_cls, "getKey", "()Ljava/lang/Object;"));
      jobject value = env->CallObjectMethod(
          entry, env->GetMethodID(entry_cls, "getValue", "()Ljava/lang/Object;"));
      if (!env->IsInstanceOf(key, string_cls)) {
        ThrowException(env, "Structured object keys must be strings");
        OgaDestroyStructuredValue(*out);
        *out = nullptr;
        return true;
      }
      ValueOwner child;
      if (BuildValueWithContext(env, value, &child.value, active, depth + 1)) {
        OgaDestroyStructuredValue(*out);
        *out = nullptr;
        return true;
      }
      CString ckey{env, static_cast<jstring>(key)};
      if (ThrowIfError(env, OgaStructuredValueObjectAppend(*out, ckey, child.value))) {
        OgaDestroyStructuredValue(*out);
        *out = nullptr;
        return true;
      }
      env->DeleteLocalRef(entry);
      env->DeleteLocalRef(key);
      env->DeleteLocalRef(value);
    }
    return false;
  }
  if (env->IsInstanceOf(input, iterable_cls)) {
    if (EnterContainer(env, input, active, depth)) return true;
    ActiveValueGuard active_guard(active);
    if (ThrowIfError(env, OgaCreateStructuredValueArray(out))) return true;
    jobject iterator = env->CallObjectMethod(
        input, env->GetMethodID(iterable_cls, "iterator", "()Ljava/util/Iterator;"));
    jclass iterator_cls = env->FindClass("java/util/Iterator");
    jmethodID has_next = env->GetMethodID(iterator_cls, "hasNext", "()Z");
    jmethodID next = env->GetMethodID(iterator_cls, "next", "()Ljava/lang/Object;");
    while (env->CallBooleanMethod(iterator, has_next)) {
      jobject item = env->CallObjectMethod(iterator, next);
      ValueOwner child;
      if (BuildValueWithContext(env, item, &child.value, active, depth + 1) ||
          ThrowIfError(env, OgaStructuredValueArrayAppend(*out, child.value))) {
        OgaDestroyStructuredValue(*out);
        *out = nullptr;
        return true;
      }
      env->DeleteLocalRef(item);
    }
    return false;
  }
  ThrowException(env, "Structured values support null, primitive numbers, strings, Maps, and Lists");
  return true;
}

bool BuildValueWithContext(
    JNIEnv* env, jobject input, OgaStructuredValueHandle** out,
    std::vector<jobject>& active, size_t depth) {
  if (env->PushLocalFrame(32) < 0) return true;
  const bool failed = BuildValueImpl(env, input, out, active, depth);
  env->PopLocalFrame(nullptr);
  return failed;
}

bool BuildValue(JNIEnv* env, jobject input, OgaStructuredValueHandle** out) {
  std::vector<jobject> active;
  return BuildValueWithContext(env, input, out, active, 0);
}

jobject ReadValue(JNIEnv* env, const OgaStructuredValueHandle* value) {
  OgaStructuredValueType type;
  if (ThrowIfError(env, OgaStructuredValueGetType(value, &type))) return nullptr;
  switch (type) {
    case OgaStructuredValueType_Null: return nullptr;
    case OgaStructuredValueType_Bool: {
      bool v;
      if (ThrowIfError(env, OgaStructuredValueGetBool(value, &v))) return nullptr;
      return BoxBoolean(env, v);
    }
    case OgaStructuredValueType_Int64: {
      int64_t v;
      if (ThrowIfError(env, OgaStructuredValueGetInt64(value, &v))) return nullptr;
      return BoxLong(env, v);
    }
    case OgaStructuredValueType_Double: {
      double v;
      if (ThrowIfError(env, OgaStructuredValueGetDouble(value, &v))) return nullptr;
      return BoxDouble(env, v);
    }
    case OgaStructuredValueType_String: {
      const char* v;
      if (ThrowIfError(env, OgaStructuredValueGetString(value, &v))) return nullptr;
      return env->NewStringUTF(v);
    }
    case OgaStructuredValueType_Array: {
      size_t count;
      if (ThrowIfError(env, OgaStructuredValueGetCount(value, &count))) return nullptr;
      jobject list = NewList(env);
      for (size_t i = 0; i < count; ++i) {
        const OgaStructuredValueHandle* item;
        if (ThrowIfError(env, OgaStructuredValueGetArrayItem(value, i, &item))) return nullptr;
        jobject converted = ReadValue(env, item);
        ListAdd(env, list, converted);
        if (converted) env->DeleteLocalRef(converted);
      }
      return list;
    }
    case OgaStructuredValueType_Object: {
      size_t count;
      if (ThrowIfError(env, OgaStructuredValueGetCount(value, &count))) return nullptr;
      jobject map = NewMap(env);
      for (size_t i = 0; i < count; ++i) {
        const char* key;
        const OgaStructuredValueHandle* item;
        if (ThrowIfError(env, OgaStructuredValueGetObjectItem(value, i, &key, &item))) return nullptr;
        jobject converted = ReadValue(env, item);
        MapPut(env, map, key, converted);
        if (converted) env->DeleteLocalRef(converted);
      }
      return map;
    }
  }
  ThrowException(env, "Native runtime returned an unknown structured value type");
  return nullptr;
}

std::vector<std::string> Providers(JNIEnv* env, jobjectArray providers) {
  std::vector<std::string> result;
  if (!providers) return result;
  for (jsize i = 0; i < env->GetArrayLength(providers); ++i) {
    jstring item = static_cast<jstring>(env->GetObjectArrayElement(providers, i));
    if (!item) {
      ThrowException(env, "providers cannot contain null");
      return {};
    }
    CString text{env, item};
    result.emplace_back(text.cstr);
    env->DeleteLocalRef(item);
  }
  return result;
}

OgaStructuredRequestHandle* BuildRequest(
    JNIEnv* env, jobject state, jobject questions, jobject temperature) {
  OgaStructuredRequestHandle* request{};
  if (ThrowIfError(env, OgaCreateStructuredRequest(&request))) return nullptr;
  ValueOwner state_value;
  if (BuildValue(env, state, &state_value.value) ||
      ThrowIfError(env, OgaStructuredRequestSetState(request, state_value.value))) {
    OgaDestroyStructuredRequest(request);
    return nullptr;
  }
  jclass map_cls = env->FindClass("java/util/Map");
  jobject entries = env->CallObjectMethod(
      questions, env->GetMethodID(map_cls, "entrySet", "()Ljava/util/Set;"));
  jclass set_cls = env->FindClass("java/util/Set");
  jobject iterator = env->CallObjectMethod(
      entries, env->GetMethodID(set_cls, "iterator", "()Ljava/util/Iterator;"));
  jclass iterator_cls = env->FindClass("java/util/Iterator");
  jmethodID has_next = env->GetMethodID(iterator_cls, "hasNext", "()Z");
  jmethodID next = env->GetMethodID(iterator_cls, "next", "()Ljava/lang/Object;");
  jclass entry_cls = env->FindClass("java/util/Map$Entry");
  while (env->CallBooleanMethod(iterator, has_next)) {
    jobject entry = env->CallObjectMethod(iterator, next);
    jstring id = static_cast<jstring>(env->CallObjectMethod(
        entry, env->GetMethodID(entry_cls, "getKey", "()Ljava/lang/Object;")));
    jobject question_map = env->CallObjectMethod(
        entry, env->GetMethodID(entry_cls, "getValue", "()Ljava/lang/Object;"));
    jstring type = static_cast<jstring>(MapGet(env, question_map, "type"));
    jobject instructions = MapGet(env, question_map, "instructions");
    jobject criteria = MapGet(env, question_map, "criteria");
    ValueOwner instruction_value, criteria_value;
    OgaQuestionHandle* question{};
    if (BuildValue(env, instructions, &instruction_value.value) ||
        BuildValue(env, criteria, &criteria_value.value)) {
      OgaDestroyStructuredRequest(request);
      return nullptr;
    }
    CString ctype{env, type};
    CString cid{env, id};
    if (ThrowIfError(env, OgaCreateQuestion(
            ctype, instruction_value.value, criteria_value.value, &question)) ||
        ThrowIfError(env, OgaStructuredRequestAddQuestion(request, cid, question))) {
      OgaDestroyQuestion(question);
      OgaDestroyStructuredRequest(request);
      return nullptr;
    }
    OgaDestroyQuestion(question);
  }
  if (temperature) {
    jclass number_cls = env->FindClass("java/lang/Number");
    jfloat value = env->CallFloatMethod(
        temperature, env->GetMethodID(number_cls, "floatValue", "()F"));
    if (ThrowIfError(env, OgaStructuredRequestSetTemperature(request, value))) {
      OgaDestroyStructuredRequest(request);
      return nullptr;
    }
  }
  return request;
}

jobject ReadModelResult(JNIEnv* env, const OgaModelResultHandle* result) {
  jobject root = NewMap(env);
  const char* model;
  size_t count;
  if (ThrowIfError(env, OgaModelResultGetModel(result, &model)) ||
      ThrowIfError(env, OgaModelResultGetAnswerCount(result, &count))) return nullptr;
  MapPut(env, root, "model", env->NewStringUTF(model));
  jobject answers = NewList(env);
  for (size_t i = 0; i < count; ++i) {
    jobject answer = NewMap(env);
    const char *id, *type, *text{};
    double number;
    bool present;
    if (ThrowIfError(env, OgaModelResultGetAnswerId(result, i, &id)) ||
        ThrowIfError(env, OgaModelResultGetAnswerType(result, i, &type))) return nullptr;
    MapPut(env, answer, "id", env->NewStringUTF(id));
    MapPut(env, answer, "type", env->NewStringUTF(type));
    if (ThrowIfError(env, OgaModelResultGetAnswerNoul(result, i, &number, &present))) return nullptr;
    MapPut(env, answer, "noul", present ? BoxDouble(env, number) : nullptr);
    if (ThrowIfError(env, OgaModelResultGetAnswerChoice(result, i, &text, &present))) return nullptr;
    MapPut(env, answer, "choice", present ? env->NewStringUTF(text) : nullptr);
    if (ThrowIfError(env, OgaModelResultGetAnswerScore(result, i, &number, &present))) return nullptr;
    MapPut(env, answer, "score", present ? BoxDouble(env, number) : nullptr);
    if (ThrowIfError(env, OgaModelResultGetAnswerConfidence(result, i, &number, &present))) return nullptr;
    MapPut(env, answer, "confidence", present ? BoxDouble(env, number) : nullptr);
    jobject probabilities = NewMap(env);
    size_t values;
    if (ThrowIfError(env, OgaModelResultGetProbabilityCount(result, i, &values))) return nullptr;
    for (size_t j = 0; j < values; ++j) {
      const char* key;
      if (ThrowIfError(env, OgaModelResultGetProbability(result, i, j, &key, &number))) return nullptr;
      MapPut(env, probabilities, key, BoxDouble(env, number));
    }
    MapPut(env, answer, "probabilities", probabilities);
    jobject legend = NewMap(env);
    if (ThrowIfError(env, OgaModelResultGetLegendCount(result, i, &values))) return nullptr;
    for (size_t j = 0; j < values; ++j) {
      const char *key, *value;
      if (ThrowIfError(env, OgaModelResultGetLegend(result, i, j, &key, &value))) return nullptr;
      MapPut(env, legend, key, env->NewStringUTF(value));
    }
    MapPut(env, answer, "legend", legend);
    ListAdd(env, answers, answer);
    env->DeleteLocalRef(answer);
  }
  MapPut(env, root, "answers", answers);
  return root;
}
}  // namespace

JNIEXPORT jlong JNICALL Java_ai_onnxruntime_genai_NonGenerativeNative_createDirectoryTokenizer(
    JNIEnv* env, jclass, jstring path) {
  if (!path) { ThrowException(env, "path must not be null"); return 0; }
  CString cpath{env, path};
  OgaDirectoryTokenizer* value{};
  return ThrowIfError(env, OgaCreateDirectoryTokenizer(cpath, &value))
      ? 0 : reinterpret_cast<jlong>(value);
}
JNIEXPORT void JNICALL Java_ai_onnxruntime_genai_NonGenerativeNative_destroyDirectoryTokenizer(
    JNIEnv*, jclass, jlong handle) {
  OgaDestroyDirectoryTokenizer(reinterpret_cast<OgaDirectoryTokenizer*>(handle));
}
JNIEXPORT jintArray JNICALL Java_ai_onnxruntime_genai_NonGenerativeNative_directoryTokenizerEncode(
    JNIEnv* env, jclass, jlong handle, jstring text) {
  CString ctext{env, text};
  OgaTokenIds* ids{};
  if (ThrowIfError(env, OgaDirectoryTokenizerEncode(
          reinterpret_cast<OgaDirectoryTokenizer*>(handle), ctext, &ids))) return nullptr;
  const int32_t* data;
  size_t count;
  if (ThrowIfError(env, OgaTokenIdsGetData(ids, &data, &count))) {
    OgaDestroyTokenIds(ids);
    return nullptr;
  }
  jintArray result = env->NewIntArray(static_cast<jsize>(count));
  env->SetIntArrayRegion(result, 0, static_cast<jsize>(count), reinterpret_cast<const jint*>(data));
  OgaDestroyTokenIds(ids);
  return result;
}
JNIEXPORT jint JNICALL Java_ai_onnxruntime_genai_NonGenerativeNative_directoryTokenizerPadTokenId(
    JNIEnv* env, jclass, jlong handle) {
  int32_t value;
  return ThrowIfError(env, OgaDirectoryTokenizerGetPadTokenId(
      reinterpret_cast<OgaDirectoryTokenizer*>(handle), &value)) ? 0 : value;
}
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_genai_NonGenerativeNative_createRankingSession(
    JNIEnv* env, jclass, jstring path, jobjectArray provider_array) {
  CString cpath{env, path};
  auto providers = Providers(env, provider_array);
  if (env->ExceptionCheck()) return 0;
  std::vector<const char*> pointers;
  for (const auto& provider : providers) pointers.push_back(provider.c_str());
  OgaRankingSessionHandle* value{};
  return ThrowIfError(env, OgaCreateRankingSession(cpath, pointers.data(), pointers.size(), &value))
      ? 0 : reinterpret_cast<jlong>(value);
}
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_genai_NonGenerativeNative_createDecisionSession(
    JNIEnv* env, jclass, jstring path, jobjectArray provider_array) {
  CString cpath{env, path};
  auto providers = Providers(env, provider_array);
  if (env->ExceptionCheck()) return 0;
  std::vector<const char*> pointers;
  for (const auto& provider : providers) pointers.push_back(provider.c_str());
  OgaDecisionSessionHandle* value{};
  return ThrowIfError(env, OgaCreateDecisionSession(cpath, pointers.data(), pointers.size(), &value))
      ? 0 : reinterpret_cast<jlong>(value);
}
JNIEXPORT void JNICALL Java_ai_onnxruntime_genai_NonGenerativeNative_destroyRankingSession(
    JNIEnv*, jclass, jlong handle) {
  OgaDestroyRankingSession(reinterpret_cast<OgaRankingSessionHandle*>(handle));
}
JNIEXPORT void JNICALL Java_ai_onnxruntime_genai_NonGenerativeNative_destroyDecisionSession(
    JNIEnv*, jclass, jlong handle) {
  OgaDestroyDecisionSession(reinterpret_cast<OgaDecisionSessionHandle*>(handle));
}
JNIEXPORT jobject JNICALL Java_ai_onnxruntime_genai_NonGenerativeNative_execute(
    JNIEnv* env, jclass, jlong handle, jboolean ranking, jboolean decide, jobject state,
    jobject questions, jobject temperature) {
  OgaStructuredRequestHandle* request = BuildRequest(env, state, questions, temperature);
  if (!request) return nullptr;
  OgaModelResultHandle* result{};
  OgaResult* status = ranking
      ? OgaRankingSessionRun(reinterpret_cast<OgaRankingSessionHandle*>(handle), request, &result)
      : (decide
             ? OgaDecisionSessionDecide(reinterpret_cast<OgaDecisionSessionHandle*>(handle), request, &result)
             : OgaDecisionSessionRun(reinterpret_cast<OgaDecisionSessionHandle*>(handle), request, &result));
  OgaDestroyStructuredRequest(request);
  if (ThrowIfError(env, status)) return nullptr;
  jobject converted = ReadModelResult(env, result);
  OgaDestroyModelResult(result);
  return converted;
}
JNIEXPORT jobject JNICALL Java_ai_onnxruntime_genai_NonGenerativeNative_rank(
    JNIEnv* env, jclass, jlong handle, jobject state, jobject instructions,
    jobject candidates, jobject temperature) {
  OgaFreeFormRankRequestHandle* request{};
  if (ThrowIfError(env, OgaCreateFreeFormRankRequest(&request))) return nullptr;
  ValueOwner state_value, instructions_value;
  if (BuildValue(env, state, &state_value.value) ||
      BuildValue(env, instructions, &instructions_value.value) ||
      ThrowIfError(env, OgaFreeFormRankRequestSetState(request, state_value.value)) ||
      ThrowIfError(env, OgaFreeFormRankRequestSetInstructions(request, instructions_value.value))) {
    OgaDestroyFreeFormRankRequest(request);
    return nullptr;
  }
  jclass map_cls = env->FindClass("java/util/Map");
  jobject entries = env->CallObjectMethod(candidates, env->GetMethodID(map_cls, "entrySet", "()Ljava/util/Set;"));
  jclass set_cls = env->FindClass("java/util/Set");
  jobject iterator = env->CallObjectMethod(entries, env->GetMethodID(set_cls, "iterator", "()Ljava/util/Iterator;"));
  jclass iterator_cls = env->FindClass("java/util/Iterator");
  jmethodID has_next = env->GetMethodID(iterator_cls, "hasNext", "()Z");
  jmethodID next = env->GetMethodID(iterator_cls, "next", "()Ljava/lang/Object;");
  jclass entry_cls = env->FindClass("java/util/Map$Entry");
  while (env->CallBooleanMethod(iterator, has_next)) {
    jobject entry = env->CallObjectMethod(iterator, next);
    jstring key = static_cast<jstring>(env->CallObjectMethod(entry, env->GetMethodID(entry_cls, "getKey", "()Ljava/lang/Object;")));
    jobject item = env->CallObjectMethod(entry, env->GetMethodID(entry_cls, "getValue", "()Ljava/lang/Object;"));
    if (!key) { ThrowException(env, "Candidate keys must not be null"); OgaDestroyFreeFormRankRequest(request); return nullptr; }
    CString ckey{env, key};
    ValueOwner value;
    if (BuildValue(env, item, &value.value) ||
        ThrowIfError(env, OgaFreeFormRankRequestAddCandidate(request, ckey, value.value))) {
      OgaDestroyFreeFormRankRequest(request);
      return nullptr;
    }
  }
  if (temperature) {
    jclass number_cls = env->FindClass("java/lang/Number");
    jfloat value = env->CallFloatMethod(temperature, env->GetMethodID(number_cls, "floatValue", "()F"));
    if (ThrowIfError(env, OgaFreeFormRankRequestSetTemperature(request, value))) {
      OgaDestroyFreeFormRankRequest(request);
      return nullptr;
    }
  }
  OgaRankingResultHandle* result{};
  OgaResult* status = OgaRankingSessionRank(
      reinterpret_cast<OgaRankingSessionHandle*>(handle), request, &result);
  OgaDestroyFreeFormRankRequest(request);
  if (ThrowIfError(env, status)) return nullptr;
  jobject root = NewMap(env);
  const char* model;
  size_t count;
  if (ThrowIfError(env, OgaRankingResultGetModel(result, &model)) ||
      ThrowIfError(env, OgaRankingResultGetCount(result, &count))) {
    OgaDestroyRankingResult(result);
    return nullptr;
  }
  MapPut(env, root, "model", env->NewStringUTF(model));
  jobject items = NewList(env);
  for (size_t i = 0; i < count; ++i) {
    size_t rank;
    const char* key;
    const OgaStructuredValueHandle* ranked_value;
    double probability;
    if (ThrowIfError(env, OgaRankingResultGetRank(result, i, &rank)) ||
        ThrowIfError(env, OgaRankingResultGetKey(result, i, &key)) ||
        ThrowIfError(env, OgaRankingResultGetValue(result, i, &ranked_value)) ||
        ThrowIfError(env, OgaRankingResultGetProbability(result, i, &probability))) {
      OgaDestroyRankingResult(result);
      return nullptr;
    }
    jobject item = NewMap(env);
    MapPut(env, item, "rank", BoxLong(env, static_cast<jlong>(rank)));
    MapPut(env, item, "key", env->NewStringUTF(key));
    MapPut(env, item, "value", ReadValue(env, ranked_value));
    MapPut(env, item, "probability", BoxDouble(env, probability));
    ListAdd(env, items, item);
  }
  MapPut(env, root, "items", items);
  OgaDestroyRankingResult(result);
  return root;
}
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_genai_NonGenerativeNative_cache(
    JNIEnv* env, jclass, jlong handle, jboolean ranking, jint operation,
    jlong entries, jlong bytes) {
  OgaResult* status{};
  OgaNonGenerativeCacheStats stats{};
  if (ranking) {
    auto* session = reinterpret_cast<OgaRankingSessionHandle*>(handle);
    if (operation == 0) status = OgaRankingSessionSetCacheCapacity(session, entries, bytes);
    else if (operation == 1) status = OgaRankingSessionGetCacheStats(session, &stats);
    else if (operation == 2) status = OgaRankingSessionClearCache(session);
    else if (operation == 3) status = OgaRankingSessionInvalidateCache(session);
    else { ThrowException(env, "Invalid cache operation"); return nullptr; }
  } else {
    auto* session = reinterpret_cast<OgaDecisionSessionHandle*>(handle);
    if (operation == 0) status = OgaDecisionSessionSetCacheCapacity(session, entries, bytes);
    else if (operation == 1) status = OgaDecisionSessionGetCacheStats(session, &stats);
    else if (operation == 2) status = OgaDecisionSessionClearCache(session);
    else if (operation == 3) status = OgaDecisionSessionInvalidateCache(session);
    else { ThrowException(env, "Invalid cache operation"); return nullptr; }
  }
  if (ThrowIfError(env, status)) return nullptr;
  jlong values[7] = {
      static_cast<jlong>(stats.hits), static_cast<jlong>(stats.misses),
      static_cast<jlong>(stats.evictions), static_cast<jlong>(stats.entries),
      static_cast<jlong>(stats.bytes), static_cast<jlong>(stats.entry_capacity),
      static_cast<jlong>(stats.byte_capacity)};
  jlongArray result = env->NewLongArray(7);
  env->SetLongArrayRegion(result, 0, 7, values);
  return result;
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_NonGenerativeNative_setDecisionPrefixReuseEnabled(
    JNIEnv* env, jclass, jlong handle, jboolean enabled) {
  ThrowIfError(env, OgaDecisionSessionSetPrefixReuseEnabled(
                        reinterpret_cast<OgaDecisionSessionHandle*>(handle),
                        enabled == JNI_TRUE));
}

JNIEXPORT jboolean JNICALL
Java_ai_onnxruntime_genai_NonGenerativeNative_getDecisionPrefixReuseEnabled(
    JNIEnv* env, jclass, jlong handle) {
  bool enabled{};
  if (ThrowIfError(env, OgaDecisionSessionGetPrefixReuseEnabled(
                            reinterpret_cast<OgaDecisionSessionHandle*>(handle),
                            &enabled))) {
    return JNI_FALSE;
  }
  return enabled ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_ai_onnxruntime_genai_NonGenerativeNative_getDecisionPrefixReuseStatus(
    JNIEnv* env, jclass, jlong handle) {
  const char* status{};
  if (ThrowIfError(env, OgaDecisionSessionGetPrefixReuseStatus(
                            reinterpret_cast<OgaDecisionSessionHandle*>(handle),
                            &status))) {
    return nullptr;
  }
  return env->NewStringUTF(status);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_NonGenerativeNative_setDecisionPrefixCacheCapacity(
    JNIEnv* env, jclass, jlong handle, jlong entries, jlong bytes) {
  ThrowIfError(env, OgaDecisionSessionSetPrefixCacheCapacity(
                        reinterpret_cast<OgaDecisionSessionHandle*>(handle),
                        static_cast<size_t>(entries), static_cast<size_t>(bytes)));
}

JNIEXPORT jlongArray JNICALL
Java_ai_onnxruntime_genai_NonGenerativeNative_getDecisionPrefixCacheStats(
    JNIEnv* env, jclass, jlong handle) {
  OgaNonGenerativeCacheStats stats{};
  if (ThrowIfError(env, OgaDecisionSessionGetPrefixCacheStats(
                            reinterpret_cast<OgaDecisionSessionHandle*>(handle),
                            &stats))) {
    return nullptr;
  }
  const jlong values[7] = {
      static_cast<jlong>(stats.hits), static_cast<jlong>(stats.misses),
      static_cast<jlong>(stats.evictions), static_cast<jlong>(stats.entries),
      static_cast<jlong>(stats.bytes), static_cast<jlong>(stats.entry_capacity),
      static_cast<jlong>(stats.byte_capacity)};
  jlongArray result = env->NewLongArray(7);
  env->SetLongArrayRegion(result, 0, 7, values);
  return result;
}

JNIEXPORT jlongArray JNICALL
Java_ai_onnxruntime_genai_NonGenerativeNative_getDecisionPrefixReuseStats(
    JNIEnv* env, jclass, jlong handle) {
  OgaKevPrefixReuseStats stats{};
  if (ThrowIfError(env, OgaDecisionSessionGetPrefixReuseStats(
                            reinterpret_cast<OgaDecisionSessionHandle*>(handle),
                            &stats))) {
    return nullptr;
  }
  const jlong values[3] = {
      static_cast<jlong>(stats.prefix_runs),
      static_cast<jlong>(stats.branch_runs),
      static_cast<jlong>(stats.fallback_runs)};
  jlongArray result = env->NewLongArray(3);
  env->SetLongArrayRegion(result, 0, 3, values);
  return result;
}
