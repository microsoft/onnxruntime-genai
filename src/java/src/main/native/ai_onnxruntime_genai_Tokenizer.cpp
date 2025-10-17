/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Tokenizer.h"

#include "ort_genai_c.h"
#include "utils.h"
#include <vector>
#include <optional>

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Tokenizer_createTokenizer(JNIEnv* env, jobject thiz, jlong model_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  OgaTokenizer* tokenizer = nullptr;

  if (ThrowIfError(env, OgaCreateTokenizer(model, &tokenizer))) {
    return 0;
  }

  return reinterpret_cast<jlong>(tokenizer);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Tokenizer_destroyTokenizer(JNIEnv* env, jobject thiz, jlong tokenizer_handle) {
  OgaTokenizer* tokenizer = reinterpret_cast<OgaTokenizer*>(tokenizer_handle);
  OgaDestroyTokenizer(tokenizer);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerEncode(JNIEnv* env, jobject thiz, jlong tokenizer_handle,
                                                    jobjectArray strings) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  auto num_strings = env->GetArrayLength(strings);

  OgaSequences* sequences = nullptr;
  if (ThrowIfError(env, OgaCreateSequences(&sequences))) {
    return 0;
  }

  for (int i = 0; i < num_strings; i++) {
    jstring string = static_cast<jstring>(env->GetObjectArrayElement(strings, i));
    CString c_string{env, string};
    if (ThrowIfError(env, OgaTokenizerEncode(tokenizer, c_string, sequences))) {
      OgaDestroySequences(sequences);
      return 0;
    }
  }

  return reinterpret_cast<jlong>(sequences);
}

JNIEXPORT jstring JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerDecode(JNIEnv* env, jobject thiz, jlong tokenizer_handle,
                                                    jintArray sequence) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  auto num_tokens = env->GetArrayLength(sequence);
  jint* jtokens = env->GetIntArrayElements(sequence, nullptr);
  const int32_t* tokens = reinterpret_cast<const int32_t*>(jtokens);  // convert between 32-bit types
  const char* decoded_text = nullptr;

  bool error = ThrowIfError(env, OgaTokenizerDecode(tokenizer, tokens, num_tokens, &decoded_text));
  env->ReleaseIntArrayElements(sequence, jtokens, JNI_ABORT);

  if (error) {
    return nullptr;
  }

  jstring result = env->NewStringUTF(decoded_text);
  OgaDestroyString(decoded_text);

  return result;
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Tokenizer_createTokenizerStream(JNIEnv* env, jobject thiz, jlong tokenizer_handle) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  OgaTokenizerStream* tokenizer_stream = nullptr;

  if (ThrowIfError(env, OgaCreateTokenizerStream(tokenizer, &tokenizer_stream))) {
    return 0;
  }

  return reinterpret_cast<jlong>(tokenizer_stream);
}

JNIEXPORT jint JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerGetBosTokenId(JNIEnv* env, jobject thiz, jlong tokenizer_handle) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  int32_t token_id = 0;

  if (ThrowIfError(env, OgaTokenizerGetBosTokenId(tokenizer, &token_id))) {
    return 0;
  }

  return static_cast<jint>(token_id);
}

JNIEXPORT jint JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerGetPadTokenId(JNIEnv* env, jobject thiz, jlong tokenizer_handle) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  int32_t token_id = 0;

  if (ThrowIfError(env, OgaTokenizerGetPadTokenId(tokenizer, &token_id))) {
    return 0;
  }

  return static_cast<jint>(token_id);
}

JNIEXPORT jintArray JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerGetEosTokenIds(JNIEnv* env, jobject thiz, jlong tokenizer_handle) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  const int32_t* eos_token_ids = nullptr;
  size_t token_count = 0;

  if (ThrowIfError(env, OgaTokenizerGetEosTokenIds(tokenizer, &eos_token_ids, &token_count))) {
    return nullptr;
  }

  // Create Java int array
  jintArray result = env->NewIntArray(static_cast<jsize>(token_count));
  if (result == nullptr) {
    return nullptr;  // OutOfMemoryError thrown
  }

  // Copy the token IDs to the Java array
  env->SetIntArrayRegion(result, 0, static_cast<jsize>(token_count),
                         reinterpret_cast<const jint*>(eos_token_ids));

  return result;
}

JNIEXPORT jint JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerToTokenId(JNIEnv* env, jobject thiz, jlong tokenizer_handle, jstring str) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  CString c_string{env, str};
  int32_t token_id = 0;

  if (ThrowIfError(env, OgaTokenizerToTokenId(tokenizer, c_string, &token_id))) {
    return 0;
  }

  return static_cast<jint>(token_id);
}

JNIEXPORT jstring JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerApplyChatTemplate(JNIEnv* env, jobject thiz, jlong tokenizer_handle,
                                                               jstring template_str, jstring messages, jstring tools, jboolean add_generation_prompt) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  CString c_template_str{env, template_str};
  CString c_messages{env, messages};

  std::optional<CString> c_tools;
  const char* c_tools_ptr = nullptr;
  if (tools != nullptr) {
    c_tools = CString{env, tools};
    c_tools_ptr = *c_tools;
  }

  const char* result = nullptr;

  if (ThrowIfError(env, OgaTokenizerApplyChatTemplate(tokenizer, c_template_str, c_messages, c_tools_ptr, add_generation_prompt, &result))) {
    return nullptr;
  }

  jstring jresult = env->NewStringUTF(result);
  OgaDestroyString(result);

  return jresult;
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Tokenizer_tokenizerUpdateOptions(JNIEnv* env, jobject thiz, jlong tokenizer_handle,
                                                           jobjectArray keys, jobjectArray values) {
  OgaTokenizer* tokenizer = reinterpret_cast<OgaTokenizer*>(tokenizer_handle);
  auto num_options = env->GetArrayLength(keys);

  if (num_options != env->GetArrayLength(values)) {
    Helpers::ThrowException(env, "Keys and values arrays must have the same length");
    return;
  }

  // Convert Java string arrays to C string arrays
  std::vector<CString> c_keys;
  std::vector<CString> c_values;
  std::vector<const char*> c_keys_ptr;
  std::vector<const char*> c_values_ptr;
  c_keys.reserve(num_options);
  c_values.reserve(num_options);
  c_keys_ptr.reserve(num_options);
  c_values_ptr.reserve(num_options);

  for (int i = 0; i < num_options; i++) {
    jstring key = static_cast<jstring>(env->GetObjectArrayElement(keys, i));
    jstring value = static_cast<jstring>(env->GetObjectArrayElement(values, i));

    c_keys.emplace_back(env, key);
    c_values.emplace_back(env, value);

    c_keys_ptr.push_back(c_keys.back());
    c_values_ptr.push_back(c_values.back());
  }

  ThrowIfError(env, OgaUpdateTokenizerOptions(tokenizer, c_keys_ptr.data(), c_values_ptr.data(), num_options));
}
