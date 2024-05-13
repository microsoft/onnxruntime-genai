/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_1genai_Tokenizer_createTokenizer(JNIEnv* env, jobject thiz, jlong model_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  OgaTokenizer* tokenizer = nullptr;

  ThrowIfError(env, OgaCreateTokenizer(model, &tokenizer));

  return reinterpret_cast<jlong>(tokenizer);
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_1genai_Tokenizer_destroyTokenizer(JNIEnv* env, jobject thiz, jlong tokenizer_handle) {
  OgaTokenizer* tokenizer = reinterpret_cast<OgaTokenizer*>(tokenizer_handle);
  OgaDestroyTokenizer(tokenizer);
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_1genai_Tokenizer_tokenizerEncode(JNIEnv* env, jobject thiz, jlong tokenizer_handle,
                                                     jobjectArray strings) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  auto num_strings = env->GetArrayLength(strings);

  OgaSequences* sequences = nullptr;
  ThrowIfError(env, OgaCreateSequences(&sequences));

  for (int i = 0; i < num_strings; i++) {
    jstring string = static_cast<jstring>(env->GetObjectArrayElement(strings, i));
    CString c_string{env, string};
    ThrowIfError(env, OgaTokenizerEncode(tokenizer, c_string, sequences));
  }

  return reinterpret_cast<jlong>(sequences);
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_onnxruntime_1genai_Tokenizer_tokenizerDecode(JNIEnv* env, jobject thiz, jlong tokenizer_handle,
                                                     jintArray sequence) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  auto num_tokens = env->GetArrayLength(sequence);
  jint* jtokens = env->GetIntArrayElements(sequence, nullptr);
  const int32_t* tokens = reinterpret_cast<const int32_t*>(jtokens);  // convert between 32-bit types
  const char* decoded_text = nullptr;

  ThrowIfError(env, OgaTokenizerDecode(tokenizer, tokens, num_tokens, &decoded_text));

  jstring result = env->NewStringUTF(decoded_text);
  OgaDestroyString(decoded_text);

  return result;
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_1genai_Tokenizer_createTokenizerStream(JNIEnv* env, jobject thiz, jlong tokenizer_handle) {
  const OgaTokenizer* tokenizer = reinterpret_cast<const OgaTokenizer*>(tokenizer_handle);
  OgaTokenizerStream* tokenizer_stream = nullptr;

  ThrowIfError(env, OgaCreateTokenizerStream(tokenizer, &tokenizer_stream));

  return reinterpret_cast<jlong>(tokenizer_stream);
}