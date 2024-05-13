/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

extern "C" JNIEXPORT jstring JNICALL
Java_ai_onnxruntime_1genai_TokenizerStream_tokenizerStreamDecode(JNIEnv* env, jobject thiz,
                                                                 jlong tokenizer_stream_handle, jint token) {
  OgaTokenizerStream* tokenizer_stream = reinterpret_cast<OgaTokenizerStream*>(tokenizer_stream_handle);
  const char* decoded_text = nullptr;

  ThrowIfError(env, OgaTokenizerStreamDecode(tokenizer_stream, token, &decoded_text));

  jstring result = env->NewStringUTF(decoded_text);
  OgaDestroyString(decoded_text);

  return result;
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_1genai_TokenizerStream_destroyTokenizerStream(JNIEnv* env, jobject thiz,
                                                                  jlong tokenizer_stream_handle) {
  OgaTokenizerStream* tokenizer_stream = reinterpret_cast<OgaTokenizerStream*>(tokenizer_stream_handle);
  OgaDestroyTokenizerStream(tokenizer_stream);
}