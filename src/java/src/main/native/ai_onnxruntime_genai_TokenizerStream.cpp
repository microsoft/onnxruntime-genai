/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_TokenizerStream.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jstring JNICALL
Java_ai_onnxruntime_genai_TokenizerStream_tokenizerStreamDecode(JNIEnv* env, jobject thiz,
                                                                jlong tokenizer_stream_handle, jint token) {
  OgaTokenizerStream* tokenizer_stream = reinterpret_cast<OgaTokenizerStream*>(tokenizer_stream_handle);
  const char* decoded_text = nullptr;

  // The const char* returned in decoded_text is the result of calling c_str on a std::string in the tokenizer cache.
  // The std::string is owned by the tokenizer cache.
  // Due to that, it is invalid to call `OgaDestroyString(decoded_text)`, and doing so will result in a crash.
  if (ThrowIfError(env, OgaTokenizerStreamDecode(tokenizer_stream, token, &decoded_text))) {
    return nullptr;
  }

  jstring result = env->NewStringUTF(decoded_text);
  return result;
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_TokenizerStream_destroyTokenizerStream(JNIEnv* env, jobject thiz,
                                                                 jlong tokenizer_stream_handle) {
  OgaTokenizerStream* tokenizer_stream = reinterpret_cast<OgaTokenizerStream*>(tokenizer_stream_handle);
  OgaDestroyTokenizerStream(tokenizer_stream);
}
