/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Generator.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Generator_createGenerator(JNIEnv* env, jobject thiz, jlong model_handle,
                                                    jlong generator_params_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  const OgaGeneratorParams* params = reinterpret_cast<const OgaGeneratorParams*>(generator_params_handle);
  OgaGenerator* generator = nullptr;
  if (ThrowIfError(env, OgaCreateGenerator(model, params, &generator))) {
    return 0;
  }

  return reinterpret_cast<jlong>(generator);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Generator_destroyGenerator(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaDestroyGenerator(reinterpret_cast<OgaGenerator*>(native_handle));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Generator_appendTokenSequences(JNIEnv* env, jobject thiz, jlong native_handle,
                                                         jlong sequences_handle) {
  OgaGenerator* generator = reinterpret_cast<OgaGenerator*>(native_handle);
  const OgaSequences* sequences = reinterpret_cast<const OgaSequences*>(sequences_handle);

  ThrowIfError(env, OgaGenerator_AppendTokenSequences(generator, sequences));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Generator_appendTokens(JNIEnv* env, jobject thiz, jlong native_handle, jintArray token_ids) {
  OgaGenerator* generator = reinterpret_cast<OgaGenerator*>(native_handle);

  jint* tokens = env->GetIntArrayElements(token_ids, nullptr);
  jsize num_tokens = env->GetArrayLength(token_ids);

  ThrowIfError(env, OgaGenerator_AppendTokens(generator, reinterpret_cast<int32_t*>(tokens), num_tokens));

  env->ReleaseIntArrayElements(token_ids, tokens, JNI_ABORT);
}

JNIEXPORT jboolean JNICALL
Java_ai_onnxruntime_genai_Generator_isDone(JNIEnv* env, jobject thiz, jlong native_handle) {
  return OgaGenerator_IsDone(reinterpret_cast<OgaGenerator*>(native_handle));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Generator_rewindTo(JNIEnv* env, jobject thiz, jlong native_handle, jlong length) {
  ThrowIfError(env, OgaGenerator_RewindTo(reinterpret_cast<OgaGenerator*>(native_handle), length));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Generator_generateNextTokenNative(JNIEnv* env, jobject thiz, jlong native_handle) {
  ThrowIfError(env, OgaGenerator_GenerateNextToken(reinterpret_cast<OgaGenerator*>(native_handle)));
}

JNIEXPORT jintArray JNICALL
Java_ai_onnxruntime_genai_Generator_getSequenceNative(JNIEnv* env, jobject thiz, jlong generator, jlong index) {
  const OgaGenerator* oga_generator = reinterpret_cast<const OgaGenerator*>(generator);

  size_t num_tokens = OgaGenerator_GetSequenceCount(oga_generator, index);
  const int32_t* tokens = OgaGenerator_GetSequenceData(oga_generator, index);

  if (num_tokens == 0) {
    ThrowException(env, "OgaGenerator_GetSequenceCount returned 0 tokens.");
    return nullptr;
  }

  // as there's no 'destroy' function in GenAI C API for the tokens we assume the OgaGenerator owns the memory.
  // copy the tokens so there's no potential for Java code to write to it (values should be treated as const)
  // or attempt to access the memory after the OgaGenerator is destroyed.
  jintArray java_int_array = env->NewIntArray(num_tokens);
  // jint is `long` on Windows and `int` on linux. 32-bit but requires reinterpret_cast.
  env->SetIntArrayRegion(java_int_array, 0, num_tokens, reinterpret_cast<const jint*>(tokens));

  return java_int_array;
}

JNIEXPORT jint JNICALL
Java_ai_onnxruntime_genai_Generator_getSequenceLastToken(JNIEnv* env, jobject thiz, jlong generator, jlong index) {
  const OgaGenerator* oga_generator = reinterpret_cast<const OgaGenerator*>(generator);

  size_t num_tokens = OgaGenerator_GetSequenceCount(oga_generator, index);
  const int32_t* tokens = OgaGenerator_GetSequenceData(oga_generator, index);

  if (num_tokens == 0) {
    ThrowException(env, "OgaGenerator_GetSequenceCount returned 0 tokens.");
    return -1;
  }

  return jint(tokens[num_tokens - 1]);
}
