/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_createGeneratorParams(JNIEnv* env, jobject thiz, jlong model_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  OgaGeneratorParams* generator_params = nullptr;
  if (ThrowIfError(env, OgaCreateGeneratorParams(model, &generator_params))) {
    return 0;
  }

  return reinterpret_cast<jlong>(generator_params);
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_destroyGeneratorParams(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  OgaDestroyGeneratorParams(generator_params);
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setSearchOptionNumber(JNIEnv* env, jobject thiz, jlong native_handle,
                                                                 jstring option_name, jdouble value) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  CString name{env, option_name};

  ThrowIfError(env, OgaGeneratorParamsSetSearchNumber(generator_params, name, value));
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setSearchOptionBool(JNIEnv* env, jobject thiz, jlong native_handle,
                                                               jstring option_name, jboolean value) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  CString name{env, option_name};

  ThrowIfError(env, OgaGeneratorParamsSetSearchBool(generator_params, name, value));
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setInputSequences(JNIEnv* env, jobject thiz, jlong native_handle,
                                                             jlong sequences_handle) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  const OgaSequences* sequences = reinterpret_cast<const OgaSequences*>(sequences_handle);

  ThrowIfError(env, OgaGeneratorParamsSetInputSequences(generator_params, sequences));
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setInputIDs(JNIEnv* env, jobject thiz, jlong native_handle,
                                                       jintArray token_ids, jint sequence_length, jint batch_size) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);

  auto num_tokens = env->GetArrayLength(token_ids);
  jboolean is_copy = false;
  jint* jtokens = env->GetIntArrayElements(token_ids, &is_copy);
  if (is_copy) {
    // we're dead as GenAI doesn't copy the inputs so the input data address will be invalid
    // when we go to generate output...
    // TODO: Figure out how we can pass in the token ids without copying here,
    // and keep the original int[] valid while running generation.
    ThrowException(env, "OgaGeneratorParamsSetInputIDs was called with temporary input data.");
    env->ReleaseIntArrayElements(token_ids, jtokens, JNI_ABORT);
    return;
  }

  const int32_t* tokens = reinterpret_cast<const int32_t*>(jtokens);  // convert between 32-bit types

  ThrowIfError(env, OgaGeneratorParamsSetInputIDs(generator_params, tokens, num_tokens, sequence_length, batch_size));
  env->ReleaseIntArrayElements(token_ids, jtokens, JNI_ABORT);
}