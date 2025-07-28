/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_GeneratorParams.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_createGeneratorParams(JNIEnv* env, jobject thiz, jlong model_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  OgaGeneratorParams* generator_params = nullptr;
  if (ThrowIfError(env, OgaCreateGeneratorParams(model, &generator_params))) {
    return 0;
  }

  return reinterpret_cast<jlong>(generator_params);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_destroyGeneratorParams(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  OgaDestroyGeneratorParams(generator_params);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setSearchOptionNumber(JNIEnv* env, jobject thiz, jlong native_handle,
                                                                jstring option_name, jdouble value) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  CString name{env, option_name};

  ThrowIfError(env, OgaGeneratorParamsSetSearchNumber(generator_params, name, value));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setSearchOptionBool(JNIEnv* env, jobject thiz, jlong native_handle,
                                                              jstring option_name, jboolean value) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  CString name{env, option_name};

  ThrowIfError(env, OgaGeneratorParamsSetSearchBool(generator_params, name, value));
}
