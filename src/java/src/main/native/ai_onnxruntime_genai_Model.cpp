/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Model.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Model_createModel(JNIEnv* env, jobject thiz, jstring model_path) {
  CString path{env, model_path};

  OgaModel* model = nullptr;
  if (ThrowIfError(env, OgaCreateModel(path, &model))) {
    return 0;
  }

  return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Model_destroyModel(JNIEnv* env, jobject thiz, jlong model_handle) {
  OgaModel* model = reinterpret_cast<OgaModel*>(model_handle);
  OgaDestroyModel(model);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Model_generate(JNIEnv* env, jobject thiz, jlong model_handle,
                                         jlong generator_params_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  const OgaGeneratorParams* params = reinterpret_cast<const OgaGeneratorParams*>(generator_params_handle);
  OgaSequences* sequences = nullptr;
  if (ThrowIfError(env, OgaGenerate(model, params, &sequences))) {
    return 0;
  }

  return reinterpret_cast<jlong>(sequences);
}
