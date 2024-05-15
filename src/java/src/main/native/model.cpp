/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_1genai_Model_createModel(JNIEnv* env, jobject thiz, jstring model_path) {
  CString path{env, model_path};

  OgaModel* model = nullptr;
  ThrowIfError(env, OgaCreateModel(path, &model));

  return reinterpret_cast<jlong>(model);
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_1genai_Model_destroyModel(JNIEnv* env, jobject thiz, jlong model_handle) {
  OgaModel* model = reinterpret_cast<OgaModel*>(model_handle);
  OgaDestroyModel(model);
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_1genai_Model_generate(JNIEnv* env, jobject thiz, jlong model_handle,
                                          jlong generator_params_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  const OgaGeneratorParams* params = reinterpret_cast<const OgaGeneratorParams*>(generator_params_handle);
  OgaSequences* sequences = nullptr;
  ThrowIfError(env, OgaGenerate(model, params, &sequences));

  return reinterpret_cast<jlong>(sequences);
}