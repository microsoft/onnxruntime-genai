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

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Model_createModelWithEp(JNIEnv* env, jobject thiz, jstring model_path, jstring ep) {
  CString path{env, model_path};
  // `ep` may be null (Java callers pass null for "use defaulting"). Avoid
  // calling GetStringUTFChars on a null jstring — its behaviour is undefined.
  const char* c_ep = nullptr;
  if (ep != nullptr) {
    c_ep = env->GetStringUTFChars(ep, nullptr);
  }

  OgaModel* model = nullptr;
  OgaResult* result = OgaCreateModelWithEp(path, c_ep, &model);
  if (ep != nullptr) {
    env->ReleaseStringUTFChars(ep, c_ep);
  }
  if (ThrowIfError(env, result)) {
    return 0;
  }

  return reinterpret_cast<jlong>(model);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Model_createModelFromConfig(JNIEnv* env, jobject thiz, jlong config_handle) {
  const OgaConfig* config = reinterpret_cast<const OgaConfig*>(config_handle);

  OgaModel* model = nullptr;
  if (ThrowIfError(env, OgaCreateModelFromConfig(config, &model))) {
    return 0;
  }

  return reinterpret_cast<jlong>(model);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Model_destroyModel(JNIEnv* env, jobject thiz, jlong model_handle) {
  OgaModel* model = reinterpret_cast<OgaModel*>(model_handle);
  OgaDestroyModel(model);
}
