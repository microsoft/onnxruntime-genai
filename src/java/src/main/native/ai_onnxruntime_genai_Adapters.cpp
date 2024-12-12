/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Adapters.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Adapters_createAdapters(JNIEnv* env, jobject thiz, jlong model_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  OgaAdapters* adapters = nullptr;
  if (ThrowIfError(env, OgaCreateAdapters(model, &adapters))) {
    return 0;
  }

  return reinterpret_cast<jlong>(adapters);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Adapters_destroyAdapters(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaDestroyAdapters(reinterpret_cast<OgaAdapters*>(native_handle));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Adapters_loadAdapter(JNIEnv* env, jobject thiz, jlong native_handle,
                                               jstring adapter_file_path, jstring adapter_name) {
  CString file_path{env, adapter_file_path};
  CString name{env, adapter_name};
  ThrowIfError(env, OgaLoadAdapter(reinterpret_cast<OgaAdapters*>(native_handle), file_path, name));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Adapters_unloadAdapter(JNIEnv* env, jobject thiz, jlong native_handle, jstring adapter_name) {
  CString name{env, adapter_name};
  ThrowIfError(env, OgaUnloadAdapter(reinterpret_cast<OgaAdapters*>(native_handle), name));
}
