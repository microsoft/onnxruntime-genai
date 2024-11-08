/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Config.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Config_createConfig(JNIEnv* env, jobject thiz, jstring model_path) {
  CString path{env, model_path};

  OgaConfig* config = nullptr;
  if (ThrowIfError(env, OgaCreateConfig(path, &config))) {
    return 0;
  }

  return reinterpret_cast<jlong>(config);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Config_destroyConfig(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaConfig* config = reinterpret_cast<OgaConfig*>(native_handle);
  OgaDestroyConfig(config);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Config_clearProviders(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaConfig* config = reinterpret_cast<OgaConfig*>(native_handle);
  ThrowIfError(env, OgaConfigClearProviders(config));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Config_appendProvider(JNIEnv* env, jobject thiz, jlong native_handle, jstring provider_name) {
  CString c_provider_name{env, provider_name};
  OgaConfig* config = reinterpret_cast<OgaConfig*>(native_handle);

  ThrowIfError(env, OgaConfigAppendProvider(config, c_provider_name));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Config_setProvider(JNIEnv* env, jobject thiz, jlong native_handle, jstring provider_name, jstring option_name, jstring option_value) {
  CString c_provider_name{env, provider_name};
  CString c_option_name{env, option_name};
  CString c_option_value{env, option_value};
  OgaConfig* config = reinterpret_cast<OgaConfig*>(native_handle);

  ThrowIfError(env, OgaConfigSetProviderOption(config, c_provider_name, c_option_name, c_option_value));
}
