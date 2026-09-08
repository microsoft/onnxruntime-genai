/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_SpeculativeStats.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_SpeculativeStats_destroySpeculativeStats(
    JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaDestroySpeculativeStats(reinterpret_cast<OgaSpeculativeStats*>(native_handle));
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_SpeculativeStats_getCount(
    JNIEnv* env, jobject thiz, jlong native_handle, jstring name) {
  CString key{env, name};
  uint64_t value{};
  if (ThrowIfError(env, OgaSpeculativeStatsGetCount(
                            reinterpret_cast<const OgaSpeculativeStats*>(native_handle), key, &value))) {
    return 0;
  }
  return static_cast<jlong>(value);
}

JNIEXPORT jdouble JNICALL
Java_ai_onnxruntime_genai_SpeculativeStats_getNumber(
    JNIEnv* env, jobject thiz, jlong native_handle, jstring name) {
  CString key{env, name};
  double value{};
  if (ThrowIfError(env, OgaSpeculativeStatsGetNumber(
                            reinterpret_cast<const OgaSpeculativeStats*>(native_handle), key, &value))) {
    return 0;
  }
  return static_cast<jdouble>(value);
}

JNIEXPORT jboolean JNICALL
Java_ai_onnxruntime_genai_SpeculativeStats_getBool(
    JNIEnv* env, jobject thiz, jlong native_handle, jstring name) {
  CString key{env, name};
  bool value{};
  if (ThrowIfError(env, OgaSpeculativeStatsGetBool(
                            reinterpret_cast<const OgaSpeculativeStats*>(native_handle), key, &value))) {
    return JNI_FALSE;
  }
  return value ? JNI_TRUE : JNI_FALSE;
}
