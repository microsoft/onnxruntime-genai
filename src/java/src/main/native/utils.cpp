/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "utils.h"

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  // To silence unused-parameter error.
  // This function must exist according to the JNI spec, but the arguments aren't necessary for the library
  // to request a specific version.
  (void)vm;
  (void)reserved;
  // Requesting 1.6 to support Android. Will need to be bumped to a later version to call interface default methods
  // from native code, or to access other new Java features.
  return JNI_VERSION_1_6;
}

namespace {
void ThrowExceptionImpl(JNIEnv* env, const char* error_message) {
  static const char* className = "ai/onnxruntime/genai/GenAIException";
  env->ThrowNew(env->FindClass(className), error_message);
}
}  // namespace

namespace Helpers {
void ThrowException(JNIEnv* env, const char* message) {
  ThrowExceptionImpl(env, message);
}

bool ThrowIfError(JNIEnv* env, OgaResult* result) {
  bool error = result != nullptr;

  if (error) {
    ThrowExceptionImpl(env, OgaResultGetError(result));
    OgaDestroyResult(result);
  }

  return error;
}
}  // namespace Helpers