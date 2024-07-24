/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <jni.h>
#include <stdlib.h>
#include <string>
#include "ort_genai_c.h"

#ifdef __cplusplus
extern "C" {
#endif

jint JNI_OnLoad(JavaVM* vm, void* reserved);

#ifdef __cplusplus
}
#endif

namespace Helpers {
void ThrowException(JNIEnv* env, const char* message);

/// @brief Throw a GenAIException if the result is an error.
/// @param env JNI environment
/// @param result Result from GenAI C API call
/// @return True if there was an error. JNI code should generally return immediately if this is true.
bool ThrowIfError(JNIEnv* env, OgaResult* result);

// handle conversion/release of jstring to const char*
struct CString {
  CString(JNIEnv* env, jstring str)
      : env_{env}, str_{str}, cstr{env->GetStringUTFChars(str, /* isCopy */ nullptr)}, len_{env->GetStringUTFLength(str)} {
  }

  const char* cstr;

  operator const char*() const { return cstr; }

  std::string utf8String() {
    return {cstr, len_};
  }

  ~CString() {
    env_->ReleaseStringUTFChars(str_, cstr);
  }

 private:
  JNIEnv* env_;
  jstring str_;
  int len_;
};
}  // namespace Helpers
