/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <jni.h>
#include <stdlib.h>
#include "ort_genai_c.h"

#ifdef __cplusplus
extern "C" {
#endif

jint JNI_OnLoad(JavaVM *vm, void *reserved);

#ifdef __cplusplus
}
#endif

namespace Helpers {
void ThrowIfError(JNIEnv *env, OgaResult *result);

// handle conversion/release of jstring to const char*
struct CString {
    CString(JNIEnv *env, jstring str)
            : env_{env}, str_{str}, cstr{env->GetStringUTFChars(str, /* isCopy */ nullptr)} {
    }

    const char *cstr;

    operator const char *() const { return cstr; }

    ~CString() {
        env_->ReleaseStringUTFChars(str_, cstr);
    }

private:
    JNIEnv *env_;
    jstring str_;
};
} // namespace Helpers
