/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <stdlib.h>
#include "src/ort_genai_c.h"

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

jint JNI_OnLoad(JavaVM *vm, void *reserved);

// void ThrowIfError(JNIEnv *env, Result *result);

#ifdef __cplusplus
}
#endif
#endif
