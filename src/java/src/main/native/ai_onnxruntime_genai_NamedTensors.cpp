/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_NamedTensors.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

/*
 * Class:     ai_onnxruntime_genai_NamedTensors
 * Method:    destroyNamedTensors
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_NamedTensors_destroyNamedTensors(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaDestroyNamedTensors(reinterpret_cast<OgaNamedTensors*>(native_handle));
}
