/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_GenAI.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GenAI_shutdown(JNIEnv* env, jclass cls) {
  OgaShutdown();
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GenAI_setTelemetryNative(JNIEnv* env, jclass cls, jboolean send_telemetry) {
  OgaSetTelemetryEnabled(send_telemetry == JNI_TRUE);
}