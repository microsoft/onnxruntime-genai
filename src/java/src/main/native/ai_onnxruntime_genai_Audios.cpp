/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Audios.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

/*
 * Class:     ai_onnxruntime_genai_Audios
 * Method:    loadAudios
 * Signature: (J)V
 */
JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Audios_loadAudios(JNIEnv* env, jobject thiz, jstring audio_path) {
  CString path(env, audio_path);

  OgaAudios* audios = nullptr;
  if (ThrowIfError(env, OgaLoadAudio(path, &audios))) {
    return 0;
  }

  return reinterpret_cast<jlong>(audios);
}

/*
 * Class:     ai_onnxruntime_genai_Audios
 * Method:    destroyAudios
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Audios_destroyAudios(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaDestroyAudios(reinterpret_cast<OgaAudios*>(native_handle));
}
