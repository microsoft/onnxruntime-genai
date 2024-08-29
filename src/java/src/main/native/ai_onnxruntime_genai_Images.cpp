/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Images.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

/*
 * Class:     ai_onnxruntime_genai_Images
 * Method:    loadImages
 * Signature: (J)V
 */
JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Images_loadImages(JNIEnv* env, jobject thiz, jstring image_path) {
  CString path(env, image_path);

  OgaImages* images = nullptr;
  if (ThrowIfError(env, OgaLoadImage(path, &images))) {
    return 0;
  }

  return reinterpret_cast<jlong>(images);
}

/*
 * Class:     ai_onnxruntime_genai_Images
 * Method:    destroyImages
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Images_destroyImages(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaDestroyImages(reinterpret_cast<OgaImages*>(native_handle));
}
