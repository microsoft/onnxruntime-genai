/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Model_setupQnnEnv(JNIEnv* env, jobject thiz, jstring jpath) {
  CString path = {env, jpath};
  std::string utf8Path = path.utf8String();
#ifdef __ANDROID__
  setenv("ADSP_LIBRARY_PATH",
         (utf8Path + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp").c_str(),
         1 /*overwrite*/);
#endif
  return 0;
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Model_createModel(JNIEnv* env, jobject thiz, jstring model_path) {
  CString path{env, model_path};

  OgaModel* model = nullptr;
  if (ThrowIfError(env, OgaCreateModel(path, &model))) {
    return 0;
  }

  return reinterpret_cast<jlong>(model);
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Model_destroyModel(JNIEnv* env, jobject thiz, jlong model_handle) {
  OgaModel* model = reinterpret_cast<OgaModel*>(model_handle);
  OgaDestroyModel(model);
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Model_generate(JNIEnv* env, jobject thiz, jlong model_handle,
                                         jlong generator_params_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  const OgaGeneratorParams* params = reinterpret_cast<const OgaGeneratorParams*>(generator_params_handle);
  OgaSequences* sequences = nullptr;
  if (ThrowIfError(env, OgaGenerate(model, params, &sequences))) {
    return 0;
  }

  return reinterpret_cast<jlong>(sequences);
}
