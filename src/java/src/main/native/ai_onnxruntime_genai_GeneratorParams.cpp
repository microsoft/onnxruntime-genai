/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_GeneratorParams.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_createGeneratorParams(JNIEnv* env, jobject thiz, jlong model_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  OgaGeneratorParams* generator_params = nullptr;
  if (ThrowIfError(env, OgaCreateGeneratorParams(model, &generator_params))) {
    return 0;
  }

  return reinterpret_cast<jlong>(generator_params);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_destroyGeneratorParams(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  OgaDestroyGeneratorParams(generator_params);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setSearchOptionNumber(JNIEnv* env, jobject thiz, jlong native_handle,
                                                                jstring option_name, jdouble value) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  CString name{env, option_name};

  ThrowIfError(env, OgaGeneratorParamsSetSearchNumber(generator_params, name, value));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setSearchOptionBool(JNIEnv* env, jobject thiz, jlong native_handle,
                                                              jstring option_name, jboolean value) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  CString name{env, option_name};

  ThrowIfError(env, OgaGeneratorParamsSetSearchBool(generator_params, name, value));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setInputSequences(JNIEnv* env, jobject thiz, jlong native_handle,
                                                            jlong sequences_handle) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  const OgaSequences* sequences = reinterpret_cast<const OgaSequences*>(sequences_handle);

  ThrowIfError(env, OgaGeneratorParamsSetInputSequences(generator_params, sequences));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setInputIDs(JNIEnv* env, jobject thiz, jlong native_handle,
                                                      jobject token_ids, jint sequence_length, jint batch_size) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);

  auto num_tokens = sequence_length * batch_size;
  const int32_t* tokens = reinterpret_cast<const int32_t*>(env->GetDirectBufferAddress(token_ids));

  ThrowIfError(env, OgaGeneratorParamsSetInputIDs(generator_params, tokens, num_tokens, sequence_length, batch_size));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setModelInput(JNIEnv* env, jobject thiz, jlong native_handle,
                                                        jstring input_name, jlong tensor) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  CString name{env, input_name};
  OgaTensor* input_tensor = reinterpret_cast<OgaTensor*>(tensor);

  ThrowIfError(env, OgaGeneratorParamsSetModelInput(generator_params, name, input_tensor));
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_GeneratorParams_setInputs(JNIEnv* env, jobject thiz, jlong native_handle,
                                                    jlong namedTensors) {
  OgaGeneratorParams* generator_params = reinterpret_cast<OgaGeneratorParams*>(native_handle);
  OgaNamedTensors* input_tensor = reinterpret_cast<OgaNamedTensors*>(namedTensors);

  ThrowIfError(env, OgaGeneratorParamsSetInputs(generator_params, input_tensor));
}
