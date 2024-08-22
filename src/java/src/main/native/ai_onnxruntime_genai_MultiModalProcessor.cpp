/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_MultiModalProcessor.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_createMultiModalProcessor(JNIEnv* env, jobject thiz, jlong model_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  OgaMultiModalProcessor* processor = nullptr;

  if (ThrowIfError(env, OgaCreateMultiModalProcessor(model, &processor))) {
    return 0;
  }

  return reinterpret_cast<jlong>(processor);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_destroyMultiModalProcessor(JNIEnv* env, jobject thiz, jlong processor_handle) {
  OgaMultiModalProcessor* processor = reinterpret_cast<OgaMultiModalProcessor*>(processor_handle);
  OgaDestroyMultiModalProcessor(processor);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorProcessImages(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                                     jstring prompt, jlong images_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);

  const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);
  OgaImages* images = reinterpret_cast<OgaImages*>(images_handle);

  OgaNamedTensors* named_tensors = nullptr;
  if (ThrowIfError(env, OgaProcessorProcessImages(processor, prompt_str, images, &named_tensors))) {
    return 0;
  }

  return reinterpret_cast<jlong>(named_tensors);
}

JNIEXPORT jstring JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorDecode(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                              jintArray sequence) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);
  auto num_tokens = env->GetArrayLength(sequence);
  jint* jtokens = env->GetIntArrayElements(sequence, nullptr);
  const int32_t* tokens = reinterpret_cast<const int32_t*>(jtokens);  // convert between 32-bit types
  const char* decoded_text = nullptr;

  bool error = ThrowIfError(env, OgaProcessorDecode(processor, tokens, num_tokens, &decoded_text));
  env->ReleaseIntArrayElements(sequence, jtokens, JNI_ABORT);

  if (error) {
    return nullptr;
  }

  jstring result = env->NewStringUTF(decoded_text);
  OgaDestroyString(decoded_text);

  return result;
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_createTokenizerStreamFromProcessor(JNIEnv* env, jobject thiz, jlong processor_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);
  OgaTokenizerStream* tokenizer_stream = nullptr;

  if (ThrowIfError(env, OgaCreateTokenizerStreamFromProcessor(processor, &tokenizer_stream))) {
    return 0;
  }

  return reinterpret_cast<jlong>(tokenizer_stream);
}
