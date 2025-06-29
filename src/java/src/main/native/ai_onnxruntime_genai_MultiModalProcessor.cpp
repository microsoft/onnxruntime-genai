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
  bool error = ThrowIfError(env, OgaProcessorProcessImages(processor, prompt_str, images, &named_tensors));

  env->ReleaseStringUTFChars(prompt, prompt_str);
  return error ? 0 : reinterpret_cast<jlong>(named_tensors);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorProcessImagesAndPrompts(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                                               jobjectArray prompts, jlong images_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);

  OgaImages* images = reinterpret_cast<OgaImages*>(images_handle);

  jsize length_jsize = env->GetArrayLength(prompts);
  int length_int = static_cast<int>(length_jsize);
  const char** prompts_ = new const char*[length_int];
  for (jsize i = 0; i < length_jsize; i++) {
    jstring prompt_jstring = reinterpret_cast<jstring>(env->GetObjectArrayElement(prompts, i));
    const char* prompt_c_str = env->GetStringUTFChars(prompt_jstring, nullptr);
    prompts_[i] = strdup(prompt_c_str);
    env->ReleaseStringUTFChars(prompt_jstring, prompt_c_str);
    env->DeleteLocalRef(prompt_jstring);
  }

  OgaNamedTensors* named_tensors = nullptr;
  OgaStringArray* strs;
  OgaCreateStringArrayFromStrings(prompts_, length_int, &strs);
  bool error = ThrowIfError(env, OgaProcessorProcessImagesAndPrompts(processor, strs, images, &named_tensors));

  for (jsize i = 0; i < length_jsize; i++) {
    free((void*)prompts_[i]);
  }
  OgaDestroyStringArray(strs);
  return error ? 0 : reinterpret_cast<jlong>(named_tensors);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorProcessAudios(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                                     jstring prompt, jlong audios_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);

  const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);
  OgaAudios* audios = reinterpret_cast<OgaAudios*>(audios_handle);

  OgaNamedTensors* named_tensors = nullptr;
  bool error = ThrowIfError(env, OgaProcessorProcessAudios(processor, prompt_str, audios, &named_tensors));

  env->ReleaseStringUTFChars(prompt, prompt_str);
  return error ? 0 : reinterpret_cast<jlong>(named_tensors);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorProcessAudiosAndPrompts(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                                               jobjectArray prompts, jlong audios_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);

  OgaAudios* audios = reinterpret_cast<OgaAudios*>(audios_handle);

  jsize length_jsize = env->GetArrayLength(prompts);
  int length_int = static_cast<int>(length_jsize);
  const char** prompts_ = new const char*[length_int];
  for (jsize i = 0; i < length_jsize; i++) {
    jstring prompt_jstring = reinterpret_cast<jstring>(env->GetObjectArrayElement(prompts, i));
    const char* prompt_c_str = env->GetStringUTFChars(prompt_jstring, nullptr);
    prompts_[i] = strdup(prompt_c_str);
    env->ReleaseStringUTFChars(prompt_jstring, prompt_c_str);
    env->DeleteLocalRef(prompt_jstring);
  }

  OgaNamedTensors* named_tensors = nullptr;
  OgaStringArray* strs;
  OgaCreateStringArrayFromStrings(prompts_, length_int, &strs);
  bool error = ThrowIfError(env, OgaProcessorProcessAudiosAndPrompts(processor, strs, audios, &named_tensors));

  for (jsize i = 0; i < length_jsize; i++) {
    free((void*)prompts_[i]);
  }
  OgaDestroyStringArray(strs);
  return error ? 0 : reinterpret_cast<jlong>(named_tensors);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorProcessImagesAndAudios(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                                              jstring prompt, jlong images_handle, jlong audios_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);

  const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);
  OgaImages* images = reinterpret_cast<OgaImages*>(images_handle);
  OgaAudios* audios = reinterpret_cast<OgaAudios*>(audios_handle);

  OgaNamedTensors* named_tensors = nullptr;
  bool error = ThrowIfError(env, OgaProcessorProcessImagesAndAudios(processor, prompt_str, images, audios, &named_tensors));

  env->ReleaseStringUTFChars(prompt, prompt_str);
  return error ? 0 : reinterpret_cast<jlong>(named_tensors);
}

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_MultiModalProcessor_processorProcessImagesAndAudiosAndPrompts(JNIEnv* env, jobject thiz, jlong processor_handle,
                                                                                        jobjectArray prompts, jlong images_handle, jlong audios_handle) {
  const OgaMultiModalProcessor* processor = reinterpret_cast<const OgaMultiModalProcessor*>(processor_handle);

  OgaImages* images = reinterpret_cast<OgaImages*>(images_handle);
  OgaAudios* audios = reinterpret_cast<OgaAudios*>(audios_handle);

  jsize length_jsize = env->GetArrayLength(prompts);
  int length_int = static_cast<int>(length_jsize);
  const char** prompts_ = new const char*[length_int];
  for (jsize i = 0; i < length_jsize; i++) {
    jstring prompt_jstring = reinterpret_cast<jstring>(env->GetObjectArrayElement(prompts, i));
    const char* prompt_c_str = env->GetStringUTFChars(prompt_jstring, nullptr);
    prompts_[i] = strdup(prompt_c_str);
    env->ReleaseStringUTFChars(prompt_jstring, prompt_c_str);
    env->DeleteLocalRef(prompt_jstring);
  }

  OgaNamedTensors* named_tensors = nullptr;
  OgaStringArray* strs;
  OgaCreateStringArrayFromStrings(prompts_, length_int, &strs);
  bool error = ThrowIfError(env, OgaProcessorProcessImagesAndAudiosAndPrompts(processor, strs, images, audios, &named_tensors));

  for (jsize i = 0; i < length_jsize; i++) {
    free((void*)prompts_[i]);
  }
  OgaDestroyStringArray(strs);
  return error ? 0 : reinterpret_cast<jlong>(named_tensors);
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
