/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_ImageGeneratorParams.h"
#include <vector>
#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_ImageGeneratorParams_createImageGeneratorParams(JNIEnv* env, jobject thiz, jlong model_handle) {
  const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
  OgaImageGeneratorParams* image_generator_params = nullptr;
  if (ThrowIfError(env, OgaCreateImageGeneratorParams(model, &image_generator_params))) {
    return 0;
  }

  return reinterpret_cast<jlong>(image_generator_params);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_ImageGeneratorParams_destroyImageGeneratorParams(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaImageGeneratorParams* image_generator_params = reinterpret_cast<OgaImageGeneratorParams*>(native_handle);
  OgaDestroyImageGeneratorParams(image_generator_params);
}

JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_ImageGeneratorParams_setPrompts(JNIEnv* env, jobject thiz, jlong native_handle, 
                                                          jobjectArray prompts, jobjectArray negative_prompts, jlong prompt_count) {
  OgaImageGeneratorParams* params = reinterpret_cast<OgaImageGeneratorParams*>(native_handle);
  
  // Convert Java string arrays to C strings
  jsize count = static_cast<jsize>(prompt_count);
  std::vector<const char*> prompt_cstrs(count);
  std::vector<const char*> negative_prompt_cstrs(count);
  
  // Process prompts
  for (jsize i = 0; i < count; i++) {
    jstring prompt_str = (jstring)env->GetObjectArrayElement(prompts, i);
    if (prompt_str != nullptr) {
      prompt_cstrs[i] = env->GetStringUTFChars(prompt_str, nullptr);
    } else {
      prompt_cstrs[i] = nullptr;
    }
    env->DeleteLocalRef(prompt_str);
  }

  
  // Process negative prompts if provided
  if (negative_prompts != nullptr) {
    for (jsize i = 0; i < count; i++) {
      jstring neg_prompt_str = (jstring)env->GetObjectArrayElement(negative_prompts, i);
      if (neg_prompt_str != nullptr) {
        negative_prompt_cstrs[i] = env->GetStringUTFChars(neg_prompt_str, nullptr);
      } else {
        negative_prompt_cstrs[i] = nullptr;
      }
      env->DeleteLocalRef(neg_prompt_str);
    }
  } else {
    // If no negative prompts provided, set all to nullptr
    for (jsize i = 0; i < count; i++) {
      negative_prompt_cstrs[i] = nullptr;
    }
  }
  
  // Call the C API
  ThrowIfError(env, OgaImageGeneratorParamsSetPrompts(params, prompt_cstrs.data(), 
                                                     negative_prompt_cstrs.data(), count));
  
  // Release allocated strings
  for (jsize i = 0; i < count; i++) {
    if (prompt_cstrs[i] != nullptr) {
      jstring prompt_str = (jstring)env->GetObjectArrayElement(prompts, i);
      env->ReleaseStringUTFChars(prompt_str, prompt_cstrs[i]);
      env->DeleteLocalRef(prompt_str);
    }
    
    if (negative_prompts != nullptr && negative_prompt_cstrs[i] != nullptr) {
      jstring neg_prompt_str = (jstring)env->GetObjectArrayElement(negative_prompts, i);
      env->ReleaseStringUTFChars(neg_prompt_str, negative_prompt_cstrs[i]);
      env->DeleteLocalRef(neg_prompt_str);
    }
  }
}