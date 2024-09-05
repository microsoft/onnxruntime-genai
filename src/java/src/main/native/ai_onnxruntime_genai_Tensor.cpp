/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ai_onnxruntime_genai_Tensor.h"

#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

/*
 * Class:     ai_onnxruntime_genai_Tensor
 * Method:    createTensor
 * Signature: (Ljava/nio/ByteBuffer;[JI)J
 */
JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_genai_Tensor_createTensor(JNIEnv* env, jobject thiz, jobject tensor_data,
                                              jlongArray shape_dims_in, jint element_type_in) {
  void* data = env->GetDirectBufferAddress(tensor_data);
  const int64_t* shape_dims = env->GetLongArrayElements(shape_dims_in, /*isCopy*/ 0);
  size_t shape_dims_count = env->GetArrayLength(shape_dims_in);
  OgaElementType element_type = static_cast<OgaElementType>(element_type_in);
  OgaTensor* tensor = nullptr;

  if (ThrowIfError(env, OgaCreateTensorFromBuffer(data, shape_dims, shape_dims_count, element_type, &tensor))) {
    return 0;
  }

  return reinterpret_cast<jlong>(tensor);
}

/*
 * Class:     ai_onnxruntime_genai_Tensor
 * Method:    destroyTensor
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_ai_onnxruntime_genai_Tensor_destroyTensor(JNIEnv* env, jobject thiz, jlong native_handle) {
  OgaDestroyTensor(reinterpret_cast<OgaTensor*>(native_handle));
}
