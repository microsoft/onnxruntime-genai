/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "ort_genai_c.h"


//        ulong sequenceLength = NativeMethods.OgaGenerator_GetSequenceCount(_generatorHandle, (UIntPtr)index).ToUInt64();
//        IntPtr sequencePtr = NativeMethods.OgaGenerator_GetSequenceData(_generatorHandle, (UIntPtr)index);
//        unsafe
//        {
//            return new ReadOnlySpan<int>(sequencePtr.ToPointer(), (int)sequenceLength);
//        }
extern "C" JNIEXPORT jobject JNICALL
Java_ai_onnxruntime_1genai_Generator_getSequence(JNIEnv *env, jobject thiz, jlong generator) {
    const int32_t* tokens = OgaGenerator_GetSequence(generator, 0);
    size_t num_tokens = OgaGenerator_GetSequenceLength(generator, 0);

    // call IntBuffer.wrap(
}