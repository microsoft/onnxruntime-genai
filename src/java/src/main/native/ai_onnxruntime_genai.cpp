/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "ort_genai_c.h"

extern "C" JNIEXPORT jobject JNICALL
Java_ai_onnxruntime_1genai_Generator_getSequence(JNIEnv *env, jobject thiz, jlong generator) {
    const OgaGenerator* oga_generator = reinterpret_cast<const OgaGenerator*>(generator);
    const int32_t* tokens = OgaGenerator_GetSequenceData(oga_generator, 0);
    size_t num_tokens = OgaGenerator_GetSequenceCount(oga_generator, 0);

    // create ByteBuffer
    jobject byte_buffer = env->NewDirectByteBuffer(static_cast<void*>(tokens), num_tokens * sizeof(*tokens));

    // jclass byteBufferClass = env->FindClass("java/nio/ByteBuffer");
    jclass bytebuffer_class = env->GetOjectClass(byte_buffer);
    jclass intbuffer_class = env->FindClass("java/nio/IntBuffer");
    jmethodID asIntBuffer_methodID = env->GetMethodID(bytebuffer_class, "asIntBuffer", "()Ljava/nio/IntBuffer;");

    // convert to IntBuffer:
    jobject int_buffer = env->CallObjectMethod(byte_buffer, asIntBuffer_methodID);

    return int_buffer;
}