/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "ort_genai_c.h"
#include "utils.h"

extern "C"
JNIEXPORT void JNICALL
Java_ai_onnxruntime_1genai_Sequences_destroySequences(JNIEnv *env, jobject thiz, jlong sequences_handle) {
    OgaSequences* sequences = reinterpret_cast<OgaSequences*>(sequences);
    OgaDestroySequences(sequences);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_1genai_Sequences_getSequencesCount(JNIEnv *env, jobject thiz, jlong sequences_handle) {
    const OgaSequences* sequences = reinterpret_cast<const OgaSequences*>(sequences);    
    size_t num_sequences = OgaSequencesCount(sequences);
    return static_cast<jlong>(num_sequences);
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_ai_onnxruntime_1genai_Sequences_getSequence(JNIEnv *env, jobject thiz, jlong sequences_handle,
                                                 jlong sequence_index) {
    const OgaSequences* sequences = reinterpret_cast<const OgaSequences*>(sequences);

    size_t num_tokens = OgaSequencesGetSequenceCount(sequences, (size_t)sequence_index);
    const int32_t* tokens = OgaSequencesGetSequenceData(sequences, (size_t)sequence_index);

    // as there's no 'destroy' function in GenAI C API for the tokens we assume the OgaSequences owns the memory.
    // copy the tokens so there's no potential for Java code to write to it (values should be treated as const)
    // or attempt to access the memory after the OgaGenerator is destroyed.
    jintArray java_int_array = env->NewIntArray(num_tokens);
    // jint is `long` on Windows and `int` on linux. 32-bit but requires reinterpret_cast. 
    env->SetIntArrayRegion(java_int_array, 0, num_tokens, reinterpret_cast<const jint*>(tokens));

    return java_int_array;
}

