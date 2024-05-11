/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "ort_genai_c.h"
#include "utils.h"

using namespace Helpers;

extern "C"
JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_1genai_Generator_createGenerator(JNIEnv *env, jobject thiz, jlong model_handle,
                                                     jlong generator_params_handle) {
    const OgaModel* model = reinterpret_cast<const OgaModel*>(model_handle);
    const OgaGeneratorParams* params = reinterpret_cast<const OgaGeneratorParams*>(generator_params_handle);
    OgaGenerator* generator = nullptr;
    ThrowIfError(env, OgaCreateGenerator(model, params, &generator));
    return (jlong)generator;
}

extern "C"
JNIEXPORT void JNICALL
Java_ai_onnxruntime_1genai_Generator_releaseGenerator(JNIEnv *env, jobject thiz, jlong native_handle) {
    OgaDestroyGenerator(reinterpret_cast<OgaGenerator*>(native_handle));
}

extern "C"
JNIEXPORT jboolean JNICALL
        Java_ai_onnxruntime_1genai_Generator_isDone(JNIEnv *env, jobject thiz, jlong native_handle) {
    return OgaGenerator_IsDone(reinterpret_cast<OgaGenerator*>(native_handle));
}

extern "C"
JNIEXPORT void JNICALL
Java_ai_onnxruntime_1genai_Generator_computeLogits(JNIEnv *env, jobject thiz, jlong native_handle) {
// TODO: implement computeLogits()
}
extern "C"
JNIEXPORT void JNICALL
Java_ai_onnxruntime_1genai_Generator_generateNextToken(JNIEnv *env, jobject thiz, jlong native_handle) {
// TODO: implement generateNextToken()
}

extern "C" JNIEXPORT jintArray JNICALL
Java_ai_onnxruntime_1genai_Generator_getSequence(JNIEnv *env, jobject thiz, jlong generator) {
    const OgaGenerator* oga_generator = reinterpret_cast<const OgaGenerator*>(generator);
    const int32_t* tokens = OgaGenerator_GetSequenceData(oga_generator, 0);
    size_t num_tokens = OgaGenerator_GetSequenceCount(oga_generator, 0);

    // copy the tokens so there's no potential for a Java user to write to it as it is still owned by GenAI C++ code

    jintArray java_int_array = env->NewIntArray(num_tokens);
    // jint is `long` on Windows, which is a 32-bit integer but requires a reinterpret_cast.
    //  On linux jint is `int`. 
    env->SetIntArrayRegion(java_int_array, 0, num_tokens, reinterpret_cast<const jint*>(tokens));

    return java_int_array;
}