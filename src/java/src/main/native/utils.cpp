/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "utils.h"

jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    // To silence unused-parameter error.
    // This function must exist according to the JNI spec, but the arguments aren't necessary for the library
    // to request a specific version.
    (void)vm; (void) reserved;
    // Requesting 1.6 to support Android. Will need to be bumped to a later version to call interface default methods
    // from native code, or to access other new Java features.
    return JNI_VERSION_1_6;
}

namespace {
void ThrowException(JNIEnv *env, OgaResult *result) {
    // copy error so we can release the OgaResult
    jstring jerr_msg = env->NewStringUTF(OgaResultGetError(result));
    OgaDestroyResult(result);

    static const char *className = "ai/onnxruntime_genai/GenAIException";
    jclass exClazz = env->FindClass(className);
    jmethodID exConstructor = env->GetMethodID(exClazz, "<init>", "(Ljava/lang/String;)V");
    jobject javaException = env->NewObject(exClazz, exConstructor, jerr_msg);
    env->Throw(static_cast<jthrowable>(javaException));
}
}

namespace Helpers {
    void ThrowIfError(JNIEnv *env, OgaResult *result) {
        if (result != nullptr) {
            ThrowException(env, result);
        }
    }
}