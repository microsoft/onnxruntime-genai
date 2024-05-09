package ai.onnxruntime_genai;

import java.lang.System;

final class GenAI {
    /** Have the native libraries been loaded */
    private static boolean loaded = false;

    static synchronized void init() {
        if (loaded) {
            return;
        }

        // TODO: Do we need to support configurable paths like https://github.com/microsoft/onnxruntime/blob/69cfcba38a60d65498f94cde30cb9c2030f7255b/java/src/main/java/ai/onnxruntime/OnnxRuntime.java#L53

        // System.load("onnxruntime");  // ORT native - should be a dependency of onnxruntime-genai so we don't need to load it
        System.load("onnxruntime-genai"); // ORT GenAI native
        System.load("onnxruntime-genai4j_jni"); // GenAI JNI layer
        loaded = true;
    }
}
