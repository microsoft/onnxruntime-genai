/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

public class Images implements AutoCloseable{
    private long nativeHandle;

    public Images(String imagePath) throws GenAIException {
        nativeHandle = loadImages(imagePath);
    }

    @Override
    public void close() {
        if (nativeHandle != 0) {
            destroyImages(nativeHandle);
            nativeHandle = 0;
        }
    }

    long nativeHandle() {
        return nativeHandle;
    }

    static {
        try {
          GenAI.init();
        } catch (Exception e) {
          throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
        }
    }

    private native long loadImages(String imagePath) throws GenAIException;

    private native void destroyImages(long imageshandle);
}
