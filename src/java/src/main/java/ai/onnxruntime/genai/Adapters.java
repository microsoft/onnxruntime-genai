/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;


public final class Adapters implements AutoCloseable {
  private long nativeHandle = 0;

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyAdapters(nativeHandle);
      nativeHandle = 0;
    }
  }

  long nativeHandle() {
    return nativeHandle;
  }

  private native long createAdapters(long modelHandle)
      throws GenAIException;

  private native void destroyAdapters(long nativeHandle);

  private native void loadAdapter(long nativeHandle, String adapterFilePath, String adapterName)
      throws GenAIException;

  private native void unloadAdapter(long nativeHandle, String adapterName)
      throws GenAIException;
}