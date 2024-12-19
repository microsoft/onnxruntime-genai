/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** This class is a list of tensors with names that match up with model input names. */
public class NamedTensors implements AutoCloseable {
  private long nativeHandle;

  /**
   * Construct a NamedTensor from native handle.
   *
   * @param handle The native handle.
   */
  public NamedTensors(long handle) {
    nativeHandle = handle;
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyNamedTensors(nativeHandle);
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

  private native void destroyNamedTensors(long handle);
}
