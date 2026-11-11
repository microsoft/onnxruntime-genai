/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** An immutable snapshot of speculative decoding statistics. */
public final class SpeculativeStats implements AutoCloseable {
  private long nativeHandle;

  SpeculativeStats(long nativeHandle) {
    if (nativeHandle == 0) {
      throw new IllegalArgumentException("nativeHandle must not be zero");
    }
    this.nativeHandle = nativeHandle;
  }

  public long getCount(String name) throws GenAIException {
    ensureValid();
    return getCount(nativeHandle, name);
  }

  public double getNumber(String name) throws GenAIException {
    ensureValid();
    return getNumber(nativeHandle, name);
  }

  public boolean getBool(String name) throws GenAIException {
    ensureValid();
    return getBool(nativeHandle, name);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroySpeculativeStats(nativeHandle);
      nativeHandle = 0;
    }
  }

  private void ensureValid() {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native void destroySpeculativeStats(long nativeHandle);

  private native long getCount(long nativeHandle, String name) throws GenAIException;

  private native double getNumber(long nativeHandle, String name) throws GenAIException;

  private native boolean getBool(long nativeHandle, String name) throws GenAIException;
}
