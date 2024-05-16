/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/**
 * A TokenizerStream is used to convert individual tokens when using Generator.generateNextToken.
 */
public class TokenizerStream implements AutoCloseable {

  private long nativeHandle = 0;

  protected TokenizerStream(long tokenizerStreamHandle) {
    assert (tokenizerStreamHandle != 0); // internal usage should never pass an invalid handle
    nativeHandle = tokenizerStreamHandle;
  }

  public String decode(int token) throws GenAIException {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }

    return tokenizerStreamDecode(nativeHandle, token);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyTokenizerStream(nativeHandle);
      nativeHandle = 0;
    }
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native String tokenizerStreamDecode(long tokenizerStreamHandle, int token)
      throws GenAIException;

  private native void destroyTokenizerStream(long tokenizerStreamHandle);
}
