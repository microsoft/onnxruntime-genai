/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

/**
 * A TokenizerStream is used to convert individual tokens when using Generator.generateNextToken.
 */
public class TokenizerStream implements AutoCloseable {

  private long tokenizerStreamHandle = 0;

  protected TokenizerStream(long nativeHandle) {
    assert (nativeHandle != 0); // internal usage should never pass an invalid handle
    tokenizerStreamHandle = nativeHandle;
  }

  public String decode(int token) throws GenAIException {
    return tokenizerStreamDecode(tokenizerStreamHandle, token);
  }

  @Override
  public void close() throws Exception {
    if (tokenizerStreamHandle != 0) {
      destroyTokenizerStream(tokenizerStreamHandle);
      tokenizerStreamHandle = 0;
    }
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native String tokenizerStreamDecode(long tokenizerStreamHandle, int token);

  private native void destroyTokenizerStream(long tokenizerStreamHandle);
}
