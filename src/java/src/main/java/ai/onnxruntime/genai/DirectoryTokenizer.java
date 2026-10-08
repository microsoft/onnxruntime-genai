/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** Tokenizer loaded directly from a non-generative package directory. */
public final class DirectoryTokenizer implements AutoCloseable {
  private long nativeHandle;

  public DirectoryTokenizer(String packagePath) throws GenAIException {
    if (packagePath == null) {
      throw new NullPointerException("packagePath");
    }
    nativeHandle = NonGenerativeNative.createDirectoryTokenizer(packagePath);
  }

  public synchronized int[] encode(String text) throws GenAIException {
    checkOpen();
    if (text == null) {
      throw new NullPointerException("text");
    }
    return NonGenerativeNative.directoryTokenizerEncode(nativeHandle, text);
  }

  public synchronized int getPadTokenId() throws GenAIException {
    checkOpen();
    return NonGenerativeNative.directoryTokenizerPadTokenId(nativeHandle);
  }

  private void checkOpen() {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
  }

  @Override
  public synchronized void close() {
    if (nativeHandle != 0) {
      NonGenerativeNative.destroyDirectoryTokenizer(nativeHandle);
      nativeHandle = 0;
    }
  }
}
