/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** This class can load audios from the given path and prepare them for processing. */
public class Audios implements AutoCloseable {
  private long nativeHandle;

  /**
   * Construct an Audios instance.
   *
   * @param audioPath The audio path.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Audios(String audioPath) throws GenAIException {
    nativeHandle = loadAudios(audioPath);
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyAudios(nativeHandle);
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

  private native long loadAudios(String audioPath) throws GenAIException;

  private native void destroyAudios(long audioshandle);
}
