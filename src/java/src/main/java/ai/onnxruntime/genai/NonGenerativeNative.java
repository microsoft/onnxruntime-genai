/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.util.Map;

final class NonGenerativeNative {
  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  static native long createDirectoryTokenizer(String path) throws GenAIException;

  static native void destroyDirectoryTokenizer(long handle);

  static native int[] directoryTokenizerEncode(long handle, String text) throws GenAIException;

  static native int directoryTokenizerPadTokenId(long handle) throws GenAIException;

  static native long createRankingSession(String path, String[] providers) throws GenAIException;

  static native long createDecisionSession(String path, String[] providers) throws GenAIException;

  static native void destroyRankingSession(long handle);

  static native void destroyDecisionSession(long handle);

  static native Map<String, Object> execute(
      long handle,
      boolean ranking,
      boolean decide,
      Object state,
      Map<String, Object> questions,
      Float temperature)
      throws GenAIException;

  static native Map<String, Object> rank(
      long handle,
      Object state,
      Object instructions,
      Map<String, Object> candidates,
      Float temperature)
      throws GenAIException;

  static native long[] cache(
      long handle, boolean ranking, int operation, long entries, long bytes)
      throws GenAIException;

  private NonGenerativeNative() {}
}
