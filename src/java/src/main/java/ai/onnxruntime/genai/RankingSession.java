/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.util.Map;

/** Typed CLM package session. */
public final class RankingSession implements AutoCloseable {
  private long nativeHandle;

  public RankingSession(String packagePath, String... providers) throws GenAIException {
    if (packagePath == null) {
      throw new NullPointerException("packagePath");
    }
    nativeHandle =
        NonGenerativeNative.createRankingSession(
            packagePath, providers == null ? new String[0] : providers);
  }

  public synchronized ModelResult run(StructuredRequest request) throws GenAIException {
    checkOpen();
    if (request == null) {
      throw new NullPointerException("request");
    }
    Map<String, Object> value =
        NonGenerativeNative.execute(
            nativeHandle,
            true,
            false,
            request.state(),
            request.nativeQuestions(),
            request.temperature());
    return new ModelResult(value);
  }

  public synchronized RankingResult rank(FreeFormRankRequest request) throws GenAIException {
    checkOpen();
    if (request == null) {
      throw new NullPointerException("request");
    }
    return new RankingResult(
        NonGenerativeNative.rank(
            nativeHandle,
            request.state,
            request.instructions,
            request.candidates,
            request.temperature));
  }

  public synchronized void setCacheCapacity(long entries, long bytes) throws GenAIException {
    checkOpen();
    if (entries < 0 || bytes < 0) {
      throw new IllegalArgumentException("Cache capacities must be non-negative");
    }
    NonGenerativeNative.cache(nativeHandle, true, 0, entries, bytes);
  }

  public synchronized CacheStats getCacheStats() throws GenAIException {
    checkOpen();
    return new CacheStats(NonGenerativeNative.cache(nativeHandle, true, 1, 0, 0));
  }

  public synchronized void clearCache() throws GenAIException {
    checkOpen();
    NonGenerativeNative.cache(nativeHandle, true, 2, 0, 0);
  }

  public synchronized void invalidateCache() throws GenAIException {
    checkOpen();
    NonGenerativeNative.cache(nativeHandle, true, 3, 0, 0);
  }

  private void checkOpen() {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
  }

  @Override
  public synchronized void close() {
    if (nativeHandle != 0) {
      NonGenerativeNative.destroyRankingSession(nativeHandle);
      nativeHandle = 0;
    }
  }
}
