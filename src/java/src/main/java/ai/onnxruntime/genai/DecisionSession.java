/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** Typed KEV package session. */
public final class DecisionSession implements AutoCloseable {
  private long nativeHandle;

  public DecisionSession(String packagePath, String... providers) throws GenAIException {
    if (packagePath == null) {
      throw new NullPointerException("packagePath");
    }
    nativeHandle =
        NonGenerativeNative.createDecisionSession(
            packagePath, providers == null ? new String[0] : providers);
  }

  public synchronized ModelResult run(StructuredRequest request) throws GenAIException {
    return execute(request, false);
  }

  public synchronized ModelResult decide(StructuredRequest request) throws GenAIException {
    return execute(request, true);
  }

  private ModelResult execute(StructuredRequest request, boolean decide) throws GenAIException {
    checkOpen();
    if (request == null) {
      throw new NullPointerException("request");
    }
    return new ModelResult(
        NonGenerativeNative.execute(
            nativeHandle,
            false,
            decide,
            request.state(),
            request.nativeQuestions(),
            request.temperature()));
  }

  public synchronized void setCacheCapacity(long entries, long bytes) throws GenAIException {
    checkOpen();
    if (entries < 0 || bytes < 0) {
      throw new IllegalArgumentException("Cache capacities must be non-negative");
    }
    NonGenerativeNative.cache(nativeHandle, false, 0, entries, bytes);
  }

  public synchronized CacheStats getCacheStats() throws GenAIException {
    checkOpen();
    return new CacheStats(NonGenerativeNative.cache(nativeHandle, false, 1, 0, 0));
  }

  public synchronized void clearCache() throws GenAIException {
    checkOpen();
    NonGenerativeNative.cache(nativeHandle, false, 2, 0, 0);
  }

  public synchronized void invalidateCache() throws GenAIException {
    checkOpen();
    NonGenerativeNative.cache(nativeHandle, false, 3, 0, 0);
  }

  private void checkOpen() {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Instance has been freed and is invalid");
    }
  }

  @Override
  public synchronized void close() {
    if (nativeHandle != 0) {
      NonGenerativeNative.destroyDecisionSession(nativeHandle);
      nativeHandle = 0;
    }
  }
}
