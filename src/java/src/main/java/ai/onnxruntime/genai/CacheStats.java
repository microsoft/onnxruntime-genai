/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** Snapshot of one session's bounded cache. */
public final class CacheStats {
  public final long hits;
  public final long misses;
  public final long evictions;
  public final long entries;
  public final long bytes;
  public final long entryCapacity;
  public final long byteCapacity;

  CacheStats(long[] values) {
    hits = values[0];
    misses = values[1];
    evictions = values[2];
    entries = values[3];
    bytes = values[4];
    entryCapacity = values[5];
    byteCapacity = values[6];
  }
}
