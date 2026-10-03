/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** Snapshot of KEV prefix-reuse execution paths. */
public final class PrefixReuseStats {
  public final long prefixRuns;
  public final long branchRuns;
  public final long fallbackRuns;

  PrefixReuseStats(long[] values) {
    prefixRuns = values[0];
    branchRuns = values[1];
    fallbackRuns = values[2];
  }
}
