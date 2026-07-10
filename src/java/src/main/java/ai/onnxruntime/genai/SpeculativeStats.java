/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** Speculative decoding work, delivery, timing, and speedup statistics. */
public final class SpeculativeStats {
  private final long rounds;
  private final long completedRounds;
  private final long interruptedRounds;
  private final long activeRounds;
  private final long draftTokensProposed;
  private final long draftTokensEvaluated;
  private final long draftTokensAccepted;
  private final long correctionTokens;
  private final long bonusTokens;
  private final long tokensQueued;
  private final long tokensEmitted;
  private final long tokensDiscarded;
  private final long tokensBuffered;
  private final long draftForwardPasses;
  private final long targetForwardPasses;
  private final boolean formulaSupported;
  private final float totalDraftMs;
  private final float totalTargetMs;
  private final float totalReconciliationMs;
  private final float avgDraftMsPerToken;
  private final float acceptanceRate;
  private final float avgDraftTokensPerRound;
  private final float meanEmittedTokensPerRound;
  private final float expectedTokensPerRound;
  private final float avgTargetMsPerRound;
  private final float targetBaselineMsPerToken;
  private final float targetOverheadRatio;
  private final float estimatedSpeedup;
  private final float observedSpeedup;

  SpeculativeStats(long[] counts, float[] values) {
    rounds = counts[0];
    completedRounds = counts[1];
    interruptedRounds = counts[2];
    activeRounds = counts[3];
    draftTokensProposed = counts[4];
    draftTokensEvaluated = counts[5];
    draftTokensAccepted = counts[6];
    correctionTokens = counts[7];
    bonusTokens = counts[8];
    tokensQueued = counts[9];
    tokensEmitted = counts[10];
    tokensDiscarded = counts[11];
    tokensBuffered = counts[12];
    draftForwardPasses = counts[13];
    targetForwardPasses = counts[14];
    formulaSupported = counts[15] != 0;
    totalDraftMs = values[0];
    totalTargetMs = values[1];
    totalReconciliationMs = values[2];
    avgDraftMsPerToken = values[3];
    acceptanceRate = values[4];
    avgDraftTokensPerRound = values[5];
    meanEmittedTokensPerRound = values[6];
    expectedTokensPerRound = values[7];
    avgTargetMsPerRound = values[8];
    targetBaselineMsPerToken = values[9];
    targetOverheadRatio = values[10];
    estimatedSpeedup = values[11];
    observedSpeedup = values[12];
  }

  public long getRounds() {
    return rounds;
  }

  public long getCompletedRounds() {
    return completedRounds;
  }

  public long getInterruptedRounds() {
    return interruptedRounds;
  }

  public long getActiveRounds() {
    return activeRounds;
  }

  public long getDraftTokensProposed() {
    return draftTokensProposed;
  }

  public long getDraftTokensEvaluated() {
    return draftTokensEvaluated;
  }

  public long getDraftTokensAccepted() {
    return draftTokensAccepted;
  }

  public long getCorrectionTokens() {
    return correctionTokens;
  }

  public long getBonusTokens() {
    return bonusTokens;
  }

  public long getTokensQueued() {
    return tokensQueued;
  }

  public long getTokensEmitted() {
    return tokensEmitted;
  }

  public long getTokensDiscarded() {
    return tokensDiscarded;
  }

  public long getTokensBuffered() {
    return tokensBuffered;
  }

  public long getDraftForwardPasses() {
    return draftForwardPasses;
  }

  public long getTargetForwardPasses() {
    return targetForwardPasses;
  }

  public boolean isFormulaSupported() {
    return formulaSupported;
  }

  public float getTotalDraftMs() {
    return totalDraftMs;
  }

  public float getTotalTargetMs() {
    return totalTargetMs;
  }

  public float getTotalReconciliationMs() {
    return totalReconciliationMs;
  }

  public float getAvgDraftMsPerToken() {
    return avgDraftMsPerToken;
  }

  public float getAcceptanceRate() {
    return acceptanceRate;
  }

  public float getAvgDraftTokensPerRound() {
    return avgDraftTokensPerRound;
  }

  public float getMeanEmittedTokensPerRound() {
    return meanEmittedTokensPerRound;
  }

  public float getExpectedTokensPerRound() {
    return expectedTokensPerRound;
  }

  public float getAvgTargetMsPerRound() {
    return avgTargetMsPerRound;
  }

  public float getTargetBaselineMsPerToken() {
    return targetBaselineMsPerToken;
  }

  public float getTargetOverheadRatio() {
    return targetOverheadRatio;
  }

  public float getEstimatedSpeedup() {
    return estimatedSpeedup;
  }

  public float getObservedSpeedup() {
    return observedSpeedup;
  }
}
