// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public readonly struct SpeculativeStats
    {
        internal SpeculativeStats(NativeMethods.OgaSpeculativeStats stats)
        {
            Rounds = stats.Rounds.ToUInt64();
            CompletedRounds = stats.CompletedRounds.ToUInt64();
            InterruptedRounds = stats.InterruptedRounds.ToUInt64();
            ActiveRounds = stats.ActiveRounds.ToUInt64();
            DraftTokensProposed = stats.DraftTokensProposed.ToUInt64();
            DraftTokensEvaluated = stats.DraftTokensEvaluated.ToUInt64();
            DraftTokensAccepted = stats.DraftTokensAccepted.ToUInt64();
            CorrectionTokens = stats.CorrectionTokens.ToUInt64();
            BonusTokens = stats.BonusTokens.ToUInt64();
            TokensQueued = stats.TokensQueued.ToUInt64();
            TokensEmitted = stats.TokensEmitted.ToUInt64();
            TokensDiscarded = stats.TokensDiscarded.ToUInt64();
            TokensBuffered = stats.TokensBuffered.ToUInt64();
            DraftForwardPasses = stats.DraftForwardPasses.ToUInt64();
            TargetForwardPasses = stats.TargetForwardPasses.ToUInt64();
            FormulaSupported = stats.FormulaSupported.ToUInt64() != 0;
            TotalDraftMs = stats.TotalDraftMs;
            TotalTargetMs = stats.TotalTargetMs;
            TotalReconciliationMs = stats.TotalReconciliationMs;
            AvgDraftMsPerToken = stats.AvgDraftMsPerToken;
            AcceptanceRate = stats.AcceptanceRate;
            AvgDraftTokensPerRound = stats.AvgDraftTokensPerRound;
            MeanEmittedTokensPerRound = stats.MeanEmittedTokensPerRound;
            ExpectedTokensPerRound = stats.ExpectedTokensPerRound;
            AvgTargetMsPerRound = stats.AvgTargetMsPerRound;
            TargetBaselineMsPerToken = stats.TargetBaselineMsPerToken;
            TargetOverheadRatio = stats.TargetOverheadRatio;
            EstimatedSpeedup = stats.EstimatedSpeedup;
            ObservedSpeedup = stats.ObservedSpeedup;
        }

        public ulong Rounds { get; }
        public ulong CompletedRounds { get; }
        public ulong InterruptedRounds { get; }
        public ulong ActiveRounds { get; }
        public ulong DraftTokensProposed { get; }
        public ulong DraftTokensEvaluated { get; }
        public ulong DraftTokensAccepted { get; }
        public ulong CorrectionTokens { get; }
        public ulong BonusTokens { get; }
        public ulong TokensQueued { get; }
        public ulong TokensEmitted { get; }
        public ulong TokensDiscarded { get; }
        public ulong TokensBuffered { get; }
        public ulong DraftForwardPasses { get; }
        public ulong TargetForwardPasses { get; }
        public bool FormulaSupported { get; }
        public float TotalDraftMs { get; }
        public float TotalTargetMs { get; }
        public float TotalReconciliationMs { get; }
        public float AvgDraftMsPerToken { get; }
        public float AcceptanceRate { get; }
        public float AvgDraftTokensPerRound { get; }
        public float MeanEmittedTokensPerRound { get; }
        public float ExpectedTokensPerRound { get; }
        public float AvgTargetMsPerRound { get; }
        public float TargetBaselineMsPerToken { get; }
        public float TargetOverheadRatio { get; }
        public float EstimatedSpeedup { get; }
        public float ObservedSpeedup { get; }
    }
}
