using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>An emitted token with an optional acoustic interval copied from a native record.</summary>
    public sealed class TokenMetadataInput
    {
        internal TokenMetadataInput(NativeTokenMetadataInput token)
        {
            TokenId = token.TokenId;
            if (token.HasTokenAcousticFrameInterval != 0)
            {
                var interval = token.TokenAcousticFrameInterval;
                TokenAcousticFrameInterval = (interval.Start, interval.Stop);
            }
        }
        public int TokenId { get; }
        public (long start, long stop)? TokenAcousticFrameInterval { get; }
    }

    /// <summary>A completed word or segment with frame bounds and times in seconds.</summary>
    public sealed class TokenMetadataTimestampRecord
    {
        internal TokenMetadataTimestampRecord(string text, long startFrame, long stopFrame, double startTime, double stopTime)
        {
            Text = text;
            StartFrame = startFrame;
            StopFrame = stopFrame;
            StartTime = startTime;
            StopTime = stopTime;
        }
        public string Text { get; }
        public long StartFrame { get; }
        public long StopFrame { get; }
        public double StartTime { get; }
        public double StopTime { get; }
    }

    /// <summary>Word and segment events completed by a single decode or finalize call.</summary>
    public sealed class TokenMetadataTimestamp
    {
        internal TokenMetadataTimestamp(IReadOnlyList<TokenMetadataTimestampRecord> words, IReadOnlyList<TokenMetadataTimestampRecord> segments)
        {
            Words = words;
            Segments = segments;
        }
        public IReadOnlyList<TokenMetadataTimestampRecord> Words { get; }
        public IReadOnlyList<TokenMetadataTimestampRecord> Segments { get; }
    }

    /// <summary>Decoded text and optional timestamps copied from one native stream operation.</summary>
    public sealed class TokenMetadataOutput
    {
        internal TokenMetadataOutput(string text, TokenMetadataTimestamp timestamps)
        {
            Text = text;
            TimestampMetadata = timestamps;
        }
        public string Text { get; }
        public TokenMetadataTimestamp TimestampMetadata { get; }
    }

}