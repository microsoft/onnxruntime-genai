using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
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
        public TokenMetadataTimestamp GetTimestamps() { return TimestampMetadata; }
    }

}