// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public sealed class TimestampRecord
    {
        internal TimestampRecord(string text, long startFrame, long stopFrame, double startTime, double stopTime)
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

    public sealed class TimestampDecodeResult
    {
        internal TimestampDecodeResult(string text, IReadOnlyList<TimestampRecord> words, IReadOnlyList<TimestampRecord> segments)
        {
            Text = text;
            Words = words;
            Segments = segments;
        }

        public string Text { get; }
        public IReadOnlyList<TimestampRecord> Words { get; }
        public IReadOnlyList<TimestampRecord> Segments { get; }
    }
}