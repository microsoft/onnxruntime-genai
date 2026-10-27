using System;
using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public sealed class TokenPostProcessorResult
    {
        private readonly TokenPostProcessor _owner;
        private readonly object _lifetime;
        private readonly IntPtr _text;
        private readonly IntPtr _timestamps;

        internal TokenPostProcessorResult(TokenPostProcessor owner, object lifetime, IntPtr text, IntPtr timestamps)
        {
            _owner = owner;
            _lifetime = lifetime;
            _text = text;
            _timestamps = timestamps;
        }

        public string GetText() => StringUtils.FromUtf8(_owner.GetResultHandle(_text, _lifetime));

        public TokenPostProcessorTimestamp GetTimestamps()
        {
            var handle = _owner.GetResultHandle(_timestamps, _lifetime);
            return handle == IntPtr.Zero ? null : new TokenPostProcessorTimestamp(_owner, _lifetime, handle);
        }
    }

    public sealed class TokenPostProcessorTimestamp
    {
        private readonly TokenPostProcessor _owner;
        private readonly object _lifetime;
        private readonly IntPtr _handle;

        internal TokenPostProcessorTimestamp(TokenPostProcessor owner, object lifetime, IntPtr handle)
        {
            _owner = owner;
            _lifetime = lifetime;
            _handle = handle;
        }

        private IntPtr Handle => _owner.GetResultHandle(_handle, _lifetime);

        public IReadOnlyList<TokenPostProcessorTimestampRecord> GetWords() => GetRecords(false);
        public IReadOnlyList<TokenPostProcessorTimestampRecord> GetSegments() => GetRecords(true);

        private TokenPostProcessorTimestampRecord[] GetRecords(bool segments)
        {
            var handle = Handle;
            var kind = segments ? NativeMethods.TimestampRecordKind.Segments : NativeMethods.TimestampRecordKind.Words;
            int count = checked((int)NativeMethods.OgaTokenPostProcessorGetTimestampRecordCount(handle, kind).ToUInt64());
            var records = new TokenPostProcessorTimestampRecord[count];
            for (int index = 0; index < count; ++index)
            {
                Result.VerifySuccess(NativeMethods.OgaTokenPostProcessorGetTimestampRecordData(handle, kind, (UIntPtr)index,
                    out IntPtr text, out long startFrame, out long stopFrame, out double startTime, out double stopTime));
                records[index] = new TokenPostProcessorTimestampRecord(_owner, _lifetime, text,
                    (startFrame, stopFrame), (startTime, stopTime));
            }
            return records;
        }
    }

    public sealed class TokenPostProcessorTimestampRecord
    {
        private readonly TokenPostProcessor _owner;
        private readonly object _lifetime;
        private readonly IntPtr _text;
        private readonly (long Start, long Stop) _frameInterval;
        private readonly (double Start, double Stop) _timeInterval;

        internal TokenPostProcessorTimestampRecord(TokenPostProcessor owner, object lifetime, IntPtr text,
            (long Start, long Stop) frameInterval, (double Start, double Stop) timeInterval)
        {
            _owner = owner;
            _lifetime = lifetime;
            _text = text;
            _frameInterval = frameInterval;
            _timeInterval = timeInterval;
        }

        public string GetText() => StringUtils.FromUtf8(_owner.GetResultHandle(_text, _lifetime));
        public (long Start, long Stop) GetFrameInterval()
        {
            _owner.GetResultHandle(_text, _lifetime);
            return _frameInterval;
        }

        public (double Start, double Stop) GetTimeInterval()
        {
            _owner.GetResultHandle(_text, _lifetime);
            return _timeInterval;
        }
    }
}