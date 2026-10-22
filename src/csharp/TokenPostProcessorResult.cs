using System;
using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public sealed class TokenPostProcessorResult
    {
        private readonly TokenPostProcessor _owner;
        private readonly object _lifetime;
        private readonly IntPtr _handle;

        internal TokenPostProcessorResult(TokenPostProcessor owner, object lifetime, IntPtr handle)
        {
            _owner = owner;
            _lifetime = lifetime;
            _handle = handle;
        }

        private IntPtr Handle => _owner.GetResultHandle(_handle, _lifetime);

        public string GetText() => StringUtils.FromUtf8(NativeMethods.OgaTokenPostProcessorResultGetText(Handle));

        public TokenPostProcessorTimestamp GetTimestamps()
        {
            var handle = NativeMethods.OgaTokenPostProcessorResultGetTimestamps(Handle);
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
            int count = checked((int)(segments ? NativeMethods.OgaTokenPostProcessorTimestampGetSegmentCount(handle)
                : NativeMethods.OgaTokenPostProcessorTimestampGetWordCount(handle)).ToUInt64());
            var records = new TokenPostProcessorTimestampRecord[count];
            for (int index = 0; index < count; ++index)
            {
                var record = segments ? NativeMethods.OgaTokenPostProcessorTimestampGetSegment(handle, (UIntPtr)index)
                    : NativeMethods.OgaTokenPostProcessorTimestampGetWord(handle, (UIntPtr)index);
                records[index] = new TokenPostProcessorTimestampRecord(_owner, _lifetime, record);
            }
            return records;
        }
    }

    public sealed class TokenPostProcessorTimestampRecord
    {
        private readonly TokenPostProcessor _owner;
        private readonly object _lifetime;
        private readonly IntPtr _handle;

        internal TokenPostProcessorTimestampRecord(TokenPostProcessor owner, object lifetime, IntPtr handle)
        {
            _owner = owner;
            _lifetime = lifetime;
            _handle = handle;
        }

        private IntPtr Handle => _owner.GetResultHandle(_handle, _lifetime);

        public string GetText() => StringUtils.FromUtf8(NativeMethods.OgaTokenPostProcessorTimestampRecordGetText(Handle));
        public bool HasTimestamp() => NativeMethods.OgaTokenPostProcessorTimestampRecordHasTimestamp(Handle);
        public string GetSpeakerLabel()
        {
            var label = NativeMethods.OgaTokenPostProcessorTimestampRecordGetSpeakerLabel(Handle);
            return label == IntPtr.Zero ? null : StringUtils.FromUtf8(label);
        }
        public long? GetStartFrame() => HasTimestamp() ? NativeMethods.OgaTokenPostProcessorTimestampRecordGetStartFrame(Handle) : (long?)null;
        public long? GetStopFrame() => HasTimestamp() ? NativeMethods.OgaTokenPostProcessorTimestampRecordGetStopFrame(Handle) : (long?)null;
        public double? GetStartTime() => HasTimestamp() ? NativeMethods.OgaTokenPostProcessorTimestampRecordGetStartTime(Handle) : (double?)null;
        public double? GetStopTime() => HasTimestamp() ? NativeMethods.OgaTokenPostProcessorTimestampRecordGetStopTime(Handle) : (double?)null;
    }
}