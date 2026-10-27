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

        public string Text => StringUtils.FromUtf8(NativeMethods.OgaTokenPostProcessorResultGetText(Handle));

        public TokenPostProcessorTimestampResult TimestampResult
        {
            get
            {
                var handle = NativeMethods.OgaTokenPostProcessorResultGetTimestampResult(Handle);
                return handle == IntPtr.Zero ? null : new TokenPostProcessorTimestampResult(_owner, _lifetime, handle);
            }
        }
    }

    public sealed class TokenPostProcessorTimestampResult
    {
        private readonly TokenPostProcessor _owner;
        private readonly object _lifetime;
        private readonly IntPtr _handle;

        internal TokenPostProcessorTimestampResult(TokenPostProcessor owner, object lifetime, IntPtr handle)
        {
            _owner = owner;
            _lifetime = lifetime;
            _handle = handle;
        }

        private IntPtr Handle => _owner.GetResultHandle(_handle, _lifetime);

        public IReadOnlyList<TokenPostProcessorTimestampRecord> Words => GetRecords(false);
        public IReadOnlyList<TokenPostProcessorTimestampRecord> Segments => GetRecords(true);

        private TokenPostProcessorTimestampRecord[] GetRecords(bool segments)
        {
            var handle = Handle;
            int count = checked((int)(segments ? NativeMethods.OgaTokenPostProcessorTimestampResultGetSegmentCount(handle)
                : NativeMethods.OgaTokenPostProcessorTimestampResultGetWordCount(handle)).ToUInt64());
            var records = new TokenPostProcessorTimestampRecord[count];
            for (int index = 0; index < count; ++index)
            {
                var kind = segments ? NativeMethods.TimestampRecordKind.Segments : NativeMethods.TimestampRecordKind.Words;
                var record = NativeMethods.OgaTokenPostProcessorTimestampResultGetTimestampRecord(handle, kind, (UIntPtr)index);
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

        public string Text => StringUtils.FromUtf8(NativeMethods.OgaTokenPostProcessorTimestampRecordGetText(Handle));
        public (long Start, long Stop) FrameInterval
        {
            get
            {
                var handle = Handle;
                return (NativeMethods.OgaTokenPostProcessorTimestampRecordGetStartFrame(handle),
                    NativeMethods.OgaTokenPostProcessorTimestampRecordGetStopFrame(handle));
            }
        }

        public (double Start, double Stop) TimeInterval
        {
            get
            {
                var handle = Handle;
                return (NativeMethods.OgaTokenPostProcessorTimestampRecordGetStartTime(handle),
                    NativeMethods.OgaTokenPostProcessorTimestampRecordGetStopTime(handle));
            }
        }
    }
}