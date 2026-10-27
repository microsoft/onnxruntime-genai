using System;
using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public sealed class ASRPostProcessorResult
    {
        private readonly ASRPostProcessor _owner;
        private readonly object _lifetime;
        private readonly IntPtr _handle;

        internal ASRPostProcessorResult(ASRPostProcessor owner, object lifetime, IntPtr handle)
        {
            _owner = owner;
            _lifetime = lifetime;
            _handle = handle;
        }

        private IntPtr Handle => _owner.GetResultHandle(_handle, _lifetime);

        public string GetText() => StringUtils.FromUtf8(NativeMethods.OgaASRPostProcessorResultGetText(Handle));

        public ASRPostProcessorTimestamp GetTimestamps()
        {
            var handle = NativeMethods.OgaASRPostProcessorResultGetTimestamps(Handle);
            return handle == IntPtr.Zero ? null : new ASRPostProcessorTimestamp(_owner, _lifetime, handle);
        }
    }

    public sealed class ASRPostProcessorTimestamp
    {
        private readonly ASRPostProcessor _owner;
        private readonly object _lifetime;
        private readonly IntPtr _handle;

        internal ASRPostProcessorTimestamp(ASRPostProcessor owner, object lifetime, IntPtr handle)
        {
            _owner = owner;
            _lifetime = lifetime;
            _handle = handle;
        }

        private IntPtr Handle => _owner.GetResultHandle(_handle, _lifetime);

        public IReadOnlyList<ASRPostProcessorTimestampRecord> GetWords() => GetRecords(false);
        public IReadOnlyList<ASRPostProcessorTimestampRecord> GetSegments() => GetRecords(true);

        private ASRPostProcessorTimestampRecord[] GetRecords(bool segments)
        {
            var handle = Handle;
            int count = checked((int)(segments ? NativeMethods.OgaASRPostProcessorTimestampGetSegmentCount(handle)
                : NativeMethods.OgaASRPostProcessorTimestampGetWordCount(handle)).ToUInt64());
            var records = new ASRPostProcessorTimestampRecord[count];
            for (int index = 0; index < count; ++index)
            {
                var record = segments ? NativeMethods.OgaASRPostProcessorTimestampGetSegment(handle, (UIntPtr)index)
                    : NativeMethods.OgaASRPostProcessorTimestampGetWord(handle, (UIntPtr)index);
                records[index] = new ASRPostProcessorTimestampRecord(_owner, _lifetime, record);
            }
            return records;
        }
    }

    public sealed class ASRPostProcessorTimestampRecord
    {
        private readonly ASRPostProcessor _owner;
        private readonly object _lifetime;
        private readonly IntPtr _handle;

        internal ASRPostProcessorTimestampRecord(ASRPostProcessor owner, object lifetime, IntPtr handle)
        {
            _owner = owner;
            _lifetime = lifetime;
            _handle = handle;
        }

        private IntPtr Handle => _owner.GetResultHandle(_handle, _lifetime);

        public string GetText() => StringUtils.FromUtf8(NativeMethods.OgaASRPostProcessorTimestampRecordGetText(Handle));
        public long GetStartFrame() => NativeMethods.OgaASRPostProcessorTimestampRecordGetStartFrame(Handle);
        public long GetStopFrame() => NativeMethods.OgaASRPostProcessorTimestampRecordGetStopFrame(Handle);
        public double GetStartTime() => NativeMethods.OgaASRPostProcessorTimestampRecordGetStartTime(Handle);
        public double GetStopTime() => NativeMethods.OgaASRPostProcessorTimestampRecordGetStopTime(Handle);
    }
}