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

        public string Text => StringUtils.FromUtf8(NativeMethods.OgaASRPostProcessorResultGetText(Handle));

        public ASRPostProcessorTimestampResult TimestampResult
        {
            get
            {
                var handle = NativeMethods.OgaASRPostProcessorResultGetTimestampResult(Handle);
                return handle == IntPtr.Zero ? null : new ASRPostProcessorTimestampResult(_owner, _lifetime, handle);
            }
        }
    }

    public sealed class ASRPostProcessorTimestampResult
    {
        private readonly ASRPostProcessor _owner;
        private readonly object _lifetime;
        private readonly IntPtr _handle;

        internal ASRPostProcessorTimestampResult(ASRPostProcessor owner, object lifetime, IntPtr handle)
        {
            _owner = owner;
            _lifetime = lifetime;
            _handle = handle;
        }

        private IntPtr Handle => _owner.GetResultHandle(_handle, _lifetime);

        public IReadOnlyList<ASRPostProcessorTimestampRecord> Words => GetRecords(NativeMethods.TimestampRecordKind.Words);
        public IReadOnlyList<ASRPostProcessorTimestampRecord> Segments => GetRecords(NativeMethods.TimestampRecordKind.Segments);

        private ASRPostProcessorTimestampRecord[] GetRecords(NativeMethods.TimestampRecordKind kind)
        {
            var handle = Handle;
            int count = checked((int)(kind == NativeMethods.TimestampRecordKind.Segments ? NativeMethods.OgaASRPostProcessorTimestampResultGetSegmentCount(handle)
                : NativeMethods.OgaASRPostProcessorTimestampResultGetWordCount(handle)).ToUInt64());
            var records = new ASRPostProcessorTimestampRecord[count];
            for (int index = 0; index < count; ++index)
            {
                var record = NativeMethods.OgaASRPostProcessorTimestampResultGetTimestampRecord(handle, kind, (UIntPtr)index);
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

        public string Text => StringUtils.FromUtf8(NativeMethods.OgaASRPostProcessorTimestampRecordGetText(Handle));
        public (long Start, long Stop) FrameInterval
        {
            get
            {
                var handle = Handle;
                return (NativeMethods.OgaASRPostProcessorTimestampRecordGetStartFrame(handle),
                    NativeMethods.OgaASRPostProcessorTimestampRecordGetStopFrame(handle));
            }
        }

        public (double Start, double Stop) TimeInterval
        {
            get
            {
                var handle = Handle;
                return (NativeMethods.OgaASRPostProcessorTimestampRecordGetStartTime(handle),
                    NativeMethods.OgaASRPostProcessorTimestampRecordGetStopTime(handle));
            }
        }
    }
}