using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public sealed class TokenPostProcessor : IDisposable
    {
        private IntPtr _handle;
        private readonly Generator _generator;

        public TokenPostProcessor(Model model, Generator generator)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            _generator = generator ?? throw new ArgumentNullException(nameof(generator));
            Result.VerifySuccess(NativeMethods.OgaCreateTokenPostProcessor(model.Handle, generator.Handle, out _handle));
        }

        private IntPtr Handle
        {
            get
            {
                if (_handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(TokenPostProcessor));
                return _handle;
            }
        }

        public void ProcessNextTokens()
        {
            Result.VerifySuccess(NativeMethods.OgaTokenPostProcessorProcessNextTokens(Handle));
            GC.KeepAlive(_generator);
        }

        public void FinalizeProcessing()
        {
            Result.VerifySuccess(NativeMethods.OgaTokenPostProcessorFinalize(Handle));
            GC.KeepAlive(_generator);
        }

        public void Reset()
        {
            Result.VerifySuccess(NativeMethods.OgaTokenPostProcessorReset(Handle));
            GC.KeepAlive(_generator);
        }

        public TokenMetadataOutput[] GetNextTokenResults()
        {
            int count = checked((int)NativeMethods.OgaTokenPostProcessorGetResultCount(Handle).ToUInt64());
            var results = new TokenMetadataOutput[count];
            for (int index = 0; index < count; ++index)
            {
                var result = NativeMethods.OgaTokenPostProcessorGetResult(Handle, (UIntPtr)index);
                TokenMetadataTimestamp timestamps = null;
                if (NativeMethods.OgaTokenPostProcessorResultHasTimestamps(result))
                    timestamps = new TokenMetadataTimestamp(CopyRecords(result, false), CopyRecords(result, true));
                results[index] = new TokenMetadataOutput(StringUtils.FromUtf8(NativeMethods.OgaTokenPostProcessorResultGetText(result)), timestamps);
            }
            return results;
        }

        private static TokenMetadataTimestampRecord[] CopyRecords(IntPtr result, bool segments)
        {
            int count = checked((int)(segments ? NativeMethods.OgaTokenPostProcessorResultGetSegmentCount(result) : NativeMethods.OgaTokenPostProcessorResultGetWordCount(result)).ToUInt64());
            var records = new TokenMetadataTimestampRecord[count];
            for (int index = 0; index < count; ++index)
            {
                var record = segments ? NativeMethods.OgaTokenPostProcessorResultGetSegment(result, (UIntPtr)index) : NativeMethods.OgaTokenPostProcessorResultGetWord(result, (UIntPtr)index);
                records[index] = new TokenMetadataTimestampRecord(
                    StringUtils.FromUtf8(NativeMethods.OgaTokenPostProcessorTimestampRecordGetText(record)),
                    NativeMethods.OgaTokenPostProcessorTimestampRecordGetStartFrame(record),
                    NativeMethods.OgaTokenPostProcessorTimestampRecordGetStopFrame(record),
                    NativeMethods.OgaTokenPostProcessorTimestampRecordGetStartTime(record),
                    NativeMethods.OgaTokenPostProcessorTimestampRecordGetStopTime(record));
            }
            return records;
        }

        ~TokenPostProcessor() { Dispose(); }

        public void Dispose()
        {
            if (_handle == IntPtr.Zero) return;
            NativeMethods.OgaDestroyTokenPostProcessor(_handle);
            _handle = IntPtr.Zero;
            GC.SuppressFinalize(this);
        }
    }
}