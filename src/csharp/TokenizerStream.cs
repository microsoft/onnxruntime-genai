// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class TokenizerStream : IDisposable
    {
        private IntPtr _tokenizerStreamHandle;
        private bool _disposed = false;

        internal TokenizerStream(IntPtr tokenizerStreamHandle)
        {
            _tokenizerStreamHandle = tokenizerStreamHandle;
        }

        internal IntPtr Handle { get { return _tokenizerStreamHandle; } }

        public string Decode(int token)
        {
            IntPtr decodedStr = IntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaTokenizerStreamDecode(_tokenizerStreamHandle, token, out decodedStr));
            return StringUtils.FromUtf8(decodedStr);
        }

        public TokenMetadataOutput DecodeWithMetadata(TokenMetadataInput token)
        {
            var timing = token.TokenAcousticFrameInterval;
            var interval = new NativeTokenMetadataAcousticFrameInterval
            {
                Start = timing?.start ?? 0,
                Stop = timing?.stop ?? 0,
            };
            var nativeToken = new NativeTokenMetadataInput
            {
                TokenId = token.TokenId,
                HasTokenAcousticFrameInterval = timing.HasValue ? 1 : 0,
                TokenAcousticFrameInterval = interval,
            };
            Result.VerifySuccess(NativeMethods.OgaTokenizerStreamDecodeWithMetadata(
                _tokenizerStreamHandle, in nativeToken, out IntPtr result));
            return CopyMetadata(result);
        }

        public TokenMetadataOutput FinalizeMetadata()
        {
            Result.VerifySuccess(NativeMethods.OgaTokenizerStreamFinalizeMetadata(
                _tokenizerStreamHandle, out IntPtr result));
            return CopyMetadata(result);
        }

        public void Reset()
        {
            Result.VerifySuccess(NativeMethods.OgaTokenizerStreamReset(_tokenizerStreamHandle));
        }

        private static TokenMetadataOutput CopyMetadata(IntPtr result)
        {
            var native = Marshal.PtrToStructure<NativeTokenMetadataOutput>(result);
            TokenMetadataTimestamp timestamps = null;
            if (native.TimestampMetadata != IntPtr.Zero)
            {
                var source = Marshal.PtrToStructure<NativeTokenMetadataTimestamp>(native.TimestampMetadata);
                timestamps = new TokenMetadataTimestamp(CopyTimestampRecords(source.Words, source.WordCount),
                    CopyTimestampRecords(source.Segments, source.SegmentCount));
            }
            return new TokenMetadataOutput(StringUtils.FromUtf8(native.Text), timestamps);
        }

        private static TokenMetadataTimestampRecord[] CopyTimestampRecords(IntPtr source, UIntPtr nativeCount)
        {
            int count = checked((int)nativeCount.ToUInt64());
            var records = new TokenMetadataTimestampRecord[count];
            int stride = Marshal.SizeOf<NativeTokenMetadataTimestampRecord>();
            for (int index = 0; index < count; index++)
            {
                var record = Marshal.PtrToStructure<NativeTokenMetadataTimestampRecord>(IntPtr.Add(source, checked(index * stride)));
                records[index] = new TokenMetadataTimestampRecord(
                    StringUtils.FromUtf8(record.Text), record.StartFrame, record.StopFrame, record.StartTime, record.StopTime);
            }
            return records;
        }

        ~TokenizerStream()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }
            NativeMethods.OgaDestroyTokenizerStream(_tokenizerStreamHandle);
            _tokenizerStreamHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
