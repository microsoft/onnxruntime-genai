// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

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

        /// <summary>
        /// Decodes one timed token and returns text plus word and segment events
        /// completed by this call.
        /// </summary>
        /// <remarks>
        /// Word and segment lists are never cumulative. Each may be empty or
        /// contain multiple records when one token spans multiple boundaries.
        /// </remarks>
        public TimestampDecodeResult DecodeWithTimestamps(TokenTiming token)
        {
            Result.VerifySuccess(NativeMethods.OgaTokenizerStreamDecodeWithTimestamps(
                _tokenizerStreamHandle, in token, out IntPtr result));
            return CopyTimestampResult(result);
        }

        public TimestampDecodeResult FinalizeTimestamps()
        {
            Result.VerifySuccess(NativeMethods.OgaTokenizerStreamFinalizeTimestamps(
                _tokenizerStreamHandle, out IntPtr result));
            return CopyTimestampResult(result);
        }

        public void Reset()
        {
            Result.VerifySuccess(NativeMethods.OgaTokenizerStreamReset(_tokenizerStreamHandle));
        }

        private static TimestampDecodeResult CopyTimestampResult(IntPtr result)
        {
            Result.VerifySuccess(NativeMethods.OgaTimestampDecodeResultGetText(result, out IntPtr text));
            var words = CopyTimestampRecords(result, true);
            var segments = CopyTimestampRecords(result, false);
            return new TimestampDecodeResult(StringUtils.FromUtf8(text), words, segments);
        }

        private static TimestampRecord[] CopyTimestampRecords(IntPtr result, bool words)
        {
            ulong count = (words
                ? NativeMethods.OgaTimestampDecodeResultGetWordCount(result)
                : NativeMethods.OgaTimestampDecodeResultGetSegmentCount(result)).ToUInt64();
            var records = new TimestampRecord[count];
            for (ulong index = 0; index < count; index++)
            {
                IntPtr status = words
                    ? NativeMethods.OgaTimestampDecodeResultGetWord(
                        result, (UIntPtr)index, out IntPtr text, out long startFrame, out long stopFrame,
                        out double startTime, out double stopTime)
                    : NativeMethods.OgaTimestampDecodeResultGetSegment(
                        result, (UIntPtr)index, out text, out startFrame, out stopFrame,
                        out startTime, out stopTime);
                Result.VerifySuccess(status);
                records[index] = new TimestampRecord(
                    StringUtils.FromUtf8(text), startFrame, stopFrame, startTime, stopTime);
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
