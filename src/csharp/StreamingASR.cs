// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class StreamingASR : IDisposable
    {
        private IntPtr _streamingASRHandle;
        private bool _disposed = false;

        public StreamingASR(Model model)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateStreamingASR(model.Handle, out _streamingASRHandle));
        }

        internal IntPtr Handle { get { return _streamingASRHandle; } }

        public string TranscribeChunk(float[] audioData)
        {
            IntPtr outStr = IntPtr.Zero;
            try
            {
                unsafe
                {
                    fixed (float* audioPtr = audioData)
                    {
                        Result.VerifySuccess(NativeMethods.OgaStreamingASRTranscribeChunk(
                            _streamingASRHandle, audioPtr, (UIntPtr)audioData.Length, out outStr));
                    }
                }
                return StringUtils.FromUtf8(outStr);
            }
            finally
            {
                NativeMethods.OgaDestroyString(outStr);
            }
        }

        public string GetTranscript()
        {
            IntPtr outStr = IntPtr.Zero;
            try
            {
                Result.VerifySuccess(NativeMethods.OgaStreamingASRGetTranscript(_streamingASRHandle, out outStr));
                return StringUtils.FromUtf8(outStr);
            }
            finally
            {
                NativeMethods.OgaDestroyString(outStr);
            }
        }

        public void Reset()
        {
            Result.VerifySuccess(NativeMethods.OgaStreamingASRReset(_streamingASRHandle));
        }

        public string Flush()
        {
            IntPtr outStr = IntPtr.Zero;
            try
            {
                Result.VerifySuccess(NativeMethods.OgaStreamingASRFlush(_streamingASRHandle, out outStr));
                return StringUtils.FromUtf8(outStr);
            }
            finally
            {
                NativeMethods.OgaDestroyString(outStr);
            }
        }

        ~StreamingASR()
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
            NativeMethods.OgaDestroyStreamingASR(_streamingASRHandle);
            _streamingASRHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
