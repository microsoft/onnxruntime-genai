// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class StreamingAudioProcessor : IDisposable
    {
        private IntPtr _processorHandle;
        private bool _disposed = false;

        public StreamingAudioProcessor(Model model)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateAudioProcessor(model.Handle, out _processorHandle));
        }

        internal IntPtr Handle { get { return _processorHandle; } }

        /// <summary>
        /// Feed a chunk of raw PCM audio (mono, float32, 16kHz).
        /// Returns a mel spectrogram Tensor if a full chunk is ready, or null if more audio is needed.
        /// </summary>
        public Tensor? Process(float[] audioData)
        {
            IntPtr melHandle = IntPtr.Zero;
            unsafe
            {
                fixed (float* audioPtr = audioData)
                {
                    Result.VerifySuccess(NativeMethods.OgaAudioProcessorProcess(
                        _processorHandle, audioPtr, (UIntPtr)audioData.Length, out melHandle));
                }
            }
            return melHandle != IntPtr.Zero ? new Tensor(melHandle) : null;
        }

        /// <summary>
        /// Flush remaining buffered audio (pads with silence).
        /// Returns a mel Tensor or null if the buffer was empty.
        /// </summary>
        public Tensor? Flush()
        {
            IntPtr melHandle = IntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaAudioProcessorFlush(_processorHandle, out melHandle));
            return melHandle != IntPtr.Zero ? new Tensor(melHandle) : null;
        }

        /// <summary>
        /// Reset processor state for a new utterance.
        /// </summary>
        public void Reset()
        {
            Result.VerifySuccess(NativeMethods.OgaAudioProcessorReset(_processorHandle));
        }

        ~StreamingAudioProcessor()
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
            NativeMethods.OgaDestroyAudioProcessor(_processorHandle);
            _processorHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
