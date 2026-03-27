// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class StreamingProcessor : IDisposable
    {
        private IntPtr _processorHandle;
        private bool _disposed = false;

        public StreamingProcessor(Model model)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateStreamingProcessor(model.Handle, out _processorHandle));
        }

        internal IntPtr Handle { get { return _processorHandle; } }

        /// <summary>
        /// Feed a chunk of raw PCM audio (mono, float32, 16kHz).
        /// Returns a NamedTensors if a full chunk is ready, or null if more audio is needed.
        /// </summary>
        public NamedTensors? Process(float[] audioData)
        {
            IntPtr outHandle = IntPtr.Zero;
            unsafe
            {
                fixed (float* audioPtr = audioData)
                {
                    Result.VerifySuccess(NativeMethods.OgaStreamingProcessorProcess(
                        _processorHandle, audioPtr, (UIntPtr)audioData.Length, out outHandle));
                }
            }
            return outHandle != IntPtr.Zero ? new NamedTensors(outHandle) : null;
        }

        /// <summary>
        /// Flush remaining buffered audio (pads with silence).
        /// Returns a NamedTensors or null if the buffer was empty.
        /// </summary>
        public NamedTensors? Flush()
        {
            IntPtr outHandle = IntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaStreamingProcessorFlush(_processorHandle, out outHandle));
            return outHandle != IntPtr.Zero ? new NamedTensors(outHandle) : null;
        }

        /// <summary>
        /// Set a processor option as a key-value pair.
        /// Supported keys: "vad_enabled", "vad_threshold", "vad_min_silence_chunks", "vad_model_path".
        /// </summary>
        public void SetOption(string key, string value)
        {
            Result.VerifySuccess(NativeMethods.OgaStreamingProcessorSetOption(
                _processorHandle,
                StringUtils.ToUtf8(key),
                StringUtils.ToUtf8(value)));
        }

        /// <summary>
        /// Get a processor option value by key.
        /// </summary>
        public string GetOption(string key)
        {
            IntPtr valuePtr = IntPtr.Zero;
            try
            {
                Result.VerifySuccess(NativeMethods.OgaStreamingProcessorGetOption(
                    _processorHandle,
                    StringUtils.ToUtf8(key),
                    out valuePtr));
                return StringUtils.FromUtf8(valuePtr);
            }
            finally
            {
                NativeMethods.OgaDestroyString(valuePtr);
            }
        }

        ~StreamingProcessor()
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
            NativeMethods.OgaDestroyStreamingProcessor(_processorHandle);
            _processorHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
