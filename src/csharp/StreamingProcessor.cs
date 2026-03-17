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
        /// Enable Voice Activity Detection. Chunks without speech will be skipped.
        /// </summary>
        /// <param name="vadModelPath">Path to the silero_vad.onnx model file.</param>
        /// <param name="threshold">Speech probability threshold (default 0.5).</param>
        public void EnableVad(string vadModelPath, float threshold = 0.5f)
        {
            Result.VerifySuccess(NativeMethods.OgaStreamingProcessorEnableVad(
                _processorHandle, System.Text.Encoding.UTF8.GetBytes(vadModelPath + '\0'), threshold));
        }

        /// <summary>
        /// Disable Voice Activity Detection. All chunks will be processed.
        /// </summary>
        public void DisableVad()
        {
            Result.VerifySuccess(NativeMethods.OgaStreamingProcessorDisableVad(_processorHandle));
        }

        /// <summary>
        /// Set the VAD speech probability threshold.
        /// </summary>
        public void SetVadThreshold(float threshold)
        {
            Result.VerifySuccess(NativeMethods.OgaStreamingProcessorSetVadThreshold(_processorHandle, threshold));
        }

        /// <summary>
        /// Returns true if VAD is currently enabled.
        /// </summary>
        public bool IsVadEnabled
        {
            get
            {
                bool enabled;
                Result.VerifySuccess(NativeMethods.OgaStreamingProcessorIsVadEnabled(_processorHandle, out enabled));
                return enabled;
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
