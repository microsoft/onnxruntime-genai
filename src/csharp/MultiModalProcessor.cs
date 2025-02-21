// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class MultiModalProcessor : IDisposable
    {
        private IntPtr _processorHandle;
        private bool _disposed = false;

        public MultiModalProcessor(Model model)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateMultiModalProcessor(model.Handle, out _processorHandle));
        }

        internal IntPtr Handle { get { return _processorHandle; } }

        public NamedTensors ProcessImages(string prompt, Images images)
        {
            IntPtr imagesHandle = images == null ? IntPtr.Zero : images.Handle;
            Result.VerifySuccess(NativeMethods.OgaProcessorProcessImages(_processorHandle, StringUtils.ToUtf8(prompt),
                                                                         imagesHandle, out IntPtr namedTensorsHandle));
            return new NamedTensors(namedTensorsHandle);
        }

        public string Decode(ReadOnlySpan<int> sequence)
        {
            IntPtr outStr = IntPtr.Zero;
            unsafe
            {
                fixed (int* sequencePtr = sequence)
                {
                    Result.VerifySuccess(NativeMethods.OgaProcessorDecode(_processorHandle, sequencePtr, (UIntPtr)sequence.Length, out outStr));
                }
            }
            try
            {
                return StringUtils.FromUtf8(outStr);
            }
            finally
            {
                NativeMethods.OgaDestroyString(outStr);
            }
        }

        public TokenizerStream CreateStream()
        {
            IntPtr tokenizerStreamHandle = IntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaCreateTokenizerStreamFromProcessor(_processorHandle, out tokenizerStreamHandle));
            return new TokenizerStream(tokenizerStreamHandle);
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
            NativeMethods.OgaDestroyMultiModalProcessor(_processorHandle);
            _processorHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
