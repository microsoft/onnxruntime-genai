// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>
    /// The MultiModalProcessor class is responsible for converting text/images into a NamedTensors list
    /// that can be fed into a Generator class instance.
    /// </summary>
    public class MultiModalProcessor : IDisposable
    {
        private IntPtr _processorHandle;
        private bool _disposed = false;

        /// <summary>
        /// Construct a MultiModalProcessor for a given model.
        /// </summary>
        /// <param name="model">The model to use.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public MultiModalProcessor(Model model)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateMultiModalProcessor(model.Handle, out _processorHandle));
        }

        internal IntPtr Handle { get { return _processorHandle; } }

        /// <summary>
        /// Processes a string and image into a NamedTensor.
        /// </summary>
        /// <param name="prompt">The text to encode as token ids.</param>
        /// <param name="images">The image input.</param>
        /// <returns>
        /// The NamedTensors object.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public NamedTensors ProcessImages(string prompt, Images images)
        {
            IntPtr imagesHandle = images == null ? IntPtr.Zero : images.Handle;
            Result.VerifySuccess(NativeMethods.OgaProcessorProcessImages(_processorHandle, StringUtils.ToUtf8(prompt),
                                                                         imagesHandle, out IntPtr namedTensorsHandle));
            return new NamedTensors(namedTensorsHandle);
        }

        /// <summary>
        /// Decodes a sequence of token ids into text.
        /// </summary>
        /// <param name="sequence">The token ids to decode to text.</param>
        /// <returns>
        /// The text representation of the sequence.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
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

        /// <summary>
        /// Creates a TokenizerStream object for streaming tokenization. This is used with Generator class
        /// to provide each token as it is generated.
        /// </summary>
        /// <returns>
        /// The new TokenizerStream instance.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
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
