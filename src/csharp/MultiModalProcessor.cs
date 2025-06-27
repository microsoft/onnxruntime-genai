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

        /// <summary>
        /// Processes a string and images into a NamedTensor.
        /// </summary>
        /// <param name="prompt">The text to encode as token ids.</param>
        /// <param name="images">The image inputs.</param>
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
        /// Processes a batch of strings and images into a NamedTensor.
        /// </summary>
        /// <param name="prompts">The texts to encode as token ids.</param>
        /// <param name="images">The image inputs.</param>
        /// <returns>
        /// The NamedTensors object.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public NamedTensors ProcessImages(string[] prompts, Images images)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateStringArray(out IntPtr stringArray));
            foreach (string prompt in prompts)
            {
                Result.VerifySuccess(NativeMethods.OgaStringArrayAddString(stringArray, StringUtils.ToUtf8(prompt)));
            }
            IntPtr imagesHandle = images == null ? IntPtr.Zero : images.Handle;
            Result.VerifySuccess(NativeMethods.OgaProcessorProcessImagesAndPrompts(_processorHandle, stringArray,
                                                                                   imagesHandle, out IntPtr namedTensorsHandle));
            NativeMethods.OgaDestroyStringArray(stringArray);
            return new NamedTensors(namedTensorsHandle);
        }

        /// <summary>
        /// Processes a string and audios into a NamedTensor.
        /// </summary>
        /// <param name="prompt">The text to encode as token ids.</param>
        /// <param name="audios">The audio inputs.</param>
        /// <returns>
        /// The NamedTensors object.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public NamedTensors ProcessAudios(string prompt, Audios audios)
        {
            IntPtr audiosHandle = audios == null ? IntPtr.Zero : audios.Handle;
            Result.VerifySuccess(NativeMethods.OgaProcessorProcessAudios(_processorHandle, StringUtils.ToUtf8(prompt),
                                                                         audiosHandle, out IntPtr namedTensorsHandle));
            return new NamedTensors(namedTensorsHandle);
        }

        /// <summary>
        /// Processes a batch of strings and audios into a NamedTensor.
        /// </summary>
        /// <param name="prompts">The texts to encode as token ids.</param>
        /// <param name="audios">The audios inputs.</param>
        /// <returns>
        /// The NamedTensors object.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public NamedTensors ProcessAudios(string[] prompts, Audios audios)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateStringArray(out IntPtr stringArray));
            foreach (string prompt in prompts)
            {
                Result.VerifySuccess(NativeMethods.OgaStringArrayAddString(stringArray, StringUtils.ToUtf8(prompt)));
            }
            IntPtr audiosHandle = audios == null ? IntPtr.Zero : audios.Handle;
            Result.VerifySuccess(NativeMethods.OgaProcessorProcessAudiosAndPrompts(_processorHandle, stringArray,
                                                                                   audiosHandle, out IntPtr namedTensorsHandle));
            NativeMethods.OgaDestroyStringArray(stringArray);
            return new NamedTensors(namedTensorsHandle);
        }

        /// <summary>
        /// Processes a string, image and audio into a NamedTensor.
        /// </summary>
        /// <param name="prompt">The text to encode as token ids.</param>
        /// <param name="images">The image inputs.</param>
        /// <param name="audios">The audio inputs.</param>
        /// <returns>
        /// The NamedTensors object.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public NamedTensors ProcessImagesAndAudios(string prompt, Images images, Audios audios)
        {
            IntPtr imagesHandle = images == null ? IntPtr.Zero : images.Handle;
            IntPtr audiosHandle = audios == null ? IntPtr.Zero : audios.Handle;
            Result.VerifySuccess(NativeMethods.OgaProcessorProcessImagesAndAudios(_processorHandle, StringUtils.ToUtf8(prompt),
                                                                                  imagesHandle, audiosHandle, out IntPtr namedTensorsHandle));
            return new NamedTensors(namedTensorsHandle);
        }

        /// <summary>
        /// Processes a batch of strings, images, and audios into a NamedTensor.
        /// </summary>
        /// <param name="prompts">The texts to encode as token ids.</param>
        /// <param name="images">The image inputs.</param>
        /// <param name="audios">The audio inputs.</param>
        /// <returns>
        /// The NamedTensors object.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public NamedTensors ProcessImagesAndAudios(string[] prompts, Images images, Audios audios)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateStringArray(out IntPtr stringArray));
            foreach (string prompt in prompts)
            {
                Result.VerifySuccess(NativeMethods.OgaStringArrayAddString(stringArray, StringUtils.ToUtf8(prompt)));
            }
            IntPtr imagesHandle = images == null ? IntPtr.Zero : images.Handle;
            IntPtr audiosHandle = audios == null ? IntPtr.Zero : audios.Handle;
            Result.VerifySuccess(NativeMethods.OgaProcessorProcessImagesAndAudiosAndPrompts(_processorHandle, stringArray,
                                                                                            imagesHandle, audiosHandle, out IntPtr namedTensorsHandle));
            NativeMethods.OgaDestroyStringArray(stringArray);
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
