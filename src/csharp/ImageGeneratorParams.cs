// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class ImageGeneratorParams : IDisposable
    {
        private IntPtr _imageGeneratorParamsHandle;
        private bool _disposed = false;

        public ImageGeneratorParams(Model model)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateImageGeneratorParams(model.Handle, out _imageGeneratorParamsHandle));
        }

        internal IntPtr Handle { get { return _imageGeneratorParamsHandle; } }

        /*
        public void SetPrompts(string prompt, string negativePrompt = null, int promptCount = 1)
        {
            // Create the byte arrays and hold references to them
            byte[] promptBytes = StringUtils.ToUtf8(prompt);
            byte[] negativePromptBytes = negativePrompt != null ? StringUtils.ToUtf8(negativePrompt) : null;

            // Pass the byte arrays to the native method
            Result.VerifySuccess(NativeMethods.OgaImageGeneratorParamsSetPrompts(
                _imageGeneratorParamsHandle,
                promptBytes,
                negativePromptBytes,
                promptCount));
        }*/

        /*
        public void SetPrompts(string[] prompts, string[] negativePrompts)
        {
            if (negativePrompts != null && prompts.Length != negativePrompts.Length)
            {
                throw new ArgumentException("Prompts and negative prompts arrays must have the same length");
            }

            // Implementation for multiple prompts would go here
            // This would require modifying the C API to accept arrays of strings
            throw new NotImplementedException("Multiple prompts are not yet supported");
        }*/
        /*

                public void SetPrompts(string prompt, string negativePrompt = null, int promptCount = 1)
                {
                    // Create the byte arrays
                    byte[] promptBytes = StringUtils.ToUtf8(prompt);
                    byte[] negativePromptBytes = negativePrompt != null ? StringUtils.ToUtf8(negativePrompt) : null;

                    // Pin memory so GC doesn't move it during the native call
                    GCHandle promptHandle = GCHandle.Alloc(promptBytes, GCHandleType.Pinned);
                    GCHandle? negativePromptHandle = null;

                    if (negativePromptBytes != null)
                    {
                        negativePromptHandle = GCHandle.Alloc(negativePromptBytes, GCHandleType.Pinned);
                    }

                    try
                    {
                        // Call the native method with pinned memory
                        Result.VerifySuccess(NativeMethods.OgaImageGeneratorParamsSetPrompts(
                            _imageGeneratorParamsHandle,
                            promptBytes,
                            negativePromptBytes,
                            promptCount));
                    }
                    finally
                    {
                        // Always unpin memory in finally block
                        promptHandle.Free();
                        if (negativePromptHandle.HasValue)
                        {
                            negativePromptHandle.Value.Free();
                        }
                    }
                }*/
        public void SetPrompts(string prompt)
        {
            Result.VerifySuccess(NativeMethods.OgaImageGeneratorParamsSetPrompts(
                    _imageGeneratorParamsHandle,
                    StringUtils.ToUtf8(prompt),
                    null,
                    1));
            
        }



        ~ImageGeneratorParams()
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
            NativeMethods.OgaDestroyImageGeneratorParams(_imageGeneratorParamsHandle);
            _imageGeneratorParamsHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}