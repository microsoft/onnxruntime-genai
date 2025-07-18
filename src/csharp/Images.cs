// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class Images : IDisposable
    {
        private IntPtr _imagesHandle;
        private bool _disposed = false;

        private Images(IntPtr imagesHandle)
        {
            _imagesHandle = imagesHandle;
        }

        internal IntPtr Handle { get { return _imagesHandle; } }

        public static Images Load(string[] imagePaths)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateStringArray(out IntPtr stringArray));
            foreach (string imagePath in imagePaths)
            {
                Result.VerifySuccess(NativeMethods.OgaStringArrayAddString(stringArray, StringUtils.ToUtf8(imagePath)));
            }
            Result.VerifySuccess(NativeMethods.OgaLoadImages(stringArray, out IntPtr imagesHandle));
            NativeMethods.OgaDestroyStringArray(stringArray);
            return new Images(imagesHandle);
        }

        public static Images OpenBytes(byte[] imageBytesDatas)
        {
            if (imageBytesDatas == null || imageBytesDatas.Length == 0)
            {
                throw new ArgumentException("Image byte data cannot be null or empty.");
            }
            // Define count variable, currently only supports one image file
            const int count = 1;
            IntPtr[] imageDatas = new IntPtr[count];
            UIntPtr[] imageDataSizes = new UIntPtr[count];
            imageDataSizes[0] = (UIntPtr)imageBytesDatas.Length;
            unsafe
            {
                fixed (byte* imageBytesPtr = imageBytesDatas)
                {
                    imageDatas[0] = (IntPtr)imageBytesPtr;
                    Result.VerifySuccess(NativeMethods.OgaLoadImagesFromBuffers(imageDatas, imageDataSizes, (UIntPtr)count, out IntPtr imagesHandle));
                    return new Images(imagesHandle);
                }

            }
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
            NativeMethods.OgaDestroyImages(_imagesHandle);
            _imagesHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
