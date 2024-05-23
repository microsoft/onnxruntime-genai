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

        public static Images Load(string imagePath)
        {
            Result.VerifySuccess(NativeMethods.OgaLoadImage(StringUtils.ToUtf8(imagePath), out IntPtr imagesHandle));
            return new Images(imagesHandle);
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
