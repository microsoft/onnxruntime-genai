// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public enum ElementType : long
    {
        undefined,
        float32,
        uint8,
        int8,
        uint16,
        int16,
        int32,
        int64,
        string_t,
        bool_t,
        float16,
        float64,
        uint32,
        uint64,
    }

    public class Tensor : IDisposable
    {
        private IntPtr _tensorHandle;
        private bool _disposed = false;

        public Tensor()
        {
            Result.VerifySuccess(NativeMethods.OgaCreateTensorFromBuffer(0, 0, 0, ElementType.float32, out _tensorHandle));
        }
        ~Tensor()
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
            NativeMethods.OgaDestroyTensor(_tensorHandle);
            _tensorHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
