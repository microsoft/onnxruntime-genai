// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    // The values in this enum must match ONNX Runtime's type values 
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

        public Tensor(IntPtr data, Int64[] shape, ElementType type)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateTensorFromBuffer(data, shape, (UIntPtr)shape.Length, type, out _tensorHandle));
        }
        internal IntPtr Handle { get { return _tensorHandle; } }

        ~Tensor()
        {
            Dispose(false);
        }
        public ElementType Type()
        {
            Result.VerifySuccess(NativeMethods.OgaTensorGetType(_tensorHandle, out ElementType element_type));
            return element_type;
        }

        public Int64[] Shape()
        {
            Result.VerifySuccess(NativeMethods.OgaTensorGetShapeRank(_tensorHandle, out UIntPtr size));
            Int64[] shape = new Int64[size.ToUInt64()];
            Result.VerifySuccess(NativeMethods.OgaTensorGetShape(_tensorHandle, shape, size));
            return shape;
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

    public class NamedTensors : IDisposable
    {
        private IntPtr _namedTensorsHandle;
        private bool _disposed = false;

        internal NamedTensors(IntPtr namedTensorsHandle)
        {
            _namedTensorsHandle = namedTensorsHandle;
        }

        internal IntPtr Handle { get { return _namedTensorsHandle; } }

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
            NativeMethods.OgaDestroyNamedTensors(_namedTensorsHandle);
            _namedTensorsHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
