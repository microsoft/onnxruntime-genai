// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Diagnostics;

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

        internal Tensor(IntPtr tensorHandle)
        {
            Debug.Assert(tensorHandle != IntPtr.Zero);
            _tensorHandle = tensorHandle;
            _disposed = false;
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

        /// <summary>
        /// Computes number of elements in the tensor
        /// given the shape
        /// </summary>
        /// <param name="shape">shape</param>
        /// <returns>product of dimensions</returns>
        public static Int64 ElementsFromShape(Int64[] shape)
        {
            Int64 size = 1;
            foreach (Int64 dim in shape)
            {
                size *= dim;
            }
            return size;
        }

        /// <summary>
        /// Computes and returns number of elements in the tensor
        /// </summary>
        /// <returns></returns>
        public Int64 NumElements()
        {
            return ElementsFromShape(Shape());
        }

        /// <summary>
        /// Return a ReadOnlySpan to tensor data
        /// no type checks are made
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns>read only span</returns>
        public ReadOnlySpan<T> GetData<T>()
        {
            var elements = NumElements();
            Result.VerifySuccess(NativeMethods.OgaTensorGetData(Handle, out IntPtr data));
            unsafe
            {
                return new ReadOnlySpan<T>(data.ToPointer(), (int)elements);
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
