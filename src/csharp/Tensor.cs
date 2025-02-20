// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Diagnostics;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>
    /// Element types that correspond to OnnxRuntime supported element types.
    /// </summary>
    /// The values in this enum must match ONNX Runtime's type values
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

    ///<summary>
    /// Currently wraps an ORT Tensor.
    ///</summary>
    public class Tensor : IDisposable
    {
        private IntPtr _tensorHandle;
        private bool _disposed = false;

        /// <summary>
        /// Constructs a Tensor with the given data pointer, shape and element type.
        /// </summary>
        /// <param name="data">The data pointer.</param>
        /// <param name="shape">The shape of the Tensor.</param>
        /// <param name="type">The type of elements in the Tensor.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
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

        /// <summary>
        /// Get the element type.
        /// </summary>
        /// <returns>
        /// The element type.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public ElementType Type()
        {
            Result.VerifySuccess(NativeMethods.OgaTensorGetType(_tensorHandle, out ElementType element_type));
            return element_type;
        }

        /// <summary>
        /// Get the tensor shape.
        /// </summary>
        /// <returns>
        /// The shape.
        /// </returns>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
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
        /// <returns>The number of elements.</returns>
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
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
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

    /// <summary>
    /// This class is a list of tensors with names that match up with model input names.
    /// </summary>
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
