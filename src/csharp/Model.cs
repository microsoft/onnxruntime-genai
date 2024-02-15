// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public enum DeviceType : long
    {
        Auto = 0,
        CPU = 1,
        CUDA = 2
    }

    public class Model : IDisposable
    {
        private IntPtr _modelHandle;
        private bool _disposed = false;

        public Model(string modelPath, DeviceType deviceType)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateModel(Utils.ToUtf8(modelPath), deviceType, out _modelHandle));
        }

        internal IntPtr Handle { get { return _modelHandle; } }

        public Sequences Generate(GeneratorParams generatorParams)
        {
            IntPtr nativeSequences = IntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaGenerate(_modelHandle, generatorParams.Handle, out nativeSequences));
            return new Sequences(nativeSequences);
        }

        ~Model()
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
            NativeMethods.OgaDestroyModel(_modelHandle);
            _modelHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
