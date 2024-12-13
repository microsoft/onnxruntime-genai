// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class Model : IDisposable
    {
        private IntPtr _modelHandle;
        private bool _disposed = false;

        public Model(string modelPath)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateModel(StringUtils.ToUtf8(modelPath), out _modelHandle));
        }

        public Model(Config config)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateModelFromConfig(config.Handle, out _modelHandle));
        }

        internal IntPtr Handle { get { return _modelHandle; } }

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
            if (_modelHandle != IntPtr.Zero)
            {
                NativeMethods.OgaDestroyModel(_modelHandle);
                _modelHandle = IntPtr.Zero;
            }
            _disposed = true;
        }
    }
}
