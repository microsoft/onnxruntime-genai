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

        // W8: explicit-EP overload. `ep` selects the execution provider for v4 model
        // packages, bypassing GenAI's compatibility-intersection defaulting. Pass null
        // or empty to fall back to defaulting. In flat-directory mode a non-empty `ep`
        // raises an error.
        public Model(string modelPath, string ep)
        {
            byte[] epBytes = string.IsNullOrEmpty(ep) ? null : StringUtils.ToUtf8(ep);
            Result.VerifySuccess(NativeMethods.OgaCreateModelWithEp(StringUtils.ToUtf8(modelPath), epBytes, out _modelHandle));
        }

        public Model(Config config)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateModelFromConfig(config.Handle, out _modelHandle));
        }

        internal IntPtr Handle { get { return _modelHandle; } }

        public string GetModelType()
        {
            IntPtr outStr = IntPtr.Zero;
            try
            {
                Result.VerifySuccess(NativeMethods.OgaModelGetType(_modelHandle, out outStr));
                return StringUtils.FromUtf8(outStr);
            }
            finally
            {
                NativeMethods.OgaDestroyString(outStr);
            }
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
            if (_modelHandle != IntPtr.Zero)
            {
                NativeMethods.OgaDestroyModel(_modelHandle);
                _modelHandle = IntPtr.Zero;
            }
            _disposed = true;
        }
    }
}
