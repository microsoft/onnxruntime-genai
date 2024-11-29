// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>
    /// An ORT GenAI model.
    /// </summary>
    public class Model : IDisposable
    {
        private IntPtr _modelHandle;
        private bool _disposed = false;

        /// <summary>
        /// Construct a Model from the given path.
        /// <param name="modelPath">The path of the model.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public Model(string modelPath)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateModel(StringUtils.ToUtf8(modelPath), out _modelHandle));
        }

        /// <summary>
        /// Construct a Model from Config.
        /// <param name="config">The config to use.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
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
