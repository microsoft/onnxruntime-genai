// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class GeneratorParams : IDisposable
    {
        private IntPtr _generatorParamsHandle;
        private bool _disposed = false;
        public GeneratorParams(Model model)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateGeneratorParams(model.Handle, out _generatorParamsHandle));
        }

        internal IntPtr Handle { get { return _generatorParamsHandle; } }

        public void SetSearchOption(string searchOption, double value)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsSetSearchNumber(_generatorParamsHandle, StringUtils.ToUtf8(searchOption), value));
        }

        public void SetSearchOption(string searchOption, bool value)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsSetSearchBool(_generatorParamsHandle, StringUtils.ToUtf8(searchOption), value));
        }

        public void TryGraphCaptureWithMaxBatchSize(int maxBatchSize)
        {
            Console.WriteLine("TryGraphCaptureWithMaxBatchSize is deprecated and will be removed in a future release.");
        }

        public void SetModelInput(string name, Tensor value)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsSetModelInput(_generatorParamsHandle, StringUtils.ToUtf8(name), value.Handle));
        }

        public void SetInputs(NamedTensors namedTensors)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsSetInputs(_generatorParamsHandle, namedTensors.Handle));
        }

        ~GeneratorParams()
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
            NativeMethods.OgaDestroyGeneratorParams(_generatorParamsHandle);
            _generatorParamsHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
