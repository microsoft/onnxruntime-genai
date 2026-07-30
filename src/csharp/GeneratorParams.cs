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

        public void SetGuidance(string type, string data, bool enableFFTokens = false)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsSetGuidance(_generatorParamsHandle, StringUtils.ToUtf8(type), StringUtils.ToUtf8(data), enableFFTokens));
        }

        public double GetSearchNumber(string searchOption)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsGetSearchNumber(_generatorParamsHandle, StringUtils.ToUtf8(searchOption), out double value));
            return value;
        }

        public bool GetSearchBool(string searchOption)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsGetSearchBool(_generatorParamsHandle, StringUtils.ToUtf8(searchOption), out bool value));
            return value;
        }

        /// <summary>
        /// Sets a numerical speculative decoding option.
        /// </summary>
        /// <param name="name">The speculative option name, such as <c>max_draft_tokens</c>.</param>
        /// <param name="value">The value to set.</param>
        public void SetSpeculativeNumber(string name, double value)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsSetSpeculativeNumber(_generatorParamsHandle, StringUtils.ToUtf8(name), value));
        }

        /// <summary>
        /// Gets the current value of a numerical speculative decoding option.
        /// </summary>
        /// <param name="name">The speculative option name.</param>
        /// <returns>The current option value.</returns>
        public double GetSpeculativeNumber(string name)
        {
            Result.VerifySuccess(NativeMethods.OgaGeneratorParamsGetSpeculativeNumber(_generatorParamsHandle, StringUtils.ToUtf8(name), out double value));
            return value;
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
