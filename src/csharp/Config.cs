// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>
    /// Use Config to set multiple ORT execution providers. The EP used will be chosen
    /// based on the insertion order.
    /// </summary>
    public class Config : IDisposable
    {
        private IntPtr _configHandle;
        private bool _disposed = false;

        /// <summary>
        /// Creates a Config
        /// Throws on error.
        /// </summary>
        /// <param name="modelPath">Path to a GenAI model</param>
        public Config(string modelPath)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateConfig(StringUtils.ToUtf8(modelPath), out _configHandle));
        }

        internal IntPtr Handle { get { return _configHandle; } }

        /// <summary>
        /// Clear all providers.
        /// </summary>
        public void ClearProviders()
        {
            Result.VerifySuccess(NativeMethods.OgaConfigClearProviders(_configHandle));
        }

        /// <summary>
        /// Append a provider with the given name
        /// </summary>
        /// <param name="provider">Name of the provider</param>
        public void AppendProvider(string provider)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigAppendProvider(_configHandle, StringUtils.ToUtf8(provider)));
        }

        /// <summary>
        /// Set options for a provider.
        /// </summary>
        /// <param name="provider">Name of the provider</param>
        /// <param name="option">Name of the option</param>
        /// <param name="value">Value of the option</param>
        public void SetProviderOption(string provider, string option, string value)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigSetProviderOption(_configHandle, StringUtils.ToUtf8(provider), StringUtils.ToUtf8(option), StringUtils.ToUtf8(value)));
        }

        ~Config()
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
            NativeMethods.OgaDestroyConfig(_configHandle);
            _configHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}
