// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>
    /// Use Config to set the ORT execution providers (EPs) and their options. The EPs are applied based on
    /// insertion order.
    /// </summary>
    public class Config : IDisposable
    {
        private IntPtr _configHandle;
        private bool _disposed = false;

        /// <summary>
        /// Creates a Config from the given configuration directory.
        /// </summary>
        /// <param name="modelPath">The path to the configuration directory.</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public Config(string modelPath)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateConfig(StringUtils.ToUtf8(modelPath), out _configHandle));
        }

        internal IntPtr Handle { get { return _configHandle; } }

        /// <summary>
        /// Clear the list of providers in the config.
        /// </summary>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void ClearProviders()
        {
            Result.VerifySuccess(NativeMethods.OgaConfigClearProviders(_configHandle));
        }

        /// <summary>
        /// Add the provider at the end of the list of providers in the given config if it doesn't already
        /// exist. If it already exists, does nothing.
        /// </summary>
        /// <param name="providerName">Name of the provider</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void AppendProvider(string providerName)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigAppendProvider(_configHandle, StringUtils.ToUtf8(providerName)));
        }

        /// <summary>
        /// Set a provider option.
        /// </summary>
        /// <param name="providerName">Name of the provider</param>
        /// <param name="optionKey">Name of the option</param>
        /// <param name="optionValue">Value of the option</param>
        /// <exception cref="OnnxRuntimeGenAIException">
        /// Thrown when the call to the GenAI native API fails.
        /// </exception>
        public void SetProviderOption(string providerName, string optionKey, string optionValue)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigSetProviderOption(_configHandle, StringUtils.ToUtf8(providerName), StringUtils.ToUtf8(optionKey), StringUtils.ToUtf8(optionValue)));
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
