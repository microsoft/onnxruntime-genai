// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class Config : IDisposable
    {
        private IntPtr _configHandle;
        private bool _disposed = false;
        public Config(string modelPath)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateConfig(StringUtils.ToUtf8(modelPath), out _configHandle));
        }

        internal IntPtr Handle { get { return _configHandle; } }
        public void ClearProviders()
        {
            Result.VerifySuccess(NativeMethods.OgaConfigClearProviders(_configHandle));
        }

        public void AppendProvider(string provider)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigAppendProvider(_configHandle, StringUtils.ToUtf8(provider)));
        }

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
