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

        public void AddModelData(string modelFilename, byte[] modelData)
        {
            unsafe
            {
                fixed (byte* modelDataBytes = modelData)
                {
                    Result.VerifySuccess(NativeMethods.OgaConfigAddModelData(_configHandle, StringUtils.ToUtf8(modelFilename), modelDataBytes, (UIntPtr)modelData.Length));
                }
            }
        }

        public void RemoveModelData(string modelFilename)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigRemoveModelData(_configHandle, StringUtils.ToUtf8(modelFilename)));
        }

        public void SetDecoderProviderOptionsHardwareDeviceType(string provider, string hardware_device_type)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigSetDecoderProviderOptionsHardwareDeviceType(_configHandle, StringUtils.ToUtf8(provider), StringUtils.ToUtf8(hardware_device_type)));
        }

        public void SetDecoderProviderOptionsHardwareDeviceId(string provider, uint hardware_device_id)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigSetDecoderProviderOptionsHardwareDeviceId(_configHandle, StringUtils.ToUtf8(provider), hardware_device_id));
        }

        public void SetDecoderProviderOptionsHardwareVendorId(string provider, uint hardware_vendor_id)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigSetDecoderProviderOptionsHardwareVendorId(_configHandle, StringUtils.ToUtf8(provider), hardware_vendor_id));
        }

        public void ClearDecoderProviderOptionsHardwareDeviceType(string provider)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigClearDecoderProviderOptionsHardwareDeviceType(_configHandle, StringUtils.ToUtf8(provider)));
        }

        public void ClearDecoderProviderOptionsHardwareDeviceId(string provider)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigClearDecoderProviderOptionsHardwareDeviceId(_configHandle, StringUtils.ToUtf8(provider)));
        }

        public void ClearDecoderProviderOptionsHardwareVendorId(string provider)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigClearDecoderProviderOptionsHardwareVendorId(_configHandle, StringUtils.ToUtf8(provider)));
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
