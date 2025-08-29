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

        public void SetHardwareDeviceType(string hardware_device_type)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigSetHardwareDeviceType(_configHandle, StringUtils.ToUtf8(hardware_device_type)));
        }

        public void SetHardwareDeviceId(uint hardware_device_id)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigSetHardwareDeviceId(_configHandle, hardware_device_id));
        }

        public void SetHardwareVendorId(uint hardware_vendor_id)
        {
            Result.VerifySuccess(NativeMethods.OgaConfigSetHardwareVendorId(_configHandle, hardware_vendor_id));
        }

        public void ClearHardwareDeviceType()
        {
            Result.VerifySuccess(NativeMethods.OgaConfigClearHardwareDeviceType(_configHandle));
        }

        public void ClearHardwareDeviceId()
        {
            Result.VerifySuccess(NativeMethods.OgaConfigClearHardwareDeviceId(_configHandle));
        }

        public void ClearHardwareVendorId()
        {
            Result.VerifySuccess(NativeMethods.OgaConfigClearHardwareVendorId(_configHandle));
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
