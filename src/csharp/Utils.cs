// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Text;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class OgaHandle: IDisposable
    {
        private bool _disposed = false;

        public OgaHandle()
        {
        }

        ~OgaHandle()
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
            NativeMethods.OgaShutdown();
            _disposed = true;
        }
    }

    public class Utils
    {
        public static void SetCurrentGpuDeviceId(int device_id)
        {
            Result.VerifySuccess(NativeMethods.OgaSetCurrentGpuDeviceId(device_id));
        }

        public static int GetCurrentGpuDeviceId()
        {
            IntPtr device_id = IntPtr.Zero;
            Result.VerifySuccess(NativeMethods.OgaGetCurrentGpuDeviceId(out device_id));
            return (int)device_id.ToInt64();
        }

        public static void SetLogBool(string name, bool value)
        {
            Result.VerifySuccess(NativeMethods.OgaSetLogBool(StringUtils.ToUtf8(name), value));
        }
        
        public static void SetLogString(string name, string value)
        {
            Result.VerifySuccess(NativeMethods.OgaSetLogString(StringUtils.ToUtf8(name), StringUtils.ToUtf8(value)));
        }
    }

    internal class StringUtils
    {
        internal static byte[] EmptyByteArray = new byte[] { 0 };

        internal static byte[] ToUtf8(string str)
        {
            if (string.IsNullOrEmpty(str))
                return EmptyByteArray;

            int arraySize = UTF8Encoding.UTF8.GetByteCount(str);
            byte[] utf8Bytes = new byte[arraySize + 1];
            UTF8Encoding.UTF8.GetBytes(str, 0, str.Length, utf8Bytes, 0);
            utf8Bytes[utf8Bytes.Length - 1] = 0;
            return utf8Bytes;
        }

        internal static string FromUtf8(IntPtr nativeUtf8)
        {
            unsafe
            {
                int len = 0;
                while (*(byte*)(nativeUtf8 + len) != 0) ++len;

                if (len == 0)
                {
                    return string.Empty;
                }
                var nativeBytes = (byte*)nativeUtf8;
                return Encoding.UTF8.GetString(nativeBytes, len);
            }
        }
    }
}
