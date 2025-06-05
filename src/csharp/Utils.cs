// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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
            Result.VerifySuccess(NativeMethods.OgaSetLogBool(StringUtils.ToNullTerminatedUtf8(name), value));
        }
        
        public static void SetLogString(string name, string value)
        {
            Result.VerifySuccess(NativeMethods.OgaSetLogString(StringUtils.ToNullTerminatedUtf8(name), StringUtils.ToNullTerminatedUtf8(value)));
        }
    }

    internal class StringUtils
    {
        internal static readonly byte[] EmptyByteArray = [0];

        internal static byte[] ToNullTerminatedUtf8(string str) => ToNullTerminatedUtf8(str.AsSpan());

        internal static unsafe byte[] ToNullTerminatedUtf8(ReadOnlySpan<char> str)
        {
            if (str.IsEmpty)
                return EmptyByteArray;

            fixed (char* pStr = str)
            {
                int byteCount = Encoding.UTF8.GetByteCount(pStr, str.Length);
                
                byte[] utf8Bytes = new byte[byteCount + 1];
                fixed (byte* pBytes = utf8Bytes)
                {
                    Encoding.UTF8.GetBytes(pStr, str.Length, pBytes, byteCount);
                    pBytes[byteCount] = 0;
                }
                
                return utf8Bytes;
            }
        }

        internal static unsafe string FromNullTerminatedUtf8(IntPtr nativeUtf8)
        {
            int len = GetNullTerminatedUtf8Length(nativeUtf8);
            return len > 0 ? Encoding.UTF8.GetString((byte*)nativeUtf8, len) : string.Empty;
        }

        internal static unsafe int GetNullTerminatedUtf8Length(IntPtr nativeUtf8)
        {
            // On .NET Core, we can use an optimized code path.
#if NETCOREAPP
            return MemoryMarshal.CreateReadOnlySpanFromNullTerminated((byte*)nativeUtf8).Length;
#else
            int len = 0;
            while (*(byte*)(nativeUtf8 + len) != 0) ++len;
            return len;
#endif
        }
    }
}
