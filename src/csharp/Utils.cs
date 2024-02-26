// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    internal class Utils
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

        internal static void ToNativeBuffer(string str, IntPtr buffer, int bufferLength)
        {
            unsafe
            {
                byte* bufferBytes = (byte*)buffer.ToPointer();
                fixed (char* chars = str)
                {
                    UTF8Encoding.UTF8.GetBytes(chars, str.Length, bufferBytes, bufferLength);
                }
            }
        }
    }
}
