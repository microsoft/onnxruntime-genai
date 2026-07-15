// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>
    /// An immutable snapshot of speculative decoding statistics.
    /// </summary>
    public sealed class SpeculativeStats : SafeHandle
    {
        internal SpeculativeStats(IntPtr nativeHandle) : base(IntPtr.Zero, true)
        {
            SetHandle(nativeHandle);
        }

        public override bool IsInvalid => handle == IntPtr.Zero;

        public ulong GetCount(string name)
        {
            Result.VerifySuccess(NativeMethods.OgaSpeculativeStatsGetCount(
                handle, StringUtils.ToUtf8(name), out ulong value));
            return value;
        }

        public double GetNumber(string name)
        {
            Result.VerifySuccess(NativeMethods.OgaSpeculativeStatsGetNumber(
                handle, StringUtils.ToUtf8(name), out double value));
            return value;
        }

        public bool GetBool(string name)
        {
            Result.VerifySuccess(NativeMethods.OgaSpeculativeStatsGetBool(
                handle, StringUtils.ToUtf8(name), out bool value));
            return value;
        }

        protected override bool ReleaseHandle()
        {
            NativeMethods.OgaDestroySpeculativeStats(handle);
            handle = IntPtr.Zero;
            return true;
        }
    }
}
