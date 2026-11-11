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
        /// <summary>
        /// Creates a managed owner for a native speculative statistics snapshot.
        /// </summary>
        /// <param name="nativeHandle">The native statistics handle.</param>
        internal SpeculativeStats(IntPtr nativeHandle) : base(IntPtr.Zero, true)
        {
            SetHandle(nativeHandle);
        }

        /// <summary>
        /// Gets whether the native statistics handle is invalid.
        /// </summary>
        public override bool IsInvalid => handle == IntPtr.Zero;

        /// <summary>
        /// Gets an unsigned integer statistic by name.
        /// </summary>
        /// <param name="name">The statistic name.</param>
        /// <returns>The statistic value.</returns>
        public ulong GetCount(string name)
        {
            Result.VerifySuccess(NativeMethods.OgaSpeculativeStatsGetCount(
                handle, StringUtils.ToUtf8(name), out ulong value));
            return value;
        }

        /// <summary>
        /// Gets a floating-point statistic by name.
        /// </summary>
        /// <param name="name">The statistic name.</param>
        /// <returns>The statistic value.</returns>
        public double GetNumber(string name)
        {
            Result.VerifySuccess(NativeMethods.OgaSpeculativeStatsGetNumber(
                handle, StringUtils.ToUtf8(name), out double value));
            return value;
        }

        /// <summary>
        /// Gets a Boolean statistic by name.
        /// </summary>
        /// <param name="name">The statistic name.</param>
        /// <returns>The statistic value.</returns>
        public bool GetBool(string name)
        {
            Result.VerifySuccess(NativeMethods.OgaSpeculativeStatsGetBool(
                handle, StringUtils.ToUtf8(name), out bool value));
            return value;
        }

        /// <summary>
        /// Releases the native statistics snapshot.
        /// </summary>
        /// <returns><see langword="true"/> when the handle has been released.</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OgaDestroySpeculativeStats(handle);
            handle = IntPtr.Zero;
            return true;
        }
    }
}
