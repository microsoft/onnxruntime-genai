// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    internal static class Result
    {
        internal static string GetErrorMessage(IntPtr nativeResult)
        {
            return StringUtils.FromNullTerminatedUtf8(NativeMethods.OgaResultGetError(nativeResult));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VerifySuccess(IntPtr nativeResult)
        {
            if (nativeResult != IntPtr.Zero)
            {
                Throw(nativeResult);
            }

            static void Throw(IntPtr nativeResult)
            {
                try
                {
                    string errorMessage = GetErrorMessage(nativeResult);
                    throw new OnnxRuntimeGenAIException(errorMessage);
                }
                finally
                {
                    NativeMethods.OgaDestroyResult(nativeResult);
                }
            }
        }
    }
}
