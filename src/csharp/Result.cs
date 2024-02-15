// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    class Result
    {
        private static string GetErrorMessage(IntPtr nativeResult)
        {

            return Utils.FromUtf8(NativeMethods.OgaResultGetError(nativeResult));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VerifySuccess(IntPtr nativeResult)
        {
            if (nativeResult != IntPtr.Zero)
            {
                try
                {
                    string errorMessage = GetErrorMessage(nativeResult);
                    throw new Exception(errorMessage);
                }
                finally
                {
                    NativeMethods.OgaDestroyResult(nativeResult);
                }
            }
        }
    }
}
