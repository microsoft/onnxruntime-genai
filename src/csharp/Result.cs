// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    class Result
    {
        private static string GetErrorMessage(IntPtr nativeResult)
        {

            return StringUtils.FromUtf8(NativeMethods.OgaResultGetError(nativeResult));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VerifySuccess(IntPtr nativeResult)
        {
            if (nativeResult != IntPtr.Zero)
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
