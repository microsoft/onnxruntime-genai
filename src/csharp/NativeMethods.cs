// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    internal static class NativeMethods
    {
        internal class NativeLib
        {
            internal const string DllName = "onnxruntime-genai";
        }

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* const char* */ OgaResultGetError(IntPtr /* OgaResult* */ result);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyResult(IntPtr /* OgaResult* */ result);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr OgaCreateModel(string /* const char* */ configPath,
                                                   DeviceType deviceType,
                                                   out IntPtr /* OgaModel** */ model);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyModel(IntPtr /* OgaModel* */ model);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr OgaCreateGeneratorParams(IntPtr /* OgaModel* */ model,
                                                             out IntPtr /* OgaGeneratorParams** */ generatorParams);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyGeneratorParams(IntPtr /* OgaGeneratorParams* */ generatorParams);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr OgaGeneratorParamsSetMaxLength(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                   int maxLength);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr OgaGeneratorParamsSetInputIDs(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                  int[] /* int32_t* */ inputIDs,
                                                                  UIntPtr inputIDsCount,
                                                                  UIntPtr sequenceLength,
                                                                  UIntPtr batchSize);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr OgaCreateGenerator(IntPtr /* OgaModel* */ model,
                                                       IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                       out IntPtr /* OgaGenerator** */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyGenerator(IntPtr /* OgaGenerator* */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern bool OgaGenerator_IsDone(IntPtr /* OgaGenerator* */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr OgaGenerator_ComputeLogits(IntPtr /* OgaGenerator* */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr OgaGenerator_GenerateNextToken_Top(IntPtr /* OgaGenerator* */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr OgaGenerator_GetSequence(IntPtr /* OgaGenerator* */ generator,
                                                             int index,
                                                             IntPtr /* int32_t* */ tokens,
                                                             out UIntPtr /* size_t* */ tokensCount);
    }
}
