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
        public static extern IntPtr /* OgaResult* */ OgaCreateModel(string /* const char* */ configPath,
                                                                    DeviceType deviceType,
                                                                    out IntPtr /* OgaModel** */ model);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyModel(IntPtr /* OgaModel* */ model);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateGeneratorParams(IntPtr /* OgaModel* */ model,
                                                                              out IntPtr /* OgaGeneratorParams** */ generatorParams);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyGeneratorParams(IntPtr /* OgaGeneratorParams* */ generatorParams);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsSetMaxLength(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                    int maxLength);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsSetInputIDs(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                   int[] /* int32_t* */ inputIDs,
                                                                                   UIntPtr inputIDsCount,
                                                                                   UIntPtr sequenceLength,
                                                                                   UIntPtr batchSize);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateGenerator(IntPtr /* OgaModel* */ model,
                                                                        IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                        out IntPtr /* OgaGenerator** */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyGenerator(IntPtr /* OgaGenerator* */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern bool OgaGenerator_IsDone(IntPtr /* OgaGenerator* */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_ComputeLogits(IntPtr /* OgaGenerator* */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_GenerateNextToken_Top(IntPtr /* OgaGenerator* */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_GetSequence(IntPtr /* OgaGenerator* */ generator,
                                                                              int index,
                                                                              IntPtr /* int32_t* */ tokens,
                                                                              out UIntPtr /* size_t* */ tokensCount);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroySequences(IntPtr /* OgaSequences* */ sequences);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern ulong OgaSequencesCount(IntPtr /* OgaSequences* */ sequences);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern ulong OgaSequencesGetSequenceCount(IntPtr /* OgaSequences* */ sequences,
                                                                ulong sequenceIndex);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* const int32_t* */ OgaSequencesGetSequenceData(IntPtr /* OgaSequences* */ sequences,
                                                                                     ulong sequenceIndex);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerate(IntPtr /* OgaModel* */ model,
                                                                 IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                 out IntPtr /* OgaSequences** */ sequences);
    }
}
