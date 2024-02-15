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

        // The returned pointer is owned by the OgaResult object and will be freed when the OgaResult
        // object is destroyed. It is expected that the caller will destroy the OgaResult object
        // when it no longer needs the result. If the error message is needed after the OgaResult
        // object is destroyed, it should be copied to a new buffer.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* const char* */ OgaResultGetError(IntPtr /* OgaResult* */ result);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyResult(IntPtr /* OgaResult* */ result);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateModel(byte[] /* const char* */ configPath,
                                                                    DeviceType /* OgaDeviceType */ deviceType,
                                                                    out IntPtr /* OgaModel** */ model);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyModel(IntPtr /* OgaModel* */ model);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateGeneratorParams(IntPtr /* OgaModel* */ model,
                                                                              out IntPtr /* OgaGeneratorParams** */ generatorParams);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyGeneratorParams(IntPtr /* OgaGeneratorParams* */ generatorParams);

        // This function is used to set the maximum length that the generated sequences can have.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsSetMaxLength(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                    IntPtr /* int */ maxLength);

        // This function is used to set the input IDs for the generator.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsSetInputIDs(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                   int[] /* const int32_t* */ inputIDs,
                                                                                   UIntPtr /* size_t */ inputIDsCount,
                                                                                   UIntPtr /* size_t */ sequenceLength,
                                                                                   UIntPtr /* size_t */ batchSize);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateGenerator(IntPtr /* OgaModel* */ model,
                                                                        IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                        out IntPtr /* OgaGenerator** */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyGenerator(IntPtr /* OgaGenerator* */ generator);

        // This function is used to check if the generator has finished generating all sequences.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern bool OgaGenerator_IsDone(IntPtr /* OgaGenerator* */ generator);

        // This function is used to compute the logits for the next token in the sequence.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_ComputeLogits(IntPtr /* OgaGenerator* */ generator);

        // This function is used to generate the next token in the sequence using the greedy search algorithm.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_GenerateNextToken_Top(IntPtr /* OgaGenerator* */ generator);

        // This function returns the length of the sequence at the given index.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern UIntPtr /* size_t */ OgaGenerator_GetSequenceLength(IntPtr /* OgaGenerator* */ generator,
                                                                                 IntPtr /* int */ index);

        // This function returns the sequence data at the given index. The returned pointer is owned by the
        // OgaGenerator object and will be freed when the OgaGenerator object is destroyed. It is expected
        // that the caller copies the data returned by this function after calling this function.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* const in32_t* */ OgaGenerator_GetSequence(IntPtr /* OgaGenerator* */ generator,
                                                                                 IntPtr /* int */ index);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroySequences(IntPtr /* OgaSequences* */ sequences);

        // This function returns the number of sequences in the OgaSequences object.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern UIntPtr OgaSequencesCount(IntPtr /* OgaSequences* */ sequences);

        // This function returns the number of tokens in the sequence at the given index of the OgaSequences object.
        // The OgaSequences object can be thought of as a 2D array of sequences, where the first dimension is the
        // number of sequences as per OgaSequencesCount and the second dimension is the sequence length
        // as per OgaSequencesGetSequenceCount.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern UIntPtr OgaSequencesGetSequenceCount(IntPtr /* OgaSequences* */ sequences,
                                                                  UIntPtr /* size_t */ sequenceIndex);

        // This function returns the sequence data at the given index of the OgaSequences object. The returned
        // pointer is owned by the OgaSequences object and will be freed when the OgaSequences object is destroyed.
        // It is expected that the caller copies the data returned by this function after calling this function.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* const int32_t* */ OgaSequencesGetSequenceData(IntPtr /* OgaSequences* */ sequences,
                                                                                     UIntPtr /* size_t */ sequenceIndex);

        // This function is used to generate sequences for the given model using the given generator parameters.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerate(IntPtr /* OgaModel* */ model,
                                                                 IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                 out IntPtr /* OgaSequences** */ sequences);
    }
}
