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
        public static extern IntPtr /* const char* */ OgaResultGetError(IntPtr /* const OgaResult* */ result);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult */ OgaSetLogBool(byte[] /* const char* */ name, bool value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult */ OgaSetLogString(byte[] /* const char* */ name, byte[] /* const char* */ value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyResult(IntPtr /* OgaResult* */ result);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateModel(byte[] /* const char* */ configPath,
                                                                    out IntPtr /* OgaModel** */ model);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyModel(IntPtr /* OgaModel* */ model);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateGeneratorParams(IntPtr /* const OgaModel* */ model,
                                                                              out IntPtr /* OgaGeneratorParams** */ generatorParams);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyGeneratorParams(IntPtr /* OgaGeneratorParams* */ generatorParams);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsSetSearchNumber(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                       byte[] /* const char* */ searchOption,
                                                                                       double value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsSetSearchBool(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                     byte[] /* const char* */ searchOption,
                                                                                     bool value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                                       int /* int32_t */ maxBatchSize);

        // This function is used to set the input IDs for the generator.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern unsafe IntPtr /* OgaResult* */ OgaGeneratorParamsSetInputIDs(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                          int* /* const int32_t* */ inputIDs,
                                                                                          UIntPtr /* size_t */ inputIDsCount,
                                                                                          UIntPtr /* size_t */ sequenceLength,
                                                                                          UIntPtr /* size_t */ batchSize);


        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsSetInputSequences(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                         IntPtr /* const OgaSequences* */ sequences);


        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsSetModelInput(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                     byte[] /* const char* */ name,                                 
                                                                                     IntPtr /* const OgaTensor* */ tensor);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsSetInputs(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                 IntPtr /* const OgaNamedTensors* */ namedTensors);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateGenerator(IntPtr /* const OgaModel* */ model,
                                                                        IntPtr /* const OgaGeneratorParams* */ generatorParams,
                                                                        out IntPtr /* OgaGenerator** */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyGenerator(IntPtr /* OgaGenerator* */ generator);

        // This function is used to check if the generator has finished generating all sequences.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern bool OgaGenerator_IsDone(IntPtr /* const OgaGenerator* */ generator);

        // This function is used to compute the logits for the next token in the sequence.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_ComputeLogits(IntPtr /* OgaGenerator* */ generator);

        // This function is used to generate the next token in the sequence using the greedy search algorithm.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_GenerateNextToken(IntPtr /* OgaGenerator* */ generator);

        // This function returns the length of the sequence at the given index.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern UIntPtr /* size_t */ OgaGenerator_GetSequenceCount(IntPtr /* const OgaGenerator* */ generator,
                                                                                UIntPtr /* size_t */ index);

        // This function returns the sequence data at the given index. The returned pointer is owned by the
        // OgaGenerator object and will be freed when the OgaGenerator object is destroyed. It is expected
        // that the caller copies the data returned by this function after calling this function.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* const in32_t* */ OgaGenerator_GetSequenceData(IntPtr /* const OgaGenerator* */ generator,
                                                                                     UIntPtr /* size_t */ index);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateSequences(out IntPtr /* OgaSequences** */ sequences);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroySequences(IntPtr /* OgaSequences* */ sequences);

        // This function returns the number of sequences in the OgaSequences object.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern UIntPtr OgaSequencesCount(IntPtr /* const OgaSequences* */ sequences);

        // This function returns the number of tokens in the sequence at the given index of the OgaSequences object.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern UIntPtr OgaSequencesGetSequenceCount(IntPtr /* const OgaSequences* */ sequences,
                                                                  UIntPtr /* size_t */ sequenceIndex);

        // This function returns the sequence data at the given index of the OgaSequences object. The returned
        // pointer is owned by the OgaSequences object and will be freed when the OgaSequences object is destroyed.
        // It is expected that the caller copies the data returned by this function after calling this function.
        // The number of sequences in the OgaSequences object can be obtained using the OgaSequencesCount function.
        // The number of tokens in the sequence at the given index can be obtained using the OgaSequencesGetSequenceCount function.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* const int32_t* */ OgaSequencesGetSequenceData(IntPtr /* const OgaSequences* */ sequences,
                                                                                     UIntPtr /* size_t */ sequenceIndex);

        // This function is used to generate sequences for the given model using the given generator parameters.
        // The OgaSequences object is an array of sequences, where each sequence is an array of tokens.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerate(IntPtr /* const OgaModel* */ model,
                                                                 IntPtr /* const OgaGeneratorParams* */ generatorParams,
                                                                 out IntPtr /* OgaSequences** */ sequences);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateTokenizer(IntPtr /* const OgaModel* */ model,
                                                                        out IntPtr /* OgaTokenizer** */ tokenizer);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyTokenizer(IntPtr /* OgaTokenizer* */ tokenizer);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaTokenizerEncode(IntPtr /* const OgaTokenizer* */ tokenizer,
                                                                        byte[] /* const char* */ strings,
                                                                        IntPtr /* OgaSequences* */ sequences);


        // This function is used to decode the given token into a string. The caller is responsible for freeing the
        // returned string using the OgaDestroyString function when it is no longer needed.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern unsafe IntPtr /* OgaResult* */ OgaTokenizerDecode(IntPtr /* const OgaTokenizer* */ tokenizer,
                                                                               int* /* const int32_t* */ sequence,
                                                                               UIntPtr /* size_t */ sequenceLength,
                                                                               out IntPtr /* const char** */ outStr);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyString(IntPtr /* const char* */ str);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateTokenizerStream(IntPtr /* const OgaTokenizer* */ tokenizer,
                                                                              out IntPtr /* OgaTokenizerStream** */ tokenizerStream);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateTokenizerStreamFromProcessor(IntPtr /* const OgaMultiModalProcessor* */ processor,
                                                                                           out IntPtr /* OgaTokenizerStream** */ tokenizerStream);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyTokenizerStream(IntPtr /* OgaTokenizerStream* */ tokenizerStream);

        // This function is used to decode the given token into a string. The returned pointer is freed when the
        // OgaTokenizerStream object is destroyed.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaTokenizerStreamDecode(IntPtr /* const OgaTokenizerStream* */ tokenizerStream,
                                                                              int /* int32_t */ token,
                                                                              out IntPtr /* const char** */ outStr);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateTensorFromBuffer(IntPtr /* data* */ data,
                                                                               long[] shapeDims,
                                                                               UIntPtr shapeDimsCount,
                                                                               ElementType elementType,
                                                                               out IntPtr /* OgaTensor** */ tensor);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyTensor(IntPtr /* OgaTensor * */ tensor);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaTensorGetType(IntPtr /* OgaTensor * */ tensor, out ElementType elementType);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaTensorGetShapeRank(IntPtr /* OgaTensor * */ tensor, out UIntPtr rank);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaTensorGetShape(IntPtr /* OgaTensor * */ tensor, long[] shapeDims, UIntPtr /* size_t */ shapeDimsCount);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaTensorGetData(IntPtr /* OgaTensor * */ tensor, out IntPtr /* void* */ data);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaSetCurrentGpuDeviceId(int /* int32_t */ deviceId);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGetCurrentGpuDeviceId(out IntPtr /* int32_t */ deviceId);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaShutdown();

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateMultiModalProcessor(IntPtr /* const OgaModel* */ model,
                                                                                  out IntPtr /* OgaMultiModalProcessor** */ processor);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyMultiModalProcessor(IntPtr /* OgaMultiModalProcessor* */ processor);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaProcessorProcessImages(IntPtr /* const OgaMultiModalProcessor* */ processor,
                                                                               byte[] /* const char* */ prompt,
                                                                               IntPtr /* const Images* */ images,
                                                                               out IntPtr /* OgaNamedTensors** */ namedTensors);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern unsafe IntPtr /* OgaResult* */ OgaProcessorDecode(IntPtr /* const OgaMultiModalProcessor* */ processor,
                                                                               int* /* const int32_t* */ sequence,
                                                                               UIntPtr /* size_t */ sequenceLength,
                                                                               out IntPtr /* const char** */ outStr);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaLoadImages(IntPtr /* const OgaStringArray* */ imagePaths,
                                                                   out IntPtr /* const OgaImages** */ images);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyImages(IntPtr /* OgaImages* */ images);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyNamedTensors(IntPtr /* OgaNamedTensors* */ namedTensors);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateStringArray(out IntPtr /* OgaStringArray** */ stringArray);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaStringArrayAddString(IntPtr /* OgaStringArray* */ stringArray,
                                                                             byte[] /* const char* */ str);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyStringArray(IntPtr /* OgaStringArray* */ stringArray);
    }
}
