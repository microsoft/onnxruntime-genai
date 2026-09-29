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
#if __ANDROID__
            // define the library name required for android
            internal const string DllName = "libonnxruntime-genai.so";
#elif __IOS__
            // define the library name required for iOS
            internal const string DllName = "__Internal";
#else
            internal const string DllName = "onnxruntime-genai";
#endif
        }

        // The returned pointer is owned by the OgaResult object and will be freed when the OgaResult
        // object is destroyed. It is expected that the caller will destroy the OgaResult object
        // when it no longer needs the result. If the error message is needed after the OgaResult
        // object is destroyed, it should be copied to a new buffer.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* const char* */ OgaResultGetError(IntPtr /* const OgaResult* */ result);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult */ OgaSetLogBool(byte[] /* const char* */ name,
                                                                  [MarshalAs(UnmanagedType.I1)] bool value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult */ OgaSetLogString(byte[] /* const char* */ name, byte[] /* const char* */ value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyResult(IntPtr /* OgaResult* */ result);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateConfig(byte[] /* const char* */ configPath,
                                                                     out IntPtr /* OgaConfig** */ config);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyConfig(IntPtr /* OgaConfig* */ config);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaConfigClearProviders(IntPtr /* OgaConfig* */ config);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaConfigAppendProvider(IntPtr /* OgaConfig* */ config, byte[] /* const char* */ provider_name);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaConfigSetProviderOption(IntPtr /* OgaConfig* */ config, byte[] /* const char* */ provider_name,
                                                                                byte[] /* const char* */ option_name, byte[] /* const char* */ option_value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern unsafe IntPtr /* OgaResult* */ OgaConfigAddModelData(IntPtr /* OgaConfig* */ config, byte[] /* const char* */ model_filename,
                                                                                  byte* /* const void* */ model_data, UIntPtr /* size_t */ model_data_length);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaConfigRemoveModelData(IntPtr /* OgaConfig* */ config, byte[] /* const char* */ model_filename);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaConfigOverlay(IntPtr /* OgaConfig* */ config, byte[] /* const char* */ json);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaConfigSetDecoderProviderOptionsHardwareDeviceType(IntPtr /* OgaConfig* */ config, byte[] /* const char* */ provider_name, byte[] /* const char* */ hardware_device_type);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaConfigSetDecoderProviderOptionsHardwareDeviceId(IntPtr /* OgaConfig* */ config, byte[] /* const char* */ provider_name, uint /* uint32_t  */ hardware_device_id);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaConfigSetDecoderProviderOptionsHardwareVendorId(IntPtr /* OgaConfig* */ config, byte[] /* const char* */ provider_name, uint /* uint32_t  */ hardware_vendor_id);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaConfigClearDecoderProviderOptionsHardwareDeviceType(IntPtr /* OgaConfig* */ config, byte[] /* const char* */ provider_name);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaConfigClearDecoderProviderOptionsHardwareDeviceId(IntPtr /* OgaConfig* */ config, byte[] /* const char* */ provider_name);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaConfigClearDecoderProviderOptionsHardwareVendorId(IntPtr /* OgaConfig* */ config, byte[] /* const char* */ provider_name);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateModel(byte[] /* const char* */ configPath,
                                                                    out IntPtr /* OgaModel** */ model);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateModelFromConfig(IntPtr /* const OgaConfig* */ config,
                                                                              out IntPtr /* OgaModel** */ model);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaModelGetType(IntPtr /* OgaModel* */ model,
                                                                     out IntPtr /* const char** */ type);

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
                                                                                     [MarshalAs(UnmanagedType.I1)] bool value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsSetGuidance(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                   byte[] /* const char* */ type,
                                                                                   byte[] /* const char* */ data,
                                                                                   [MarshalAs(UnmanagedType.I1)] bool /* boolean */ enable_ff_tokens);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsGetSearchNumber(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                       byte[] /* const char* */ searchOption,
                                                                                       out double /* const double* */ value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsGetSearchBool(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                     byte[] /* const char* */ searchOption,
                                                                                     [MarshalAs(UnmanagedType.I1)] out bool /* const bool* */ value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsSetSpeculativeNumber(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                           byte[] /* const char* */ name,
                                                                                           double value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsGetSpeculativeNumber(IntPtr /* const OgaGeneratorParams* */ generatorParams,
                                                                                           byte[] /* const char* */ name,
                                                                                           out double /* double* */ value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsSetSpeculativeBool(IntPtr /* OgaGeneratorParams* */ generatorParams,
                                                                                         byte[] /* const char* */ name,
                                                                                         [MarshalAs(UnmanagedType.I1)] bool value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGeneratorParamsGetSpeculativeBool(IntPtr /* const OgaGeneratorParams* */ generatorParams,
                                                                                         byte[] /* const char* */ name,
                                                                                         [MarshalAs(UnmanagedType.I1)] out bool /* bool* */ value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateGenerator(IntPtr /* const OgaModel* */ model,
                                                                        IntPtr /* const OgaGeneratorParams* */ generatorParams,
                                                                        out IntPtr /* OgaGenerator** */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyGenerator(IntPtr /* OgaGenerator* */ generator);

        // This function is used to check if the generator has finished generating all sequences.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern byte OgaGenerator_IsDone(IntPtr /* const OgaGenerator* */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_GetNextTokens(IntPtr /* const OgaGenerator* */ generator,
                                                                                out IntPtr /* const int32_t** */ outTokenIds,
                                                                                out UIntPtr /* size_t* */ outTokenCount);

        // This function is used to generate the next token in the sequence using the greedy search algorithm.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_GenerateNextToken(IntPtr /* OgaGenerator* */ generator);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_SetModelInput(IntPtr /* OgaGenerator* */ generator,
                                                                                byte[] /* const char* */ name,
                                                                                IntPtr /* const OgaTensor* */ tensor);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_SetInputs(IntPtr /* OgaGenerator* */ generator,
                                                                            IntPtr /* const OgaNamedTensors* */ namedTensors);

        // This function is used to append tokens to the sequence.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern unsafe IntPtr /* OgaResult* */ OgaGenerator_AppendTokens(IntPtr /* OgaGenerator* */ generator,
                                                                                      int* /* const int32_t* */ inputIDs,
                                                                                      UIntPtr /* size_t */ tokenCount);

        // This function is used to append a Sequences
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_AppendTokenSequences(IntPtr /* OgaGenerator* */ generator,
                                                                                       IntPtr /* const OgaSequences* */ sequences);
                                                                                       
        // This function is used to get the number of tokens in the generator.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern UIntPtr OgaGenerator_TokenCount(IntPtr /* const OgaGenerator* */ generator);


        // This function is used to rewind the generator to the given newLength.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_RewindTo(IntPtr /* OgaGenerator* */ generator,
                                                                            UIntPtr /* size_t */ newLength);

        // This function returns the length of the sequence at the given index.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern UIntPtr /* size_t */ OgaGenerator_GetSequenceCount(IntPtr /* const OgaGenerator* */ generator,
                                                                                UIntPtr /* size_t */ index);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_GetSpeculativeStats(IntPtr /* const OgaGenerator* */ generator,
                                                                                     out IntPtr /* OgaSpeculativeStats* */ stats);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroySpeculativeStats(IntPtr /* OgaSpeculativeStats* */ stats);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaSpeculativeStatsGetCount(IntPtr /* const OgaSpeculativeStats* */ stats,
                                                                                byte[] /* const char* */ name,
                                                                                out ulong value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaSpeculativeStatsGetNumber(IntPtr /* const OgaSpeculativeStats* */ stats,
                                                                                 byte[] /* const char* */ name,
                                                                                 out double value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaSpeculativeStatsGetBool(IntPtr /* const OgaSpeculativeStats* */ stats,
                                                                               byte[] /* const char* */ name,
                                                                               [MarshalAs(UnmanagedType.I1)] out bool value);

        // This function returns the sequence data at the given index. The returned pointer is owned by the
        // OgaGenerator object and will be freed when the OgaGenerator object is destroyed. It is expected
        // that the caller copies the data returned by this function after calling this function.
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* const in32_t* */ OgaGenerator_GetSequenceData(IntPtr /* const OgaGenerator* */ generator,
                                                                                     UIntPtr /* size_t */ index);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_GetInput(IntPtr /* const OgaGenerator* */ generator,
                                                                           byte[] /* const char* */ inputName,
                                                                           out IntPtr /* OgaTensor** */ tensor);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_GetOutput(IntPtr /* const OgaGenerator* */ generator,
                                                                            byte[] /* const char* */ outputName,
                                                                            out IntPtr /* OgaTensor** */ tensor);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaSetActiveAdapter(IntPtr /* OgaGenerator* */ generator,
                                                                         IntPtr /* OgaAdapters* */ adapters,
                                                                         byte[] /*const char**/ adapterName);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaGenerator_SetRuntimeOption(IntPtr /* OgaGenerator* */ generator,
                                                                                   byte[] /* const char* */ key,
                                                                                   byte[] /* const char* */ value);

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

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaAppendTokenToSequence(int token /* int32_t */,
                                                                              IntPtr /* const OgaSequences* */ sequences,
                                                                              UIntPtr /* size_t** */ sequenceIndex);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateTokenizer(IntPtr /* const OgaModel* */ model,
                                                                        out IntPtr /* OgaTokenizer** */ tokenizer);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateTokenizerFromConfig(IntPtr /* const OgaConfig* */ config,
                                                                                  out IntPtr /* OgaTokenizer** */ tokenizer);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateTokenizerFromPath(byte[] /* const char* */ configPath,
                                                                                out IntPtr /* OgaTokenizer** */ tokenizer);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyTokenizer(IntPtr /* OgaTokenizer* */ tokenizer);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaUpdateTokenizerOptions(
                             IntPtr /* const OgaTokenizer* */ tokenizer,
                             string[] /* const char*[] */ keys,
                             string[] /* const char*[] */ values,
                             UIntPtr /* size_t */ numOptions);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaTokenizerGetBosTokenId(IntPtr /* const OgaTokenizer* */ tokenizer,
                                                                               out int /* const int32_t* */ outBosTokenId);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaTokenizerGetEosTokenIds(IntPtr /* const OgaTokenizer* */ tokenizer,
                                                                                out IntPtr /* const int32_t** */ outEosTokenIds,
                                                                                out UIntPtr /* size_t* */ outTokenCount);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaTokenizerGetPadTokenId(IntPtr /* const OgaTokenizer* */ tokenizer,
                                                                               out int /* const int32_t* */ outPadTokenId);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaTokenizerGetBotTokenId(IntPtr /* const OgaTokenizer* */ tokenizer,
                                                                               out int /* const int32_t* */ outBotTokenId);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaTokenizerGetEotTokenId(IntPtr /* const OgaTokenizer* */ tokenizer,
                                                                               out int /* const int32_t* */ outEotTokenId);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaTokenizerGetBorTokenId(IntPtr /* const OgaTokenizer* */ tokenizer,
                                                                               out int /* const int32_t* */ outBorTokenId);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaTokenizerGetEorTokenId(IntPtr /* const OgaTokenizer* */ tokenizer,
                                                                               out int /* const int32_t* */ outEorTokenId);

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
        public static extern IntPtr /* OgaResult* */ OgaTokenizerApplyChatTemplate(IntPtr /* const OgaTokenizer* */ tokenizer,
                                                                                   byte[] /* const char* */ template_string,
                                                                                   byte[] /* const char* */ message,
                                                                                   byte[] /* const char* */ tool_calls,
                                                                                   [MarshalAs(UnmanagedType.I1)] bool /* bool */ add_gen_prompt,
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
        public static extern void OgaSetTelemetryEnabled([MarshalAs(UnmanagedType.I1)] bool enabled);

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
        public static extern IntPtr /* OgaResult* */ OgaProcessorProcessImagesAndPrompts(IntPtr /* const OgaMultiModalProcessor* */ processor,
                                                                                         IntPtr /* const OgaStringArray* */ prompts,
                                                                                         IntPtr /* const Images* */ images,
                                                                                         out IntPtr /* OgaNamedTensors** */ namedTensors);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaProcessorProcessAudios(IntPtr /* const OgaMultiModalProcessor* */ processor,
                                                                               byte[] /* const char* */ prompt,
                                                                               IntPtr /* const Audios* */ audios,
                                                                               out IntPtr /* OgaNamedTensors** */ namedTensors);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaProcessorProcessAudiosAndPrompts(IntPtr /* const OgaMultiModalProcessor* */ processor,
                                                                                         IntPtr /* const OgaStringArray* */ prompts,
                                                                                         IntPtr /* const Audios* */ audios,
                                                                                         out IntPtr /* OgaNamedTensors** */ namedTensors);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaProcessorProcessImagesAndAudios(IntPtr /* const OgaMultiModalProcessor* */ processor,
                                                                                        byte[] /* const char* */ prompt,
                                                                                        IntPtr /* const Images* */ images,
                                                                                        IntPtr /* const Audios* */ audios,
                                                                                        out IntPtr /* OgaNamedTensors** */ namedTensors);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaProcessorProcessImagesAndAudiosAndPrompts(IntPtr /* const OgaMultiModalProcessor* */ processor,
                                                                                                  IntPtr /* const OgaStringArray* */ prompts,
                                                                                                  IntPtr /* const Images* */ images,
                                                                                                  IntPtr /* const Audios* */ audios,
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
        public static extern unsafe IntPtr /* OgaResult* */ OgaLoadImagesFromBuffers(IntPtr[] /* const void** */ image_data,
                                                                              UIntPtr[] /* const size_t* */ image_data_sizes,
                                                                              UIntPtr /* size_t */ count,
                                                                              out IntPtr /* const OgaImages** */ images);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaLoadAudios(IntPtr /* const OgaStringArray* */ audioPaths,
                                                                   out IntPtr /* const OgaAudios** */ audios);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern unsafe IntPtr /* OgaResult* */ OgaLoadAudiosFromBuffers(IntPtr[] /* const void** */ audio_data,
                                                                              UIntPtr[] /* const size_t* */ audio_data_sizes,
                                                                              UIntPtr /* size_t */ count,
                                                                              out IntPtr /* const OgaAudios** */ audios);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyImages(IntPtr /* OgaImages* */ images);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyAudios(IntPtr /* OgaAudios* */ audios);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyNamedTensors(IntPtr /* OgaNamedTensors* */ namedTensors);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateStringArray(out IntPtr /* OgaStringArray** */ stringArray);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaStringArrayAddString(IntPtr /* OgaStringArray* */ stringArray,
                                                                             byte[] /* const char* */ str);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyStringArray(IntPtr /* OgaStringArray* */ stringArray);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateAdapters(IntPtr /* const OgaModel* */ model,
                                                                       out IntPtr /* OgaAdapters** */ adapters);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyAdapters(IntPtr /* OgaAdapters* */ adapters);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaLoadAdapter(IntPtr /* OgaAdapters* */ adapters,
                                                                    byte[] /* const char* */ adapterFilePath,
                                                                    byte[] /* const char* */ adapterName);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaUnloadAdapter(IntPtr /* OgaAdapters* */ adapters,
                                                                      byte[] /* const char* */ adapterName);

        // StreamingProcessor API
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaCreateStreamingProcessor(IntPtr /* const OgaModel* */ model,
                                                                              out IntPtr /* OgaStreamingProcessor** */ processor);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern void OgaDestroyStreamingProcessor(IntPtr /* OgaStreamingProcessor* */ processor);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern unsafe IntPtr /* OgaResult* */ OgaStreamingProcessorProcess(IntPtr /* OgaStreamingProcessor* */ processor,
                                                                                      float* /* const float* */ audioData,
                                                                                      UIntPtr /* size_t */ numSamples,
                                                                                      out IntPtr /* OgaNamedTensors** */ out_named_tensors);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaStreamingProcessorFlush(IntPtr /* OgaStreamingProcessor* */ processor,
                                                                             out IntPtr /* OgaNamedTensors** */ out_named_tensors);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaStreamingProcessorSetOption(IntPtr /* OgaStreamingProcessor* */ processor,
                                                                                  byte[] /* const char* */ key,
                                                                                  byte[] /* const char* */ value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        public static extern IntPtr /* OgaResult* */ OgaStreamingProcessorGetOption(IntPtr /* OgaStreamingProcessor* */ processor,
                                                                                  byte[] /* const char* */ key,
                                                                                  out IntPtr /* const char** */ value);

        // Stable non-generative C ABI.
        [StructLayout(LayoutKind.Sequential)]
        internal struct NonGenerativeCacheStats
        {
            internal ulong Hits, Misses, Evictions;
            internal UIntPtr Entries, Bytes, EntryCapacity, ByteCapacity;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct KevPrefixReuseStats
        {
            internal ulong PrefixRuns, BranchRuns, FallbackRuns;
        }

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateDirectoryTokenizer(byte[] path, out IntPtr tokenizer);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern void OgaDestroyDirectoryTokenizer(IntPtr tokenizer);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDirectoryTokenizerEncode(IntPtr tokenizer, byte[] text, out IntPtr ids);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDirectoryTokenizerGetPadTokenId(IntPtr tokenizer, out int id);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaTokenIdsGetData(IntPtr ids, out IntPtr data, out UIntPtr count);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern void OgaDestroyTokenIds(IntPtr ids);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateStructuredValueNull(out IntPtr value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateStructuredValueBool([MarshalAs(UnmanagedType.I1)] bool value, out IntPtr result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateStructuredValueInt64(long value, out IntPtr result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateStructuredValueDouble(double value, out IntPtr result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateStructuredValueString(byte[] value, out IntPtr result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateStructuredValueArray(out IntPtr value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateStructuredValueObject(out IntPtr value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaStructuredValueArrayAppend(IntPtr array, IntPtr value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaStructuredValueObjectAppend(IntPtr obj, byte[] key, IntPtr value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaStructuredValueGetType(IntPtr value, out int type);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaStructuredValueGetBool(IntPtr value, [MarshalAs(UnmanagedType.I1)] out bool result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaStructuredValueGetInt64(IntPtr value, out long result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaStructuredValueGetDouble(IntPtr value, out double result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaStructuredValueGetString(IntPtr value, out IntPtr result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaStructuredValueGetCount(IntPtr value, out UIntPtr result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaStructuredValueGetArrayItem(IntPtr value, UIntPtr index, out IntPtr result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaStructuredValueGetObjectItem(IntPtr value, UIntPtr index, out IntPtr key, out IntPtr result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern void OgaDestroyStructuredValue(IntPtr value);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateQuestion(byte[] type, IntPtr instructions, IntPtr criteria, out IntPtr question);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern void OgaDestroyQuestion(IntPtr question);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateStructuredRequest(out IntPtr request);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaStructuredRequestSetState(IntPtr request, IntPtr state);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaStructuredRequestAddQuestion(IntPtr request, byte[] id, IntPtr question);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaStructuredRequestSetTemperature(IntPtr request, float temperature);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern void OgaDestroyStructuredRequest(IntPtr request);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateFreeFormRankRequest(out IntPtr request);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaFreeFormRankRequestSetState(IntPtr request, IntPtr state);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaFreeFormRankRequestSetInstructions(IntPtr request, IntPtr instructions);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaFreeFormRankRequestAddCandidate(IntPtr request, byte[] key, IntPtr value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaFreeFormRankRequestSetTemperature(IntPtr request, float temperature);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern void OgaDestroyFreeFormRankRequest(IntPtr request);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateRankingSession(byte[] path, IntPtr providers, UIntPtr count, out IntPtr session);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern void OgaDestroyRankingSession(IntPtr session);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaRankingSessionRun(IntPtr session, IntPtr request, out IntPtr result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaRankingSessionRank(IntPtr session, IntPtr request, out IntPtr result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaRankingSessionSetCacheCapacity(IntPtr session, UIntPtr entries, UIntPtr bytes);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaRankingSessionGetCacheStats(IntPtr session, out NonGenerativeCacheStats stats);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaRankingSessionClearCache(IntPtr session);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaRankingSessionInvalidateCache(IntPtr session);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateDecisionSession(byte[] path, IntPtr providers, UIntPtr count, out IntPtr session);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern void OgaDestroyDecisionSession(IntPtr session);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDecisionSessionRun(IntPtr session, IntPtr request, out IntPtr result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDecisionSessionDecide(IntPtr session, IntPtr request, out IntPtr result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDecisionSessionSetCacheCapacity(IntPtr session, UIntPtr entries, UIntPtr bytes);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDecisionSessionGetCacheStats(IntPtr session, out NonGenerativeCacheStats stats);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDecisionSessionSetPrefixReuseEnabled(IntPtr session, [MarshalAs(UnmanagedType.I1)] bool enabled);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDecisionSessionGetPrefixReuseEnabled(IntPtr session, [MarshalAs(UnmanagedType.I1)] out bool enabled);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDecisionSessionCopyPrefixReuseStatus(IntPtr session, IntPtr buffer, UIntPtr bufferCapacity, out UIntPtr requiredSize);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDecisionSessionSetPrefixCacheCapacity(IntPtr session, UIntPtr entries, UIntPtr bytes);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDecisionSessionGetPrefixCacheStats(IntPtr session, out NonGenerativeCacheStats stats);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDecisionSessionGetPrefixReuseStats(IntPtr session, out KevPrefixReuseStats stats);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDecisionSessionClearCache(IntPtr session);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaDecisionSessionInvalidateCache(IntPtr session);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaModelResultGetModel(IntPtr result, out IntPtr model);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaModelResultGetAnswerCount(IntPtr result, out UIntPtr count);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaModelResultGetAnswerId(IntPtr result, UIntPtr answer, out IntPtr value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaModelResultGetAnswerType(IntPtr result, UIntPtr answer, out IntPtr value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaModelResultGetAnswerNoul(IntPtr result, UIntPtr answer, out double value, [MarshalAs(UnmanagedType.I1)] out bool present);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaModelResultGetAnswerChoice(IntPtr result, UIntPtr answer, out IntPtr value, [MarshalAs(UnmanagedType.I1)] out bool present);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaModelResultGetAnswerScore(IntPtr result, UIntPtr answer, out double value, [MarshalAs(UnmanagedType.I1)] out bool present);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaModelResultGetAnswerConfidence(IntPtr result, UIntPtr answer, out double value, [MarshalAs(UnmanagedType.I1)] out bool present);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaModelResultGetProbabilityCount(IntPtr result, UIntPtr answer, out UIntPtr count);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaModelResultGetProbability(IntPtr result, UIntPtr answer, UIntPtr index, out IntPtr key, out double value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaModelResultGetLegendCount(IntPtr result, UIntPtr answer, out UIntPtr count);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaModelResultGetLegend(IntPtr result, UIntPtr answer, UIntPtr index, out IntPtr key, out IntPtr value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern void OgaDestroyModelResult(IntPtr result);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaRankingResultGetModel(IntPtr result, out IntPtr model);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaRankingResultGetCount(IntPtr result, out UIntPtr count);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaRankingResultGetRank(IntPtr result, UIntPtr index, out UIntPtr rank);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaRankingResultGetKey(IntPtr result, UIntPtr index, out IntPtr key);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaRankingResultGetValue(IntPtr result, UIntPtr index, out IntPtr value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaRankingResultGetProbability(IntPtr result, UIntPtr index, out double probability);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern void OgaDestroyRankingResult(IntPtr result);

        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateComponentSession(byte[] path, byte[] component, IntPtr providers, UIntPtr count, out IntPtr session);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern void OgaDestroyComponentSession(IntPtr session);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentSessionGetInputCount(IntPtr session, out UIntPtr count);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentSessionGetOutputCount(IntPtr session, out UIntPtr count);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentSessionGetInputName(IntPtr session, UIntPtr index, out IntPtr name);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentSessionGetOutputName(IntPtr session, UIntPtr index, out IntPtr name);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentSessionGetInputType(IntPtr session, UIntPtr index, out ElementType type);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentSessionGetInputShapeRank(IntPtr session, UIntPtr index, out UIntPtr rank);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentSessionGetInputShapeDimension(IntPtr session, UIntPtr index, UIntPtr dimension, out long value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentSessionGetInputSymbolicDimension(IntPtr session, UIntPtr index, UIntPtr dimension, out IntPtr value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaCreateComponentInputs(out IntPtr inputs);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern unsafe IntPtr OgaComponentInputsAdd(IntPtr inputs, byte[] name, void* data, UIntPtr byteCount, long[] shape, UIntPtr rank, ElementType type);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern void OgaDestroyComponentInputs(IntPtr inputs);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentSessionRun(IntPtr session, IntPtr inputs, IntPtr outputNames, UIntPtr outputCount, out IntPtr tensors);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentTensorsGetCount(IntPtr tensors, out UIntPtr count);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentTensorsGetName(IntPtr tensors, UIntPtr index, out IntPtr name);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentTensorsGetType(IntPtr tensors, UIntPtr index, out ElementType type);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentTensorsGetShapeRank(IntPtr tensors, UIntPtr index, out UIntPtr rank);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentTensorsGetShapeDimension(IntPtr tensors, UIntPtr index, UIntPtr dimension, out long value);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern IntPtr OgaComponentTensorsGetData(IntPtr tensors, UIntPtr index, out IntPtr data, out UIntPtr bytes);
        [DllImport(NativeLib.DllName, CallingConvention = CallingConvention.Winapi)]
        internal static extern void OgaDestroyComponentTensors(IntPtr tensors);
    }
}
