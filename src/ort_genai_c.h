// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#ifdef BUILDING_ORT_GENAI_C
#define OGA_EXPORT __declspec(dllexport)
#else
#define OGA_EXPORT __declspec(dllimport)
#endif
#define OGA_API_CALL _stdcall
#else
// To make symbols visible on macOS/iOS
#ifdef __APPLE__
#define OGA_EXPORT __attribute__((visibility("default")))
#else
#define OGA_EXPORT
#endif
#define OGA_API_CALL
#endif

// ONNX Runtime Generative AI C API
// This API is not thread safe.

typedef enum OgaElementType {
  OgaElementType_undefined,
  OgaElementType_float32,  // maps to c type float
  OgaElementType_uint8,    // maps to c type uint8_t
  OgaElementType_int8,     // maps to c type int8_t
  OgaElementType_uint16,   // maps to c type uint16_t
  OgaElementType_int16,    // maps to c type int16_t
  OgaElementType_int32,    // maps to c type int32_t
  OgaElementType_int64,    // maps to c type int64_t
  OgaElementType_string,   // string type (not currently supported by Oga)
  OgaElementType_bool,     // maps to c type bool
  OgaElementType_float16,  // IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
  OgaElementType_float64,  // maps to c type double
  OgaElementType_uint32,   // maps to c type uint32_t
  OgaElementType_uint64,   // maps to c type uint64_t
} OgaElementType;

typedef struct OgaResult OgaResult;
typedef struct OgaGeneratorParams OgaGeneratorParams;
typedef struct OgaGenerator OgaGenerator;
typedef struct OgaRuntimeSettings OgaRuntimeSettings;
typedef struct OgaModel OgaModel;
// OgaSequences is an array of token arrays where the number of token arrays can be obtained using
// OgaSequencesCount and the number of tokens in each token array can be obtained using OgaSequencesGetSequenceCount.
typedef struct OgaSequences OgaSequences;
typedef struct OgaTokenizer OgaTokenizer;
typedef struct OgaTokenizerStream OgaTokenizerStream;
typedef struct OgaTensor OgaTensor;
typedef struct OgaImages OgaImages;
typedef struct OgaNamedTensors OgaNamedTensors;
typedef struct OgaMultiModalProcessor OgaMultiModalProcessor;
typedef struct OgaAudios OgaAudios;
typedef struct OgaStringArray OgaStringArray;
typedef struct OgaAdapters OgaAdapters;

/* \brief Call this on process exit to cleanly shutdown the genai library & its onnxruntime usage
 */
OGA_EXPORT void OGA_API_CALL OgaShutdown();

/*
 * \param[in] result OgaResult that contains the error message.
 * \return Error message contained in the OgaResult. The const char* is owned by the OgaResult
 *         and can will be freed when the OgaResult is destroyed.
 */
OGA_EXPORT const char* OGA_API_CALL OgaResultGetError(const OgaResult* result);

/*
 * \param[in] Set logging options, see logging.h 'struct LogItems' for the list of available options
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaSetLogBool(const char* name, bool value);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSetLogString(const char* name, const char* value);

/*
 * \param[in] result OgaResult to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyResult(OgaResult*);
OGA_EXPORT void OGA_API_CALL OgaDestroyString(const char*);
OGA_EXPORT void OGA_API_CALL OgaDestroyNamedTensors(OgaNamedTensors*);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateSequences(OgaSequences** out);

/*
 * \param[in] sequences OgaSequences to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroySequences(OgaSequences* sequences);

/*
 * \brief Returns the number of sequences in the OgaSequences
 * \param[in] sequences
 * \return The number of sequences in the OgaSequences
 */
OGA_EXPORT size_t OGA_API_CALL OgaSequencesCount(const OgaSequences* sequences);

/*
 * \brief Appends token_cnt number of tokens from token_ptr to sequence
 * \param[in] token_ptr constant pointer to int32 tokens
 * \param[in] token_cnt number of tokens to read from token_ptr
 * \param[in] sequences OgaSequences object to append the tokens to
 * \return OgaResult containing the error message when tokens could not been added, else nullptr.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaAppendTokenSequence(const int32_t* token_ptr, size_t token_cnt, OgaSequences* sequence);

/*
 * \brief Appends the given token to the sequence at the given index.
          If the sequence at the given index does not exist, a new sequence is
          created at the given index if sequence_idx is equal to the current sequences count.
 * \param[in] token token to append to the sequence
 * \param[in] sequences OgaSequences object to append the token to
 * \param[in] sequence_index index of the sequence to append the token to
 * \return OgaResult containing the error message when tokens could not been added, else nullptr.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaAppendTokenToSequence(int32_t token, OgaSequences* sequence, size_t sequence_index);

/*
 * \brief Returns the number of tokens in the sequence at the given index
 * \param[in] sequences
 * \return The number of tokens in the sequence at the given index
 */
OGA_EXPORT size_t OGA_API_CALL OgaSequencesGetSequenceCount(const OgaSequences* sequences, size_t sequence_index);

/*
 * \brief Returns a pointer to the sequence data at the given index. The number of tokens in the sequence
 *        is given by OgaSequencesGetSequenceCount
 * \param[in] sequences
 * \return The pointer to the sequence data at the given index. The pointer is valid until the OgaSequences is destroyed.
 */
OGA_EXPORT const int32_t* OGA_API_CALL OgaSequencesGetSequenceData(const OgaSequences* sequences, size_t sequence_index);

OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadImage(const char* image_path, OgaImages** images);

OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadImages(const OgaStringArray* image_paths, OgaImages** images);

OGA_EXPORT void OGA_API_CALL OgaDestroyImages(OgaImages* images);

OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadAudio(const char* audio_path, OgaAudios** audios);

OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadAudios(const OgaStringArray* audio_paths, OgaAudios** audios);

OGA_EXPORT void OGA_API_CALL OgaDestroyAudios(OgaAudios* audios);

/*
 * \brief Creates a runtime settings instance to be used to create a model.
 * \param[out] out The created runtime settings.
 * \return OgaResult containing the error message if the creation of the runtime settings failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateRuntimeSettings(OgaRuntimeSettings** out);
/*
 * \brief Destroys the given runtime settings.
 * \param[in] settings The runtime settings to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyRuntimeSettings(OgaRuntimeSettings* settings);

/*
 * \brief Sets a specific runtime handle for the runtime settings.
 * \param[in] settings The runtime settings to set the device type.
 * \param[in] handle_name The name of the handle to set for the runtime settings.
 * \param[in] handle The value of handle to set for the runtime settings.
 * \return OgaResult containing the error message if the setting of the device type failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaRuntimeSettingsSetHandle(OgaRuntimeSettings* settings, const char* handle_name, void* handle);

/*
 * \brief Creates a model from the given configuration directory and device type.
 * \param[in] config_path The path to the model configuration directory. The path is expected to be encoded in UTF-8.
 * \param[in] device_type The device type to use for the model.
 * \param[out] out The created model.
 * \return OgaResult containing the error message if the model creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaModel** out);

/*
 * \brief Creates a model from the given configuration directory, runtime settings and device type.
 * \param[in] config_path The path to the model configuration directory. The path is expected to be encoded in UTF-8.
 * \param[in] settings The runtime settings to use for the model.
 * \param[in] device_type The device type to use for the model.
 * \param[out] out The created model.
 * \return OgaResult containing the error message if the model creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateModelWithRuntimeSettings(const char* config_path, const OgaRuntimeSettings* settings, OgaModel** out);

/*
 * \brief Destroys the given model.
 * \param[in] model The model to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyModel(OgaModel* model);

/*
 * \brief Generates an array of token arrays from the model execution based on the given generator params.
 * \param[in] model The model to use for generation.
 * \param[in] generator_params The parameters to use for generation.
 * \param[out] out The generated sequences of tokens. The caller is responsible for freeing the sequences using OgaDestroySequences
 *             after it is done using the sequences.
 * \return OgaResult containing the error message if the generation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerate(const OgaModel* model, const OgaGeneratorParams* generator_params, OgaSequences** out);

/*
 * \brief Creates a OgaGeneratorParams from the given model.
 * \param[in] model The model to use for generation.
 * \param[out] out The created generator params.
 * \return OgaResult containing the error message if the generator params creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** out);

/*
 * \brief Destroys the given generator params.
 * \param[in] generator_params The generator params to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyGeneratorParams(OgaGeneratorParams* generator_params);

OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetSearchNumber(OgaGeneratorParams* generator_params, const char* name, double value);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetSearchBool(OgaGeneratorParams* generator_params, const char* name, bool value);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize(OgaGeneratorParams* generator_params, int32_t max_batch_size);

/*
 * \brief Sets the input ids for the generator params. The input ids are used to seed the generation.
 * \param[in] generator_params The generator params to set the input ids on.
 * \param[in] input_ids The input ids array of size input_ids_count = batch_size * sequence_length.
 * \param[in] input_ids_count The total number of input ids.
 * \param[in] sequence_length The sequence length of the input ids.
 * \param[in] batch_size The batch size of the input ids.
 * \return OgaResult containing the error message if the setting of the input ids failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputIDs(OgaGeneratorParams* generator_params, const int32_t* input_ids,
                                                                 size_t input_ids_count, size_t sequence_length, size_t batch_size);

/*
 * \brief Sets the input id sequences for the generator params. The input id sequences are used to seed the generation.
 * \param[in] generator_params The generator params to set the input ids on.
 * \param[in] sequences The input id sequences.
 * \return OgaResult containing the error message if the setting of the input id sequences failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputSequences(OgaGeneratorParams* generator_params, const OgaSequences* sequences);

OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputs(OgaGeneratorParams* generator_params, const OgaNamedTensors* named_tensors);

/*
 * \brief For additional model inputs that genai does not handle, this lets the user set their values. For example LoRA models handle
 * fine tuning through model inputs. This lets the user supply the fine tuning inputs, while genai handles the standard inputs.
 * \param[in] generator_params The generator params to set the input on
 * \param[in] name Name of the model input (this must match the model's input name)
 * \param[in] tensor The OgaTensor of the input data
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetModelInput(OgaGeneratorParams* generator_params, const char* name, OgaTensor* tensor);

OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetWhisperInputFeatures(OgaGeneratorParams*, OgaTensor* tensor);

/*
 * \brief Creates a generator from the given model and generator params.
 * \param[in] model The model to use for generation.
 * \param[in] params The parameters to use for generation.
 * \param[out] out The created generator.
 * \return OgaResult containing the error message if the generator creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGenerator(const OgaModel* model, const OgaGeneratorParams* params, OgaGenerator** out);

/*
 * \brief Destroys the given generator.
 * \param[in] generator The generator to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyGenerator(OgaGenerator* generator);

/*
 * \brief Returns true if the generator has finished generating all the sequences.
 * \param[in] generator The generator to check if it is done with generating all sequences.
 * \return True if the generator has finished generating all the sequences, false otherwise.
 */
OGA_EXPORT bool OGA_API_CALL OgaGenerator_IsDone(const OgaGenerator* generator);
OGA_EXPORT bool OGA_API_CALL OgaGenerator_IsSessionTerminated(const OgaGenerator* generator);

/*
 * \brief Computes the logits from the model based on the input ids and the past state. The computed logits are stored in the generator.
 * \param[in] generator The generator to compute the logits for.
 * \return OgaResult containing the error message if the computation of the logits failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_ComputeLogits(OgaGenerator* generator);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken(OgaGenerator* generator);

OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_SetRuntimeOption(OgaGenerator* generator, const char* key, const char* value);

/*
 * \brief Returns a copy of the model output identified by the given name as an OgaTensor on CPU. The buffer is owned by returned OgaTensor
 *       and will be released when the OgaTensor is destroyed
 * \param[in] generator The generator to run the GetOutput on the name provided and the out pointer to store the output
 * \return OgaResult containing the error message if the computation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GetOutput(const OgaGenerator* oga_generator, const char* name, OgaTensor** out);

/*
 * \brief Returns the number of tokens in the sequence at the given index.
 * \param[in] generator The generator to get the count of the tokens for the sequence at the given index.
 * \return The number tokens in the sequence at the given index.
 */
OGA_EXPORT size_t OGA_API_CALL OgaGenerator_GetSequenceCount(const OgaGenerator* generator, size_t index);

/*
 * \brief Returns a pointer to the sequence data at the given index. The number of tokens in the sequence
 *        is given by OgaGenerator_GetSequenceCount
 * \param[in] generator The generator to get the sequence data for the sequence at the given index.
 * \return The pointer to the sequence data at the given index. The sequence data is owned by the OgaGenerator
 *         and will be freed when the OgaGenerator is destroyed. The caller must copy the data if it needs to
 *         be used after the OgaGenerator is destroyed.
 */
OGA_EXPORT const int32_t* OGA_API_CALL OgaGenerator_GetSequenceData(const OgaGenerator* generator, size_t index);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizer(const OgaModel* model, OgaTokenizer** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizer(OgaTokenizer*);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateMultiModalProcessor(const OgaModel* model, OgaMultiModalProcessor** out);

OGA_EXPORT void OGA_API_CALL OgaDestroyMultiModalProcessor(OgaMultiModalProcessor* processor);

/* Encodes a single string and adds the encoded sequence of tokens to the OgaSequences. The OgaSequences must be freed with OgaDestroySequences
   when it is no longer needed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerEncode(const OgaTokenizer*, const char* str, OgaSequences* sequences);

/*
 * \brief Converts the given string to a single token id.
 * \param[in] tokenizer The tokenizer to use to convert the string to a token id.
 * \param[in] str The string to convert to a token id.
 * \param[in] token_id The converted token id.
 * \return OgaResult containing the error message if the conversion of the string to a token id failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerToTokenId(const OgaTokenizer* tokenizer, const char* str, int32_t* token_id);

OGA_EXPORT OgaResult* OGA_API_CALL OgaProcessorProcessImages(const OgaMultiModalProcessor*, const char* prompt, const OgaImages* images, OgaNamedTensors** input_tensors);

OGA_EXPORT OgaResult* OGA_API_CALL OgaProcessorProcessAudios(const OgaMultiModalProcessor*, const OgaAudios* audios, OgaNamedTensors** input_tensors);

/* Decode a single token sequence and returns a null terminated utf8 string. out_string must be freed with OgaDestroyString
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerDecode(const OgaTokenizer*, const int32_t* tokens, size_t token_count, const char** out_string);
OGA_EXPORT OgaResult* OGA_API_CALL OgaProcessorDecode(const OgaMultiModalProcessor*, const int32_t* tokens, size_t token_count, const char** out_string);

/* OgaTokenizerStream is to decoded token strings incrementally, one token at a time.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizerStream(const OgaTokenizer*, OgaTokenizerStream** out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizerStreamFromProcessor(const OgaMultiModalProcessor*, OgaTokenizerStream** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizerStream(OgaTokenizerStream*);

/*
 * Decode a single token in the stream. If this results in a word being generated, it will be returned in 'out'.
 * The caller is responsible for concatenating each chunk together to generate the complete result.
 * 'out' is valid until the next call to OgaTokenizerStreamDecode or when the OgaTokenizerStream is destroyed
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerStreamDecode(OgaTokenizerStream*, int32_t token, const char** out);

/* Create an OgaTensor from a user owned buffer. The OgaTensor does not own the memory (as it has no way to free it) so
 * the 'data' parameter must be valid for the lifetime of the OgaTensor.
 *
 * \param[in] data User supplied memory pointer, must remain valid for lifetime of the OgaTensor
 * \param[in] shape_dims Pointer to array of int64_t values that define the tensor shape, example [1 20 30] would be equivalent to a C array of [1][20][30]
 * \param[in] shape_dims_count Count of elements in the shape_dims array
 * \param[in] element_type The data type that 'data' points to.
 * \param[out] out Writes the newly created OgaTensor into this, must be destroyed with OgaDestroyTensor
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTensorFromBuffer(void* data, const int64_t* shape_dims, size_t shape_dims_count, OgaElementType element_type, OgaTensor** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTensor(OgaTensor* tensor);

/* Get the OgaElementType of the data stored in the OgaTensor
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTensorGetType(OgaTensor*, OgaElementType* out);

/* Get the number of dimensions of the OgaTensor's shape, typically used to allocate a buffer of this size then calling OgaTensorGetShape with it
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTensorGetShapeRank(OgaTensor*, size_t* out);

/* Copies the shape dimensions into the shape_dims parameters. shape_dims_count must match the value returned by OgaTensorGetShapeRank
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTensorGetShape(OgaTensor*, int64_t* shape_dims, size_t shape_dims_count);

/* A pointer to the tensor data, it is typically cast into the actual data type of the tensor
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTensorGetData(OgaTensor*, void** out);

OGA_EXPORT OgaResult* OGA_API_CALL OgaSetCurrentGpuDeviceId(int device_id);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGetCurrentGpuDeviceId(int* device_id);

/*
 * \brief Creates an object of type OgaStringArray.
 * \return The result of the operation. If the operation is successful, a nullptr is returned.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateStringArray(OgaStringArray** out);

/*
 * \brief Creates an object of type OgaStringArray from the given strings.
 * \return The result of the operation. If the operation is successful, a nullptr is returned.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateStringArrayFromStrings(const char* const* strs, size_t count, OgaStringArray** out);

/*
 * \brief Destroys OgaStringArray.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyStringArray(OgaStringArray* string_array);

/*
 * \brief Adds the given string to the string_array.
 * \param[inout] string_array The string array to which the string is to be added
 * \param[in] str The string to be added to the string_array.
 * \return The result of the operation. If the operation is successful, a nullptr is returned.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaStringArrayAddString(OgaStringArray* string_array, const char* str);

/*
 * \brief Gets the number of strings in the string_array.
 * \param[in] string_array The OgaStringArray object to get the count of the strings.
 * \return The number of strings in the string_array.
 */
OGA_EXPORT size_t OGA_API_CALL OgaStringArrayGetCount(const OgaStringArray* string_array);

/*
 * \brief Creates the OgaAdapters object that manages the adapters.
          - The OgaAdapters object is used to load all the model adapters.
          - It is responsible for reference counting the loaded adapters.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateAdapters(const OgaModel* model, OgaAdapters** out);

/*
 * \brief Destroys the OgaAdapters object.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyAdapters(OgaAdapters* adapters);

/*
 * \brief Loads the model adapter from the given adapter file path and adapter name.
 * \param[in] adapters The OgaAdapters object to load the adapter.
 * \param[in] adapter_file_path The file path of the adapter to load.
 * \param[in] adapter_name A unique identifier for the adapter chosed by the function invoker.
 *                         This name is used for querying the adapter.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaLoadAdapter(OgaAdapters* adapters, const char* adapter_file_path,
                                                  const char* adapter_name);

/*
 * \brief Unloads the adapter with the given identifier from the previosly loaded adapters.
          If the adapter is not found, or if it cannot be unloaded (when it is in use), an error is returned.
 * \param[in] adapters The OgaAdapters object to unload the adapter.
 * \param[in] adapter_name The name of the adapter to unload.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaUnloadAdapter(OgaAdapters* adapters, const char* adapter_name);

/*
 * \brief Sets the adapter with the given adapter name as active for the given OgaGenerator object.
 * \param[in] generator The OgaGenerator object to set the active adapter.
 * \param[in] adapters The OgaAdapters object that manages the model adapters.
 * \param[in] adapter_name The name of the adapter to set as active.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaSetActiveAdapter(OgaGenerator* generator, OgaAdapters* adapters,
                                                       const char* adapter_name);

#ifdef __cplusplus
}
#endif
