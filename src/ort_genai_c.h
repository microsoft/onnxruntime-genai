// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define OGA_EXPORT
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

typedef enum OgaDeviceType {
  OgaDeviceTypeAuto,
  OgaDeviceTypeCPU,
  OgaDeviceTypeCUDA,
} OgaDeviceType;

typedef enum OgaDataType {
  OgaDataTypeFloat32,
  OgaDataTypeInt64,
} OgaDataType;

typedef struct OgaResult OgaResult;
typedef struct OgaArray OgaArray;
typedef struct OgaSearchParams OgaSearchParams;
typedef struct OgaSearch OgaSearch;
typedef struct OgaModel OgaModel;
typedef struct OgaState OgaState;
typedef struct OgaRoamingArray OgaRoamingArray;

OGA_EXPORT void OGA_API_CALL OgaDestroyArray(OgaArray*);
OGA_EXPORT size_t OGA_API_CALL OgaArrayGetSize(OgaArray*);
OGA_EXPORT OgaDataType OGA_API_CALL OgaArrayGetType(OgaArray*);
OGA_EXPORT OgaDeviceType OGA_API_CALL OgaArrayGetNativeDeviceType(OgaArray*);
OGA_EXPORT void* OGA_API_CALL OgaArrayGetData(OgaArray*, OgaDeviceType*);

OGA_EXPORT const char* OGA_API_CALL OgaResultGetError(OgaResult*);
OGA_EXPORT void OGA_API_CALL OgaDestroyResult(OgaResult*);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaDeviceType device_type, OgaModel** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyModel(OgaModel*);
OGA_EXPORT OgaResult* OGA_API_CALL OgaModelGenerate(OgaSearchParams* search_params);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateSearchParams(OgaModel* model, OgaSearchParams** out);
OGA_EXPORT void OGA_API_CALL OgaDestroySearchParams(OgaSearchParams*);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchParamsCreateSearch(OgaSearchParams*, OgaSearch** out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchParamsSetInputIDs(OgaSearchParams*, int32_t* input_ids, size_t input_ids_count, int num_batches);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchParamsSetWhisperInputFeatures(OgaSearchParams*, int32_t* inputs, size_t count);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchParamsSetWhisperDecoderInputIDs(OgaSearchParams*, int32_t* input_ids, size_t input_ids_count);

OGA_EXPORT void OGA_API_CALL OgaDestroySearch(OgaSearch*);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchSetLogits(OgaSearch*, OgaArray*);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchGetSequenceLength(OgaSearch*, size_t* sequence_length);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchGetSequenceLengths(OgaSearch*, size_t* sequence_length);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchGetNextTokens(OgaSearch*, OgaArray**);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchGetNextIndices(OgaSearch*, OgaArray**);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchIsDone(OgaSearch*, bool* out);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchSelectTop(OgaSearch*);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchSampleTopK(OgaSearch*, int k, float t);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchSampleTopP(OgaSearch*, float p, float t);
OGA_EXPORT OgaResult* OGA_API_CALL OgaSearchGetSequence(OgaSearch*, int index, OgaArray**);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateState(OgaModel* model, int32_t* sequence_lengths, size_t sequence_lengths_count, const OgaSearchParams* search_params, OgaState** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyState(OgaState*);
OGA_EXPORT OgaResult* OGA_API_CALL OgaStateRun(int current_length, int32_t* next_tokens, size_t next_tokens_count, int32_t* next_indices, size_t next_indices_count, float* logits, float** logits_count);

#ifdef __cplusplus
}
#endif
