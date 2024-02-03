// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <stdint.h>
#include <cstddef>

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

typedef struct OgaResult OgaResult;
typedef struct OgaGeneratorParams OgaGeneratorParams;
typedef struct OgaGenerator OgaGenerator;
typedef struct OgaModel OgaModel;

OGA_EXPORT const char* OGA_API_CALL OgaResultGetError(OgaResult*);
OGA_EXPORT void OGA_API_CALL OgaDestroyResult(OgaResult*);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaDeviceType device_type, OgaModel** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyModel(OgaModel*);
OGA_EXPORT OgaResult* OGA_API_CALL OgaModelGenerate(OgaGeneratorParams* generator_params);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGeneratorParams(OgaModel* model, OgaGeneratorParams** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyGeneratorParams(OgaGeneratorParams*);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetMaxLength(OgaGeneratorParams*, int max_length);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputIDs(OgaGeneratorParams*, const int32_t* input_ids, size_t input_ids_count, size_t sequence_length, size_t batch_size);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetWhisperInputFeatures(OgaGeneratorParams*, const int32_t* inputs, size_t count);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetWhisperDecoderInputIDs(OgaGeneratorParams*, const int32_t* input_ids, size_t input_ids_count);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateGenerator(OgaModel* model, const OgaGeneratorParams* params, OgaGenerator** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyGenerator(OgaGenerator*);
OGA_EXPORT bool OGA_API_CALL OgaGenerator_IsDone(const OgaGenerator*);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_ComputeLogits(OgaGenerator*);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_Top(OgaGenerator*);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_TopK(OgaGenerator*, int k, float t);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_TopP(OgaGenerator*, float p, float t);

/* Writes the sequence into the provided buffer 'tokens' and writes the count into 'count'. If 'tokens' is nullptr just writes the count
*/
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GetSequence(OgaGenerator*, int index, int32_t* tokens, size_t* count);

#ifdef __cplusplus
}
#endif
