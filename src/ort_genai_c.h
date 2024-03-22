// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
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

typedef struct OgaResult OgaResult;
typedef struct OgaGeneratorParams OgaGeneratorParams;
typedef struct OgaGenerator OgaGenerator;
typedef struct OgaModel OgaModel;
// OgaSequences is an array of token arrays where the number of token arrays can be obtained using
// OgaSequencesCount and the number of tokens in each token array can be obtained using OgaSequencesGetSequenceCount.
typedef struct OgaSequences OgaSequences;
typedef struct OgaTokenizer OgaTokenizer;
typedef struct OgaTokenizerStream OgaTokenizerStream;

/*
 * \param[in] result OgaResult that contains the error message.
 * \return Error message contained in the OgaResult. The const char* is owned by the OgaResult
 *         and can will be freed when the OgaResult is destroyed.
 */
OGA_EXPORT const char* OGA_API_CALL OgaResultGetError(const OgaResult* result);

/*
 * \param[in] result OgaResult to be destroyed.
 */
OGA_EXPORT void OGA_API_CALL OgaDestroyResult(OgaResult*);
OGA_EXPORT void OGA_API_CALL OgaDestroyString(const char*);

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

/*
 * \brief Creates a model from the given configuration directory and device type.
 * \param[in] config_path The path to the model configuration directory. The path is expected to be encoded in UTF-8.
 * \param[in] device_type The device type to use for the model.
 * \param[out] out The created model.
 * \return OgaResult containing the error message if the model creation failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaModel** out);

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

OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetWhisperInputFeatures(OgaGeneratorParams*, const int32_t* inputs, size_t count);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGeneratorParamsSetWhisperDecoderInputIDs(OgaGeneratorParams*, const int32_t* input_ids, size_t input_ids_count);

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

/*
 * \brief Computes the logits from the model based on the input ids and the past state. The computed logits are stored in the generator.
 * \param[in] generator The generator to compute the logits for.
 * \return OgaResult containing the error message if the computation of the logits failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_ComputeLogits(OgaGenerator* generator);

/*
 * \brief Generates the next token based on the computed logits using the greedy search.
 * \param[in] generator The generator to generate the next token for.
 * \return OgaResult containing the error message if the generation of the next token failed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_Top(OgaGenerator* generator);

/* Top-K sampling: most probable words from the model's output probability distribution for the next word
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_TopK(OgaGenerator* generator, int k, float t);

/*Top-P sampling selects words from the smallest set of words whose cumulative probability exceeds a predefined threshold (p)
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_TopP(OgaGenerator* generator, float p, float t);

OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_TopK_TopP(OgaGenerator* generator, int k, float p, float t);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken(OgaGenerator* generator);

/*
 * \brief Returns the number of tokens in the sequence at the given index.
 * \param[in] generator The generator to get the count of the tokens for the sequence at the given index.
 * \return The number tokens in the sequence at the given index.
 */
OGA_EXPORT size_t OGA_API_CALL OgaGenerator_GetSequenceLength(const OgaGenerator* generator, size_t index);

/*
 * \brief Returns a pointer to the sequence data at the given index. The number of tokens in the sequence
 *        is given by OgaGenerator_GetSequenceLength
 * \param[in] generator The generator to get the sequence data for the sequence at the given index.
 * \return The pointer to the sequence data at the given index. The sequence data is owned by the OgaGenerator
 *         and will be freed when the OgaGenerator is destroyed. The caller must copy the data if it needs to
 *         be used after the OgaGenerator is destroyed.
 */
OGA_EXPORT const int32_t* OGA_API_CALL OgaGenerator_GetSequence(const OgaGenerator* generator, size_t index);

OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizer(const OgaModel* model, OgaTokenizer** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizer(OgaTokenizer*);

/* Encodes a single string and adds the encoded sequence of tokens to the OgaSequences. The OgaSequences must be freed with OgaDestroySequences
   when it is no longer needed.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerEncode(const OgaTokenizer*, const char* str, OgaSequences* sequences);

/* Decode a single token sequence and returns a null terminated utf8 string. out_string must be freed with OgaDestroyString
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerDecode(const OgaTokenizer*, const int32_t* tokens, size_t token_count, const char** out_string);

/* OgaTokenizerStream is to decoded token strings incrementally, one token at a time.
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateTokenizerStream(const OgaTokenizer*, OgaTokenizerStream** out);
OGA_EXPORT void OGA_API_CALL OgaDestroyTokenizerStream(OgaTokenizerStream*);

/*
 * Decode a single token in the stream. If this results in a word being generated, it will be returned in 'out'.
 * The caller is responsible for concatenating each chunk together to generate the complete result.
 * 'out' is valid until the next call to OgaTokenizerStreamDecode or when the OgaTokenizerStream is destroyed
 */
OGA_EXPORT OgaResult* OGA_API_CALL OgaTokenizerStreamDecode(OgaTokenizerStream*, int32_t token, const char** out);

OGA_EXPORT OgaResult* OGA_API_CALL OgaSetCurrentGpuDeviceId(int device_id);
OGA_EXPORT OgaResult* OGA_API_CALL OgaGetCurrentGpuDeviceId(int* device_id);

#ifdef __cplusplus
}

// This is added as a member variable in the wrapped types to prevent accidental construction/copying
// Since the wrapped types are never instantiated by value, this member doesn't really exist. The types are still opaque.
struct OgaAbstract {
  OgaAbstract() = delete;
  OgaAbstract(const OgaAbstract&) = delete;
  void operator=(const OgaAbstract&) = delete;
};

struct OgaResult : OgaAbstract {
  static void operator delete(void* p) { OgaDestroyResult(reinterpret_cast<OgaResult*>(p)); }
  const char* GetError() const { return OgaResultGetError(this); }
};

inline void OgaCheckResult(OgaResult* result) {
  if (result) {
    std::unique_ptr<OgaResult> p_result{result};  // Take ownership so it's destroyed properly
    throw std::runtime_error(p_result->GetError());
  }
}

struct OgaModel : OgaAbstract {
  static std::unique_ptr<OgaModel> Create(const char* config_path) {
    OgaModel* p;
    OgaCheckResult(OgaCreateModel(config_path, &p));
    return std::unique_ptr<OgaModel>(p);
  }

  std::unique_ptr<OgaSequences> Generate(const OgaGeneratorParams& params) {
    OgaSequences* p;
    OgaCheckResult(OgaGenerate(this, &params, &p));
    return std::unique_ptr<OgaSequences>(p);
  }

  static void operator delete(void* p) { OgaDestroyModel(reinterpret_cast<OgaModel*>(p)); }
};

struct OgaString {
  OgaString(const char* p) : p_{p} {}
  ~OgaString() { OgaDestroyString(p_); }

  operator const char*() const { return p_; }

  const char* p_;
};

struct OgaSequences : OgaAbstract {
  static std::unique_ptr<OgaSequences> Create() {
    OgaSequences* p;
    OgaCheckResult(OgaCreateSequences(&p));
    return std::unique_ptr<OgaSequences>(p);
  }

  size_t Count() const {
    return OgaSequencesCount(this);
  }

  std::span<const int32_t> Get(size_t index) const {
    return {OgaSequencesGetSequenceData(this, index), OgaSequencesGetSequenceCount(this, index)};
  }

  static void operator delete(void* p) { OgaDestroySequences(reinterpret_cast<OgaSequences*>(p)); }
};

struct OgaTokenizer : OgaAbstract {
  static std::unique_ptr<OgaTokenizer> Create(const OgaModel& model) {
    OgaTokenizer* p;
    OgaCheckResult(OgaCreateTokenizer(&model, &p));
    return std::unique_ptr<OgaTokenizer>(p);
  }

  void Encode(const char* str, OgaSequences& sequences) const {
    OgaCheckResult(OgaTokenizerEncode(this, str, &sequences));
  }

  OgaString Decode(std::span<const int32_t> tokens) const {
    const char* p;
    OgaCheckResult(OgaTokenizerDecode(this, tokens.data(), tokens.size(), &p));
    return p;
  }

  static void operator delete(void* p) { OgaDestroyTokenizer(reinterpret_cast<OgaTokenizer*>(p)); }
};

struct OgaTokenizerStream : OgaAbstract {
  static std::unique_ptr<OgaTokenizerStream> Create(const OgaTokenizer& tokenizer) {
    OgaTokenizerStream* p;
    OgaCheckResult(OgaCreateTokenizerStream(&tokenizer, &p));
    return std::unique_ptr<OgaTokenizerStream>(p);
  }

  /*
   * Decode a single token in the stream. If this results in a word being generated, it will be returned in 'out'.
   * The caller is responsible for concatenating each chunk together to generate the complete result.
   * 'out' is valid until the next call to OgaTokenizerStreamDecode or when the OgaTokenizerStream is destroyed
   */
  const char* Decode(int32_t token) {
    const char* out;
    OgaCheckResult(OgaTokenizerStreamDecode(this, token, &out));
    return out;
  }

  static void operator delete(void* p) { OgaDestroyTokenizerStream(reinterpret_cast<OgaTokenizerStream*>(p)); }
};

struct OgaGeneratorParams : OgaAbstract {
  static std::unique_ptr<OgaGeneratorParams> Create(const OgaModel& model) {
    OgaGeneratorParams* p;
    OgaCheckResult(OgaCreateGeneratorParams(&model, &p));
    return std::unique_ptr<OgaGeneratorParams>(p);
  }

  void SetSearchOption(const char* name, int value) {
    OgaCheckResult(OgaGeneratorParamsSetSearchNumber(this, name, value));
  }

  void SetSearchOption(const char* name, double value) {
    OgaCheckResult(OgaGeneratorParamsSetSearchNumber(this, name, value));
  }

  void SetSearchOption(const char* name, bool value) {
    OgaCheckResult(OgaGeneratorParamsSetSearchBool(this, name, value));
  }

  void SetInputIDs(const int32_t* input_ids, size_t input_ids_count, size_t sequence_length, size_t batch_size) {
    OgaCheckResult(OgaGeneratorParamsSetInputIDs(this, input_ids, input_ids_count, sequence_length, batch_size));
  }

  void SetInputSequences(const OgaSequences& sequences) {
    OgaCheckResult(OgaGeneratorParamsSetInputSequences(this, &sequences));
  }

  static void operator delete(void* p) { OgaDestroyGeneratorParams(reinterpret_cast<OgaGeneratorParams*>(p)); }
};

struct OgaGenerator : OgaAbstract {
  static std::unique_ptr<OgaGenerator> Create(const OgaModel& model, const OgaGeneratorParams& params) {
    OgaGenerator* p;
    OgaCheckResult(OgaCreateGenerator(&model, &params, &p));
    return std::unique_ptr<OgaGenerator>(p);
  }

  bool IsDone() const {
    return OgaGenerator_IsDone(this);
  }

  void ComputeLogits() {
    OgaCheckResult(OgaGenerator_ComputeLogits(this));
  }

  void GenerateNextToken() {
    OgaCheckResult(OgaGenerator_GenerateNextToken(this));
  }

  std::span<const int32_t> GetSequence(size_t index) const {
    return {OgaGenerator_GetSequence(this, index), OgaGenerator_GetSequenceLength(this, index)};
  }

  static void operator delete(void* p) { OgaDestroyGenerator(reinterpret_cast<OgaGenerator*>(p)); }
};

#endif
