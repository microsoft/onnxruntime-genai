// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#if __cplusplus >= 202002L
#include <span>
#endif

#include "ort_genai_c.h"

// GenAI C++ API
//
// This is a zero cost wrapper around the C API, and provides for a set of C++ classes with automatic resource management

/* A simple end to end example of how to generate an answer from a prompt:
 *
 * auto model = OgaModel::Create("phi-2");
 * auto tokenizer = OgaTokenizer::Create(*model);
 *
 * auto sequences = OgaSequences::Create();
 * tokenizer->Encode("A great recipe for Kung Pao chicken is ", *sequences);
 *
 * auto params = OgaGeneratorParams::Create(*model);
 * params->SetSearchOption("max_length", 200);
 * params->SetSearchOption("batch_size", 1);
 *
 * auto generator = OgaGenerator::Create(*model, *params);
 * generator->AppendTokenSequences(*sequences);
 * while (!generator->IsDone()) {
 *  generator->GenerateNextToken();
 * }
 * auto output_sequence = generator->GetSequenceData(0);
 * auto output_string = tokenizer->Decode(output_sequence, generator->GetSequenceCount(0));
 *
 * std::cout << "Output: " << std::endl << output_string << std::endl;
 */

// The types defined in this file are to give us zero overhead C++ style interfaces around an opaque C pointer.
// For example, there is no actual 'OgaModel' type defined anywhere, so we create a fake definition here
// that lets users have a C++ style OgaModel type that can be held in a std::unique_ptr.
//
// This OgaAbstract struct is to prevent accidentally trying to use them by value.
struct OgaAbstract {
  OgaAbstract() = delete;
  OgaAbstract(const OgaAbstract&) = delete;
  void operator=(const OgaAbstract&) = delete;
};

struct OgaResult : OgaAbstract {
  const char* GetError() const { return OgaResultGetError(this); }
  static void operator delete(void* p) { OgaDestroyResult(reinterpret_cast<OgaResult*>(p)); }
};

// This is used to turn OgaResult return values from the C API into std::runtime_error exceptions
inline void OgaCheckResult(OgaResult* result) {
  if (result) {
    std::unique_ptr<OgaResult> p_result{result};  // Take ownership so it's destroyed properly
    throw std::runtime_error(p_result->GetError());
  }
}

struct OgaRuntimeSettings : OgaAbstract {
  static std::unique_ptr<OgaRuntimeSettings> Create() {
    OgaRuntimeSettings* p;
    OgaCheckResult(OgaCreateRuntimeSettings(&p));
    return std::unique_ptr<OgaRuntimeSettings>(p);
  }

  void SetHandle(const char* name, void* handle) {
    OgaCheckResult(OgaRuntimeSettingsSetHandle(this, name, handle));
  }
  void SetHandle(const std::string& name, void* handle) {
    SetHandle(name.c_str(), handle);
  }

  static void operator delete(void* p) { OgaDestroyRuntimeSettings(reinterpret_cast<OgaRuntimeSettings*>(p)); }
};

struct OgaConfig : OgaAbstract {
  static std::unique_ptr<OgaConfig> Create(const char* config_path) {
    OgaConfig* p;
    OgaCheckResult(OgaCreateConfig(config_path, &p));
    return std::unique_ptr<OgaConfig>(p);
  }

  void ClearProviders() {
    OgaCheckResult(OgaConfigClearProviders(this));
  }

  void AppendProvider(const char* provider) {
    OgaCheckResult(OgaConfigAppendProvider(this, provider));
  }

  void SetProviderOption(const char* provider, const char* name, const char* value) {
    OgaCheckResult(OgaConfigSetProviderOption(this, provider, name, value));
  }

  static void operator delete(void* p) { OgaDestroyConfig(reinterpret_cast<OgaConfig*>(p)); }
};

struct OgaModel : OgaAbstract {
  static std::unique_ptr<OgaModel> Create(const char* config_path) {
    OgaModel* p;
    OgaCheckResult(OgaCreateModel(config_path, &p));
    return std::unique_ptr<OgaModel>(p);
  }
  static std::unique_ptr<OgaModel> Create(const char* config_path, const OgaRuntimeSettings& settings) {
    OgaModel* p;
    OgaCheckResult(OgaCreateModelWithRuntimeSettings(config_path, &settings, &p));
    return std::unique_ptr<OgaModel>(p);
  }
  static std::unique_ptr<OgaModel> Create(OgaConfig& config) {
    OgaModel* p;
    OgaCheckResult(OgaCreateModelFromConfig(&config, &p));
    return std::unique_ptr<OgaModel>(p);
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

  size_t SequenceCount(size_t index) const {
    return OgaSequencesGetSequenceCount(this, index);
  }

  const int32_t* SequenceData(size_t index) const {
    return OgaSequencesGetSequenceData(this, index);
  }

  void Append(const int32_t* tokens, size_t token_cnt) {
    OgaCheckResult(OgaAppendTokenSequence(tokens, token_cnt, this));
  }

  void Append(int32_t token, size_t sequence_index) {
    OgaCheckResult(OgaAppendTokenToSequence(token, this, sequence_index));
  }

#if __cplusplus >= 202002L
  std::span<const int32_t> Get(size_t index) const {
    return {SequenceData(index), SequenceCount(index)};
  }
  void Append(const std::span<const int32_t>& sequence) {
    OgaCheckResult(OgaAppendTokenSequence(sequence.data(), sequence.size(), this));
  }
  void Append(const std::vector<int32_t>& sequence) {
    OgaCheckResult(OgaAppendTokenSequence(sequence.data(), sequence.size(), this));
  }
#endif

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

  int32_t ToTokenId(const char* str) const {
    int32_t token_id;
    OgaCheckResult(OgaTokenizerToTokenId(this, str, &token_id));
    return token_id;
  }

  OgaString Decode(const int32_t* tokens_data, size_t tokens_length) const {
    const char* p;
    OgaCheckResult(OgaTokenizerDecode(this, tokens_data, tokens_length, &p));
    return p;
  }

#if __cplusplus >= 202002L
  OgaString Decode(std::span<const int32_t> tokens) const {
    const char* p;
    OgaCheckResult(OgaTokenizerDecode(this, tokens.data(), tokens.size(), &p));
    return p;
  }
#endif

  static void operator delete(void* p) { OgaDestroyTokenizer(reinterpret_cast<OgaTokenizer*>(p)); }
};

struct OgaTokenizerStream : OgaAbstract {
  static std::unique_ptr<OgaTokenizerStream> Create(const OgaTokenizer& tokenizer) {
    OgaTokenizerStream* p;
    OgaCheckResult(OgaCreateTokenizerStream(&tokenizer, &p));
    return std::unique_ptr<OgaTokenizerStream>(p);
  }

  static std::unique_ptr<OgaTokenizerStream> Create(const OgaMultiModalProcessor& processor) {
    OgaTokenizerStream* p;
    OgaCheckResult(OgaCreateTokenizerStreamFromProcessor(&processor, &p));
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

  void SetSearchOption(const char* name, double value) {
    OgaCheckResult(OgaGeneratorParamsSetSearchNumber(this, name, value));
  }

  void SetSearchOptionBool(const char* name, bool value) {
    OgaCheckResult(OgaGeneratorParamsSetSearchBool(this, name, value));
  }

  void SetModelInput(const char* name, OgaTensor& tensor) {
    OgaCheckResult(OgaGeneratorParamsSetModelInput(this, name, &tensor));
  }

  void SetInputs(OgaNamedTensors& named_tensors) {
    OgaCheckResult(OgaGeneratorParamsSetInputs(this, &named_tensors));
  }

  void TryGraphCaptureWithMaxBatchSize(int max_batch_size) {
    OgaCheckResult(OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize(this, max_batch_size));
  }

  static void operator delete(void* p) { OgaDestroyGeneratorParams(reinterpret_cast<OgaGeneratorParams*>(p)); }
};

struct OgaGenerator : OgaAbstract {
  static std::unique_ptr<OgaGenerator> Create(const OgaModel& model, OgaGeneratorParams& params) {
    OgaGenerator* p;
    OgaCheckResult(OgaCreateGenerator(&model, &params, &p));
    return std::unique_ptr<OgaGenerator>(p);
  }

  bool IsDone() const {
    return OgaGenerator_IsDone(this);
  }

  void AppendTokenSequences(const OgaSequences& sequences) {
    OgaCheckResult(OgaGenerator_AppendTokenSequences(this, &sequences));
  }

  void AppendTokens(int32_t* input_ids, size_t input_ids_count) {
    OgaCheckResult(OgaGenerator_AppendTokens(this, input_ids, input_ids_count));
  }

  bool IsSessionTerminated() const {
    return OgaGenerator_IsSessionTerminated(this);
  }

  void GenerateNextToken() {
    OgaCheckResult(OgaGenerator_GenerateNextToken(this));
  }

  void RewindTo(size_t new_length) {
    OgaCheckResult(OgaGenerator_RewindTo(this, new_length));
  }

  void SetRuntimeOption(const char* key, const char* value) {
    OgaCheckResult(OgaGenerator_SetRuntimeOption(this, key, value));
  }

  size_t GetSequenceCount(size_t index) const {
    return OgaGenerator_GetSequenceCount(this, index);
  }

  const int32_t* GetSequenceData(size_t index) const {
    return OgaGenerator_GetSequenceData(this, index);
  }

  std::unique_ptr<OgaTensor> GetOutput(const char* name) {
    OgaTensor* out;
    OgaCheckResult(OgaGenerator_GetOutput(this, name, &out));
    return std::unique_ptr<OgaTensor>(out);
  }

  std::unique_ptr<OgaTensor> GetLogits() {
    OgaTensor* out;
    OgaCheckResult(OgaGenerator_GetLogits(this, &out));
    return std::unique_ptr<OgaTensor>(out);
  }

  void SetLogits(OgaTensor& tensor) {
    OgaCheckResult(OgaGenerator_SetLogits(this, &tensor));
  }

#if __cplusplus >= 202002L
  std::span<const int32_t> GetSequence(size_t index) const {
    return {GetSequenceData(index), GetSequenceCount(index)};
  }
#endif

  void SetActiveAdapter(OgaAdapters& adapters, const char* adapter_name) {
    OgaCheckResult(OgaSetActiveAdapter(this, &adapters, adapter_name));
  }

  static void operator delete(void* p) { OgaDestroyGenerator(reinterpret_cast<OgaGenerator*>(p)); }
};

struct OgaTensor : OgaAbstract {
#if __cplusplus >= 202002L
  static std::unique_ptr<OgaTensor> Create(void* data, std::span<const int64_t> shape, OgaElementType element_type) {
    OgaTensor* p;
    OgaCheckResult(OgaCreateTensorFromBuffer(data, shape.data(), shape.size(), element_type, &p));
    return std::unique_ptr<OgaTensor>(p);
  }
#endif
  static std::unique_ptr<OgaTensor> Create(void* data, const int64_t* shape_dims, size_t shape_dims_count, OgaElementType element_type) {
    OgaTensor* p;
    OgaCheckResult(OgaCreateTensorFromBuffer(data, shape_dims, shape_dims_count, element_type, &p));
    return std::unique_ptr<OgaTensor>(p);
  }

  OgaElementType Type() {
    OgaElementType type;
    OgaCheckResult(OgaTensorGetType(this, &type));
    return type;
  }

  std::vector<int64_t> Shape() {
    size_t size;
    OgaCheckResult(OgaTensorGetShapeRank(this, &size));
    std::vector<int64_t> shape(size);
    OgaCheckResult(OgaTensorGetShape(this, shape.data(), shape.size()));
    return shape;
  }

  void* Data() {
    void* data;
    OgaCheckResult(OgaTensorGetData(this, &data));
    return data;
  }

  static void operator delete(void* p) { OgaDestroyTensor(reinterpret_cast<OgaTensor*>(p)); }
};

struct OgaImages : OgaAbstract {
  static std::unique_ptr<OgaImages> Load(const std::vector<const char*>& image_paths) {
    OgaImages* p;
    OgaStringArray* strs;
    OgaCheckResult(OgaCreateStringArrayFromStrings(image_paths.data(), image_paths.size(), &strs));
    OgaCheckResult(OgaLoadImages(strs, &p));
    OgaDestroyStringArray(strs);
    return std::unique_ptr<OgaImages>(p);
  }

#if __cplusplus >= 202002L
  static std::unique_ptr<OgaImages> Load(std::span<const char* const> image_paths) {
    OgaImages* p;
    OgaStringArray* strs;
    OgaCheckResult(OgaCreateStringArrayFromStrings(image_paths.data(), image_paths.size(), &strs));
    OgaCheckResult(OgaLoadImages(strs, &p));
    OgaDestroyStringArray(strs);
    return std::unique_ptr<OgaImages>(p);
  }
#endif

  static void operator delete(void* p) { OgaDestroyImages(reinterpret_cast<OgaImages*>(p)); }
};

struct OgaAudios : OgaAbstract {
  static std::unique_ptr<OgaAudios> Load(const std::vector<const char*>& audio_paths) {
    OgaAudios* p;
    OgaStringArray* strs;
    OgaCheckResult(OgaCreateStringArrayFromStrings(audio_paths.data(), audio_paths.size(), &strs));
    OgaCheckResult(OgaLoadAudios(strs, &p));
    OgaDestroyStringArray(strs);
    return std::unique_ptr<OgaAudios>(p);
  }

#if __cplusplus >= 202002L
  static std::unique_ptr<OgaAudios> Load(std::span<const char* const> audio_paths) {
    OgaAudios* p;
    OgaStringArray* strs;
    OgaCheckResult(OgaCreateStringArrayFromStrings(audio_paths.data(), audio_paths.size(), &strs));
    OgaCheckResult(OgaLoadAudios(strs, &p));
    OgaDestroyStringArray(strs);
    return std::unique_ptr<OgaAudios>(p);
  }
#endif

  static void operator delete(void* p) { OgaDestroyAudios(reinterpret_cast<OgaAudios*>(p)); }
};

struct OgaNamedTensors : OgaAbstract {
  static void operator delete(void* p) { OgaDestroyNamedTensors(reinterpret_cast<OgaNamedTensors*>(p)); }
};

struct OgaMultiModalProcessor : OgaAbstract {
  static std::unique_ptr<OgaMultiModalProcessor> Create(const OgaModel& model) {
    OgaMultiModalProcessor* p;
    OgaCheckResult(OgaCreateMultiModalProcessor(&model, &p));
    return std::unique_ptr<OgaMultiModalProcessor>(p);
  }

  std::unique_ptr<OgaNamedTensors> ProcessImages(const char* str, const OgaImages* images = nullptr) const {
    OgaNamedTensors* p;
    OgaCheckResult(OgaProcessorProcessImages(this, str, images, &p));
    return std::unique_ptr<OgaNamedTensors>(p);
  }

  std::unique_ptr<OgaNamedTensors> ProcessAudios(const OgaAudios* audios) const {
    OgaNamedTensors* p;
    OgaCheckResult(OgaProcessorProcessAudios(this, audios, &p));
    return std::unique_ptr<OgaNamedTensors>(p);
  }

  OgaString Decode(const int32_t* tokens_data, size_t tokens_length) const {
    const char* p;
    OgaCheckResult(OgaProcessorDecode(this, tokens_data, tokens_length, &p));
    return p;
  }

#if __cplusplus >= 202002L
  OgaString Decode(std::span<const int32_t> tokens) const {
    const char* p;
    OgaCheckResult(OgaProcessorDecode(this, tokens.data(), tokens.size(), &p));
    return p;
  }
#endif

  static void operator delete(void* p) { OgaDestroyMultiModalProcessor(reinterpret_cast<OgaMultiModalProcessor*>(p)); }
};

struct OgaAdapters : OgaAbstract {
  static std::unique_ptr<OgaAdapters> Create(const OgaModel& model) {
    OgaAdapters* p;
    OgaCheckResult(OgaCreateAdapters(&model, &p));
    return std::unique_ptr<OgaAdapters>(p);
  }

  void LoadAdapter(const char* adapter_file_path,
                   const char* adapter_name) {
    OgaCheckResult(OgaLoadAdapter(this, adapter_file_path, adapter_name));
  }

  void UnloadAdapter(const char* adapter_name) {
    OgaCheckResult(OgaUnloadAdapter(this, adapter_name));
  }

  static void operator delete(void* p) { OgaDestroyAdapters(reinterpret_cast<OgaAdapters*>(p)); }
};

struct OgaHandle {
  OgaHandle() = default;
  ~OgaHandle() noexcept {
    OgaShutdown();
  }
};

// Global Oga functions
namespace Oga {

inline void SetLogBool(const char* name, bool value) {
  OgaCheckResult(OgaSetLogBool(name, value));
}

inline void SetLogString(const char* name, const char* value) {
  OgaCheckResult(OgaSetLogString(name, value));
}

inline void SetCurrentGpuDeviceId(int device_id) {
  OgaCheckResult(OgaSetCurrentGpuDeviceId(device_id));
}

inline int GetCurrentGpuDeviceId() {
  int device_id;
  OgaCheckResult(OgaGetCurrentGpuDeviceId(&device_id));
  return device_id;
}

}  // namespace Oga
