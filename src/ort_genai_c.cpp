// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include "span.h"
#include "ort_genai_c.h"
#include "generators.h"
#include "models/model.h"
#include "runtime_settings.h"
#include "search.h"
#include "smartptrs.h"

namespace Generators {

struct Result {
  explicit Result(const char* what) : what_{what} {}
  std::string what_;
};

}  // namespace Generators

// Allocate a null terminated C string from a std::string
const char* AllocOgaString(const std::string& string) {
  auto length = string.length() + 1;
  auto cstr_buffer = std::make_unique<char[]>(length);
  memcpy(cstr_buffer.get(), string.c_str(), length);
  return cstr_buffer.release();
}

// This type can't be created or copied by value, only by pointer. It's used for the definitions below to ensure nobody
// accidentally creates/copies one of the types that happens to have a default constructor.
struct OgaAbstract {
  OgaAbstract() = delete;
  OgaAbstract(const OgaAbstract&) = delete;
  void operator=(const OgaAbstract&) = delete;
};

// As the Oga* types are just typedefs, we can use them as the actual types in the C API.
// We still need to cast from internal types to the external ones, but these definitions ensure that the types are correct.
// But do not use reinterpret_cast!
struct OgaAdapters : Generators::Adapters, OgaAbstract {};
struct OgaAudios : Generators::Audios, OgaAbstract {};
struct OgaConfig : Generators::Config, OgaAbstract {};
struct OgaGenerator : Generators::Generator, OgaAbstract {};
struct OgaGeneratorParams : Generators::GeneratorParams, OgaAbstract {};
struct OgaImages : Generators::Images, OgaAbstract {};
struct OgaModel : Generators::Model, OgaAbstract {};
struct OgaMultiModalProcessor : Generators::MultiModalProcessor, OgaAbstract {};
struct OgaNamedTensors : Generators::NamedTensors, OgaAbstract {};
struct OgaResult : Generators::Result, OgaAbstract {};
struct OgaRuntimeSettings : Generators::RuntimeSettings, OgaAbstract {};
struct OgaSequences : Generators::TokenSequences, OgaAbstract {};
struct OgaStringArray : std::vector<std::string>, OgaAbstract {};
struct OgaTensor : Generators::Tensor, OgaAbstract {};
struct OgaTokenizer : Generators::Tokenizer, OgaAbstract {};
struct OgaTokenizerStream : Generators::TokenizerStream, OgaAbstract {};

// Helper function to return a shared pointer as a raw pointer. It won't compile if the types are wrong.
// Exposed types that are internally owned by shared_ptrs inherit from ExternalRefCounted. Then we
// manage external C API ownership through ExternalAddRef/ExternalRelease. This function to return
// a value to an external C API owner does the ExternalAddRef, and the OgaDestroy* method has the
// corresponding ExternalRelease.
template <typename T, typename U>
T* ReturnShared(std::shared_ptr<U>& p) {
  p->ExternalAddRef();
  return static_cast<T*>(p.get());
}

// Helper function to return a unique pointer as a raw pointer. It won't compile if the types are wrong.
template <typename T, typename U>
T* ReturnUnique(std::unique_ptr<U> p) {
  return static_cast<T*>(p.release());
}

extern "C" {

#define OGA_TRY try {
#define OGA_CATCH                                                                   \
  }                                                                                 \
  catch (const std::exception& e) {                                                 \
    return ReturnUnique<OgaResult>(std::make_unique<Generators::Result>(e.what())); \
  }

void OGA_API_CALL OgaShutdown() {
  Generators::Shutdown();
}

const char* OGA_API_CALL OgaResultGetError(const OgaResult* result) {
  return result->what_.c_str();
}

OgaResult* OGA_API_CALL OgaSetLogBool(const char* name, bool value) {
  OGA_TRY
  Generators::SetLogBool(name, value);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaSetLogString(const char* name, const char* value) {
  OGA_TRY
  // Turn nullptr into an empty std::string (nullptr directly will crash the std::string constructor)
  Generators::SetLogString(name, value ? value : std::string{});
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateSequences(OgaSequences** out) {
  OGA_TRY
  *out = ReturnUnique<OgaSequences>(std::make_unique<Generators::TokenSequences>());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaAppendTokenSequence(const int32_t* token_ptr, size_t token_cnt, OgaSequences* sequence) {
  OGA_TRY
  std::vector<int32_t> tmp(token_cnt);
  for (size_t i = 0; i < token_cnt; i++) {
    tmp[i] = token_ptr[i];
  }
  sequence->emplace_back(std::move(tmp));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaAppendTokenToSequence(int32_t token, OgaSequences* sequences, size_t sequence_index) {
  OGA_TRY
  if (sequence_index > sequences->size()) {
    throw std::runtime_error("sequence index out of bounds");
  }
  if (sequence_index == sequences->size()) {
    sequences->emplace_back();
  }

  sequences->at(sequence_index).push_back(token);

  return nullptr;
  OGA_CATCH
}

size_t OGA_API_CALL OgaSequencesCount(const OgaSequences* p) {
  return p->size();
}

size_t OGA_API_CALL OgaSequencesGetSequenceCount(const OgaSequences* p, size_t sequence) {
  return (*p)[sequence].size();
}

const int32_t* OGA_API_CALL OgaSequencesGetSequenceData(const OgaSequences* p, size_t sequence) {
  return (*p)[sequence].data();
}

OgaResult* OGA_API_CALL OgaLoadImage(const char* image_path, OgaImages** images) {
  OGA_TRY
  const std::vector<const char*> image_paths_vector{image_path};
  *images = ReturnUnique<OgaImages>(Generators::LoadImages(image_paths_vector));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaLoadImages(const OgaStringArray* image_paths, OgaImages** images) {
  OGA_TRY
  std::vector<const char*> image_paths_vector_c;
  for (const auto& image_path : *image_paths)
    image_paths_vector_c.push_back(image_path.c_str());
  *images = ReturnUnique<OgaImages>(Generators::LoadImages(image_paths_vector_c));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaLoadImagesFromBuffers(const void** image_data, const size_t* image_data_sizes, size_t count, OgaImages** images) {
  OGA_TRY
  *images = ReturnUnique<OgaImages>(Generators::LoadImagesFromBuffers(std::span<const void*>(image_data, count), std::span<const size_t>(image_data_sizes, count)));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaLoadAudio(const char* audio_path, OgaAudios** audios) {
  OGA_TRY
  const std::vector<const char*> audio_paths_vector{audio_path};
  *audios = ReturnUnique<OgaAudios>(Generators::LoadAudios(audio_paths_vector));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaLoadAudios(const OgaStringArray* audio_paths, OgaAudios** audios) {
  OGA_TRY
  std::vector<const char*> audio_paths_vector_c;
  for (const auto& audio_path : *audio_paths)
    audio_paths_vector_c.push_back(audio_path.c_str());
  *audios = ReturnUnique<OgaAudios>(Generators::LoadAudios(audio_paths_vector_c));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaLoadAudiosFromBuffers(const void** audio_data, const size_t* audio_data_sizes, size_t count, OgaAudios** audios) {
  OGA_TRY
  *audios = ReturnUnique<OgaAudios>(Generators::LoadAudiosFromBuffers(std::span<const void*>(audio_data, count), std::span<const size_t>(audio_data_sizes, count)));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateRuntimeSettings(OgaRuntimeSettings** out) {
  OGA_TRY
  *out = ReturnUnique<OgaRuntimeSettings>(Generators::CreateRuntimeSettings());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateModelWithRuntimeSettings(const char* config_path, const OgaRuntimeSettings* settings, OgaModel** out) {
  OGA_TRY
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), config_path, settings);
  *out = ReturnShared<OgaModel>(model);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateConfig(const char* config_path, OgaConfig** out) {
  OGA_TRY
  *out = ReturnUnique<OgaConfig>(std::make_unique<Generators::Config>(fs::path(config_path), std::string_view{}));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaConfigClearProviders(OgaConfig* config) {
  OGA_TRY
  Generators::ClearProviders(*config);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaConfigAppendProvider(OgaConfig* config, const char* provider) {
  OGA_TRY
  Generators::SetProviderOption(*config, provider, {}, {});
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaConfigSetProviderOption(OgaConfig* config, const char* provider, const char* key, const char* value) {
  OGA_TRY
  Generators::SetProviderOption(*config, provider, key, value);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaConfigOverlay(OgaConfig* config, const char* json) {
  OGA_TRY
  Generators::OverlayConfig(*config, json);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateModelFromConfig(const OgaConfig* config, OgaModel** out) {
  OGA_TRY
  auto config_copy = std::make_unique<Generators::Config>(*config);
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), std::move(config_copy));
  *out = ReturnShared<OgaModel>(model);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaModel** out) {
  return OgaCreateModelWithRuntimeSettings(config_path, nullptr, out);
}

OgaResult* OGA_API_CALL OgaModelGetType(const OgaModel* model, const char** out) {
  OGA_TRY
  *out = AllocOgaString(model->config_->model.type.c_str());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaModelGetDeviceType(const OgaModel* model, const char** out) {
  OGA_TRY
  *out = AllocOgaString(to_string(model->p_device_->GetType()));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** out) {
  OGA_TRY
  auto params = std::make_shared<Generators::GeneratorParams>(*model);
  *out = ReturnShared<OgaGeneratorParams>(params);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaRuntimeSettingsSetHandle(OgaRuntimeSettings* settings, const char* handle_name, void* handle) {
  OGA_TRY
  settings->handles_[handle_name] = handle;
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsSetSearchNumber(OgaGeneratorParams* generator_params, const char* name, double value) {
  OGA_TRY
  Generators::SetSearchNumber(generator_params->search, name, value);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsSetSearchBool(OgaGeneratorParams* generator_params, const char* name, bool value) {
  OGA_TRY
  Generators::SetSearchBool(generator_params->search, name, value);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize(OgaGeneratorParams* generator_params, int32_t max_batch_size) {
  OGA_TRY
  printf("TryGraphCaptureWithMaxBatchSize is deprecated and will be removed in a future release\n");
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputs(OgaGeneratorParams* params, const OgaNamedTensors* p_named_tensors) {
  OGA_TRY
  params->SetInputs(*p_named_tensors);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsSetModelInput(OgaGeneratorParams* params, const char* name, OgaTensor* tensor) {
  OGA_TRY
  params->extra_inputs.push_back({std::string{name}, tensor->shared_from_this()});
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsSetWhisperInputFeatures(OgaGeneratorParams* params, OgaTensor* tensor) {
  OGA_TRY
  Generators::GeneratorParams::Whisper& whisper = params->inputs.emplace<Generators::GeneratorParams::Whisper>();
  whisper.input_features = tensor->shared_from_this();
  return nullptr;
  OGA_CATCH
}

OgaResult* OgaCreateGenerator(const OgaModel* model, const OgaGeneratorParams* generator_params, OgaGenerator** out) {
  OGA_TRY
  *out = ReturnUnique<OgaGenerator>(CreateGenerator(*model, *generator_params));
  return nullptr;
  OGA_CATCH
}

bool OGA_API_CALL OgaGenerator_IsDone(const OgaGenerator* generator) {
  return generator->IsDone();
}

bool OGA_API_CALL OgaGenerator_IsSessionTerminated(const OgaGenerator* generator) {
  return generator->IsSessionTerminated();
}

OgaResult* OGA_API_CALL OgaGenerator_AppendTokenSequences(OgaGenerator* generator, const OgaSequences* sequences) {
  OGA_TRY

  if (sequences->empty()) {
    throw std::runtime_error("input sequences are empty");
  } else if (sequences->size() != generator->state_->params_->search.batch_size) {
    throw std::runtime_error("input sequences count does not match batch size");
  }
  std::vector<std::span<const int32_t>> span_sequences;
  for (size_t i = 0; i < sequences->size(); i++) {
    span_sequences.emplace_back((*sequences)[i]);
  }

  auto input_ids = Generators::PadInputs(span_sequences, generator->model_->config_->model.pad_token_id);
  generator->AppendTokens(input_ids);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_AppendTokens(OgaGenerator* generator, const int32_t* input_ids, size_t input_ids_count) {
  OGA_TRY
  generator->AppendTokens(Generators::cpu_span<const int32_t>(input_ids, input_ids_count));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken(OgaGenerator* generator) {
  OGA_TRY
  generator->GenerateNextToken();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_GetNextTokens(const OgaGenerator* generator, const int32_t** out, size_t* out_count) {
  OGA_TRY
  auto tokens = generator->search_->GetNextTokens().CopyDeviceToCpu();
  *out = tokens.data();
  *out_count = tokens.size();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_RewindTo(OgaGenerator* generator, size_t new_length) {
  OGA_TRY
  generator->RewindToLength(new_length);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_SetRuntimeOption(OgaGenerator* generator, const char* key, const char* value) {
  OGA_TRY
  generator->SetRuntimeOption(key, value);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_GetOutput(const OgaGenerator* generator, const char* name, OgaTensor** out) {
  OGA_TRY
  auto* ortvalue_output = generator->state_->GetOutput(name);
  auto type_info = ortvalue_output->GetTensorTypeAndShapeInfo();
  auto ortvalue_clone = OrtValue::CreateTensor(generator->model_->allocator_cpu_, type_info->GetShape(), type_info->GetElementType());

  // Copy data to ortvalue_clone
  bool is_cpu = ortvalue_output->GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU;
  auto output_span = Generators::ByteWrapTensor(is_cpu ? *Generators::GetDeviceInterface(Generators::DeviceType::CPU) : *generator->model_->p_device_, *ortvalue_output);
  auto copy_span = Generators::ByteWrapTensor(*Generators::GetDeviceInterface(Generators::DeviceType::CPU), *ortvalue_clone);
  copy_span.CopyFrom(output_span);

  auto tensor = std::make_shared<Generators::Tensor>(std::move(ortvalue_clone));
  *out = ReturnShared<OgaTensor>(tensor);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_GetLogits(OgaGenerator* generator, OgaTensor** out) {
  OGA_TRY
  auto logits_span = generator->GetLogits();
  const std::array<int64_t, 3> shape{generator->state_->params_->search.batch_size, 1, generator->model_->config_->model.vocab_size};
  std::span<const float> cpu_logits_span = logits_span.CopyDeviceToCpu();

  // Copy logits to cpu tensor
  std::unique_ptr<OrtValue> ortvalue_clone = OrtValue::CreateTensor<float>(generator->model_->allocator_cpu_, shape);
  auto clone_span = std::span<float>(ortvalue_clone->GetTensorMutableData<float>(), cpu_logits_span.size());
  Generators::copy(cpu_logits_span, clone_span);
  auto tensor = std::make_shared<Generators::Tensor>(std::move(ortvalue_clone));
  *out = ReturnShared<OgaTensor>(tensor);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_SetLogits(OgaGenerator* generator, OgaTensor* tensor) {
  OGA_TRY
  auto logits = generator->search_->GetLogits();
  if (!generator->computed_logits_ && logits.size() != 0) {
    throw std::runtime_error("logits are not computed yet. Please call GenerateNextToken or AppendTokens before calling SetLogits.");
  }
  size_t element_count = tensor->GetElementCount();
  auto new_logits_span = std::span<const float>(tensor->GetData<float>(), element_count);
  if (logits.size() == 0) {
    logits = generator->model_->p_device_inputs_->Allocate<float>(element_count);
    generator->SetLogits(logits);
  } else if (new_logits_span.size() != logits.size()) {
    throw std::runtime_error("Generator::SetLogits passed an array of size " +
                             std::to_string(new_logits_span.size()) + " but should be size " + std::to_string(logits.size()));
  }
  Generators::copy(new_logits_span, logits.CpuSpan());
  logits.CopyCpuToDevice();
  generator->computed_logits_ = true;
  return nullptr;
  OGA_CATCH
}

size_t OGA_API_CALL OgaGenerator_GetSequenceCount(const OgaGenerator* generator, size_t index) {
  return generator->GetSequence(static_cast<int>(index)).size();
}

const int32_t* OGA_API_CALL OgaGenerator_GetSequenceData(const OgaGenerator* generator, size_t index) {
  return generator->GetSequence(static_cast<int>(index)).CopyDeviceToCpu().data();
}

OgaResult* OGA_API_CALL OgaCreateTokenizer(const OgaModel* model, OgaTokenizer** out) {
  OGA_TRY
  auto tokenizer = model->CreateTokenizer();
  *out = ReturnShared<OgaTokenizer>(tokenizer);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTokenizerEncode(const OgaTokenizer* tokenizer, const char* str, OgaSequences* sequences) {
  OGA_TRY
  sequences->emplace_back(tokenizer->Encode(str));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTokenizerEncodeBatch(const OgaTokenizer* tokenizer, const char** strings, size_t count, OgaTensor** out) {
  OGA_TRY
  auto tensor = tokenizer->EncodeBatch(std::span<const char*>(strings, count));
  *out = ReturnShared<OgaTensor>(tensor);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTokenizerToTokenId(const OgaTokenizer* tokenizer, const char* str, int32_t* token_id) {
  OGA_TRY
  *token_id = tokenizer->TokenToTokenId(str);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTokenizerDecode(const OgaTokenizer* tokenizer, const int32_t* tokens, size_t token_count, const char** out_string) {
  OGA_TRY
  *out_string = AllocOgaString(tokenizer->Decode({tokens, token_count}));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTokenizerApplyChatTemplate(const OgaTokenizer* tokenizer, const char* template_str, const char* messages, bool add_generation_prompt, const char** out_string) {
  OGA_TRY
  *out_string = AllocOgaString(tokenizer->ApplyChatTemplate(template_str, messages, add_generation_prompt));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTokenizerDecodeBatch(const OgaTokenizer* tokenizer, const OgaTensor* tensor, OgaStringArray** out) {
  OGA_TRY
  auto shape = tensor->GetShape();
  if (shape.size() != 2)
    throw std::runtime_error("Expected a 2D tensor");
  auto strings = tokenizer->DecodeBatch(std::span<const int32_t>{tensor->GetData<int32_t>(), tensor->GetElementCount()}, shape[0]);
  *out = ReturnUnique<OgaStringArray>(std::make_unique<std::vector<std::string>>(std::move(strings)));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaProcessorDecode(const OgaMultiModalProcessor* processor, const int32_t* tokens, size_t token_count, const char** out_string) {
  OGA_TRY
  *out_string = AllocOgaString(processor->tokenizer_->Decode({tokens, token_count}));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateTokenizerStream(const OgaTokenizer* p, OgaTokenizerStream** out) {
  OGA_TRY
  *out = ReturnUnique<OgaTokenizerStream>(p->CreateStream());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateTokenizerStreamFromProcessor(const OgaMultiModalProcessor* p, OgaTokenizerStream** out) {
  OGA_TRY
  *out = ReturnUnique<OgaTokenizerStream>(p->tokenizer_->CreateStream());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTokenizerStreamDecode(OgaTokenizerStream* p, int32_t token, const char** out) {
  OGA_TRY
  *out = p->Decode(token).c_str();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateTensorFromBuffer(void* data, const int64_t* shape_dims, size_t shape_dims_count, OgaElementType element_type, OgaTensor** out) {
  OGA_TRY
  auto p_memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto ort_element_type = static_cast<ONNXTensorElementDataType>(element_type);
  size_t byte_count = Ort::SizeOf(ort_element_type);
  auto shape = std::span<const int64_t>{shape_dims, shape_dims_count};
  for (size_t i = 0; i < shape_dims_count; i++)
    byte_count *= shape_dims[i];
  std::unique_ptr<OrtValue> ort_tensor;
  if (data)
    ort_tensor = OrtValue::CreateTensor(*p_memory_info, data, byte_count, shape, ort_element_type);
  else
    ort_tensor = OrtValue::CreateTensor(Ort::Allocator::GetWithDefaultOptions(), shape, ort_element_type);
  auto tensor = std::make_shared<Generators::Tensor>(std::move(ort_tensor));
  *out = ReturnShared<OgaTensor>(tensor);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTensorGetType(OgaTensor* tensor, OgaElementType* out) {
  OGA_TRY
  *out = static_cast<OgaElementType>(tensor->GetType());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTensorGetShapeRank(OgaTensor* tensor, size_t* out) {
  OGA_TRY
  *out = tensor->GetShape().size();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTensorGetShape(OgaTensor* tensor, int64_t* shape_dims, size_t rank) {
  OGA_TRY
  auto shape = tensor->GetShape();
  if (rank != shape.size())
    throw std::runtime_error("shape_dims_count doesn't match result of OgaTensorGetShapeRank");
  std::copy(shape.begin(), shape.end(), shape_dims);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTensorGetData(OgaTensor* tensor, void** out) {
  OGA_TRY
  *out = tensor->GetMutableRawData();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateNamedTensors(OgaNamedTensors** out) {
  OGA_TRY
  *out = ReturnUnique<OgaNamedTensors>(std::make_unique<Generators::NamedTensors>());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaNamedTensorsGet(OgaNamedTensors* named_tensors, const char* name, OgaTensor** out) {
  OGA_TRY
  auto iter = named_tensors->find(name);
  if (iter == named_tensors->end())
    *out = nullptr;
  else
    *out = ReturnShared<OgaTensor>(iter->second);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaNamedTensorsSet(OgaNamedTensors* named_tensors, const char* name, OgaTensor* tensor) {
  OGA_TRY(*named_tensors)
  [name] = tensor->shared_from_this();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaNamedTensorsDelete(OgaNamedTensors* named_tensors, const char* name) {
  OGA_TRY
  named_tensors->erase(name);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaNamedTensorsCount(const OgaNamedTensors* named_tensors, size_t* out) {
  OGA_TRY
  *out = named_tensors->size();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaNamedTensorsGetNames(const OgaNamedTensors* named_tensors, OgaStringArray** out) {
  OGA_TRY
  auto names = std::make_unique<std::vector<std::string>>();
  for (const auto& pair : *named_tensors)
    names->push_back(pair.first);
  *out = ReturnUnique<OgaStringArray>(std::move(names));
  return nullptr;
  OGA_CATCH
}

OGA_EXPORT OgaResult* OGA_API_CALL OgaSetCurrentGpuDeviceId(int device_id) {
  OGA_TRY
  Ort::SetCurrentGpuDeviceId(device_id);
  return nullptr;
  OGA_CATCH
}

OGA_EXPORT OgaResult* OGA_API_CALL OgaGetCurrentGpuDeviceId(int* device_id) {
  OGA_TRY
  *device_id = Ort::GetCurrentGpuDeviceId();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateMultiModalProcessor(const OgaModel* model, OgaMultiModalProcessor** out) {
  OGA_TRY
  auto processor = model->CreateMultiModalProcessor();
  *out = ReturnShared<OgaMultiModalProcessor>(processor);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaProcessorProcessImages(const OgaMultiModalProcessor* processor, const char* prompt, const OgaImages* images, OgaNamedTensors** input_tensors) {
  OGA_TRY
  if (!processor->processor_)
    throw std::runtime_error("Image processor is not available for this model.");

  *input_tensors = ReturnUnique<OgaNamedTensors>(processor->Process(prompt, images, nullptr));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaProcessorProcessAudios(const OgaMultiModalProcessor* processor, const OgaAudios* audios, OgaNamedTensors** input_tensors) {
  OGA_TRY
  if (!processor->processor_)
    throw std::runtime_error("Audio processor not available for this model.");

  *input_tensors = ReturnUnique<OgaNamedTensors>(processor->Process(std::string(), nullptr, audios));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaProcessorProcessImagesAndAudios(const OgaMultiModalProcessor* processor, const char* prompt, const OgaImages* images,
                                                           const OgaAudios* audios, OgaNamedTensors** input_tensors) {
  OGA_TRY
  if (!processor->processor_)
    throw std::runtime_error("Audio processor not available for this model.");

  *input_tensors = ReturnUnique<OgaNamedTensors>(processor->Process(prompt, images, audios));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateStringArray(OgaStringArray** out) {
  OGA_TRY
  *out = ReturnUnique<OgaStringArray>(std::make_unique<std::vector<std::string>>());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateStringArrayFromStrings(const char* const* strs, size_t count, OgaStringArray** out) {
  OGA_TRY
  auto string_array = std::make_unique<std::vector<std::string>>();
  for (size_t i = 0; i < count; i++)
    string_array->push_back(strs[i]);
  *out = ReturnUnique<OgaStringArray>(std::move(string_array));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaStringArrayAddString(OgaStringArray* string_array, const char* str) {
  OGA_TRY
  string_array->push_back(str);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaStringArrayGetCount(const OgaStringArray* string_array, size_t* out) {
  OGA_TRY
  *out = string_array->size();
  return nullptr;
  OGA_CATCH
}

OgaResult* OgaStringArrayGetString(const OgaStringArray* string_array, size_t index, const char** out) {
  OGA_TRY
  *out = string_array->at(index).c_str();
  return nullptr;
  OGA_CATCH
}

OgaResult* OgaCreateAdapters(const OgaModel* model, OgaAdapters** out) {
  OGA_TRY
  auto adapters = std::make_shared<Generators::Adapters>(model);
  *out = ReturnShared<OgaAdapters>(adapters);
  return nullptr;
  OGA_CATCH
}

OgaResult* OgaLoadAdapter(OgaAdapters* adapters, const char* adapter_file_path, const char* adapter_name) {
  OGA_TRY
  adapters->LoadAdapter(adapter_file_path, adapter_name);
  return nullptr;
  OGA_CATCH
}

OgaResult* OgaUnloadAdapter(OgaAdapters* adapters, const char* adapter_name) {
  OGA_TRY
  adapters->UnloadAdapter(adapter_name);
  return nullptr;
  OGA_CATCH
}

OgaResult* OgaSetActiveAdapter(OgaGenerator* generator, OgaAdapters* adapters, const char* adapter_name) {
  OGA_TRY
  generator->state_->SetActiveAdapter(adapters, adapter_name);
  return nullptr;
  OGA_CATCH
}

void OGA_API_CALL OgaDestroyStringArray(OgaStringArray* string_array) { delete string_array; }
void OGA_API_CALL OgaDestroyResult(OgaResult* p) { delete p; }
void OGA_API_CALL OgaDestroyString(const char* p) { delete p; }
void OGA_API_CALL OgaDestroySequences(OgaSequences* p) { delete p; }
void OGA_API_CALL OgaDestroyConfig(OgaConfig* p) { delete p; }
void OGA_API_CALL OgaDestroyModel(OgaModel* p) { p->ExternalRelease(); }
void OGA_API_CALL OgaDestroyGeneratorParams(OgaGeneratorParams* p) { p->ExternalRelease(); }
void OGA_API_CALL OgaDestroyGenerator(OgaGenerator* p) { delete p; }
void OGA_API_CALL OgaDestroyTokenizer(OgaTokenizer* p) { p->ExternalRelease(); }
void OGA_API_CALL OgaDestroyTokenizerStream(OgaTokenizerStream* p) { delete p; }
void OGA_API_CALL OgaDestroyTensor(OgaTensor* p) { p->ExternalRelease(); }
void OGA_API_CALL OgaDestroyMultiModalProcessor(OgaMultiModalProcessor* p) { p->ExternalRelease(); }
void OGA_API_CALL OgaDestroyImages(OgaImages* p) { delete p; }
void OGA_API_CALL OgaDestroyAudios(OgaAudios* p) { delete p; }
void OGA_API_CALL OgaDestroyNamedTensors(OgaNamedTensors* p) { delete p; }
void OGA_API_CALL OgaDestroyAdapters(OgaAdapters* p) { p->ExternalRelease(); }
void OGA_API_CALL OgaDestroyRuntimeSettings(OgaRuntimeSettings* p) { delete p; }

}  // extern "C"
