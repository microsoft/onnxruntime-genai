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
#include "models/sd.h"

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

std::unique_ptr<OrtValue> initial_latents_tensor(int64_t batch_size, int64_t unet_channels, int64_t latent_height, int64_t latent_width, Ort::Allocator* allocator) {
  // Create the tensor of latentss
  std::vector<int64_t> latents_shape{batch_size, unet_channels, latent_height, latent_width};
  std::unique_ptr<OrtValue> latents_tensor = OrtValue::CreateTensor<float>(*allocator, std::span{latents_shape});
  float* latents_data = latents_tensor->GetTensorMutableData<float>();

  // Create a random number generator and normal distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0, 1.0);

  // Fill the latents_tensor_data with random values

  for (int i = 0; i < batch_size * unet_channels * latent_height * latent_width; i++) {
    // Scale the initial noise by the standard deviation required by the scheduler
    latents_data[i] = dist(gen);
  }
  return latents_tensor;
}

OGA_EXPORT OgaResult* OGA_API_CALL OgaSelenaTest(const char* prompt, const char* model_path, uint8_t** result) {
  OGA_TRY
  auto text_encoder_path = std::string(model_path) + "/text_encoder/model.onnx";
  auto vae_model_path = std::string(model_path) + "/vae_decoder/model.onnx";
  auto unet_model_path = std::string(model_path) + "/unet/model.onnx";

  Generators::Config config{fs::path{model_path}, std::string_view{}};

  auto tokenizer = std::make_shared<Generators::Tokenizer>(config);

  auto input_ids = tokenizer->Encode(prompt);

  int32_t* sequences_data = input_ids.data();
  size_t sequences_size = input_ids.size();
  // The following are params (currently hardcoded) for the model
  int32_t batch_size = 1;
  int32_t max_sequence_length = 77;
  int32_t hidden_size = 1024;
  int32_t unet_channels = 4;
  int32_t image_height = 512;
  int32_t image_width = 512;

  float init_noise_sigma = 1.0;
  float vae_scaling_factor = 0.18215;

  // create the OrtSession
  Ort::InitApi();
  std::unique_ptr<OrtEnv> p_env = OrtEnv::Create(ORT_LOGGING_LEVEL_WARNING, "test");

  std::unique_ptr<OrtSessionOptions> session_options = OrtSessionOptions::Create();

  std::unique_ptr<OrtSession> p_session_ = OrtSession::Create(
      *p_env,
      text_encoder_path.c_str(),
      session_options.get());

  std::unique_ptr<OrtMemoryInfo> p_memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto allocator = Ort::Allocator::Create(*p_session_, *p_memory_info);

  // enable_cuda_graph is false in prototype version
  // create input_ids tensor
  std::vector<int64_t> input_ids_shape{batch_size, max_sequence_length};

  std::unique_ptr<OrtValue> p_input_tensor = OrtValue::CreateTensor<int32_t>(*allocator, std::span{input_ids_shape});
  int32_t* input_ids_data = p_input_tensor->GetTensorMutableData<int32_t>();

  // if the length of input_ids is larger than max_sequence_length, we need to truncate it
  if (sequences_size > max_sequence_length) {
    std::copy(sequences_data, sequences_data + max_sequence_length, input_ids_data);
  }

  std::copy(sequences_data, sequences_data + sequences_size, input_ids_data);

  // if the length of input_ids is smaller than max_sequence_length, we need to pad it
  if (sequences_size < max_sequence_length) {
    std::fill(input_ids_data + sequences_size, input_ids_data + max_sequence_length, 0);
  }
  // Bind input tensors and run inference
  auto io_binding = OrtIoBinding::Create(*p_session_);
  io_binding->BindInput("input_ids", *p_input_tensor);

  // Bind output text_embeddings tensor

  std::vector<int64_t> output_embeddings_shape{batch_size, max_sequence_length, hidden_size};
  std::unique_ptr<OrtValue> p_output_tensor = OrtValue::CreateTensor<float>(*allocator, std::span{output_embeddings_shape});
  io_binding->BindOutput("text_embeddings", *p_output_tensor);

  std::unique_ptr<OrtRunOptions> run_options = OrtRunOptions::Create();

  io_binding->SynchronizeInputs();
  p_session_->Run(run_options.get(), *io_binding);
  io_binding->SynchronizeOutputs();

  // Get output text_embeddings tensor
  auto text_embeddings_tensor = io_binding->GetOutputValues();

  auto text_embeddings_tensor_data = text_embeddings_tensor[0]->GetTensorMutableData<float>();

  // Create a new OrtValue for the latents tensor
  int64_t latent_height = image_height / 8;
  int64_t latent_width = image_width / 8;
  std::vector<int64_t> latents_shape{batch_size, unet_channels, latent_height, latent_width};
  // latents = torch.randn(latents_shape, device=self.device, dtype=latents_dtype, generator=self.generator)
  //  Create the tensor of latentss

  std::unique_ptr<OrtValue> latents_tensor = OrtValue::CreateTensor<float>(*allocator, std::span{latents_shape});
  float* latents_data = latents_tensor->GetTensorMutableData<float>();

  // Create a random number generator and normal distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0, 1.0);

  // Fill the latents_tensor_data with random values
  // size_t step_index = 0;
  for (int i = 0; i < batch_size * unet_channels * latent_height * latent_width; i++) {
    // Scale the initial noise by the standard deviation required by the scheduler
    latents_data[i] = dist(gen) * init_noise_sigma;
  }

  // Denoising latents
  std::vector<int64_t> tiemsteps = {999, 759, 519, 279};
  // beta_start**0.5, beta_end**0.5, torch.linspace()
  std::vector<float> alphas_cumprod = {0.9991, 0.9983, 0.9974, 0.9966, 0.9957, 0.9948, 0.9940, 0.9931, 0.9922,
                                       0.9913, 0.9904, 0.9895, 0.9886, 0.9877, 0.9868, 0.9859, 0.9850, 0.9841,
                                       0.9832, 0.9822, 0.9813, 0.9804, 0.9794, 0.9785, 0.9776, 0.9766, 0.9757,
                                       0.9747, 0.9737, 0.9728, 0.9718, 0.9708, 0.9698, 0.9689, 0.9679, 0.9669,
                                       0.9659, 0.9649, 0.9639, 0.9629, 0.9619, 0.9609, 0.9599, 0.9588, 0.9578,
                                       0.9568, 0.9557, 0.9547, 0.9537, 0.9526, 0.9516, 0.9505, 0.9495, 0.9484,
                                       0.9473, 0.9463, 0.9452, 0.9441, 0.9430, 0.9420, 0.9409, 0.9398, 0.9387,
                                       0.9376, 0.9365, 0.9354, 0.9343, 0.9332, 0.9320, 0.9309, 0.9298, 0.9287,
                                       0.9275, 0.9264, 0.9252, 0.9241, 0.9229, 0.9218, 0.9206, 0.9195, 0.9183,
                                       0.9171, 0.9160, 0.9148, 0.9136, 0.9124, 0.9112, 0.9100, 0.9089, 0.9077,
                                       0.9065, 0.9052, 0.9040, 0.9028, 0.9016, 0.9004, 0.8992, 0.8979, 0.8967,
                                       0.8955, 0.8942, 0.8930, 0.8917, 0.8905, 0.8892, 0.8880, 0.8867, 0.8854,
                                       0.8842, 0.8829, 0.8816, 0.8804, 0.8791, 0.8778, 0.8765, 0.8752, 0.8739,
                                       0.8726, 0.8713, 0.8700, 0.8687, 0.8674, 0.8661, 0.8647, 0.8634, 0.8621,
                                       0.8607, 0.8594, 0.8581, 0.8567, 0.8554, 0.8540, 0.8527, 0.8513, 0.8500,
                                       0.8486, 0.8473, 0.8459, 0.8445, 0.8431, 0.8418, 0.8404, 0.8390, 0.8376,
                                       0.8362, 0.8348, 0.8334, 0.8320, 0.8306, 0.8292, 0.8278, 0.8264, 0.8250,
                                       0.8236, 0.8221, 0.8207, 0.8193, 0.8179, 0.8164, 0.8150, 0.8136, 0.8121,
                                       0.8107, 0.8092, 0.8078, 0.8063, 0.8049, 0.8034, 0.8019, 0.8005, 0.7990,
                                       0.7975, 0.7960, 0.7946, 0.7931, 0.7916, 0.7901, 0.7886, 0.7871, 0.7856,
                                       0.7842, 0.7827, 0.7812, 0.7796, 0.7781, 0.7766, 0.7751, 0.7736, 0.7721,
                                       0.7706, 0.7690, 0.7675, 0.7660, 0.7645, 0.7629, 0.7614, 0.7599, 0.7583,
                                       0.7568, 0.7552, 0.7537, 0.7521, 0.7506, 0.7490, 0.7475, 0.7459, 0.7444,
                                       0.7428, 0.7412, 0.7397, 0.7381, 0.7365, 0.7350, 0.7334, 0.7318, 0.7302,
                                       0.7286, 0.7271, 0.7255, 0.7239, 0.7223, 0.7207, 0.7191, 0.7175, 0.7159,
                                       0.7143, 0.7127, 0.7111, 0.7095, 0.7079, 0.7063, 0.7047, 0.7031, 0.7015,
                                       0.6999, 0.6982, 0.6966, 0.6950, 0.6934, 0.6918, 0.6901, 0.6885, 0.6869,
                                       0.6852, 0.6836, 0.6820, 0.6803, 0.6787, 0.6771, 0.6754, 0.6738, 0.6722,
                                       0.6705, 0.6689, 0.6672, 0.6656, 0.6639, 0.6623, 0.6606, 0.6590, 0.6573,
                                       0.6557, 0.6540, 0.6524, 0.6507, 0.6490, 0.6474, 0.6457, 0.6441, 0.6424,
                                       0.6407, 0.6391, 0.6374, 0.6357, 0.6341, 0.6324, 0.6307, 0.6291, 0.6274,
                                       0.6257, 0.6241, 0.6224, 0.6207, 0.6190, 0.6174, 0.6157, 0.6140, 0.6123,
                                       0.6107, 0.6090, 0.6073, 0.6056, 0.6039, 0.6023, 0.6006, 0.5989, 0.5972,
                                       0.5955, 0.5939, 0.5922, 0.5905, 0.5888, 0.5871, 0.5855, 0.5838, 0.5821,
                                       0.5804, 0.5787, 0.5770, 0.5754, 0.5737, 0.5720, 0.5703, 0.5686, 0.5669,
                                       0.5652, 0.5636, 0.5619, 0.5602, 0.5585, 0.5568, 0.5551, 0.5535, 0.5518,
                                       0.5501, 0.5484, 0.5467, 0.5450, 0.5434, 0.5417, 0.5400, 0.5383, 0.5366,
                                       0.5350, 0.5333, 0.5316, 0.5299, 0.5282, 0.5266, 0.5249, 0.5232, 0.5215,
                                       0.5199, 0.5182, 0.5165, 0.5148, 0.5132, 0.5115, 0.5098, 0.5082, 0.5065,
                                       0.5048, 0.5032, 0.5015, 0.4998, 0.4982, 0.4965, 0.4948, 0.4932, 0.4915,
                                       0.4898, 0.4882, 0.4865, 0.4849, 0.4832, 0.4816, 0.4799, 0.4782, 0.4766,
                                       0.4749, 0.4733, 0.4716, 0.4700, 0.4684, 0.4667, 0.4651, 0.4634, 0.4618,
                                       0.4601, 0.4585, 0.4569, 0.4552, 0.4536, 0.4520, 0.4503, 0.4487, 0.4471,
                                       0.4455, 0.4438, 0.4422, 0.4406, 0.4390, 0.4374, 0.4357, 0.4341, 0.4325,
                                       0.4309, 0.4293, 0.4277, 0.4261, 0.4245, 0.4229, 0.4213, 0.4197, 0.4181,
                                       0.4165, 0.4149, 0.4133, 0.4117, 0.4101, 0.4086, 0.4070, 0.4054, 0.4038,
                                       0.4022, 0.4007, 0.3991, 0.3975, 0.3960, 0.3944, 0.3928, 0.3913, 0.3897,
                                       0.3882, 0.3866, 0.3850, 0.3835, 0.3819, 0.3804, 0.3789, 0.3773, 0.3758,
                                       0.3742, 0.3727, 0.3712, 0.3697, 0.3681, 0.3666, 0.3651, 0.3636, 0.3621,
                                       0.3605, 0.3590, 0.3575, 0.3560, 0.3545, 0.3530, 0.3515, 0.3500, 0.3485,
                                       0.3470, 0.3456, 0.3441, 0.3426, 0.3411, 0.3396, 0.3382, 0.3367, 0.3352,
                                       0.3338, 0.3323, 0.3308, 0.3294, 0.3279, 0.3265, 0.3250, 0.3236, 0.3222,
                                       0.3207, 0.3193, 0.3178, 0.3164, 0.3150, 0.3136, 0.3122, 0.3107, 0.3093,
                                       0.3079, 0.3065, 0.3051, 0.3037, 0.3023, 0.3009, 0.2995, 0.2981, 0.2967,
                                       0.2953, 0.2940, 0.2926, 0.2912, 0.2899, 0.2885, 0.2871, 0.2858, 0.2844,
                                       0.2831, 0.2817, 0.2804, 0.2790, 0.2777, 0.2763, 0.2750, 0.2737, 0.2723,
                                       0.2710, 0.2697, 0.2684, 0.2671, 0.2658, 0.2645, 0.2631, 0.2618, 0.2606,
                                       0.2593, 0.2580, 0.2567, 0.2554, 0.2541, 0.2528, 0.2516, 0.2503, 0.2490,
                                       0.2478, 0.2465, 0.2453, 0.2440, 0.2428, 0.2415, 0.2403, 0.2391, 0.2378,
                                       0.2366, 0.2354, 0.2341, 0.2329, 0.2317, 0.2305, 0.2293, 0.2281, 0.2269,
                                       0.2257, 0.2245, 0.2233, 0.2221, 0.2209, 0.2198, 0.2186, 0.2174, 0.2163,
                                       0.2151, 0.2139, 0.2128, 0.2116, 0.2105, 0.2093, 0.2082, 0.2071, 0.2059,
                                       0.2048, 0.2037, 0.2026, 0.2014, 0.2003, 0.1992, 0.1981, 0.1970, 0.1959,
                                       0.1948, 0.1937, 0.1926, 0.1915, 0.1905, 0.1894, 0.1883, 0.1872, 0.1862,
                                       0.1851, 0.1841, 0.1830, 0.1820, 0.1809, 0.1799, 0.1788, 0.1778, 0.1768,
                                       0.1757, 0.1747, 0.1737, 0.1727, 0.1717, 0.1707, 0.1696, 0.1686, 0.1677,
                                       0.1667, 0.1657, 0.1647, 0.1637, 0.1627, 0.1618, 0.1608, 0.1598, 0.1589,
                                       0.1579, 0.1569, 0.1560, 0.1550, 0.1541, 0.1532, 0.1522, 0.1513, 0.1504,
                                       0.1494, 0.1485, 0.1476, 0.1467, 0.1458, 0.1449, 0.1440, 0.1431, 0.1422,
                                       0.1413, 0.1404, 0.1395, 0.1386, 0.1378, 0.1369, 0.1360, 0.1352, 0.1343,
                                       0.1334, 0.1326, 0.1317, 0.1309, 0.1301, 0.1292, 0.1284, 0.1276, 0.1267,
                                       0.1259, 0.1251, 0.1243, 0.1235, 0.1227, 0.1219, 0.1211, 0.1203, 0.1195,
                                       0.1187, 0.1179, 0.1171, 0.1163, 0.1155, 0.1148, 0.1140, 0.1132, 0.1125,
                                       0.1117, 0.1110, 0.1102, 0.1095, 0.1087, 0.1080, 0.1073, 0.1065, 0.1058,
                                       0.1051, 0.1044, 0.1036, 0.1029, 0.1022, 0.1015, 0.1008, 0.1001, 0.0994,
                                       0.0987, 0.0980, 0.0973, 0.0967, 0.0960, 0.0953, 0.0946, 0.0940, 0.0933,
                                       0.0926, 0.0920, 0.0913, 0.0907, 0.0900, 0.0894, 0.0887, 0.0881, 0.0875,
                                       0.0868, 0.0862, 0.0856, 0.0850, 0.0844, 0.0837, 0.0831, 0.0825, 0.0819,
                                       0.0813, 0.0807, 0.0801, 0.0795, 0.0789, 0.0784, 0.0778, 0.0772, 0.0766,
                                       0.0761, 0.0755, 0.0749, 0.0744, 0.0738, 0.0732, 0.0727, 0.0721, 0.0716,
                                       0.0711, 0.0705, 0.0700, 0.0694, 0.0689, 0.0684, 0.0679, 0.0673, 0.0668,
                                       0.0663, 0.0658, 0.0653, 0.0648, 0.0643, 0.0638, 0.0633, 0.0628, 0.0623,
                                       0.0618, 0.0613, 0.0608, 0.0604, 0.0599, 0.0594, 0.0589, 0.0585, 0.0580,
                                       0.0575, 0.0571, 0.0566, 0.0562, 0.0557, 0.0553, 0.0548, 0.0544, 0.0539,
                                       0.0535, 0.0531, 0.0526, 0.0522, 0.0518, 0.0514, 0.0509, 0.0505, 0.0501,
                                       0.0497, 0.0493, 0.0489, 0.0485, 0.0481, 0.0477, 0.0473, 0.0469, 0.0465,
                                       0.0461, 0.0457, 0.0453, 0.0450, 0.0446, 0.0442, 0.0438, 0.0435, 0.0431,
                                       0.0427, 0.0424, 0.0420, 0.0416, 0.0413, 0.0409, 0.0406, 0.0402, 0.0399,
                                       0.0395, 0.0392, 0.0389, 0.0385, 0.0382, 0.0379, 0.0375, 0.0372, 0.0369,
                                       0.0365, 0.0362, 0.0359, 0.0356, 0.0353, 0.0350, 0.0347, 0.0343, 0.0340,
                                       0.0337, 0.0334, 0.0331, 0.0328, 0.0325, 0.0323, 0.0320, 0.0317, 0.0314,
                                       0.0311, 0.0308, 0.0305, 0.0303, 0.0300, 0.0297, 0.0295, 0.0292, 0.0289,
                                       0.0286, 0.0284, 0.0281, 0.0279, 0.0276, 0.0274, 0.0271, 0.0268, 0.0266,
                                       0.0264, 0.0261, 0.0259, 0.0256, 0.0254, 0.0251, 0.0249, 0.0247, 0.0244,
                                       0.0242, 0.0240, 0.0237, 0.0235, 0.0233, 0.0231, 0.0229, 0.0226, 0.0224,
                                       0.0222, 0.0220, 0.0218, 0.0216, 0.0214, 0.0212, 0.0210, 0.0207, 0.0205,
                                       0.0203, 0.0201, 0.0200, 0.0198, 0.0196, 0.0194, 0.0192, 0.0190, 0.0188,
                                       0.0186, 0.0184, 0.0182, 0.0181, 0.0179, 0.0177, 0.0175, 0.0174, 0.0172,
                                       0.0170, 0.0168, 0.0167, 0.0165, 0.0163, 0.0162, 0.0160, 0.0158, 0.0157,
                                       0.0155, 0.0154, 0.0152, 0.0151, 0.0149, 0.0147, 0.0146, 0.0144, 0.0143,
                                       0.0142, 0.0140, 0.0139, 0.0137, 0.0136, 0.0134, 0.0133, 0.0132, 0.0130,
                                       0.0129, 0.0127, 0.0126, 0.0125, 0.0123, 0.0122, 0.0121, 0.0120, 0.0118,
                                       0.0117, 0.0116, 0.0115, 0.0113, 0.0112, 0.0111, 0.0110, 0.0109, 0.0107,
                                       0.0106, 0.0105, 0.0104, 0.0103, 0.0102, 0.0101, 0.0100, 0.0098, 0.0097,
                                       0.0096, 0.0095, 0.0094, 0.0093, 0.0092, 0.0091, 0.0090, 0.0089, 0.0088,
                                       0.0087, 0.0086, 0.0085, 0.0084, 0.0083, 0.0082, 0.0082, 0.0081, 0.0080,
                                       0.0079, 0.0078, 0.0077, 0.0076, 0.0075, 0.0074, 0.0074, 0.0073, 0.0072,
                                       0.0071, 0.0070, 0.0070, 0.0069, 0.0068, 0.0067, 0.0066, 0.0066, 0.0065,
                                       0.0064, 0.0063, 0.0063, 0.0062, 0.0061, 0.0061, 0.0060, 0.0059, 0.0058,
                                       0.0058, 0.0057, 0.0056, 0.0056, 0.0055, 0.0054, 0.0054, 0.0053, 0.0053,
                                       0.0052, 0.0051, 0.0051, 0.0050, 0.0049, 0.0049, 0.0048, 0.0048, 0.0047,
                                       0.0047};

  for (size_t i = 0; i < tiemsteps.size(); i++) {
    std::vector<int64_t> timestep_shape{1};
    std::vector<float> timestep_data{static_cast<float>(tiemsteps[i])};
    std::unique_ptr<OrtValue> timestep_tensor = OrtValue::CreateTensor<float>(*p_memory_info, std::span{timestep_data}, std::span{timestep_shape});

    // Create the input params map for the denoising step
    std::map<std::string, OrtValue*> params;
    params["sample"] = latents_tensor.get();
    params["timestep"] = timestep_tensor.get();
    params["encoder_hidden_states"] = text_embeddings_tensor[0].get();

    // Bind the input tensors and run inference
    std::unique_ptr<OrtSession> p_unet_session_ = OrtSession::Create(
        *p_env,
        unet_model_path.c_str(),
        session_options.get());
    auto unet_allocator = Ort::Allocator::Create(*p_unet_session_, *p_memory_info);
    auto io_binding_unet = OrtIoBinding::Create(*p_unet_session_);

    io_binding_unet->BindInput("sample", *params["sample"]);
    io_binding_unet->BindInput("timestep", *params["timestep"]);
    io_binding_unet->BindInput("encoder_hidden_states", *params["encoder_hidden_states"]);

    // Bind the output latent tensor
    std::vector<int64_t> output_latent_shape{batch_size, unet_channels, latent_height, latent_width};
    std::unique_ptr<OrtValue> output_latent_tensor = OrtValue::CreateTensor<float>(*unet_allocator, std::span{output_latent_shape});
    io_binding_unet->BindOutput("latent", *output_latent_tensor);

    // Run the unet model

    io_binding_unet->SynchronizeInputs();
    p_unet_session_->Run(run_options.get(), *io_binding_unet);
    io_binding_unet->SynchronizeOutputs();

    // Get the output latent tensor
    auto unet_output_tensor = io_binding_unet->GetOutputValues();
    auto noise_pred_data = unet_output_tensor[0]->GetTensorMutableData<float>();

    // Post-process noise_pred to step it
    // 1. get previous step value
    size_t prev_step_index = i + 1;
    int64_t prev_timestep = 0;
    if (prev_step_index < tiemsteps.size()) {
      prev_timestep = tiemsteps[prev_step_index];
    } else {
      prev_timestep = tiemsteps[i];
    }

    // 2. compute alphas, betas
    auto alpha_prod_t = alphas_cumprod[tiemsteps[i]];
    auto alpha_prod_t_prev = alphas_cumprod[prev_timestep] ? prev_timestep >= 0 : alphas_cumprod[0];

    auto beta_prod_t = 1 - alpha_prod_t;
    auto beta_prod_t_prev = 1 - alpha_prod_t_prev;

    // 3. Get scalings for boundary conditions
    float sigma_data = 0.5;  // Default: 0.5
    float timestep_scaling = 10.0;
    float scaled_timestep = tiemsteps[i] * timestep_scaling;

    float c_skip = (sigma_data * sigma_data) / (scaled_timestep * scaled_timestep + sigma_data * sigma_data);
    float c_out = scaled_timestep / std::sqrt(scaled_timestep * scaled_timestep + sigma_data * sigma_data);

    // 4. Compute the predicted original sample x_0 based on the model parameterization
    // prediction_type == "epsilon":  # noise-prediction
    // predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    std::vector<int64_t> predicted_sample_shape{batch_size, unet_channels, latent_height, latent_width};
    std::vector<float> predicted_sample_data(batch_size * unet_channels * latent_height * latent_width);
    for (int i = 0; i < predicted_sample_data.size(); i++) {
      predicted_sample_data[i] = (latents_data[i] - std::sqrt(beta_prod_t) * noise_pred_data[i]) / std::sqrt(alpha_prod_t);
    }
    std::unique_ptr<OrtValue> predicted_sample_tensor = OrtValue::CreateTensor<float>(*p_memory_info, std::span{predicted_sample_data}, std::span{predicted_sample_shape});

    // 5. Clip or threshold "predicted x_0"
    // In SD turbo, this step is skipped

    // 6. Denoise model output using boundary conditions
    // denoised = c_out * predicted_original_sample + c_skip * sample
    for (int i = 0; i < predicted_sample_data.size(); i++) {
      predicted_sample_data[i] = c_out * predicted_sample_data[i] + c_skip * latents_data[i];
    }

    // 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
    // Noise is not used on the final timestep of the timestep schedule.
    // This also means that noise is not used for one-step sampling.
    //  prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
    if (i != tiemsteps.size() - 1) {
      auto noise = initial_latents_tensor(batch_size, unet_channels, latent_height, latent_width, unet_allocator.get());
      auto noise_data = noise->GetTensorMutableData<float>();
      for (int i = 0; i < predicted_sample_data.size(); i++) {
        latents_data[i] = std::sqrt(alpha_prod_t_prev) * predicted_sample_data[i] + std::sqrt(beta_prod_t_prev) * noise_data[i];
      }
    } else {
      for (int i = 0; i < predicted_sample_data.size(); i++) {
        latents_data[i] = predicted_sample_data[i] / vae_scaling_factor;
      }
    }
  }

  // VAE decode latents
  // Bind the vae input tensors
  std::unique_ptr<OrtSession> p_vae_session_ = OrtSession::Create(
      *p_env,
      vae_model_path.c_str(),
      session_options.get());
  auto vae_allocator = Ort::Allocator::Create(*p_vae_session_, *p_memory_info);
  auto io_binding_vae = OrtIoBinding::Create(*p_vae_session_);

  io_binding_vae->BindInput("latent", *latents_tensor);

  // Bind the output latent tensor
  std::vector<int64_t> output_image_shape{batch_size, 3, image_height, image_width};
  std::unique_ptr<OrtValue> output_image_tensor = OrtValue::CreateTensor<float>(*vae_allocator, std::span{output_image_shape});
  // auto x = output_image_tensor->GetTensorMutableData<float>();

  io_binding_vae->BindOutput("images", *output_image_tensor);

  // Run the vae decoder model

  io_binding_vae->SynchronizeInputs();
  p_vae_session_->Run(run_options.get(), *io_binding_vae);
  io_binding_vae->SynchronizeOutputs();

  // Get the output image tensor
  auto vae_output_tensor = io_binding_vae->GetOutputValues();

  auto image_tensor_data = vae_output_tensor[0]->GetTensorMutableData<float>();

  auto images = std::move(vae_output_tensor[0]);

  // std::vector<int64_t> post_processed_image_shape{static_cast<int64_t>(batch_size), image_height, image_width, 3};
  // auto post_processed_image_tensor = OrtValue::CreateTensor<uint8_t>(*vae_allocator, post_processed_image_shape);
  // uint8_t* image_data = post_processed_image_tensor->GetTensorMutableData<uint8_t>();

  // temporary code: leaked memory
  uint8_t* image_data = new uint8_t[batch_size * image_height * image_width * 3];

  for (size_t B = 0; B < batch_size; ++B) {
    for (size_t H = 0; H < image_height; ++H) {
      for (size_t W = 0; W < image_width; ++W) {
        for (size_t C = 0; C < 3; ++C) {
          size_t index_images = B * image_height * image_width * 3 + C * image_height * image_width + H * image_width + W;
          size_t index_post_processed = B * image_height * image_width * 3 + H * image_width * 3 + W * 3 + C;

          float image_value = images->GetTensorMutableData<float>()[index_images];
          uint8_t post_processed_value = static_cast<uint8_t>(std::clamp((image_value + 1.0f) * 255.0f / 2.0f, 0.f, 255.f));

          // if (C == 2) {
          //   std::cout << (int)post_processed_value << ", ";
          // }

          image_data[index_post_processed] = post_processed_value;
        }
      }
    }
  }

  //auto final_output_tensor = std::make_shared<Generators::Tensor>(std::move(post_processed_image_tensor));
  //*result = ReturnShared<OgaTensor>(final_output_tensor);
  *result = image_data;

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
