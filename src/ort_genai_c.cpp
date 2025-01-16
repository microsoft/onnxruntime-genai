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

extern "C" {

#define OGA_TRY try {
#define OGA_CATCH                                                                                  \
  }                                                                                                \
  catch (const std::exception& e) {                                                                \
    return reinterpret_cast<OgaResult*>(std::make_unique<Generators::Result>(e.what()).release()); \
  }

void OGA_API_CALL OgaShutdown() {
  Generators::Shutdown();
}

const char* OGA_API_CALL OgaResultGetError(const OgaResult* result) {
  return reinterpret_cast<const Generators::Result*>(result)->what_.c_str();
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
  *out = reinterpret_cast<OgaSequences*>(std::make_unique<Generators::TokenSequences>().release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaAppendTokenSequence(const int32_t* token_ptr, size_t token_cnt, OgaSequences* sequence) {
  OGA_TRY
  Generators::TokenSequences* toks = reinterpret_cast<Generators::TokenSequences*>(sequence);
  std::vector<int32_t> tmp(token_cnt);
  for (size_t i = 0; i < token_cnt; i++) {
    tmp[i] = token_ptr[i];
  }
  toks->emplace_back(std::move(tmp));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaAppendTokenToSequence(int32_t token, OgaSequences* sequences, size_t sequence_index) {
  OGA_TRY
  Generators::TokenSequences* toks = reinterpret_cast<Generators::TokenSequences*>(sequences);
  if (sequence_index > toks->size()) {
    throw std::runtime_error("sequence index out of bounds");
  }
  if (sequence_index == toks->size()) {
    toks->emplace_back();
  }

  toks->at(sequence_index).push_back(token);

  return nullptr;
  OGA_CATCH
}

size_t OGA_API_CALL OgaSequencesCount(const OgaSequences* p) {
  return reinterpret_cast<const Generators::TokenSequences*>(p)->size();
}

size_t OGA_API_CALL OgaSequencesGetSequenceCount(const OgaSequences* p, size_t sequence) {
  return (*reinterpret_cast<const Generators::TokenSequences*>(p))[sequence].size();
}

const int32_t* OGA_API_CALL OgaSequencesGetSequenceData(const OgaSequences* p, size_t sequence) {
  return (*reinterpret_cast<const Generators::TokenSequences*>(p))[sequence].data();
}

OgaResult* OGA_API_CALL OgaLoadImage(const char* image_path, OgaImages** images) {
  OGA_TRY
  const std::vector<const char*> image_paths_vector{image_path};
  *images = reinterpret_cast<OgaImages*>(Generators::LoadImages(image_paths_vector).release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaLoadImages(const OgaStringArray* image_paths, OgaImages** images) {
  OGA_TRY
  const auto& image_paths_vector = *reinterpret_cast<const std::vector<std::string>*>(image_paths);
  std::vector<const char*> image_paths_vector_c;
  for (const auto& image_path : image_paths_vector) image_paths_vector_c.push_back(image_path.c_str());
  *images = reinterpret_cast<OgaImages*>(Generators::LoadImages(image_paths_vector_c).release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaLoadAudio(const char* audio_path, OgaAudios** audios) {
  OGA_TRY
  const std::vector<const char*> audio_paths_vector{audio_path};
  *audios = reinterpret_cast<OgaAudios*>(Generators::LoadAudios(audio_paths_vector).release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaLoadAudios(const OgaStringArray* audio_paths, OgaAudios** audios) {
  OGA_TRY
  const auto& audio_paths_vector = *reinterpret_cast<const std::vector<std::string>*>(audio_paths);
  std::vector<const char*> audio_paths_vector_c;
  for (const auto& audio_path : audio_paths_vector) audio_paths_vector_c.push_back(audio_path.c_str());
  *audios = reinterpret_cast<OgaAudios*>(Generators::LoadAudios(audio_paths_vector_c).release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateRuntimeSettings(OgaRuntimeSettings** out) {
  OGA_TRY
  *out = reinterpret_cast<OgaRuntimeSettings*>(Generators::CreateRuntimeSettings().release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateModelWithRuntimeSettings(const char* config_path, const OgaRuntimeSettings* settings, OgaModel** out) {
  OGA_TRY
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), config_path, reinterpret_cast<const Generators::RuntimeSettings*>(settings));
  model->external_owner_ = model;
  *out = reinterpret_cast<OgaModel*>(model.get());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateConfig(const char* config_path, OgaConfig** out) {
  OGA_TRY
  *out = reinterpret_cast<OgaConfig*>(std::make_unique<Generators::Config>(fs::path(config_path), std::string_view{}).release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaConfigClearProviders(OgaConfig* config) {
  OGA_TRY
  Generators::ClearProviders(*reinterpret_cast<Generators::Config*>(config));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaConfigAppendProvider(OgaConfig* config, const char* provider) {
  OGA_TRY
  Generators::SetProviderOption(*reinterpret_cast<Generators::Config*>(config), provider, {}, {});
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaConfigSetProviderOption(OgaConfig* config, const char* provider, const char* key, const char* value) {
  OGA_TRY
  Generators::SetProviderOption(*reinterpret_cast<Generators::Config*>(config), provider, key, value);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateModelFromConfig(const OgaConfig* config, OgaModel** out) {
  OGA_TRY
  auto config_copy = std::make_unique<Generators::Config>(*reinterpret_cast<const Generators::Config*>(config));
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), std::move(config_copy));
  model->external_owner_ = model;
  *out = reinterpret_cast<OgaModel*>(model.get());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaModel** out) {
  return OgaCreateModelWithRuntimeSettings(config_path, nullptr, out);
}

OgaResult* OGA_API_CALL OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** out) {
  OGA_TRY
  auto params = std::make_shared<Generators::GeneratorParams>(*reinterpret_cast<const Generators::Model*>(model));
  params->external_owner_ = params;
  *out = reinterpret_cast<OgaGeneratorParams*>(params.get());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaRuntimeSettingsSetHandle(OgaRuntimeSettings* settings, const char* handle_name, void* handle) {
  OGA_TRY
  Generators::RuntimeSettings* settings_ = reinterpret_cast<Generators::RuntimeSettings*>(settings);
  settings_->handles_[handle_name] = handle;
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsSetSearchNumber(OgaGeneratorParams* generator_params, const char* name, double value) {
  OGA_TRY
  Generators::SetSearchNumber(reinterpret_cast<Generators::GeneratorParams*>(generator_params)->search, name, value);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsSetSearchBool(OgaGeneratorParams* generator_params, const char* name, bool value) {
  OGA_TRY
  Generators::SetSearchBool(reinterpret_cast<Generators::GeneratorParams*>(generator_params)->search, name, value);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize(OgaGeneratorParams* generator_params, int32_t max_batch_size) {
  OGA_TRY
  auto* params = reinterpret_cast<Generators::GeneratorParams*>(generator_params);
  params->TryGraphCapture(max_batch_size);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputs(OgaGeneratorParams* oga_params, const OgaNamedTensors* p_named_tensors) {
  OGA_TRY
  auto& params = *reinterpret_cast<Generators::GeneratorParams*>(oga_params);
  auto& named_tensors = *reinterpret_cast<const Generators::NamedTensors*>(p_named_tensors);

  params.SetInputs(named_tensors);

  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsSetModelInput(OgaGeneratorParams* oga_params, const char* name, OgaTensor* tensor) {
  OGA_TRY
  auto& params = *reinterpret_cast<Generators::GeneratorParams*>(oga_params);
  params.extra_inputs.push_back({std::string{name}, reinterpret_cast<Generators::Tensor*>(tensor)->shared_from_this()});
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsSetWhisperInputFeatures(OgaGeneratorParams* oga_params, OgaTensor* tensor) {
  OGA_TRY
  auto& params = *reinterpret_cast<Generators::GeneratorParams*>(oga_params);
  Generators::GeneratorParams::Whisper& whisper = params.inputs.emplace<Generators::GeneratorParams::Whisper>();
  whisper.input_features = reinterpret_cast<Generators::Tensor*>(tensor)->shared_from_this();
  return nullptr;
  OGA_CATCH
}

OgaResult* OgaCreateGenerator(const OgaModel* model, const OgaGeneratorParams* generator_params, OgaGenerator** out) {
  OGA_TRY
  *out = reinterpret_cast<OgaGenerator*>(CreateGenerator(*reinterpret_cast<const Generators::Model*>(model), *reinterpret_cast<const Generators::GeneratorParams*>(generator_params)).release());
  return nullptr;
  OGA_CATCH
}

bool OGA_API_CALL OgaGenerator_IsDone(const OgaGenerator* generator) {
  return reinterpret_cast<const Generators::Generator*>(generator)->IsDone();
}

bool OGA_API_CALL OgaGenerator_IsSessionTerminated(const OgaGenerator* generator) {
  return reinterpret_cast<const Generators::Generator*>(generator)->IsSessionTerminated();
}

OgaResult* OGA_API_CALL OgaGenerator_AppendTokenSequences(OgaGenerator* oga_generator, const OgaSequences* p_sequences) {
  OGA_TRY
  auto& generator = *reinterpret_cast<Generators::Generator*>(oga_generator);
  auto& sequences = *reinterpret_cast<const Generators::TokenSequences*>(p_sequences);

  if (sequences.empty()) {
    throw std::runtime_error("input sequences are empty");
  } else if (sequences.size() != generator.state_->params_->search.batch_size) {
    throw std::runtime_error("input sequences count does not match batch size");
  }
  std::vector<std::span<const int32_t>> span_sequences;
  for (size_t i = 0; i < sequences.size(); i++) {
    span_sequences.emplace_back(sequences[i]);
  }

  auto input_ids = Generators::PadInputs(span_sequences, generator.model_->config_->model.pad_token_id);
  generator.AppendTokens(input_ids);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_AppendTokens(OgaGenerator* oga_generator, const int32_t* input_ids, size_t input_ids_count) {
  OGA_TRY
  auto& generator = *reinterpret_cast<Generators::Generator*>(oga_generator);
  generator.AppendTokens(Generators::cpu_span<const int32_t>(input_ids, input_ids_count));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken(OgaGenerator* generator) {
  OGA_TRY
  reinterpret_cast<Generators::Generator*>(generator)->GenerateNextToken();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_RewindTo(OgaGenerator* generator, size_t new_length) {
  OGA_TRY
  reinterpret_cast<Generators::Generator*>(generator)->RewindToLength(new_length);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_SetRuntimeOption(OgaGenerator* generator, const char* key, const char* value) {
  OGA_TRY
  reinterpret_cast<Generators::Generator*>(generator)->SetRuntimeOption(key, value);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_GetOutput(const OgaGenerator* oga_generator, const char* name, OgaTensor** out) {
  OGA_TRY
  auto& generator = *reinterpret_cast<const Generators::Generator*>(oga_generator);
  auto* ortvalue_output = generator.state_->GetOutput(name);
  auto type_info = ortvalue_output->GetTensorTypeAndShapeInfo();
  std::unique_ptr<OrtValue> ortvalue_clone = OrtValue::CreateTensor(generator.model_->allocator_cpu_,
                                                                    type_info->GetShape(),
                                                                    type_info->GetElementType());
  // Copy data to ortvalue_clone
  auto element_size = Generators::SizeOf(type_info->GetElementType());
  auto data_size = type_info->GetElementCount() * element_size;
  const auto device_type = ortvalue_output->GetTensorMemoryInfo().GetDeviceType();
  if (device_type == OrtMemoryInfoDeviceType_CPU) {
    std::copy(static_cast<uint8_t*>(ortvalue_output->GetTensorMutableRawData()),
              static_cast<uint8_t*>(ortvalue_output->GetTensorMutableRawData()) + data_size,
              static_cast<uint8_t*>(ortvalue_clone->GetTensorMutableRawData()));
  } else if (device_type == OrtMemoryInfoDeviceType_GPU) {
#if USE_CUDA
    cudaMemcpy(ortvalue_clone->GetTensorMutableRawData(), ortvalue_output->GetTensorMutableRawData(), data_size, cudaMemcpyDeviceToHost);
#else
    throw std::runtime_error("Unexpected error. Trying to access GPU memory but the project is not compiled with CUDA.");
#endif
  } else if (static_cast<int>(device_type) == 4) {
#if USE_DML
    ComPtr<ID3D12Resource> gpu_resource;
    Ort::ThrowOnError(generator.model_->GetOrtDmlApi()->GetD3D12ResourceFromAllocation(
        generator.model_->allocator_device_,
        ortvalue_output->GetTensorMutableRawData(),
        &gpu_resource));
    auto cpu_tensor = ortvalue_clone->GetTensorMutableRawData();
    generator.model_->GetDmlReadbackHeap()->ReadbackFromGpu(
        std::span(reinterpret_cast<uint8_t*>(cpu_tensor), data_size),
        gpu_resource.Get(),
        0,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
#else
    throw std::runtime_error("Unexpected error. Trying to access DML memory but the project is not compiled with DML.");
#endif
  } else {
    throw std::runtime_error("Unsupported device type: " + std::to_string(static_cast<int>(device_type)));
  }

  auto tensor = std::make_shared<Generators::Tensor>(std::move(ortvalue_clone));
  tensor->external_owner_ = tensor;
  *out = reinterpret_cast<OgaTensor*>(tensor.get());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_GetLogits(OgaGenerator* oga_generator, OgaTensor** out) {
  OGA_TRY
  auto generator = reinterpret_cast<Generators::Generator*>(oga_generator);
  auto logits_span = generator->GetLogits();
  const std::array<int64_t, 3> shape{generator->state_->params_->search.batch_size, 1, generator->model_->config_->model.vocab_size};
  std::span<const float> cpu_logits_span = logits_span.CopyDeviceToCpu();

  // Copy logits to cpu tensor
  std::unique_ptr<OrtValue> ortvalue_clone = OrtValue::CreateTensor<float>(generator->model_->allocator_cpu_, shape);
  auto clone_span = std::span<float>(ortvalue_clone->GetTensorMutableData<float>(), cpu_logits_span.size());
  Generators::copy(cpu_logits_span, clone_span);
  auto tensor = std::make_shared<Generators::Tensor>(std::move(ortvalue_clone));
  tensor->external_owner_ = tensor;
  *out = reinterpret_cast<OgaTensor*>(tensor.get());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_SetLogits(OgaGenerator* oga_generator, OgaTensor* tensor) {
  OGA_TRY
  auto generator = reinterpret_cast<Generators::Generator*>(oga_generator);
  if (!generator->computed_logits_) {
    throw std::runtime_error("logits are not computed yet. Please call GenerateNextToken or AppendTokens before calling SetLogits.");
  }
  auto logits_tensor = reinterpret_cast<Generators::Tensor*>(tensor);
  size_t element_count = logits_tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementCount();
  auto new_logits_span = std::span<const float>(logits_tensor->ort_tensor_->GetTensorData<float>(), element_count);
  auto logits = generator->search_->GetLogits();
  if (new_logits_span.size() != logits.size()) {
    throw std::runtime_error("Generator::SetLogits passed an array of size " +
                             std::to_string(new_logits_span.size()) + " but should be size " + std::to_string(logits.size()));
  }
  Generators::copy(new_logits_span, logits.CpuSpan());
  logits.CopyCpuToDevice();
  generator->computed_logits_ = true;
  return nullptr;
  OGA_CATCH
}

size_t OGA_API_CALL OgaGenerator_GetSequenceCount(const OgaGenerator* oga_generator, size_t index) {
  auto& generator = *reinterpret_cast<const Generators::Generator*>(oga_generator);
  return generator.GetSequence(static_cast<int>(index)).size();
}

const int32_t* OGA_API_CALL OgaGenerator_GetSequenceData(const OgaGenerator* oga_generator, size_t index) {
  auto& generator = *reinterpret_cast<const Generators::Generator*>(oga_generator);
  return generator.GetSequence(static_cast<int>(index)).CopyDeviceToCpu().data();
}

OgaResult* OGA_API_CALL OgaCreateTokenizer(const OgaModel* model, OgaTokenizer** out) {
  OGA_TRY
  auto tokenizer = reinterpret_cast<const Generators::Model*>(model)->CreateTokenizer();
  tokenizer->external_owner_ = tokenizer;
  *out = reinterpret_cast<OgaTokenizer*>(tokenizer.get());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTokenizerEncode(const OgaTokenizer* p, const char* str, OgaSequences* sequences) {
  OGA_TRY
  auto& tokenizer = *reinterpret_cast<const Generators::Tokenizer*>(p);
  auto& token_sequences = *reinterpret_cast<Generators::TokenSequences*>(sequences);
  token_sequences.emplace_back(tokenizer.Encode(str));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTokenizerToTokenId(const OgaTokenizer* p, const char* str, int32_t* token_id) {
  OGA_TRY
  auto& tokenizer = *reinterpret_cast<const Generators::Tokenizer*>(p);
  *token_id = tokenizer.TokenToTokenId(str);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTokenizerDecode(const OgaTokenizer* p, const int32_t* tokens, size_t token_count, const char** out_string) {
  OGA_TRY
  auto& tokenizer = *reinterpret_cast<const Generators::Tokenizer*>(p);

  auto string = tokenizer.Decode({tokens, token_count});
  auto length = string.length() + 1;
  auto cstr_buffer = std::make_unique<char[]>(length);
  memcpy(cstr_buffer.get(), string.c_str(), length);
  *out_string = cstr_buffer.release();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaProcessorDecode(const OgaMultiModalProcessor* p, const int32_t* tokens, size_t token_count, const char** out_string) {
  OGA_TRY
  auto& processor = *reinterpret_cast<const Generators::MultiModalProcessor*>(p);

  auto string = processor.tokenizer_->Decode({tokens, token_count});
  auto length = string.length() + 1;
  auto cstr_buffer = std::make_unique<char[]>(length);
  memcpy(cstr_buffer.get(), string.c_str(), length);
  *out_string = cstr_buffer.release();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateTokenizerStream(const OgaTokenizer* p, OgaTokenizerStream** out) {
  OGA_TRY
  *out = reinterpret_cast<OgaTokenizerStream*>(reinterpret_cast<const Generators::Tokenizer*>(p)->CreateStream().release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateTokenizerStreamFromProcessor(const OgaMultiModalProcessor* p, OgaTokenizerStream** out) {
  OGA_TRY
  *out = reinterpret_cast<OgaTokenizerStream*>(
      reinterpret_cast<const Generators::MultiModalProcessor*>(
          p)
          ->tokenizer_->CreateStream()
          .release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTokenizerStreamDecode(OgaTokenizerStream* p, int32_t token, const char** out) {
  OGA_TRY
  *out = reinterpret_cast<Generators::TokenizerStream*>(p)->Decode(token).c_str();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateTensorFromBuffer(void* data, const int64_t* shape_dims, size_t shape_dims_count, OgaElementType element_type, OgaTensor** out) {
  OGA_TRY
  auto tensor = std::make_shared<Generators::Tensor>();
  auto p_memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto ort_element_type = static_cast<ONNXTensorElementDataType>(element_type);
  size_t byte_count = Generators::SizeOf(ort_element_type);
  for (size_t i = 0; i < shape_dims_count; i++)
    byte_count *= shape_dims[i];
  tensor->ort_tensor_ = OrtValue::CreateTensor(*p_memory_info, data, byte_count, std::span<const int64_t>{shape_dims, shape_dims_count}, ort_element_type);
  tensor->external_owner_ = tensor;
  *out = reinterpret_cast<OgaTensor*>(tensor.get());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTensorGetType(OgaTensor* tensor, OgaElementType* out) {
  OGA_TRY
  *out = static_cast<OgaElementType>(reinterpret_cast<Generators::Tensor*>(tensor)->ort_tensor_->GetTensorTypeAndShapeInfo()->GetElementType());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTensorGetShapeRank(OgaTensor* tensor, size_t* out) {
  OGA_TRY
  *out = reinterpret_cast<Generators::Tensor*>(tensor)->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape().size();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTensorGetShape(OgaTensor* tensor, int64_t* shape_dims, size_t rank) {
  OGA_TRY
  auto shape = reinterpret_cast<Generators::Tensor*>(tensor)->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape();
  if (rank != shape.size())
    throw std::runtime_error("shape_dims_count doesn't match result of OgaTensorGetShapeRank");
  std::copy(shape.begin(), shape.end(), shape_dims);
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaTensorGetData(OgaTensor* tensor, void** out) {
  OGA_TRY
  *out = reinterpret_cast<Generators::Tensor*>(tensor)->ort_tensor_->GetTensorMutableRawData();
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
  auto processor = reinterpret_cast<const Generators::Model*>(model)->CreateMultiModalProcessor();
  processor->external_owner_ = processor;
  *out = reinterpret_cast<OgaMultiModalProcessor*>(processor.get());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaProcessorProcessImages(const OgaMultiModalProcessor* p, const char* prompt, const OgaImages* images_p, OgaNamedTensors** input_tensors) {
  OGA_TRY
  auto& processor = *reinterpret_cast<const Generators::MultiModalProcessor*>(p);
  auto* images = images_p ? reinterpret_cast<const Generators::Images*>(images_p) : nullptr;
  if (processor.image_processor_ == nullptr)
    throw std::runtime_error("Image processor is not available for this model.");

  auto named_tensors = processor.image_processor_->Process(*processor.tokenizer_, prompt, images);
  *input_tensors = reinterpret_cast<OgaNamedTensors*>(named_tensors.release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaProcessorProcessAudios(const OgaMultiModalProcessor* p, const OgaAudios* audios_p, OgaNamedTensors** input_tensors) {
  OGA_TRY
  auto& processor = *reinterpret_cast<const Generators::MultiModalProcessor*>(p);
  auto* audios = reinterpret_cast<const Generators::Audios*>(audios_p);

  if (!processor.audio_processor_)
    throw std::runtime_error("Audio processor not available for this model.");

  auto named_tensors = processor.audio_processor_->Process(audios);
  *input_tensors = reinterpret_cast<OgaNamedTensors*>(named_tensors.release());

  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateStringArray(OgaStringArray** out) {
  OGA_TRY
  *out = reinterpret_cast<OgaStringArray*>(std::make_unique<std::vector<std::string>>().release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateStringArrayFromStrings(const char* const* strs, size_t count, OgaStringArray** out) {
  OGA_TRY
  auto string_array = std::make_unique<std::vector<std::string>>();
  for (size_t i = 0; i < count; i++)
    string_array->push_back(strs[i]);
  *out = reinterpret_cast<OgaStringArray*>(string_array.release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaStringArrayAddString(OgaStringArray* string_array, const char* str) {
  OGA_TRY
  reinterpret_cast<std::vector<std::string>*>(string_array)->push_back(str);
  return nullptr;
  OGA_CATCH
}

size_t OGA_API_CALL OgaStringArrayGetCount(const OgaStringArray* string_array) {
  return reinterpret_cast<const std::vector<std::string>*>(string_array)->size();
}

OgaResult* OgaCreateAdapters(const OgaModel* model, OgaAdapters** out) {
  OGA_TRY
  auto adapters = std::make_shared<Generators::Adapters>(reinterpret_cast<const Generators::Model*>(model));
  *out = reinterpret_cast<OgaAdapters*>(adapters.get());
  adapters->external_owner_ = adapters;
  return nullptr;
  OGA_CATCH
}

OgaResult* OgaLoadAdapter(OgaAdapters* adapters, const char* adapter_file_path,
                          const char* adapter_name) {
  OGA_TRY
  reinterpret_cast<Generators::Adapters*>(adapters)->LoadAdapter(adapter_file_path, adapter_name);
  return nullptr;
  OGA_CATCH
}

OgaResult* OgaUnloadAdapter(OgaAdapters* adapters, const char* adapter_name) {
  OGA_TRY
  reinterpret_cast<Generators::Adapters*>(adapters)->UnloadAdapter(adapter_name);
  return nullptr;
  OGA_CATCH
}

OgaResult* OgaSetActiveAdapter(OgaGenerator* generator, OgaAdapters* adapters,
                               const char* adapter_name) {
  OGA_TRY
  reinterpret_cast<Generators::Generator*>(generator)->state_->SetActiveAdapter(
      reinterpret_cast<Generators::Adapters*>(adapters), adapter_name);
  return nullptr;
  OGA_CATCH
}

void OGA_API_CALL OgaDestroyStringArray(OgaStringArray* string_array) {
  delete reinterpret_cast<std::vector<std::string>*>(string_array);
}

void OGA_API_CALL OgaDestroyResult(OgaResult* p) {
  delete reinterpret_cast<Generators::Result*>(p);
}

void OGA_API_CALL OgaDestroyString(const char* p) {
  delete p;
}

void OGA_API_CALL OgaDestroySequences(OgaSequences* p) {
  delete reinterpret_cast<Generators::TokenSequences*>(p);
}

void OGA_API_CALL OgaDestroyConfig(OgaConfig* p) {
  delete reinterpret_cast<Generators::Config*>(p);
}

void OGA_API_CALL OgaDestroyModel(OgaModel* p) {
  reinterpret_cast<Generators::Model*>(p)->external_owner_ = nullptr;
}

void OGA_API_CALL OgaDestroyGeneratorParams(OgaGeneratorParams* p) {
  reinterpret_cast<Generators::GeneratorParams*>(p)->external_owner_ = nullptr;
}

void OGA_API_CALL OgaDestroyGenerator(OgaGenerator* p) {
  delete reinterpret_cast<Generators::Generator*>(p);
}

void OGA_API_CALL OgaDestroyTokenizer(OgaTokenizer* p) {
  reinterpret_cast<Generators::Tokenizer*>(p)->external_owner_ = nullptr;
}

void OGA_API_CALL OgaDestroyTokenizerStream(OgaTokenizerStream* p) {
  delete reinterpret_cast<Generators::TokenizerStream*>(p);
}

void OGA_API_CALL OgaDestroyTensor(OgaTensor* p) {
  reinterpret_cast<Generators::Tensor*>(p)->external_owner_ = nullptr;
}

void OGA_API_CALL OgaDestroyMultiModalProcessor(OgaMultiModalProcessor* p) {
  reinterpret_cast<Generators::MultiModalProcessor*>(p)->external_owner_ = nullptr;
}

void OGA_API_CALL OgaDestroyImages(OgaImages* p) {
  delete reinterpret_cast<Generators::Images*>(p);
}

void OGA_API_CALL OgaDestroyAudios(OgaAudios* p) {
  delete reinterpret_cast<Generators::Audios*>(p);
}

void OGA_API_CALL OgaDestroyNamedTensors(OgaNamedTensors* p) {
  delete reinterpret_cast<Generators::NamedTensors*>(p);
}

void OGA_API_CALL OgaDestroyAdapters(OgaAdapters* p) {
  reinterpret_cast<Generators::Adapters*>(p)->external_owner_ = nullptr;
}

void OGA_API_CALL OgaDestroyRuntimeSettings(OgaRuntimeSettings* p) {
  delete reinterpret_cast<Generators::RuntimeSettings*>(p);
}

}  // extern "C"
