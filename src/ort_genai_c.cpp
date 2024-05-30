// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <memory>
#include <onnxruntime_c_api.h>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include "span.h"
#include "ort_genai_c.h"
#include "generators.h"
#include "models/model.h"
#include "search.h"

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
  *images = reinterpret_cast<OgaImages*>(Generators::LoadImageImpl(image_path).release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaModel** out) {
  OGA_TRY
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), config_path);
  model->external_owner_ = model;
  *out = reinterpret_cast<OgaModel*>(model.get());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** out) {
  OGA_TRY
  auto params = std::make_shared<Generators::GeneratorParams>(*reinterpret_cast<const Generators::Model*>(model));
  params->external_owner_ = params;
  *out = reinterpret_cast<OgaGeneratorParams*>(params.get());
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

OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputIDs(OgaGeneratorParams* oga_params, const int32_t* input_ids, size_t input_ids_count, size_t sequence_length, size_t batch_size) {
  OGA_TRY
  auto& params = *reinterpret_cast<Generators::GeneratorParams*>(oga_params);
  params.input_ids = std::span<const int32_t>(input_ids, input_ids_count);
  params.sequence_length = static_cast<int>(sequence_length);
  params.batch_size = static_cast<int>(batch_size);
  if (params.sequence_length * params.batch_size != input_ids_count)
    throw std::runtime_error("sequence length * batch size is not equal to input_ids_count");
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsSetInputSequences(OgaGeneratorParams* oga_params, const OgaSequences* p_sequences) {
  OGA_TRY
  auto& params = *reinterpret_cast<Generators::GeneratorParams*>(oga_params);
  auto& sequences = *reinterpret_cast<const Generators::TokenSequences*>(p_sequences);

  std::vector<std::span<const int32_t>> span_sequences;
  for (size_t i = 0; i < sequences.size(); i++) {
    span_sequences.emplace_back(sequences[i]);
  }

  params.input_ids_owner = Generators::PadInputs(span_sequences, params.pad_token_id);
  params.batch_size = static_cast<int>(sequences.size());
  params.sequence_length = static_cast<int>(params.input_ids_owner.size() / params.batch_size);
  params.input_ids = params.input_ids_owner;
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

OgaResult* OGA_API_CALL OgaGenerate(const OgaModel* model, const OgaGeneratorParams* generator_params, OgaSequences** out) {
  OGA_TRY
  auto result = Generators::Generate(*reinterpret_cast<const Generators::Model*>(model), *reinterpret_cast<const Generators::GeneratorParams*>(generator_params));
  *out = reinterpret_cast<OgaSequences*>(std::make_unique<Generators::TokenSequences>(std::move(result)).release());
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

OgaResult* OGA_API_CALL OgaGenerator_ComputeLogits(OgaGenerator* generator) {
  OGA_TRY
  reinterpret_cast<Generators::Generator*>(generator)->ComputeLogits();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken(OgaGenerator* generator) {
  OGA_TRY
  reinterpret_cast<Generators::Generator*>(generator)->GenerateNextToken();
  return nullptr;
  OGA_CATCH
}

size_t OGA_API_CALL OgaGenerator_GetSequenceCount(const OgaGenerator* oga_generator, size_t index) {
  auto& generator = *reinterpret_cast<const Generators::Generator*>(oga_generator);
  return generator.GetSequence(static_cast<int>(index)).GetCPU().size();
}

const int32_t* OGA_API_CALL OgaGenerator_GetSequenceData(const OgaGenerator* oga_generator, size_t index) {
  auto& generator = *reinterpret_cast<const Generators::Generator*>(oga_generator);
  return generator.GetSequence(static_cast<int>(index)).GetCPU().data();
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

OgaResult* OGA_API_CALL OgaTokenizerDecode(const OgaTokenizer* p, const int32_t* tokens, size_t token_count, const char** out_string) {
  OGA_TRY
  auto& tokenizer = *reinterpret_cast<const Generators::Tokenizer*>(p);

  auto string = tokenizer.Decode({tokens, token_count});
  auto length = string.length() + 1;
  auto cstr_buffer = std::make_unique<char[]>(length);
#if _MSC_VER
  strcpy_s(cstr_buffer.get(), length, string.c_str());
#else
  strncpy(cstr_buffer.get(), string.c_str(), length);
  cstr_buffer[length] = 0;
#endif
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
#if _MSC_VER
  strcpy_s(cstr_buffer.get(), length, string.c_str());
#else
  strncpy(cstr_buffer.get(), string.c_str(), length);
  cstr_buffer[length] = 0;
#endif
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
  auto named_tensors = processor.image_processor_->Process(*processor.tokenizer_, prompt, images);
  *input_tensors = reinterpret_cast<OgaNamedTensors*>(named_tensors.release());
  return nullptr;
  OGA_CATCH
}

void OGA_API_CALL OgaDestroyResult(OgaResult* p) {
  delete reinterpret_cast<Generators::Result*>(p);
}

void OGA_API_CALL OgaDestroyString(const char* p) {
  delete p;
}

void OGA_API_CALL OgaDestroySequences(OgaSequences* p) {
  delete reinterpret_cast<Generators::Sequences*>(p);
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

void OGA_API_CALL OgaDestroyNamedTensors(OgaNamedTensors* p) {
  delete reinterpret_cast<Generators::NamedTensors*>(p);
}
}
