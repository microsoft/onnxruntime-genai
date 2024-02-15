// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "ort_genai_c.h"
#include <memory>
#include <onnxruntime_c_api.h>
#include <exception>
#include <cstdint>
#include <cstddef>
#include "generators.h"
#include "models/model.h"
#include "search.h"

namespace Generators {

std::unique_ptr<OrtEnv> g_ort_env;

OrtEnv& GetOrtEnv() {
  if (!g_ort_env) {
    g_ort_env = OrtEnv::Create();
  }
  return *g_ort_env;
}

}  // namespace Generators

using Sequences = std::vector<std::vector<int32_t>>;

extern "C" {

#define OGA_TRY try {
#define OGA_CATCH                   \
  }                                 \
  catch (const std::exception& e) { \
    return new OgaResult{e.what()}; \
  }

struct OgaResult {
  explicit OgaResult(const char* what) : what_{what} {}
  std::string what_;
};

const char* OGA_API_CALL OgaResultGetError(OgaResult* result) {
  return result->what_.c_str();
}

struct OgaBuffer {
  std::unique_ptr<uint8_t[]> data_;
  std::vector<size_t> dims_;
  OgaDataType type_;
};

OgaDataType OGA_API_CALL OgaBufferGetType(const OgaBuffer* p) {
  return p->type_;
}

size_t OGA_API_CALL OgaBufferGetDimCount(const OgaBuffer* p) {
  return p->dims_.size();
}

OgaResult* OGA_API_CALL OgaBufferGetDims(const OgaBuffer* p, size_t* dims, size_t dim_count) {
  OGA_TRY
  if (dim_count != p->dims_.size())
    throw std::runtime_error("OgaBufferGetDims - passed in buffer size does not match dim count");

  std::copy(p->dims_.begin(), p->dims_.end(), dims);
  return nullptr;
  OGA_CATCH
}

const void* OGA_API_CALL OgaBufferGetData(const OgaBuffer* p) {
  return p->data_.get();
}

size_t OGA_API_CALL OgaSequencesCount(const OgaSequences* p) {
  return reinterpret_cast<const Sequences*>(p)->size();
}

size_t OGA_API_CALL OgaSequencesGetSequenceCount(const OgaSequences* p, size_t sequence) {
  return (*reinterpret_cast<const Sequences*>(p))[sequence].size();
}

const int32_t* OGA_API_CALL OgaSequencesGetSequenceData(const OgaSequences* p, size_t sequence) {
  return (*reinterpret_cast<const Sequences*>(p))[sequence].data();
}

OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaDeviceType device_type, OgaModel** out) {
  OGA_TRY
  auto provider_options = Generators::GetDefaultProviderOptions(static_cast<Generators::DeviceType>(device_type));
  *out = reinterpret_cast<OgaModel*>(Generators::CreateModel(Generators::GetOrtEnv(), config_path, &provider_options).release());
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** out) {
  OGA_TRY
  *out = reinterpret_cast<OgaGeneratorParams*>(new Generators::GeneratorParams(*reinterpret_cast<const Generators::Model*>(model)));
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGeneratorParamsSetMaxLength(OgaGeneratorParams* params, int max_length) {
  reinterpret_cast<Generators::GeneratorParams*>(params)->max_length = static_cast<int>(max_length);
  return nullptr;
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

OgaResult* OGA_API_CALL OgaGenerate(const OgaModel* model, OgaGeneratorParams* generator_params, OgaSequences** out) {
  OGA_TRY
  Sequences result = Generators::Generate(*reinterpret_cast<const Generators::Model*>(model), *reinterpret_cast<const Generators::GeneratorParams*>(generator_params));
  *out = reinterpret_cast<OgaSequences*>(std::make_unique<Sequences>(std::move(result)).release());
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

OgaResult* OGA_API_CALL OgaGenerator_GenerateNextToken_Top(OgaGenerator* generator) {
  OGA_TRY
  reinterpret_cast<Generators::Generator*>(generator)->GenerateNextToken_Top();
  return nullptr;
  OGA_CATCH
}

OgaResult* OGA_API_CALL OgaGenerator_GetSequence(const OgaGenerator* oga_generator, int index, int32_t* tokens, size_t* count) {
  OGA_TRY
  auto& generator = *reinterpret_cast<const Generators::Generator*>(oga_generator);
  auto sequence = generator.GetSequence(index);
  auto sequence_cpu = sequence.GetCPU();
  *count = sequence_cpu.size();
  if (tokens)
    std::copy(sequence_cpu.begin(), sequence_cpu.end(), tokens);
  return nullptr;
  OGA_CATCH
}

void OGA_API_CALL OgaDestroyResult(OgaResult* p) {
  delete p;
}

void OGA_API_CALL OgaDestroyBuffer(OgaBuffer* p) {
  delete p;
}

void OGA_API_CALL OgaDestroySequences(OgaSequences* p) {
  delete reinterpret_cast<Sequences*>(p);
}

void OGA_API_CALL OgaDestroyModel(OgaModel* p) {
  delete reinterpret_cast<Generators::Model*>(p);
}

void OGA_API_CALL OgaDestroyGeneratorParams(OgaGeneratorParams* p) {
  delete reinterpret_cast<Generators::GeneratorParams*>(p);
}

void OGA_API_CALL OgaDestroyGenerator(OgaGenerator* p) {
  delete reinterpret_cast<Generators::Generator*>(p);
}
}
