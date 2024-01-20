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

namespace Generators {

std::unique_ptr<OrtEnv> g_ort_env;

OrtEnv& GetOrtEnv() {
  if (!g_ort_env) {
    g_ort_env = OrtEnv::Create();
}
  return *g_ort_env;
}

}  // namespace Generators

extern "C" {

struct OgaResult {
  explicit OgaResult(const char *what){}
 // TODO: implement this constructor !!!!
};

OgaResult* OgaCreateModel(const char* config_path, OgaDeviceType device_type, OgaModel** out) {
  try {
    auto provider_options = Generators::GetDefaultProviderOptions(static_cast<Generators::DeviceType>(device_type));
    *out=reinterpret_cast<OgaModel*>(Generators::CreateModel(Generators::GetOrtEnv(), config_path, &provider_options).release());
    return nullptr;
  } catch (const std::exception& e) {
    return new OgaResult { e.what()};
  }
}

void OgaDestroyModel(OgaModel* model) {
  delete reinterpret_cast<Generators::Model*>(model);
}

OgaResult* OgaCreateState(OgaModel* model, int32_t* sequence_lengths, size_t sequence_lengths_count, const OgaSearchParams* search_params, OgaState** out) {
  try {
    *out = reinterpret_cast<OgaState*>(reinterpret_cast<Generators::Model*>(model)->CreateState(Generators::cpu_span<int32_t>{sequence_lengths, sequence_lengths_count}, *reinterpret_cast<const Generators::SearchParams*>(search_params)).release());
    return nullptr;
  } catch (const std::exception& e) {
    return new OgaResult{e.what()};
  }
}

void OgaDestroyState(OgaState* state) {
  delete reinterpret_cast<Generators::State*>(state);
}

}
