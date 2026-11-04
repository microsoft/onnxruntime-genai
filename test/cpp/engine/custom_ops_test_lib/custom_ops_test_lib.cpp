// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "onnxruntime_c_api.h"

#if defined(_WIN32)
#define ORTGENAI_TEST_CUSTOM_OPS_EXPORT __declspec(dllexport)
#else
#define ORTGENAI_TEST_CUSTOM_OPS_EXPORT __attribute__((visibility("default")))
#endif

extern "C" ORTGENAI_TEST_CUSTOM_OPS_EXPORT OrtStatus* ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  constexpr uint32_t kMinOrtApiVersion = 18;
  const OrtApi* api = api_base ? api_base->GetApi(kMinOrtApiVersion) : nullptr;
  if (!api) {
    return nullptr;
  }

  for (size_t index = 0;; ++index) {
    const std::string key = "genai.test.custom_ops_loaded." + std::to_string(index);
    int has_entry = 0;
    if (OrtStatus* status = api->HasSessionConfigEntry(options, key.c_str(), &has_entry)) {
      return status;
    }
    if (!has_entry) {
      return api->AddSessionConfigEntry(options, key.c_str(), ORTGENAI_TEST_CUSTOM_OPS_TAG);
    }
  }
}
