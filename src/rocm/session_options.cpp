// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session_options.h"
#include "../models/session_options.h"

namespace Generators::ROCmExecutionProvider {

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& /*config*/,
                                         bool /*disable_graph_capture*/) {
  OrtROCMProviderOptions ort_provider_options;

  std::vector<const char*> keys, values;
  for (auto& option : provider_options.options) {
    keys.emplace_back(option.first.c_str());
    values.emplace_back(option.second.c_str());
  }

  Ort::ThrowOnError(Ort::api->UpdateROCMProviderOptions(&ort_provider_options, keys.data(), values.data(), keys.size()));
  session_options.AppendExecutionProvider_ROCM(ort_provider_options);

  return nullptr;
}

}  // namespace Generators::ROCmExecutionProvider
