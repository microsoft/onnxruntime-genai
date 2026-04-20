// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "session_options.h"

#include "../models/session_options.h"

namespace Generators::AMDGPUExecutionProvider {

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool /*disable_graph_capture*/) {
  // AMDGPU does not have a device type specific allocator in OGA yet,
  // so we use CPU as the device type. The AMDGPU EP handles CPU<->GPU
  // transfers internally.
  // Note: ORT recognizes "MIGraphXExecutionProvider" — do not change these strings.
  if (!AppendExecutionProviderV2(session_options, provider_options,
                                 DeviceType::CPU, "MIGraphXExecutionProvider")) {
    // V1 fallback: ORT recognizes "MIGraphX", not "AMDGPU"
    std::vector<const char*> keys, values;
    for (auto& option : provider_options.options) {
      keys.emplace_back(option.first.c_str());
      values.emplace_back(option.second.c_str());
    }
    session_options.AppendExecutionProvider("MIGraphX", keys.data(), values.data(), keys.size());
  }

  return nullptr;
}

}  // namespace Generators::AMDGPUExecutionProvider
