// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session_options.h"
#include "../models/session_options.h"

namespace Generators::WebGPUExecutionProvider {

// Always retrieves the WebGPU device interface so the caller can use it for
// device memory allocations, regardless of whether the EP is registered as a
// plugin (V2) or via the legacy (V1) path.
DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool /*disable_graph_capture*/) {
  auto device = GetDeviceInterface(DeviceType::WEBGPU);
  if (!AppendExecutionProviderV2(session_options, provider_options,
                                 DeviceType::WEBGPU, "WebGpuExecutionProvider")) {
    AppendExecutionProviderV1(session_options, provider_options);
  }

  return device;
}

}  // namespace Generators::WebGPUExecutionProvider
