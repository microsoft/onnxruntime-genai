// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session_options.h"
#include "models/session_options.h"

namespace Generators::WebGPUExecutionProvider {

// Always retrieves the WebGPU device interface so the caller can use it for
// device memory allocations, regardless of whether the EP is registered as a
// plugin (V2) or via the legacy (V1) path.
DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool disable_graph_capture) {
  auto device = GetDeviceInterface(DeviceType::WEBGPU);
  auto session_provider_options = provider_options;
  // Graph capture applies to the decoder only. Auxiliary embedding and vision sessions pass
  // disable_graph_capture because they are not fully partitioned to WebGPU.
  if (disable_graph_capture) {
    auto graph_capture_option = std::find_if(
        session_provider_options.options.begin(), session_provider_options.options.end(),
        [](const auto& option) { return option.first == "enableGraphCapture"; });
    if (graph_capture_option == session_provider_options.options.end()) {
      session_provider_options.options.emplace_back("enableGraphCapture", "0");
    } else {
      graph_capture_option->second = "0";
    }
  }

  if (!AppendExecutionProviderV2(session_options, session_provider_options,
                                 DeviceType::WEBGPU, "WebGpuExecutionProvider")) {
    AppendExecutionProviderV1(session_options, session_provider_options);
  }

  return device;
}

}  // namespace Generators::WebGPUExecutionProvider
