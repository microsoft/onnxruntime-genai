// Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "session_options.h"
#include "interface.h"

namespace Generators::AMDGPUExecutionProvider {

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& /*config*/,
                                         bool /*disable_graph_capture*/) {
  auto* device = GetDeviceInterface(DeviceType::AMDGPU);
  static_cast<AMDGPUInterface*>(device)->SetupProvider(session_options, provider_options.options);

  return device;
}

}  // namespace Generators::AMDGPUExecutionProvider
