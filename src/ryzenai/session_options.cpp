// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session_options.h"
#include "interface.h"

namespace Generators::RyzenAIExecutionProvider {

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool /*disable_graph_capture*/) {
  auto device = GetDeviceInterface(DeviceType::RyzenAI);
  session_options.AddConfigEntry("model_root", config.config_path.string().c_str());
  GetRyzenAIInterface()->SetupProvider(session_options, provider_options.options);

  return device;
}

}  // namespace Generators::RyzenAIExecutionProvider
