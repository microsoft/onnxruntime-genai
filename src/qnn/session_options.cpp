// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session_options.h"
#include "../models/session_options.h"

namespace Generators::QNNExecutionProvider {

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool /*disable_graph_capture*/) {
  DeviceInterface* device = nullptr;
  session_options.AddConfigEntry("ep.share_ep_contexts", "1");
  if (const auto opt_it = std::find_if(
          provider_options.options.begin(), provider_options.options.end(),
          [](const auto& pair) { return pair.first == "enable_htp_shared_memory_allocator"; });
      opt_it != provider_options.options.end() && opt_it->second == "1") {
    device = GetDeviceInterface(DeviceType::QNN);
  }
  // is_primary_session_options is set to false because the device is set based on
  // the presence of the "enable_htp_shared_memory_allocator" option,
  // not based on whether this is the primary session options or not.
  if (!AppendExecutionProviderV2(session_options, provider_options,
                                 DeviceType::QNN, "QNNExecutionProvider")) {
    AppendExecutionProviderV1(session_options, provider_options);
  }

  return device;
}

}  // namespace Generators::QNNExecutionProvider
