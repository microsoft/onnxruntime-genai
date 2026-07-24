// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cctype>

#include "session_options.h"
#include "../models/session_options.h"

#if USE_DML
#include "../dml/interface.h"
#endif
namespace Generators::DMLExecutionProvider {

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& /*config*/,
                                         bool disable_graph_capture) {
#if USE_DML
  if (!GetDmlInterface()) {
    LUID device_luid{};
    LUID* p_device_luid{};
    uint32_t device_index{};
    uint32_t* p_device_index{};
    for (const auto& [name, value] : provider_options.options) {
      if (name == "luid") {
        if (auto separator_position = value.find(":"); separator_position != std::string::npos) {
          device_luid.HighPart = std::stol(value.substr(0, separator_position));
          device_luid.LowPart = std::stol(value.substr(separator_position + 1));
          p_device_luid = &device_luid;
        }
      } else if (name == "device_index") {
        device_index = std::stoi(value);
        p_device_index = &device_index;
      }
    }

    InitDmlInterface(p_device_luid, p_device_index);
  }

  // Graph capture can also be opted out per model via the provider option
  // "enable_graph_capture": "0" — captured-command-list replay computes wrong
  // logits on some D3D12 devices (see IsGraphCaptureEnabled in config.cpp).
  bool graph_capture_opt_out = false;
  for (const auto& [name, value] : provider_options.options) {
    if (name == "enable_graph_capture") {
      std::string lower_value = value;
      std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(),
                     [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });
      if (lower_value == "0" || lower_value == "false") {
        graph_capture_opt_out = true;
      }
    }
  }

  // Non-decoder sessions (vision, speech, embedding) have control-flow nodes
  // that are incompatible with graph capture, so the caller sets
  // disable_graph_capture=true for those sessions.
  if (!disable_graph_capture && !graph_capture_opt_out) {
    session_options.AddConfigEntry("ep.dml.enable_graph_capture", "1");
  }

  SetDmlProvider(session_options);

  auto device = GetDeviceInterface(DeviceType::DML);  // We use a DML allocator for input/output caches, but other tensors will use CPU tensors
  return device;
#else
  throw std::runtime_error("DML provider requested, but the installed GenAI has not been built with DML support");
#endif
}

}  // namespace Generators::DMLExecutionProvider
