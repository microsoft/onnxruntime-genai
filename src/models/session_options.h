// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../generators.h"

namespace Generators {

// Filters a list of EP devices according to the device_filtering_options specified
// in provider_options (hardware_device_id, hardware_vendor_id, hardware_device_type).
// Returns the full list unchanged if no filtering criteria are set.
// Throws std::runtime_error if criteria are set but no devices match.
std::vector<const OrtEpDevice*> ApplyDeviceFiltering(const Config::ProviderOptions& provider_options,
                                                     const std::vector<const OrtEpDevice*>& devices);

// Returns all OrtEpDevice instances whose EP name matches |ep_name|.
// Returns an empty vector if the provider is not registered as a plugin.
std::vector<const OrtEpDevice*> FindRegisteredEpDevices(const std::string& ep_name);

// Attempts to append an execution provider via the V2 plugin API.
// Discovers registered EP devices for |ep_name|, applies device filtering, and
// calls AppendExecutionProvider_V2. Returns true on success, false if the
// provider is not registered (caller should fall back to V1).
bool AppendExecutionProviderV2(OrtSessionOptions& session_options,
                               const Config::ProviderOptions& provider_options,
                               DeviceType device_type,
                               const std::string& ep_name);

// Appends an execution provider using the legacy V1 API (key/value string pairs).
void AppendExecutionProviderV1(OrtSessionOptions& session_options,
                               const Config::ProviderOptions& provider_options);

// Iterates over the requested providers, dispatches to provider-specific
// AppendExecutionProvider implementations, and returns the DeviceInterface
// for the first provider that supplies one (or nullptr if none do).
DeviceInterface* SetProviderSessionOptions(OrtSessionOptions& session_options,
                                           const std::vector<std::string>& providers,
                                           const std::vector<Config::ProviderOptions>& provider_options_list,
                                           bool is_primary_session_options,
                                           const Config& config,
                                           bool disable_graph_capture = false);

// Gets a trivial ONNX model that just returns a single float constant.
inline auto GetTrivialModel() {
  static const auto trivial_model = std::array<uint8_t, 96>{
      0x08, 0x0a, 0x12, 0x01, 0x61, 0x3a, 0x53, 0x0a, 0x38, 0x12, 0x06, 0x76, 0x61, 0x6c, 0x75, 0x65,
      0x73, 0x22, 0x08, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x2a, 0x24, 0x0a, 0x05, 0x76,
      0x61, 0x6c, 0x75, 0x65, 0x2a, 0x18, 0x08, 0x01, 0x10, 0x01, 0x42, 0x0c, 0x63, 0x6f, 0x6e, 0x73,
      0x74, 0x5f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x4a, 0x04, 0x00, 0x00, 0x00, 0x00, 0xa0, 0x01,
      0x04, 0x12, 0x01, 0x62, 0x62, 0x14, 0x0a, 0x06, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x73, 0x12, 0x0a,
      0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x15};

  return std::span<const uint8_t>{trivial_model};
}

}  // namespace Generators
