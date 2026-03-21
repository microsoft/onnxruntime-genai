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

}  // namespace Generators
