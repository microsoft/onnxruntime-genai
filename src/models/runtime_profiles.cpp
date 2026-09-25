// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "runtime_profiles.h"

#include "smartptrs.h"

namespace Generators {

void ApplyRuntimeProfileForSelectedDevice(Config& config, DeviceInterface& device) {
  if (config.runtime_profiles.empty()) return;
  if (device.GetType() != DeviceType::CUDA) {
    throw std::runtime_error(
        "runtime_profiles require CUDA to be the primary execution provider");
  }
  for (const auto& provider : config.model.decoder.session_options.provider_options) {
    if (provider.name != "cuda") continue;
    if (provider.device_filtering_options && provider.device_filtering_options->hardware_device_id) {
      throw std::runtime_error(
          "runtime_profiles do not support CUDA hardware_device_id filtering; use CUDA_VISIBLE_DEVICES instead");
    }
    for (const auto& [name, value] : provider.options) {
      if (name == "device_id" && value != "0") {
        throw std::runtime_error(
            "runtime_profiles require CUDA device_id 0; use CUDA_VISIBLE_DEVICES instead");
      }
    }
  }
  if (device.GetDeviceId(nullptr) != 0) {
    throw std::runtime_error(
        "runtime_profiles require current CUDA device 0; use CUDA_VISIBLE_DEVICES instead");
  }

  size_t available_device_memory_bytes{};
  size_t total_device_memory_bytes{};
  device.GetAvailableMemory(available_device_memory_bytes, total_device_memory_bytes);
  ApplyRuntimeProfile(config, total_device_memory_bytes);
  config.runtime_profiles.clear();
}

}  // namespace Generators