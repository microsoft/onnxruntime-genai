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

  size_t available_device_memory_bytes{};
  size_t total_device_memory_bytes{};
  device.GetAvailableMemory(available_device_memory_bytes, total_device_memory_bytes);
  ApplyRuntimeProfile(config, total_device_memory_bytes);
  config.runtime_profiles.clear();
}

}  // namespace Generators