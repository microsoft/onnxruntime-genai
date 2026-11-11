// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "models/onnxruntime_api.h"  // for OrtHardwareDeviceType

namespace Generators {

using NamedString = std::pair<std::string, std::string>;

struct DeviceFilteringOptions {
  std::optional<OrtHardwareDeviceType> hardware_device_type;  // OrtHardwareDeviceType_CPU, OrtHardwareDeviceType_GPU, OrtHardwareDeviceType_NPU
  std::optional<uint32_t> hardware_device_id;
  std::optional<uint32_t> hardware_vendor_id;
};

struct ProviderOptions {
  std::string name;
  std::vector<NamedString> options;
  std::optional<DeviceFilteringOptions> device_filtering_options;
};

}  // namespace Generators
