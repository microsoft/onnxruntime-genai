// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "device.h"

#include <stdexcept>

namespace Generators {

const char* DeviceTypeToName(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::CUDA:
      return "CUDA";
    case DeviceType::DML:
      return "DirectML";
    case DeviceType::WEBGPU:
      return "WebGpu";
    case DeviceType::QNN:
      return "QnnWithSharedMemory";
    default:
      throw std::runtime_error("Unknown device type");
  }
}
}  // namespace Generators
