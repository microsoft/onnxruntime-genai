// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Cross-platform device information collection for telemetry.
// Provides privacy-safe device fingerprinting (hashed IDs, no PII).

#pragma once

#include <string>

namespace Generators {

struct DeviceInfo {
  std::string device_id;       // "c:" + stable hash of a locally-generated UUID (no hardware id)
  std::string os;              // "Windows", "Linux", "macOS"
  std::string os_version;      // OS version string
  std::string os_architecture; // "x64", "arm64"
  int processor_count;         // CPU core count
  int total_memory_mb;         // System RAM in MB
  std::string cpu_model;       // Processor model string
};

// Collect device information. Thread-safe, results are cached after first call.
const DeviceInfo& GetDeviceInfo();

}  // namespace Generators
