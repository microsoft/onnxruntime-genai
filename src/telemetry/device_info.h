// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Cross-platform device information collection for telemetry.

#pragma once

#include <cstdint>
#include <string>

namespace Generators {

struct DeviceInfo {
  std::string device_id;         // Empty on Android/iOS so the 1DS SDK uses its platform device id.
  std::string os_architecture;   // "x64", "arm64"
  int processor_count;           // CPU core count
  int64_t total_memory_mb;       // System RAM in MB
  std::string cpu_model;         // Processor model string
  std::string process_name;      // Host executable name (basename only), e.g. "python.exe"
  std::string device_id_status;  // Provenance of device_id: "New"/"Existing"/"Corrupted"/"Failed"/"Platform"
};

// Collect device information. Thread-safe, results are cached after first call.
const DeviceInfo& GetDeviceInfo();

// Per-user telemetry storage directory. Empty when no per-user location is available.
std::string GetTelemetryStorageDir();

}  // namespace Generators
