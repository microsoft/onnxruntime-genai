// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include "config.h"

namespace Generators {

struct DeviceInterface;

void ApplyRuntimeProfileForSelectedDevice(Config& config, DeviceInterface& device);

// Host memory observations in bytes. A value the platform did not report is left empty.
struct HostMemoryReadings {
  std::optional<uint64_t> physical_total;
  std::optional<uint64_t> physical_available;  // includes reclaimable memory, e.g. MemAvailable on Linux
  std::optional<uint64_t> cgroup_limit;
  std::optional<uint64_t> cgroup_usage;
};

// Derives the host memory facts used for runtime profile eligibility. The effective memory is the physical memory
// capped by the cgroup limit. The available memory is capped by the cgroup headroom and never exceeds the effective
// memory.
void ResolveHostMemoryFacts(const HostMemoryReadings& readings, std::optional<uint64_t>& effective_host_bytes,
                            std::optional<uint64_t>& available_host_bytes) noexcept;

// Returns the value of `key` (for example "MemAvailable") from /proc/meminfo formatted text, in bytes.
std::optional<uint64_t> ParseMeminfoBytes(std::string_view meminfo, std::string_view key);

}  // namespace Generators