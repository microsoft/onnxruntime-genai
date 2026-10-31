// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "config.h"

namespace Generators {

struct DeviceInterface;

void ApplyRuntimeProfileForSelectedDevice(Config& config, DeviceInterface& device);

// The memory bounds that the cgroup v2 hierarchy puts on the calling process.
struct CgroupMemoryLimits {
  enum class Status {
    kNotApplicable,  // no cgroup v2 hierarchy to read (cgroup v1 or no cgroup2 mount): the physical readings stand
    kResolved,       // the process's own cgroup was read; an empty bound means every level is unlimited
    kUnresolved,     // cgroup v2 applies but the process's cgroup could not be read or located
  };
  Status status{Status::kNotApplicable};
  std::optional<uint64_t> limit;     // the tightest memory.max from the process's cgroup up to the cgroup mount root
  std::optional<uint64_t> headroom;  // the tightest memory.max - memory.current over the same cgroups
};

// Host memory observations in bytes. A value the platform did not report is left empty.
struct HostMemoryReadings {
  std::optional<uint64_t> physical_total;
  std::optional<uint64_t> physical_available;  // includes reclaimable memory, e.g. MemAvailable on Linux
  CgroupMemoryLimits cgroup;
};

// Derives the host memory facts used for runtime profile eligibility. The effective memory is the physical memory
// capped by the cgroup limit. The available memory is the physical availability capped by the cgroup headroom, and
// never exceeds the effective memory; it stays empty when the physical availability was not reported, because the
// cgroup headroom says nothing about how much memory the host has free. A zero cgroup limit is a hard limit. Both facts
// are left empty when the process's cgroup is unresolved, because the host-wide values would then overstate what the
// process may use.
void ResolveHostMemoryFacts(const HostMemoryReadings& readings, std::optional<uint64_t>& effective_host_bytes,
                            std::optional<uint64_t>& available_host_bytes) noexcept;

// Returns the text of the file at `path`, or nothing when it does not exist or cannot be read.
using HostFileReader = std::function<std::optional<std::string>(const std::string& path)>;

// Resolves the cgroup v2 memory bounds of the calling process. The process's own cgroup comes from /proc/self/cgroup
// and is located through the cgroup2 mount in /proc/self/mountinfo, whose root may be a sub-cgroup (a container that
// sees only its own cgroup). The limit and headroom are the tightest over that cgroup and each ancestor up to the
// mount root; a level whose memory.max is "max" or that has no memory controller does not bound the process. A level
// that has the memory controller (it is listed in its cgroup.controllers) but whose memory.max cannot be read leaves
// the result unresolved, as the hierarchy root is the only cgroup that has the controller yet no memory.max. The
// files are read through `read_file` so that the resolution does not depend on the machine it runs on.
CgroupMemoryLimits ResolveProcessCgroupMemory(const HostFileReader& read_file);

// Returns the value of `key` (for example "MemAvailable") from /proc/meminfo formatted text, in bytes.
std::optional<uint64_t> ParseMeminfoBytes(std::string_view meminfo, std::string_view key);

}  // namespace Generators