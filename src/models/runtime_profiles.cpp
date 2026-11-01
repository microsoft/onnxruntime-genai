// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "runtime_profiles.h"

#include "smartptrs.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace Generators {

std::optional<uint64_t> ParseMeminfoBytes(std::string_view meminfo, std::string_view key) {
  while (!meminfo.empty()) {
    const size_t end = meminfo.find('\n');
    const std::string_view line = meminfo.substr(0, end);
    meminfo = end == std::string_view::npos ? std::string_view{} : meminfo.substr(end + 1);
    if (line.size() <= key.size() || line.compare(0, key.size(), key) != 0 || line[key.size()] != ':') {
      continue;
    }
    std::istringstream fields{std::string{line.substr(key.size() + 1)}};
    uint64_t value = 0;
    std::string unit;
    if (!(fields >> value)) return std::nullopt;
    fields >> unit;
    if (unit.empty()) return value;
    if (unit != "kB" || value > std::numeric_limits<uint64_t>::max() / 1024) return std::nullopt;
    return value * 1024;
  }
  return std::nullopt;
}

void ResolveHostMemoryFacts(const HostMemoryReadings& readings, std::optional<uint64_t>& effective_host_bytes,
                            std::optional<uint64_t>& available_host_bytes) noexcept {
  const bool has_cgroup_limit = readings.cgroup_limit && *readings.cgroup_limit > 0;

  effective_host_bytes = readings.physical_total;
  if (has_cgroup_limit && (!effective_host_bytes || *readings.cgroup_limit < *effective_host_bytes)) {
    effective_host_bytes = *readings.cgroup_limit;
  }

  available_host_bytes = readings.physical_available;
  if (has_cgroup_limit && readings.cgroup_usage) {
    const uint64_t headroom =
        *readings.cgroup_limit > *readings.cgroup_usage ? *readings.cgroup_limit - *readings.cgroup_usage : 0;
    available_host_bytes = available_host_bytes ? std::min(*available_host_bytes, headroom) : headroom;
  }
  if (available_host_bytes && effective_host_bytes) {
    available_host_bytes = std::min(*available_host_bytes, *effective_host_bytes);
  }
}

namespace {

#if defined(__linux__)
std::optional<uint64_t> ReadCgroupBytes(const char* path) {
  std::ifstream file(path);
  uint64_t value = 0;
  if (file && (file >> value)) return value;
  return std::nullopt;  // memory.max holds "max" when the cgroup has no limit
}

std::optional<uint64_t> ReadMeminfoBytes(std::string_view key) {
  std::ifstream file("/proc/meminfo");
  if (!file) return std::nullopt;
  std::ostringstream contents;
  contents << file.rdbuf();
  return ParseMeminfoBytes(contents.str(), key);
}
#endif

void QueryHostMemoryFacts(std::optional<uint64_t>& effective_host_bytes,
                          std::optional<uint64_t>& available_host_bytes) noexcept {
  HostMemoryReadings readings;
#if defined(_WIN32)
  MEMORYSTATUSEX stat{};
  stat.dwLength = sizeof(stat);
  if (GlobalMemoryStatusEx(&stat)) {
    readings.physical_total = static_cast<uint64_t>(stat.ullTotalPhys);
    readings.physical_available = static_cast<uint64_t>(stat.ullAvailPhys);
  }
#else
  const long page_sz = sysconf(_SC_PAGESIZE);
  const long phys_pages = sysconf(_SC_PHYS_PAGES);
  if (page_sz > 0 && phys_pages > 0) {
    readings.physical_total = static_cast<uint64_t>(page_sz) * static_cast<uint64_t>(phys_pages);
  }
#if defined(__linux__)
  readings.physical_available = ReadMeminfoBytes("MemAvailable");
  readings.cgroup_limit = ReadCgroupBytes("/sys/fs/cgroup/memory.max");
  readings.cgroup_usage = ReadCgroupBytes("/sys/fs/cgroup/memory.current");
#endif
#endif
  ResolveHostMemoryFacts(readings, effective_host_bytes, available_host_bytes);
}

}  // namespace

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
  std::optional<bool> is_integrated;
  if (std::any_of(config.runtime_profiles.begin(), config.runtime_profiles.end(), [](const auto& profile) {
        return profile.eligibility.is_integrated.has_value();
      })) {
    std::string error;
    is_integrated = device.GetIsIntegrated(error);
    if (!is_integrated) {
      std::cerr << "Warning: runtime profile integrated-device detection failed: " << error
                << "; using base configuration\n";
      config.runtime_profiles.clear();
      return;
    }
  }
  std::optional<uint64_t> effective_host_bytes;
  std::optional<uint64_t> available_host_bytes;
  QueryHostMemoryFacts(effective_host_bytes, available_host_bytes);
  ApplyRuntimeProfile(config, RuntimeProfileDeviceFacts{
                                  total_device_memory_bytes,
                                  is_integrated,
                                  available_device_memory_bytes,
                                  effective_host_bytes,
                                  available_host_bytes});
  config.runtime_profiles.clear();
}

}  // namespace Generators