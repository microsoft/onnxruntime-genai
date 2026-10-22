// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "runtime_profiles.h"

#include "smartptrs.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace Generators {

namespace {

void QueryHostMemoryFacts(std::optional<uint64_t>& effective_host_bytes,
                          std::optional<uint64_t>& available_host_bytes) noexcept {
#if defined(_WIN32)
  MEMORYSTATUSEX stat{};
  stat.dwLength = sizeof(stat);
  if (GlobalMemoryStatusEx(&stat)) {
    effective_host_bytes = static_cast<uint64_t>(stat.ullTotalPhys);
    available_host_bytes = static_cast<uint64_t>(stat.ullAvailPhys);
  }
#else
  const long page_sz = sysconf(_SC_PAGESIZE);
  const long phys_pages = sysconf(_SC_PHYS_PAGES);
  const long avphys_pages = sysconf(_SC_AVPHYS_PAGES);
  if (page_sz > 0 && phys_pages > 0) {
    uint64_t total_phys = static_cast<uint64_t>(page_sz) * static_cast<uint64_t>(phys_pages);
    std::ifstream cgroup_max("/sys/fs/cgroup/memory.max");
    uint64_t cg_limit = 0;
    if (cgroup_max && (cgroup_max >> cg_limit) && cg_limit > 0 && cg_limit < total_phys) {
      total_phys = cg_limit;
    }
    effective_host_bytes = total_phys;
  }
  if (page_sz > 0 && avphys_pages > 0) {
    available_host_bytes = static_cast<uint64_t>(page_sz) * static_cast<uint64_t>(avphys_pages);
  }
#endif
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