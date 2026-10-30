// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include "../span.h"
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Generators {

inline constexpr size_t kMaxHardwareInventoryEntries = 64;

struct HardwareDeviceInfo {
  std::string type;
  std::string name{"unknown"};
  std::string vendor{"unknown"};
  std::string driver_version{"unknown"};
  std::string source;
  int64_t vendor_id{-1};
  int64_t device_id{-1};
  int64_t dedicated_memory_mb{-1};
  int64_t shared_memory_mb{-1};
  int64_t free_memory_mb{-1};
  int64_t memory_budget_mb{-1};
  int64_t memory_budget_available_mb{-1};
  int64_t physical_core_count{-1};
  int64_t logical_core_count{-1};
  int64_t nominal_clock_mhz{-1};
  int64_t max_clock_mhz{-1};
  int64_t memory_clock_mhz{-1};
  int64_t memory_bus_width_bits{-1};
};

struct HardwareInfo {
  int64_t available_system_memory_mb{-1};
  bool inventory_truncated{};
  std::vector<HardwareDeviceInfo> devices;
};

struct ProcessResourceUsage {
  int64_t resident_memory_mb{-1};
  int64_t cpu_time_ms{-1};
  int64_t gpu_memory_mb{-1};
  int64_t npu_memory_mb{-1};
  std::string_view accelerator_memory_source{"unavailable"};
};

HardwareInfo CollectHardwareInfo() noexcept;
ProcessResourceUsage CollectProcessResourceUsage() noexcept;
int64_t ReadProcessCpuTimeMs() noexcept;

inline int64_t ProcessCpuTimeDelta(int64_t start, int64_t end) {
  return start >= 0 && end >= start ? end - start : -1;
}

namespace TelemetryInternal {

int64_t ParseHardwareInteger(std::string_view value, int base = 10);
HardwareDeviceInfo ParseCpuInfo(std::string_view cpu_info);

class ResourceSampleWindow {
 public:
  void Begin(ProcessResourceUsage sample) {
    start_ = std::move(sample);
    end_ = {};
    complete_ = false;
    started_ = true;
  }

  void Complete(ProcessResourceUsage sample) {
    if (!complete_) {
      end_ = std::move(sample);
      complete_ = true;
    }
  }

  void Reset() {
    start_ = {};
    end_ = {};
    complete_ = false;
    started_ = false;
  }

  bool IsComplete() const {
    return complete_;
  }

  bool IsStarted() const {
    return started_;
  }

  const ProcessResourceUsage& Start() const {
    return start_;
  }

  const ProcessResourceUsage& End() const {
    return end_;
  }

 private:
  ProcessResourceUsage start_;
  ProcessResourceUsage end_;
  bool started_{};
  bool complete_{};
};

HardwareDeviceInfo* FindOrtHardwareDeviceMatch(
    std::span<HardwareDeviceInfo> candidates,
    const HardwareDeviceInfo& ort_device,
    std::span<const HardwareDeviceInfo> ort_devices);

bool AccumulateProcessMemoryBytes(uint64_t& total, uint64_t used);

inline bool ShouldQueryNvmlProcessMemory(bool is_container, bool is_wsl,
                                         uint64_t pid_namespace_inode) {
  // Linux UAPI linux/nsfs.h reserves PID_NS_INIT_INO for the initial PID namespace.
  constexpr uint64_t kInitialPidNamespaceInode = 0xeffffffc;
  return !is_container && !is_wsl && pid_namespace_inode == kInitialPidNamespaceInode;
}

inline bool ShouldBeginResourceSample(int64_t prompt_tokens,
                                      bool first_token_logged,
                                      bool sample_started,
                                      bool session_sampled) {
  return session_sampled && prompt_tokens == 0 && !first_token_logged && !sample_started;
}

}  // namespace TelemetryInternal
}  // namespace Generators
