// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

namespace benchmark::utils {

size_t GetPeakWorkingSetSizeInBytes();

#ifdef _WIN32
struct GpuMemoryInfo {
  uint64_t dedicated;  // VRAM (DXGI_MEMORY_SEGMENT_GROUP_LOCAL)
  uint64_t shared;     // System RAM used by GPU (DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)
  uint64_t Total() const { return dedicated + shared; }
};

GpuMemoryInfo GetGpuMemoryUsage();
#endif

}  // namespace benchmark::utils
