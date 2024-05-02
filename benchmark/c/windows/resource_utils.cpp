// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "resource_utils.h"

#include <Windows.h>
#include <psapi.h>

#include <stdexcept>
#include <string>

namespace benchmark::utils {

size_t GetPeakWorkingSetSizeInBytes() {
  PROCESS_MEMORY_COUNTERS pmc;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    throw std::runtime_error("GetProcessMemoryInfo failed with error code " + std::to_string(GetLastError()));
  }

  return pmc.PeakWorkingSetSize;
}

}  // namespace benchmark::utils
