// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "resource_utils.h"

#include <sys/resource.h>

#include <cerrno>
#include <stdexcept>
#include <string>

namespace benchmark::utils {

size_t GetPeakWorkingSetSizeInBytes() {
  struct rusage rusage;
  if (getrusage(RUSAGE_SELF, &rusage) != 0) {
    throw std::runtime_error("getrusage failed with error code " + std::to_string(errno));
  }

#if defined(__APPLE__)
  constexpr size_t kBytesPerMaxRssUnit = 1;
#else
  constexpr size_t kBytesPerMaxRssUnit = 1024;
#endif

  return static_cast<size_t>(rusage.ru_maxrss) * kBytesPerMaxRssUnit;
}

}  // namespace benchmark::utils
