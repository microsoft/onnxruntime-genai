// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cpu_utils.h"

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

  return static_cast<size_t>(rusage.ru_maxrss * 1024L);
}

}
