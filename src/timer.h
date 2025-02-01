// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <chrono>

namespace Generators {

using Clock = std::chrono::steady_clock;

class Timer {
 public:
  Clock::duration Elapsed() const {
    return Clock::now() - start_;
  }

  template <typename DurationType>
  DurationType Elapsed() const {
    return std::chrono::duration_cast<DurationType>(Elapsed());
  }

  void Reset() {
    start_ = Clock::now();
  }

 private:
  Clock::time_point start_{Clock::now()};
};

}  // namespace Generators

#include <string_view>
#include "logging.h"

namespace Generators {

inline void LogTiming(std::string_view label, const Clock::duration& timing) {
  if (g_log.enabled) {
    const auto elapsed_usec = std::chrono::duration_cast<std::chrono::microseconds>(timing).count();
    Log("performance", MakeString(label, " time: ", elapsed_usec, " usec"));
  }
}

}  // namespace Generators
