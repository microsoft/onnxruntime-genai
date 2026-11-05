// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <chrono>
#include <filesystem>
#include <system_error>
#include <thread>

namespace Generators::test {

// The threads of this process, or -1 where the platform has no cheap way to count them (only Linux does).
inline int ProcessThreadCount() {
#if defined(__linux__)
  std::error_code error;
  std::filesystem::directory_iterator it{"/proc/self/task", error};
  int count = 0;
  for (; !error && it != std::filesystem::directory_iterator{}; it.increment(error)) ++count;
  return error ? -1 : count;
#else
  return -1;
#endif
}

// Waits for the process to be down to `baseline` threads or fewer: the kernel reaps a thread a moment after it has been joined.
// True when it is, and always where the threads cannot be counted (`baseline` is -1).
inline bool ProcessThreadCountSettlesAtOrBelow(int baseline) {
  if (baseline < 0) return true;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (ProcessThreadCount() > baseline) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

}  // namespace Generators::test
