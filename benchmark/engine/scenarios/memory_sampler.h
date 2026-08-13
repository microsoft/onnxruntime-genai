// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace engine_benchmark {

/// Polls device and host memory usage on a background thread.
///
/// Device usage is read from NVML, which is loaded lazily so the benchmark still runs on machines
/// without an NVIDIA driver; usage attributed to this process is preferred and the device-wide
/// delta from the pre-load baseline is used when the driver does not report per-process numbers.
/// All values are 0 when no source is available.
class MemorySampler {
 public:
  explicit MemorySampler(std::chrono::milliseconds interval = std::chrono::milliseconds(100));
  ~MemorySampler();

  MemorySampler(const MemorySampler&) = delete;
  MemorySampler& operator=(const MemorySampler&) = delete;

  /// Records the pre-load baseline and starts sampling. Call before the model is created.
  void Start();
  void Stop();

  uint64_t PeakDeviceBytes() const;
  /// Mean of the trailing samples, i.e. usage once allocations have settled.
  uint64_t SteadyStateDeviceBytes() const;
  uint64_t PeakHostBytes() const;

 private:
  void Loop();

  std::chrono::milliseconds interval_;
  std::vector<uint64_t> samples_;
  uint64_t baseline_device_bytes_{0};
  bool running_{false};
  mutable std::mutex mutex_;
  std::condition_variable stop_signal_;
  std::thread thread_;
};

}  // namespace engine_benchmark
