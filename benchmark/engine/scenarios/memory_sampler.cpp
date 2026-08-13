// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scenarios/memory_sampler.h"

#include <algorithm>

#if defined(__linux__)
#include <dlfcn.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace engine_benchmark {
namespace {

#if defined(__linux__)

// Minimal subset of the NVML ABI so the benchmark does not need the CUDA toolkit headers.
using nvmlDevice_t = void*;

struct nvmlMemory_t {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

struct nvmlProcessInfo_t {
  unsigned int pid;
  unsigned long long used_gpu_memory;
  unsigned int gpu_instance_id;
  unsigned int compute_instance_id;
};

constexpr int kNvmlSuccess = 0;
constexpr unsigned long long kNvmlValueNotAvailable = ~0ULL;

class Nvml {
 public:
  static const Nvml& Instance() {
    static const Nvml instance;
    return instance;
  }

  bool available() const { return handle_ != nullptr; }

  /// Bytes used on all devices by this process, or the device-wide total when the driver does not
  /// attribute usage per process.
  uint64_t UsedBytes(bool* per_process) const {
    *per_process = false;
    if (!available()) {
      return 0;
    }

    unsigned int device_count = 0;
    if (device_count_(&device_count) != kNvmlSuccess) {
      return 0;
    }

    const auto pid = static_cast<unsigned int>(getpid());
    uint64_t process_bytes = 0;
    uint64_t device_bytes = 0;

    for (unsigned int i = 0; i < device_count; ++i) {
      nvmlDevice_t device = nullptr;
      if (device_handle_(i, &device) != kNvmlSuccess) {
        continue;
      }

      if (running_processes_ != nullptr) {
        unsigned int count = 64;
        nvmlProcessInfo_t infos[64] = {};
        if (running_processes_(device, &count, infos) == kNvmlSuccess) {
          for (unsigned int p = 0; p < count; ++p) {
            if (infos[p].pid == pid && infos[p].used_gpu_memory != kNvmlValueNotAvailable) {
              process_bytes += infos[p].used_gpu_memory;
              *per_process = true;
            }
          }
        }
      }

      nvmlMemory_t memory = {};
      if (device_memory_ != nullptr && device_memory_(device, &memory) == kNvmlSuccess) {
        device_bytes += memory.used;
      }
    }

    return *per_process ? process_bytes : device_bytes;
  }

 private:
  Nvml() {
    handle_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (handle_ == nullptr) {
      return;
    }

    auto* init = reinterpret_cast<int (*)()>(dlsym(handle_, "nvmlInit_v2"));
    device_count_ = reinterpret_cast<int (*)(unsigned int*)>(dlsym(handle_, "nvmlDeviceGetCount_v2"));
    device_handle_ =
        reinterpret_cast<int (*)(unsigned int, nvmlDevice_t*)>(dlsym(handle_, "nvmlDeviceGetHandleByIndex_v2"));
    device_memory_ = reinterpret_cast<int (*)(nvmlDevice_t, nvmlMemory_t*)>(dlsym(handle_, "nvmlDeviceGetMemoryInfo"));
    running_processes_ = reinterpret_cast<int (*)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*)>(
        dlsym(handle_, "nvmlDeviceGetComputeRunningProcesses_v3"));
    shutdown_ = reinterpret_cast<int (*)()>(dlsym(handle_, "nvmlShutdown"));

    if (init == nullptr || device_count_ == nullptr || device_handle_ == nullptr || init() != kNvmlSuccess) {
      dlclose(handle_);
      handle_ = nullptr;
    }
  }

  ~Nvml() {
    if (handle_ != nullptr) {
      if (shutdown_ != nullptr) {
        shutdown_();
      }
      dlclose(handle_);
    }
  }

  void* handle_{nullptr};
  int (*device_count_)(unsigned int*){nullptr};
  int (*device_handle_)(unsigned int, nvmlDevice_t*){nullptr};
  int (*device_memory_)(nvmlDevice_t, nvmlMemory_t*){nullptr};
  int (*running_processes_)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*){nullptr};
  int (*shutdown_)(){nullptr};
};

#endif  // defined(__linux__)

uint64_t ReadDeviceBytes(bool* per_process) {
#if defined(__linux__)
  return Nvml::Instance().UsedBytes(per_process);
#else
  *per_process = false;
  return 0;
#endif
}

uint64_t ReadPeakHostBytes() {
#if defined(__linux__)
  rusage usage = {};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0;
  }
  return static_cast<uint64_t>(usage.ru_maxrss) * 1024;  // ru_maxrss is in kilobytes on Linux.
#else
  return 0;
#endif
}

}  // namespace

MemorySampler::MemorySampler(std::chrono::milliseconds interval) : interval_(interval) {}

MemorySampler::~MemorySampler() {
  Stop();
}

void MemorySampler::Start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) {
    return;
  }

  bool per_process = false;
  baseline_device_bytes_ = ReadDeviceBytes(&per_process);
  samples_.clear();
  running_ = true;
  thread_ = std::thread(&MemorySampler::Loop, this);
}

void MemorySampler::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
      return;
    }
    running_ = false;
  }

  stop_signal_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void MemorySampler::Loop() {
  std::unique_lock<std::mutex> lock(mutex_);
  do {
    bool per_process = false;
    const uint64_t used = ReadDeviceBytes(&per_process);
    // Without per-process attribution the reading includes other tenants, so report the growth
    // since the pre-load baseline instead.
    samples_.push_back(per_process ? used : (used > baseline_device_bytes_ ? used - baseline_device_bytes_ : 0));
  } while (!stop_signal_.wait_for(lock, interval_, [this] { return !running_; }));
}

uint64_t MemorySampler::PeakDeviceBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (samples_.empty()) {
    return 0;
  }
  return *std::max_element(samples_.begin(), samples_.end());
}

uint64_t MemorySampler::SteadyStateDeviceBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (samples_.empty()) {
    return 0;
  }

  const size_t tail = std::min<size_t>(10, samples_.size());
  uint64_t sum = 0;
  for (size_t i = samples_.size() - tail; i < samples_.size(); ++i) {
    sum += samples_[i];
  }
  return sum / tail;
}

uint64_t MemorySampler::PeakHostBytes() const {
  return ReadPeakHostBytes();
}

}  // namespace engine_benchmark
