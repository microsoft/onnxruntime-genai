// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <assert.h>
#include <memory>
#include "span.h"

namespace Generators {
struct Search;
struct GeneratorParams;

struct DeviceMemoryBase : std::enable_shared_from_this<DeviceMemoryBase> {
  virtual ~DeviceMemoryBase() {}
  virtual const char* GetType() const = 0;  // Returns "cuda" "cuda_cpu" "directml" etc
  virtual bool IsCpuAccessible() const = 0;
  virtual void GetOnCpu() = 0;  // Allocates p_cpu_ if necessary and copies p_device_ memory into it
  virtual void CopyFromDevice(size_t begin_dest, DeviceMemoryBase& source, size_t begin_source, size_t size_in_bytes) = 0;

  uint8_t* p_device_{};
  uint8_t* p_cpu_{};
  size_t size_in_bytes_{};
};

template <typename T>
struct DeviceMemorySpan;

template <typename T>
struct DeviceMemory : DeviceMemoryBase {
  size_t Size() const { return size_in_bytes_ / sizeof(T); }
  std::span<T> DeviceSpan() { return std::span<T>(reinterpret_cast<T*>(p_device_), Size()); }
  std::span<T> CpuSpan() {
    if (!p_cpu_) GetOnCpu();
    return std::span<T>(reinterpret_cast<T*>(p_cpu_), Size());
  }

  DeviceMemorySpan<T> subspan() { return DeviceMemorySpan<T>(*this, 0, Size()); }
  DeviceMemorySpan<T> subspan(size_t begin, size_t length) { return DeviceMemorySpan<T>(*this, begin, length); }
  DeviceMemorySpan<T> subspan_cpu(std::span<T> span) { return DeviceMemorySpan<T>(*this, span.data() - CpuSpan().data(), span.size()); }
  DeviceMemorySpan<T> subspan_device(std::span<T> span) { return DeviceMemorySpan<T>(*this, span.data() - DeviceSpan().data(), span.size()); }

 private:
  static void CheckSize() {
    // Ensure we're the same size as DeviceMemory so we can be cast to it
    // Has to be in a method because sizeof(TDeviceMemory) is not known outside of methods
    static_assert(sizeof(DeviceMemory) == sizeof(DeviceMemoryBase));
  }
};

// Many times we want to pass a subspan of device memory, this handles the memory differences
template <typename T>
struct DeviceMemorySpan {
  DeviceMemorySpan() = default;
  DeviceMemorySpan(DeviceMemory<T>& memory, size_t begin, size_t length)
      : p_device_memory_{std::static_pointer_cast<DeviceMemory<T>>(memory.shared_from_this())}, begin_{begin}, length_{length} {}

  size_t size() const { return length_; }
  std::span<T> CpuSpan() { return p_device_memory_->CpuSpan().subspan(begin_, length_); }

 private:
  std::shared_ptr<DeviceMemory<T>> p_device_memory_;
  size_t begin_, length_;  // Subspan of p_device_memory_, relative to original memory block
};

template <typename T>
void copy(DeviceMemorySpan<const T> source, DeviceMemorySpan<T> dest) {
  assert(source.size() == dest.size());
  dest.p_device_memory_->CopyFromDevice(dest.begin, *source.p_device_memory_, source.begin, source.size * sizeof(T));
}

struct DeviceInterface {
  virtual ~DeviceInterface() {}

  virtual std::shared_ptr<DeviceMemoryBase> AllocateBase(size_t size, bool cpu_accessible) = 0;
  template <typename T>
  std::shared_ptr<DeviceMemory<T>> Allocate(size_t count, bool cpu_accessible) {
    return std::static_pointer_cast<DeviceMemory<T>>(AllocateBase(sizeof(T) * count, cpu_accessible));
  }

  virtual std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) = 0;
  virtual std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) = 0;
};

DeviceInterface& GetCpuDeviceInterface();

namespace Location {
struct CPU {};
struct GPU {};
}  // namespace Location

template <typename T>
struct cpu_span : std::span<T> {
  using std::span<T>::span;
  explicit cpu_span(std::span<T> v) : std::span<T>(v) {}
};
template <typename T>
struct gpu_span : std::span<T> {
  using std::span<T>::span;
};

template <typename T>
void copy(std::span<const T> source, std::span<T> dest) {
  assert(source.size() == dest.size());
  std::copy(source.begin(), source.end(), dest.begin());
}

template <typename T>
void copy(cpu_span<const T> source, cpu_span<T> dest) {
  assert(source.size() == dest.size());
  std::copy(source.begin(), source.end(), dest.begin());
}

template <typename T>
std::unique_ptr<T[]> AllocateArray(size_t count, std::span<T>* p_span = nullptr) {
  T* p = new T[count];
  if (p_span)
    *p_span = std::span<T>(p, count);
  return std::unique_ptr<T[]>{p};
}

#if USE_CUDA
struct CudaDeleter {
  void operator()(void* p) {
    cudaFree(p);
  }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
cuda_unique_ptr<T> CudaMallocArray(size_t count, gpu_span<T>* p_span = nullptr) {
  T* p;
  ::cudaMalloc(&p, sizeof(T) * count);
  if (p_span)
    *p_span = gpu_span<T>(p, count);
  return cuda_unique_ptr<T>{p};
}

struct CudaHostDeleter {
  void operator()(void* p) {
    ::cudaFreeHost(p);
  }
};

template <typename T>
using cuda_host_unique_ptr = std::unique_ptr<T, CudaHostDeleter>;

template <typename T>
cuda_host_unique_ptr<T> CudaMallocHostArray(size_t count, cpu_span<T>* p_span = nullptr) {
  T* p;
  ::cudaMallocHost(&p, sizeof(T) * count);
  if (p_span)
    *p_span = cpu_span<T>(p, count);
  return cuda_host_unique_ptr<T>{p};
}

struct cuda_stream_holder {
  void Create() {
    assert(!v_);
    cudaStreamCreate(&v_);
  }

  ~cuda_stream_holder() {
    if (v_)
      (void)cudaStreamDestroy(v_);
  }

  operator cudaStream_t() const { return v_; }
  cudaStream_t get() const { return v_; }

 private:
  cudaStream_t v_{};
};
#else
struct cuda_stream_holder {
  void Create() {
    throw std::runtime_error("Trying to create a cuda stream in a non cuda build");
  }

  operator cudaStream_t() const { return v_; }
  cudaStream_t get() const { return v_; }

 private:
  cudaStream_t v_{};
};
#endif

#if USE_CUDA
// A roaming array is one that can be in CPU or GPU memory, and will copy the memory as needed to be used from anywhere
// It does not own the original memory, only the on-demand copy memory.
template <typename T>
struct RoamingArray {
  RoamingArray() = default;
  RoamingArray(const RoamingArray& v) { Assign(v); }

  bool empty() const { return cpu_.empty() && device_.empty(); }

  RoamingArray(cpu_span<T> v) {
    SetCPU(v);
  }

  RoamingArray(gpu_span<T> v) {
    SetGPU(v);
  }

  operator cpu_span<T>() { return GetCPU(); }
  operator gpu_span<T>() { return GetGPU(); }

  void SetCPU(cpu_span<T> cpu) {
    cpu_ = cpu;
    device_ = {};
  }

  void SetGPU(gpu_span<T> device) {
    device_ = device;
    cpu_ = {};
  }

  cpu_span<T> GetCPU() {
    if (cpu_.empty() && !device_.empty()) {
      cpu_owner_ = CudaMallocHostArray<T>(device_.size(), &cpu_);
      cudaMemcpy(cpu_.data(), device_.data(), cpu_.size_bytes(), cudaMemcpyDeviceToHost);
    }

    return cpu_;
  }

  gpu_span<T> GetGPU() {
    if (device_.empty() && !cpu_.empty()) {
      device_owner_ = CudaMallocArray<T>(cpu_.size(), &device_);
      cudaMemcpy(device_.data(), cpu_.data(), cpu_.size_bytes(), cudaMemcpyHostToDevice);
    }
    return device_;
  }

  void Assign(const RoamingArray<T>& v) {
    cpu_ = v.cpu_;
    device_ = v.device_;
  }

  cpu_span<T> cpu_;
  cuda_host_unique_ptr<T> cpu_owner_;
  gpu_span<T> device_;
  cuda_unique_ptr<T> device_owner_;
};
#else
// A roaming array is one that can be in CPU or GPU memory, and will copy the memory as needed to be used from anywhere
template <typename T>
struct RoamingArray {
  RoamingArray() = default;
  RoamingArray(const RoamingArray& v) { Assign(v); }

  RoamingArray(cpu_span<T> v) {
    SetCPU(v);
  }

  bool empty() const { return cpu_.empty(); }

  operator cpu_span<T>() { return GetCPU(); }

  void SetCPU(cpu_span<T> cpu) {
    cpu_ = cpu;
  }

  cpu_span<T> GetCPU() {
    return cpu_;
  }

  void Assign(const RoamingArray<T>& v) {
    cpu_ = v.cpu_;
  }

  cpu_span<T> cpu_;
};
#endif

}  // namespace Generators
