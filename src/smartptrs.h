// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <assert.h>
#include <memory>
#include "span.h"

namespace Generators {
struct Search;
struct Sequences;
struct GeneratorParams;

// A DeviceBuffer is an abstract interface to a block of device memory (can be cuda/dml/cpu memory)
// Note: For a CPU DeviceBuffer, there's only one block of memory on CPU, the copy methods are no-ops
// Do not use DeviceBuffer directly, use a DeviceSpan (the Allocate/WrapMemory methods return DeviceSpans)
struct DeviceBuffer : std::enable_shared_from_this<DeviceBuffer> {
  virtual ~DeviceBuffer() {}
  virtual const char* GetType() const = 0;  // Returns "cuda" "cuda_cpu" "directml" etc

  virtual void AllocateCpu() = 0;      // Allocates p_cpu_ if necessary (using appropriate memory type for interop)
  virtual void CopyDeviceToCpu() = 0;  // Allocates p_cpu_ if necessary and copies p_device_ memory into it
  virtual void CopyCpuToDevice() = 0;
  virtual void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) = 0;

  uint8_t* p_device_{};
  uint8_t* p_cpu_{};
  size_t size_in_bytes_{};
};

// A DeviceSpan is how a DeviceBuffer is used. It can be thought of as a std::span for device memory with
// utilities to interop with CPU memory. It is what Allocate<T> returns and what should be passed around by value.
template <typename T>
struct DeviceSpan {
  DeviceSpan() = default;
  DeviceSpan(std::shared_ptr<DeviceBuffer>&& memory)
      : p_device_memory_{std::move(memory)}, begin_{}, length_{p_device_memory_->size_in_bytes_ / sizeof(T)} {}

  bool empty() const { return length_ == 0; }
  size_t size() const { return length_; }

  DeviceSpan<T> subspan(size_t begin, size_t length) { return DeviceSpan<T>(*p_device_memory_, begin_ + begin, length); }

  // Return the device accessible memory. Should only be done in device specific code, as it's not CPU accessible
  std::span<T> Span() { return std::span<T>{reinterpret_cast<T*>(p_device_memory_->p_device_) + begin_, length_}; }

  // Return the CPU accessible memory, allocating if necessary (note, to get the current device memory on CPU, use 'CopyDeviceToCpu' instead)
  std::span<T> CpuSpan() {
    p_device_memory_->AllocateCpu();
    return std::span<T>{reinterpret_cast<T*>(p_device_memory_->p_cpu_) + begin_, length_};
  }

  // Copy device memory to CPU memory and return the CPU accessible memory
  std::span<T> CopyDeviceToCpu() {
    p_device_memory_->CopyDeviceToCpu();
    return std::span<T>{reinterpret_cast<T*>(p_device_memory_->p_cpu_) + begin_, length_};
  }

  // Copy CPU memory to device memory, typically used after calling CpuSpan or CopyDeviceToCpu to update the device memory with the modifications made
  void CopyCpuToDevice() { p_device_memory_->CopyCpuToDevice(); }

 private:
  DeviceSpan(DeviceBuffer& memory, size_t begin, size_t length)
      : p_device_memory_{memory.shared_from_this()}, begin_{begin}, length_{length} {}

  std::shared_ptr<DeviceBuffer> p_device_memory_;
  size_t begin_{}, length_{};  // Subspan of p_device_memory_, relative to original memory block
};

struct DeviceInterface {
  virtual ~DeviceInterface() {}

  template <typename T>
  DeviceSpan<T> Allocate(size_t count, bool cpu_accessible = false) { return DeviceSpan<T>(AllocateBase(sizeof(T) * count, cpu_accessible)); }
  virtual std::shared_ptr<DeviceBuffer> AllocateBase(size_t size, bool cpu_accessible) = 0;

  // Wraps an existing memory block, useful for tensors. Use WrapTensor for OrtValue vs calling this directly
  template <typename T>
  DeviceSpan<T> WrapMemory(std::span<T> memory) { return DeviceSpan<T>(WrapMemoryBase(memory.data(), memory.size_bytes())); }
  virtual std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* memory, size_t size) = 0;

  virtual std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) = 0;
  virtual std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) = 0;

  virtual void Synchronize() = 0;  // Synchronize the device, typically used for timing or debugging

  virtual cudaStream_t GetCudaStream() {
    assert(false);
    return nullptr;
  }  // Temporary until we fully factor out providers
};

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
  explicit gpu_span(std::span<T> v) : std::span<T>(v) {}
};

template <typename T>
void copy(std::span<const T> source, std::span<T> dest) {
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

}  // namespace Generators
