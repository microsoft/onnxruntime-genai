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

struct DeviceBuffer : std::enable_shared_from_this<DeviceBuffer> {
  virtual ~DeviceBuffer() {}
  virtual const char* GetType() const = 0;  // Returns "cuda" "cuda_cpu" "directml" etc

  bool IsCpuAccessible() const { return p_device_ == p_cpu_; }  // Device memory is CPU accessible if it's the same memory
  virtual void AllocateCpu() = 0;                               // Allocates p_cpu_ if necessary (using appropriate memory type for interop)
  virtual void CopyDeviceToCpu() = 0;                           // Allocates p_cpu_ if necessary and copies p_device_ memory into it
  virtual void CopyCpuToDevice() = 0;
  virtual void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) = 0;

  uint8_t* p_device_{};
  uint8_t* p_cpu_{};
  size_t size_in_bytes_{};
};

// Many times we want to pass a subspan of device memory, this handles the memory differences
template <typename T>
struct DeviceSpan {
  DeviceSpan() = default;
  DeviceSpan(std::shared_ptr<DeviceBuffer>&& memory)
      : p_device_memory_{std::move(memory)}, begin_{}, length_{p_device_memory_->size_in_bytes_ / sizeof(T)} {}
  DeviceSpan(DeviceBuffer& memory, size_t begin, size_t length)
      : p_device_memory_{memory.shared_from_this()}, begin_{begin}, length_{length} {}

  bool empty() const { return length_ == 0; }
  size_t size() const { return length_; }

  std::span<T> Span() { return std::span<T>{reinterpret_cast<T*>(p_device_memory_->p_device_) + begin_, length_}; }
  std::span<T> CpuSpan() {
    p_device_memory_->AllocateCpu();
    return std::span<T>{reinterpret_cast<T*>(p_device_memory_->p_cpu_) + begin_, length_};
  }
  std::span<T> CopyDeviceToCpu() {
    p_device_memory_->CopyDeviceToCpu();
    return std::span<T>{reinterpret_cast<T*>(p_device_memory_->p_cpu_) + begin_, length_};
  }
  void CopyCpuToDevice() { p_device_memory_->CopyCpuToDevice(); }

  DeviceSpan<T> subspan(size_t begin, size_t length) { return DeviceSpan<T>(*p_device_memory_, begin_ + begin, length); }
  DeviceSpan<T> subspan_device(std::span<T> span) { return DeviceSpan<T>(*p_device_memory_, span.data() - reinterpret_cast<T*>(p_device_memory_->p_device_), span.size()); }

  DeviceBuffer& GetDeviceMemory() { return *p_device_memory_; }

 private:
  std::shared_ptr<DeviceBuffer> p_device_memory_;
  size_t begin_{}, length_{};  // Subspan of p_device_memory_, relative to original memory block
};

#if 0
template <typename T>
void copy(DeviceSpan<const T> source, DeviceSpan<T> dest) {
  assert(source.size() == dest.size());
  dest.p_device_memory_->CopyFromDevice(dest.begin, *source.p_device_memory_, source.begin, source.size * sizeof(T));
}
#endif

struct DeviceInterface {
  virtual ~DeviceInterface() {}

  template <typename T>
  DeviceSpan<T> Allocate(size_t count, bool cpu_accessible) { return DeviceSpan<T>(AllocateBase(sizeof(T) * count, cpu_accessible)); }
  virtual std::shared_ptr<DeviceBuffer> AllocateBase(size_t size, bool cpu_accessible) = 0;

  // Wraps an existing memory block, useful for tensors. Use WrapTensor for OrtValue vs calling this directly
  template <typename T>
  DeviceSpan<T> WrapMemory(std::span<T> memory) { return DeviceSpan<T>(WrapMemoryBase(memory.data(), memory.size_bytes())); }
  virtual std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* memory, size_t size) = 0;

  virtual std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) = 0;
  virtual std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) = 0;

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
