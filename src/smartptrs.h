// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <assert.h>
#include <memory>
#include "span.h"

#if USE_DML
#include "models/dml/dml_readback_heap.h"
#endif

namespace Generators {

template <typename... T>
void Unreferenced(const T&...) {}

namespace Location {
struct CPU {};
struct GPU {};
}  // namespace Location

template <typename T>
struct cpu_span : std::span<T> {
  using std::span<T>::span;
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

struct cuda_event_holder {
  cuda_event_holder() {
    cudaEventCreate(&v_);
  }

  cuda_event_holder(unsigned flags) {
    cudaEventCreateWithFlags(&v_, flags);
  }

  ~cuda_event_holder() {
    if (v_)
      (void)cudaEventDestroy(v_);
  }

  operator cudaEvent_t() { return v_; }

 private:
  cudaEvent_t v_{};
};

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
    assert(false);
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

  void FlushCPUChanges() {
    if (!device_.empty())
      cudaMemcpy(device_.data(), cpu_.data(), cpu_.size_bytes(), cudaMemcpyHostToDevice);
  }

  void FlushGPUChanges() {
    if (!cpu_.empty())
      cudaMemcpy(cpu_.data(), device_.data(), cpu_.size_bytes(), cudaMemcpyDeviceToHost);
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
#elif USE_DML

template <typename T>
struct RoamingArray {
  RoamingArray() = default;
  RoamingArray(const RoamingArray& v) { Assign(v); }

  RoamingArray(DmlReadbackHeap* readback_heap, ID3D12Resource* gpu_resource, size_t element_count)
      : readback_heap_(readback_heap),
        gpu_resource_(gpu_resource),
        element_count_(element_count) {
  }

  RoamingArray(cpu_span<T> v) {
    SetCPU(v);
  }

  bool empty() const { return cpu_data_.empty() && !gpu_resource_; }

  operator cpu_span<T>() { return GetCPU(); }

  void SetCPU(cpu_span<T> cpu_data) {
    cpu_data_ = cpu_data;
  }

  void SetGPU(ID3D12Resource* gpu_resource) {
    gpu_resource_ = gpu_resource;
  }

  cpu_span<T> GetCPU() {
    if (cpu_data_.empty() && gpu_resource_) {
      owned_cpu_data_.resize(element_count_);

      readback_heap_->ReadbackFromGpu(
          std::span(reinterpret_cast<uint8_t*>(owned_cpu_data_.data()), element_count_ * sizeof(T)),
          gpu_resource_.Get(),
          0,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

      cpu_data_ = cpu_span<T>{owned_cpu_data_.data(), element_count_};
    }

    return cpu_data_;
  }

  ComPtr<ID3D12Resource> GetGPU() {
    return gpu_resource_;
  }

  void Assign(const RoamingArray<T>& v) {
    cpu_data_ = v.cpu_data_;
    gpu_resource_ = v.gpu_resource_;
    readback_heap_ = v.readback_heap_;
    element_count_ = v.element_count_;
  }

  std::vector<T> owned_cpu_data_;
  cpu_span<T> cpu_data_;
  DmlReadbackHeap* readback_heap_;
  ComPtr<ID3D12Resource> gpu_resource_;
  size_t element_count_;
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
