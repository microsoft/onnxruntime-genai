// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <stdexcept>
#include <string>
#include <sstream>
#include <memory>
#include <cassert>

#include <cuda_runtime.h>
#include "span.h"

namespace Generators {

cudaStream_t GetStream();

// Device the NvTensorRtRtx interface is bound to, or -1 while unbound. Defined inline so that targets
// using these helpers without the add-on's interface.cpp still link.
inline std::atomic<int> g_bound_device_id{-1};

// Device of the process-wide stream, or -1 while there is no stream. Writers hold g_stream_mutex (interface.cpp) and
// store the id with release semantics only after the stream exists.
inline std::atomic<int> g_stream_device_id{-1};

// Makes the device of the shared stream current on the calling thread: the device the NvTensorRtRtx interface is bound
// to, otherwise the device the stream was created on. CUDA's current device is per thread, so a thread that did not
// bind the interface would otherwise allocate or create events on another device than the shared stream's.
inline void UseBoundDevice() noexcept {
  int device_id = g_bound_device_id.load(std::memory_order_acquire);
  if (device_id < 0)
    device_id = g_stream_device_id.load(std::memory_order_acquire);
  if (device_id < 0)
    return;
  int current_device_id{};
  // Errors are left to the CUDA call that follows, as callers include noexcept paths.
  if (::cudaGetDevice(&current_device_id) != cudaSuccess || current_device_id != device_id)
    (void)::cudaSetDevice(device_id);
}

#define CeilDiv(a, b) ((a + (b - 1)) / b)

class CudaError : public std::runtime_error {
 public:
  explicit CudaError(const std::string& msg, cudaError_t code)
      : std::runtime_error(msg), code_(code) {}

  cudaError_t code() const noexcept { return code_; }

 private:
  cudaError_t code_;
};

#define CUDA_CHECK(call)                                         \
  do {                                                           \
    cudaError_t err = (call);                                    \
    if (err != cudaSuccess) {                                    \
      std::stringstream ss;                                      \
      ss << "CUDA error in " << __func__ << " at " << __FILE__   \
         << ":" << __LINE__ << " - " << cudaGetErrorString(err); \
      (void)cudaGetLastError();                                  \
      throw Generators::CudaError(ss.str(), err);                \
    }                                                            \
  } while (0)

#define CUDA_CHECK_LAUNCH()                               \
  do {                                                    \
    cudaError_t err = cudaPeekAtLastError();              \
    if (err != cudaSuccess) {                             \
      std::stringstream ss;                               \
      ss << "CUDA launch error in " << __func__ << " at " \
         << __FILE__ << ":" << __LINE__ << " - "          \
         << cudaGetErrorString(err);                      \
      (void)cudaGetLastError();                           \
      throw Generators::CudaError(ss.str(), err);         \
    }                                                     \
  } while (0)

struct cuda_event_holder {
  cuda_event_holder() {
    UseBoundDevice();
    CUDA_CHECK(cudaEventCreate(&v_));
  }

  cuda_event_holder(unsigned flags) {
    UseBoundDevice();
    CUDA_CHECK(cudaEventCreateWithFlags(&v_, flags));
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
    CUDA_CHECK(cudaStreamCreate(&v_));
  }

  // Replaces the stream with one created on the current device. The old stream is kept if creation fails.
  void Recreate() {
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (v_)
      (void)cudaStreamDestroy(v_);
    v_ = stream;
  }

  // Releases the stream, if there is one; Create() may be called again afterwards.
  void Destroy() noexcept {
    if (v_) {
      (void)cudaStreamDestroy(v_);
      v_ = nullptr;
    }
  }

  bool has_stream() const noexcept { return v_ != nullptr; }

  ~cuda_stream_holder() { Destroy(); }

  operator cudaStream_t() const {
    assert(v_);
    return v_;
  }
  cudaStream_t get() const {
    assert(v_);
    return v_;
  }

 private:
  cudaStream_t v_{};
};

struct CudaHostDeleter {
  void operator()(void* p) {
    ::cudaFreeHost(p);
  }
};

template <typename T>
using cuda_host_unique_ptr = std::unique_ptr<T, CudaHostDeleter>;

template <typename T>
cuda_host_unique_ptr<T> CudaMallocHostArray(size_t count, std::span<T>* p_span = nullptr) {
  T* p;
  UseBoundDevice();
  CUDA_CHECK(::cudaMallocHost(&p, sizeof(T) * count));
  if (p_span)
    *p_span = std::span<T>(p, count);
  return cuda_host_unique_ptr<T>{p};
}

struct CudaDeleter {
  void operator()(void* p) {
    cudaFree(p);
  }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
cuda_unique_ptr<T> CudaMallocArray(size_t count, std::span<T>* p_span = nullptr) {
  T* p;
  UseBoundDevice();
  CUDA_CHECK(::cudaMalloc(&p, sizeof(T) * count));
  if (p_span)
    *p_span = std::span<T>(p, count);
  return cuda_unique_ptr<T>{p};
}

}  // namespace Generators
