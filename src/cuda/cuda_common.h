// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <memory>
#include <cassert>

#include <cuda_runtime.h>
#include "span.h"

namespace Generators {

cudaStream_t GetStream();

void OnCudaError(cudaError_t error);

struct CudaCheck {
  void operator==(cudaError_t error) {
    if (error != cudaSuccess)
      OnCudaError(error);
  }
};

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
  ::cudaMallocHost(&p, sizeof(T) * count);
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
  ::cudaMalloc(&p, sizeof(T) * count);
  if (p_span)
    *p_span = std::span<T>(p, count);
  return cuda_unique_ptr<T>{p};
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
      throw Generators::CudaError(ss.str(), err);                \
    }                                                            \
  } while (0)

#ifdef NDEBUG
#define CUDA_CHECK_LAUNCH()                               \
  do {                                                    \
    cudaError_t err = cudaPeekAtLastError();              \
    if (err != cudaSuccess) {                             \
      std::stringstream ss;                               \
      ss << "CUDA launch error in " << __func__ << " at " \
         << __FILE__ << ":" << __LINE__ << " - "          \
         << cudaGetErrorString(err);                      \
      throw Generators::CudaError(ss.str(), err);         \
    }                                                     \
  } while (0)
#else
#define CUDA_CHECK_LAUNCH()                               \
  do {                                                    \
    cudaError_t err = cudaGetLastError();                 \
    if (err != cudaSuccess) {                             \
      std::stringstream ss;                               \
      ss << "CUDA launch error in " << __func__ << " at " \
         << __FILE__ << ":" << __LINE__ << " - "          \
         << cudaGetErrorString(err);                      \
      throw Generators::CudaError(ss.str(), err);         \
    }                                                     \
  } while (0)
#endif

}  // namespace Generators
