// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <assert.h>
#include <functional>
#include <span>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>

#include "SafeInt.hpp"
#include "onnxruntime_cxx_api_2.h"
#include "debugging.h"

using ScoreType = float;

#if USE_CUDA
#include <cuda_runtime.h>

struct CudaDeleter {
  void operator()(void* p) {
    cudaFree(p);
  }
};

template<typename T>
void copy(std::span<const T> source, std::span<T> dest) {
  assert(source.size()==dest.size());
  copy(source.begin(), source.end(), dest.begin());
}

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
cuda_unique_ptr<T> CudaMallocArray(size_t count) {
  T* p;
  cudaMallocManaged(&p, sizeof(T) * count);
      //  cudaMalloc(&p, sizeof(T) * count);
  return cuda_unique_ptr<T>{p};
}

struct CudaHostDeleter {
  void operator()(void* p) {
    cudaFreeHost(p);
  }
};

template <typename T>
using cuda_host_unique_ptr = std::unique_ptr<T, CudaHostDeleter>;

template <typename T>
cuda_host_unique_ptr<T> CudaMallocHostArray(size_t count) {
  T* p;
  cudaMallocHost(&p, sizeof(T) * count);
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

#endif

struct Tensor;
struct Stream;
struct IConsoleDumper;

struct OpKernelContextInternal {};
struct SessionState {};
struct NodeArg {};
struct Node {};
struct TensorShapeProto {};

// Macros to disable the copy and/or move ctor and assignment methods
// These are usually placed in the private: declarations for a class.

#define ORT_DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete

#define ORT_DISALLOW_ASSIGNMENT(TypeName) TypeName& operator=(const TypeName&) = delete

#define ORT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName) \
  ORT_DISALLOW_COPY(TypeName);                     \
  ORT_DISALLOW_ASSIGNMENT(TypeName)

#define ORT_DISALLOW_MOVE(TypeName) \
  TypeName(TypeName&&) = delete;    \
  TypeName& operator=(TypeName&&) = delete

#define ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TypeName) \
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName);           \
  ORT_DISALLOW_MOVE(TypeName)

#define ORT_ENFORCE(condition, ...) assert(condition)

// TODO: Do we need this class or is IAllocator::MakeUniquePtr sufficient/better
struct  BufferDeleter {
  BufferDeleter() = default;
  explicit BufferDeleter(OrtAllocator* alloc)
      : alloc_(alloc) {}

  void operator()(void* p) const {
    if (alloc_)
      alloc_->Free(alloc_, p);
  }

 private:
  // TODO: we may need consider the lifetime of alloc carefully
  // The alloc_ here is the allocator that used to allocate the buffer
  // And need go with the unique_ptr together. If it is using our internal
  // allocator, it is ok as our allocators are global managed. But if it
  // is provide by user, user need to be very careful about it.
  // A weak_ptr may be a choice to reduce the impact, but that require to
  // change our current allocator mgr to use shared_ptr. Will revisit it
  // later.
  OrtAllocator* alloc_{};
};

using BufferUniquePtr = std::unique_ptr<void, BufferDeleter>;
using BufferNakedPtr = void*;

template <typename T>
std::span<T> AllocateBuffer(OrtAllocator* allocator,
                            BufferUniquePtr& buffer,
                            size_t elements,
                            bool fill = false,
                            T fill_value = T{}) {
  size_t bytes = SafeInt<size_t>(sizeof(T)) * elements;
  void* data = allocator->Alloc(allocator, bytes);
  BufferUniquePtr temp_buffer(data, BufferDeleter(allocator));
  buffer = std::move(temp_buffer);
  T* first = reinterpret_cast<T*>(buffer.get());
  auto span = std::span(first, elements);

  if (fill) {
    std::fill_n(first, elements, fill_value);
  }

  return span;
}

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

/** Allocate a unique_ptr using allocator_, and return a span to the allocated memory so usage is safe
@param allocator IAllocator to use for the allocation.
@param size Allocation size. Number of elements of type TAlloc, or total size if TAlloc is 'void'.
@param unique_ptr unique_ptr that will control the lifetime of the allocated memory.
@param fill If true, fill the allocated memory with fill_value.
@param fill_value Value to use if 'fill' is true.
@returns A span to provide bounds checked access to the allocated memory.
*/
template <typename TAlloc>
std::span<TAlloc> Allocate(OrtAllocator& allocator,
                           size_t size,
                           IAllocatorUniquePtr<TAlloc>& unique_ptr,
                           bool fill = false, TAlloc fill_value = TAlloc{}) {
  unique_ptr = IAllocatorUniquePtr<TAlloc>(reinterpret_cast<TAlloc*>(allocator.Alloc(&allocator, size * sizeof(TAlloc))), [&allocator](void *p) { allocator.Free(&allocator, p);} );
  auto span = std::span(unique_ptr.get(), size);

  if (fill) {
    std::fill_n(unique_ptr.get(), size, fill_value);
  }

  return span;
}

#include "Sequences.h"
#include "Search.h"
#include "gpt.h"
