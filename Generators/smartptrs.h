// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Generators {

template <typename T>
void copy(std::span<const T> source, std::span<T> dest) {
  assert(source.size() == dest.size());
  copy(source.begin(), source.end(), dest.begin());
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
cuda_unique_ptr<T> CudaMallocArray(size_t count, std::span<T>* p_span = nullptr) {
  T* p;
  ::cudaMalloc(&p, sizeof(T) * count);
  if (p_span)
    *p_span = std::span<T>(p, count);
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
cuda_host_unique_ptr<T> CudaMallocHostArray(size_t count) {
  T* p;
  ::cudaMallocHost(&p, sizeof(T) * count);
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

struct AllocatorDeleter {
  AllocatorDeleter() = default;
  explicit AllocatorDeleter(OrtAllocator* allocator)
      : allocator_(allocator) {}

  void operator()(void* p) const {
    if (allocator_)
      allocator_->Free(allocator_, p);
  }

 private:
  OrtAllocator* allocator_{};
};

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, AllocatorDeleter>;

template <typename TAlloc>
std::span<TAlloc> Allocate(OrtAllocator& allocator,
                           size_t size,
                           IAllocatorUniquePtr<TAlloc>& unique_ptr) {
  assert(!unique_ptr.get());  // Ensure pointer is empty, to avoid accidentally passing the wrong pointer and overwriting things
  unique_ptr = IAllocatorUniquePtr<TAlloc>(reinterpret_cast<TAlloc*>(allocator.Alloc(&allocator, size * sizeof(TAlloc))), AllocatorDeleter(&allocator));
  return std::span(unique_ptr.get(), size);
}

}  // namespace Generators
