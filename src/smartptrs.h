// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Generators {

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
cuda_host_unique_ptr<T> CudaMallocHostArray(size_t count, std::span<T>* p_span=nullptr) {
  T* p;
  ::cudaMallocHost(&p, sizeof(T) * count);
  if (p_span)
    *p_span = std::span<T>(p, count);
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



}  // namespace Generators
