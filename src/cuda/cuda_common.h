namespace Generators {

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

}  // namespace Generators
