namespace Generators {

void OnCudaError(cudaError_t error);

struct CudaCheck {
  void operator==(cudaError_t error) {
    if (error!=cudaSuccess)
      OnCudaError(error);
  }
};

}
