#include <cuda_runtime.h>

__global__ void Test(float* test) {
  *test = 1.2345f;
}

void LaunchTest(float *test, cudaStream_t stream) {

  Test<<<1, 1, 0, stream>>>(test);
}
