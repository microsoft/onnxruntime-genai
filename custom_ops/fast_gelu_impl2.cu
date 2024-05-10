// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_prop.cuh"
#include "utils.cuh"
#include "fast_gelu_impl2.cuh"

using namespace Ort::Custom;

constexpr float A = 0.5f;

constexpr float B = 0.7978845608028654f;  // sqrt(2.0/M_PI)

constexpr float C = 0.035677408136300125f;  // 0.044715 * sqrt(2.0/M_PI)

template <typename T, unsigned TPB>
__global__ void FastGeluKernel2(const T a, const T b, const T c, int input_length, int bias_length,
                               const T* input, const T* bias, T* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;

  if (idx < input_length) {
    const T x = input[idx];
    const T in = (bias == nullptr) ? x : (T)(x + bias[idx % bias_length]);
    const T cdf = a + a * _Tanh(in * (c * in * in + b));
    output[idx] = in * cdf;
  }
}

template <unsigned TPB>
__global__ void FastGeluKernel22(const half2 a, const half2 b, const half2 c, int input_length, int bias_length,
                                const half2* input, const half2* bias, half2* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < input_length) {
    const half2 x = input[idx];
    const half2 in = (bias == nullptr) ? x : (x + bias[idx % bias_length]);
    const half2 cdf = a + a * _Tanh(in * (c * in * in + b));
    output[idx] = in * cdf;
  }
}

template <>
cudaError_t LaunchFastGeluKernel2(cudaStream_t stream, int input_length, int bias_length,
                                 const float* input, const float* bias, float* output, bool /*use_half2*/) {
  constexpr int blockSize = 256;
  const int gridSize = (input_length + blockSize - 1) / blockSize;
  FastGeluKernel2<float, blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, C, input_length, bias_length,
                                                                       input, bias, output);

  return cudaGetLastError();
}

template <>
cudaError_t LaunchFastGeluKernel2(cudaStream_t stream, int input_length, int bias_length,
                                 const half* input, const half* bias, half* output, bool use_half2) {
  constexpr int blockSize = 256;
  if (use_half2 && 0 == (bias_length & 1) && DeviceProp::GetCapability() >= 7) {
    const int n = input_length / 2;
    const int gridSize = (n + blockSize - 1) / blockSize;
    const half2 A2 = __floats2half2_rn(A, A);
    const half2 B2 = __floats2half2_rn(B, B);
    const half2 C2 = __floats2half2_rn(C, C);
    const half2* input2 = reinterpret_cast<const half2*>(input);
    const half2* bias2 = reinterpret_cast<const half2*>(bias);
    half2* output2 = reinterpret_cast<half2*>(output);
    FastGeluKernel22<blockSize><<<gridSize, blockSize, 0, stream>>>(A2, B2, C2, n, bias_length / 2,
                                                                   input2, bias2, output2);
  } else {
    const int gridSize = (input_length + blockSize - 1) / blockSize;
    FastGeluKernel2<half, blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, C, input_length, bias_length,
                                                                        input, bias, output);
  }

  return cudaGetLastError();
}

template <>
cudaError_t LaunchFastGeluKernel2(cudaStream_t stream, int input_length, int bias_length,
                                 const BFloat16* input, const BFloat16* bias, BFloat16* output, bool /*use_half2*/) {
  constexpr int blockSize = 256;

  // remove nv_bfloat162 implementation for now to fix build issue
  // we can decide whether to add it back if there's perf concern
  const int gridSize = (input_length + blockSize - 1) / blockSize;
  FastGeluKernel2<BFloat16, blockSize>
      <<<gridSize, blockSize, 0, stream>>>(A, B, C, input_length, bias_length, input, bias, output);

  return cudaGetLastError();
}