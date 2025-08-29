// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include "span.h"
#include "beam_search_topk.h"
#include "cuda_sampling.cuh"
#include "models/onnxruntime_api.h"
#include "smartptrs.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <limits>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

// Robust CUDA error checking macro
#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t err = call;                                 \
    if (err != cudaSuccess) {                               \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n",          \
              cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  } while (0)

namespace Generators {
namespace cuda {

constexpr int kMaxThreads = 1024;
constexpr int kGPUWarpSize = 32;

__global__ void InitCurandStates(unsigned long long seed, curandState* states, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= batch_size)
    return;

  curand_init(seed, index, 0, &states[index]);
}

SamplingData::SamplingData(unsigned long long random_seed, int batch_size, int vocab_size, cudaStream_t stream) {
  const size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;

  // The intermediate buffers are used by bitonic sort algorithms. We need to allocate
  // them to be large enough for the worst-case scenario from the benchmarks.
  const size_t intermediate_buffer_elements = static_cast<size_t>(batch_size) * kBitonicSortMaxPartitions * kBitonicSortMaxK;

  // These buffers are used for intermediate results in bitonic sort, which can be larger than the vocab size.
  const size_t max_buffer_elements = std::max(vocab_batch_size, intermediate_buffer_elements);
  indices_sorted = CudaMallocArray<int>(max_buffer_elements);
  scores_sorted = CudaMallocArray<float>(max_buffer_elements);
  scores_buffer = CudaMallocArray<float>(max_buffer_elements);
  scores_temp = CudaMallocArray<float>(max_buffer_elements);
  indices_in = CudaMallocArray<int>(max_buffer_elements);

  // These buffers are used in sampling and are safe with vocab_batch_size
  prefix_sums = CudaMallocArray<float>(vocab_batch_size);
  scores_adjusted = CudaMallocArray<float>(vocab_batch_size);
  prefix_sums_adjusted = CudaMallocArray<float>(vocab_batch_size);

  thresholds = CudaMallocArray<float>(batch_size);
  offsets = CudaMallocArray<int>(batch_size + 1);
  curand_states = CudaMallocArray<curandState>(batch_size);

  // The temp buffer is used by both the full sort (over vocab_size) and the
  // map-reduce bitonic sort (over an intermediate buffer). We need to allocate a buffer
  // large enough for the biggest possible sort operation.

  // Case 1: Temp storage for full sort.
  size_t temp_storage_bytes_full_sort = 0;
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_storage_bytes_full_sort,
                                                                (float*)nullptr, (float*)nullptr, (int*)nullptr, (int*)nullptr,
                                                                vocab_size * batch_size, batch_size,
                                                                (int*)nullptr, (int*)nullptr, 0, sizeof(float) * 8, stream));

  // Case 2: Temp storage for map-reduce bitonic sort's worst case.
  size_t temp_storage_bytes_bitonic_sort = 0;
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_storage_bytes_bitonic_sort,
                                                                (float*)nullptr, (float*)nullptr, (int*)nullptr, (int*)nullptr,
                                                                intermediate_buffer_elements, batch_size,
                                                                (int*)nullptr, (int*)nullptr, 0, sizeof(float) * 8, stream));

  temp_storage_bytes = std::max(temp_storage_bytes_full_sort, temp_storage_bytes_bitonic_sort);

  // Allocate the temporary buffer with the exact number of bytes required by CUB.
  // The original code used `temp_storage_bytes / sizeof(float)`, which truncated the size.
  temp_buffer = CudaMallocArray<unsigned char>(temp_storage_bytes);

  InitCurandStates<<<int(batch_size / 128) + 1, 128, 0, stream>>>(random_seed, curand_states.get(), batch_size);
  CUDA_CHECK(cudaGetLastError());
}

// Softmax Kernels and Launchers

template <typename T, typename AccumT>
struct MaxFloat {
  __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
    return ::max(max, (AccumT)v);
  }
};

template <typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T, typename AccumT>
struct SumExpFloat {
  __device__ __forceinline__ SumExpFloat(AccumT v)
      : max_k(v) {}

  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + exp((AccumT)v - max_k);
  }

  const AccumT max_k;
};

template <typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

// aligned vector generates vectorized load/store on CUDA
template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_vector {
  T val[vec_size];
};

template <template <typename, typename> class Reduction, int ILP, typename T, typename AccumT>
__device__ __forceinline__ AccumT IlpReduce(int shift, T* data, int size, const Reduction<T, AccumT>& r, AccumT defaultVal) {
  using LoadT = aligned_vector<T, ILP>;
  AccumT threadVal = defaultVal;
  int offset = threadIdx.x;
  // shift and do 1
  if (shift > 0) {
    data -= shift;
    size += shift;
    if (threadIdx.x >= shift && threadIdx.x < size) {
      threadVal = r(threadVal, data[offset]);
    }
    size -= blockDim.x;
    data += blockDim.x;
  }
  if (size <= 0) return threadVal;
  int last = size % (ILP * blockDim.x);
  T v[ILP];
  LoadT* value = reinterpret_cast<LoadT*>(&v);
  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *value = reinterpret_cast<LoadT*>(data)[offset];
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      threadVal = r(threadVal, v[j]);
    }
  }
  offset = size - last + threadIdx.x;
  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);
  return threadVal;
}

template <template <typename> class Reduction, typename AccumT>
__device__ __forceinline__ AccumT SoftmaxReduce(AccumT* smem, AccumT val, const Reduction<AccumT>& r, AccumT defaultVal) {
  // To avoid RaW races from chaining SoftmaxReduce calls together, we need a sync here
  __syncthreads();
  smem[threadIdx.x] = val;
  __syncthreads();
  AccumT warpVal = defaultVal;
  // First warp will perform per-warp reductions for the remaining warps
  if (threadIdx.x < kGPUWarpSize) {
    int warps_per_block = blockDim.x / kGPUWarpSize;
    for (int i = 0; i < warps_per_block; ++i) {
      warpVal = r(warpVal, smem[i * kGPUWarpSize + threadIdx.x]);
    }
    smem[threadIdx.x] = warpVal;
  }
  __syncthreads();
  // First thread will perform a reduction of the above per-warp reductions
  AccumT blockVal = defaultVal;
  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < kGPUWarpSize; ++i) {
      blockVal = r(blockVal, smem[i]);
    }
    smem[0] = blockVal;
  }
  // Sync and broadcast
  __syncthreads();
  return smem[0];
}

dim3 SoftmaxGetBlockSize(int ILP, uint64_t size) {
  uint64_t block_size = 1;
  uint64_t max_block_size = min(size / ILP, static_cast<uint64_t>(kMaxThreads));
  // In the vectorized case we want to trade off allowing more of the buffers to be accessed
  // in a vectorized way against wanting a larger block size to get better utilisation.
  // In general with ILP you can have (ILP-1)/ILP of the buffer accessed vectorised, at the risk
  // of having a very small block size. We choose to keep >= 1/2 of the buffer vectorised while
  // allowing a larger block size.
  if (ILP > 1) {
    max_block_size /= 2;
  }
  while (block_size < max_block_size) block_size *= 2;
  // Launch at least a single warp - the kernel assumes that.
  block_size = max(block_size, static_cast<uint64_t>(kGPUWarpSize));
  return dim3(static_cast<unsigned int>(block_size));
}

template <typename T, typename AccumT, typename OutT>
struct LogSoftmaxForwardEpilogue {
  __device__ __forceinline__ LogSoftmaxForwardEpilogue(AccumT max_input, AccumT sum)
      : max_input(max_input), logsum(log(sum)) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>((AccumT)input - max_input - logsum);
  }

  const AccumT max_input;
  const AccumT logsum;
};

template <typename T, typename AccumT, typename OutT>
struct SoftmaxForwardEpilogue {
  __device__ __forceinline__ SoftmaxForwardEpilogue(AccumT max_input, AccumT sum)
      : max_input(max_input), sum(sum) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(exp((AccumT)input - max_input) / sum);
  }

  const AccumT max_input;
  const AccumT sum;
};

/**
 * This will apply the Epilogue with vectorized reads & writes when input & output have the same shift
 */
template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__device__ __forceinline__ void WriteFpropResultsVectorized(int size,
                                                            const int shift,
                                                            scalar_t* input,
                                                            outscalar_t* output,
                                                            Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
  using LoadT = aligned_vector<scalar_t, ILP>;
  using StoreT = aligned_vector<outscalar_t, ILP>;
  int offset = threadIdx.x;
  // if unaligned, do one value / thread and move on, guaranteeing aligned reads/writes later
  if (shift > 0) {
    input -= shift;
    output -= shift;
    size += shift;
    if (threadIdx.x >= shift && threadIdx.x < size) {
      output[offset] = epilogue(input[offset]);
    }
    size -= blockDim.x;
    input += blockDim.x;
    output += blockDim.x;
  }
  if (size <= 0) return;
  const int last = size % (ILP * blockDim.x);
  scalar_t in_v[ILP];
  LoadT* in_value = reinterpret_cast<LoadT*>(&in_v);
  outscalar_t out_v[ILP];
  StoreT* out_value = reinterpret_cast<StoreT*>(&out_v);
  for (; offset * ILP < (size - last); offset += blockDim.x) {
    *in_value = reinterpret_cast<LoadT*>(input)[offset];
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      out_v[j] = epilogue(in_v[j]);
    }
    reinterpret_cast<StoreT*>(output)[offset] = *out_value;
  }
  offset = size - last + threadIdx.x;
  // handle the tail
  for (; offset < size; offset += blockDim.x) {
    output[offset] = epilogue(input[offset]);
  }
}

/**
 * This will apply the Epilogue with non-vectrorized reads & writes for the general case
 */
template <int ILP, typename scalar_t, typename accum_t, typename outscalar_t, template <typename, typename, typename> class Epilogue>
__device__ __forceinline__ void WriteFpropResults(int classes,
                                                  scalar_t* input,
                                                  outscalar_t* output,
                                                  Epilogue<scalar_t, accum_t, outscalar_t> epilogue) {
  int offset = threadIdx.x;
  int last = classes % (ILP * blockDim.x);
  // Main bulk of loop with ILP
  for (; offset < classes - last; offset += blockDim.x * ILP) {
    scalar_t tmp[ILP];
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmp[j] = input[offset + j * blockDim.x];
    }
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      output[offset + j * blockDim.x] = epilogue(tmp[j]);
    }
  }
  // Remainder - no ILP
  for (; offset < classes; offset += blockDim.x) {
    output[offset] = epilogue(input[offset]);
  }
}

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t,
          template <typename, typename, typename> class Epilogue>
__global__ void SoftmaxBlockForward(outscalar_t* output, scalar_t* input, int classes,
                                    int input_stride, int output_stride) {
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * input_stride;
  output += blockIdx.x * output_stride;
  const int input_align_bytes = ILP * sizeof(scalar_t);
  const int output_align_bytes = ILP * sizeof(outscalar_t);
  const int shift = ((uint64_t)input) % input_align_bytes / sizeof(scalar_t);
  const int output_shift = ((uint64_t)output) % output_align_bytes / sizeof(outscalar_t);
  // find the max
  accscalar_t threadMax = IlpReduce<MaxFloat, ILP, scalar_t, accscalar_t>(
      shift, input, classes, MaxFloat<scalar_t, accscalar_t>(), -std::numeric_limits<accscalar_t>::max());
  accscalar_t max_k = SoftmaxReduce<Max, accscalar_t>(
      sdata, threadMax, Max<accscalar_t>(), -std::numeric_limits<accscalar_t>::max());
  // reduce all values
  accscalar_t threadExp = IlpReduce<SumExpFloat, ILP, scalar_t, accscalar_t>(
      shift, input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
  accscalar_t sumAll = SoftmaxReduce<Add, accscalar_t>(
      sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));
  Epilogue<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll);
  if (shift == output_shift) {
    WriteFpropResultsVectorized<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, shift, input, output, epilogue);
  } else {
    WriteFpropResults<ILP, scalar_t, accscalar_t, outscalar_t, Epilogue>(classes, input, output, epilogue);
  }
}

template <bool is_log_softmax>
void DispatchBlockwiseSoftmaxForward(cudaStream_t stream, float* output, const float* input, int softmax_elements,
                                     int input_stride, int output_stride, int batch_count) {
  dim3 grid(batch_count);
  constexpr int ILP = sizeof(float4) / sizeof(float);
  dim3 block = SoftmaxGetBlockSize(ILP, softmax_elements);
  if (is_log_softmax) {
    SoftmaxBlockForward<ILP, float, float, float, LogSoftmaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(float), stream>>>(output, const_cast<float*>(input),
                                                           softmax_elements, input_stride, output_stride);
  } else {
    SoftmaxBlockForward<ILP, float, float, float, SoftmaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(float), stream>>>(output, const_cast<float*>(input),
                                                           softmax_elements, input_stride, output_stride);
  }
  CUDA_CHECK(cudaGetLastError());
}
template void DispatchBlockwiseSoftmaxForward<true>(cudaStream_t, float*, const float*, int, int, int, int);

// Populate Kernels and Launchers

__global__ void PopulateIndices(int* indices, int size, int batch_size) {
  int global_index = threadIdx.x + blockIdx.x * blockDim.x;
  int index = global_index % size;
  if (global_index < size * batch_size) {
    indices[global_index] = index;
  }
}

void LaunchPopulateIndices(int* indices, int size, int batch_size, cudaStream_t stream) {
  dim3 grid((batch_size * size / 256) + 1, 1, 1);
  dim3 block(256, 1, 1);
  PopulateIndices<<<grid, block, 0, stream>>>(indices, size, batch_size);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void PopulateOffsets(int* offsets, int size, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < batch_size + 1)
    offsets[index] = index * size;
}

void LaunchPopulateOffsets(int* offsets, int size, int batch_size, cudaStream_t stream) {
  dim3 grid(int(batch_size / 128) + 1, 1, 1);
  dim3 block(128, 1, 1);
  PopulateOffsets<<<grid, block, 0, stream>>>(offsets, size, batch_size);
  CUDA_CHECK(cudaGetLastError());
}

// Sorting Kernel Launcher

template <typename T>
void LaunchSortPairs(void* d_temp_storage,
                     size_t temp_storage_bytes,
                     const T* d_keys_in,
                     T* d_keys_out,
                     const int* d_values_in,
                     int* d_values_out,
                     int num_items,
                     int num_segments,
                     int* d_offsets,
                     cudaStream_t stream,
                     bool is_descending) {
  if (is_descending) {
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
                                                                  d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream));
  } else {
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
                                                        d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream));
  }
}

template <typename T>
void GetTempStorageSize(const T* d_keys_in,
                        const int* d_values_in,
                        int* d_offsets,
                        int num_items,
                        int num_segments,
                        cudaStream_t stream,
                        bool is_descending,
                        size_t& temp_storage_bytes) {
  if (is_descending) {
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, d_keys_in, (T*)nullptr,
                                                                  d_values_in, (int*)nullptr, num_items, num_segments, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream));
  } else {
    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(nullptr, temp_storage_bytes, d_keys_in, (T*)nullptr,
                                                        d_values_in, (int*)nullptr, num_items, num_segments, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream));
  }
}

template <int kBlockSize>
__global__ void PrefixSumKernel(float* scores, float* prefix_sums, int sample_range, int batch_size) {
  int batch = blockIdx.x;
  float prefix_sum = 0.0f;

  typedef cub::BlockScan<float, kBlockSize> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  for (int i = 0; i < sample_range; i += blockDim.x) {
    int global_index = threadIdx.x + i + batch * sample_range;
    int local_index = threadIdx.x + i;
    float score = (local_index < sample_range) ? scores[global_index] : 0.0f;
    float sum = score;
    BlockScan(temp_storage).InclusiveSum(sum, sum);
    prefix_sum += sum;
    __syncthreads();
    if (local_index < sample_range) {
      prefix_sums[global_index] = prefix_sum;
    }
  }
}

template <int kBlockSize>
__global__ void FilterOnTopP(float* scores, float* prefix_sums, float* scores_temp, float* actual_values, int sample_range, int batch_size, float p) {
  int batch = blockIdx.x;
  float prefix_sum = 0.0f;
  float saferNegative = std::numeric_limits<float>::lowest() / 1000.0f;

  typedef cub::BlockScan<float, kBlockSize> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  for (int i = 0; i < sample_range; i += blockDim.x) {
    int global_index = threadIdx.x + i + batch * sample_range;
    int local_index = threadIdx.x + i;
    float score = (local_index < sample_range) ? scores[global_index] : 0.0f;
    float sum = score;
    BlockScan(temp_storage).InclusiveSum(sum, sum);
    prefix_sum += sum;
    __syncthreads();
    if (local_index < sample_range) {
      scores_temp[global_index] = prefix_sum;
    }
    __syncthreads();
    if (local_index == 0) {
      prefix_sums[global_index] = actual_values[global_index];
    } else if (local_index < sample_range) {
      if (scores_temp[global_index - 1] < p) {
        prefix_sums[global_index] = actual_values[global_index];
      } else {
        prefix_sums[global_index] = saferNegative;
      }
    }
  }
}

// START of improved Top-K kernel and helpers from cuda_sampling_improved.cu
// This new kernel finds the top-k elements iteratively. For each of the k
// elements, it performs a parallel reduction to find the max score, records it,
// and then effectively removes it from the next iteration by setting its
// score to a very low value. This avoids complex heap structures.
struct TopK_2 {
  int p = INT_MAX;
  float u = -FLT_MAX;

  __device__ __forceinline__ void insert(float elem, int elem_id) {
    if (elem > u || (elem == u && elem_id < p)) {
      u = elem;
      p = elem_id;
    }
  }

  __device__ __forceinline__ void init() {
    u = -FLT_MAX;
    p = -1;
  }
};

__device__ __forceinline__ TopK_2 reduce_topk_op_2(TopK_2 const& a, TopK_2 const& b) {
  return a.u > b.u ? a : (a.u == b.u && a.p < b.p) ? a
                                                   : b;
}

template <int kBlockSize>
__global__ void GetTopKKernel(int* indices_out, float* scores_in, float* scores_out, int batch_size, int vocab_size, int k, float temperature) {
  int batch = blockIdx.x;
  int tid = threadIdx.x;
  TopK_2 partial;

  float const MAX_T_VAL = FLT_MAX;

  for (int ite = 0; ite < k; ite++) {
    partial.init();
    for (auto elemId = tid; elemId < vocab_size; elemId += kBlockSize) {
      float elem = scores_in[elemId + batch * vocab_size];
      partial.insert(elem, elemId);
    }
    // reduce in thread block
    typedef cub::BlockReduce<TopK_2, kBlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopK_2 top_k_sequence = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2);

    if (tid == 0) {
      scores_out[ite + batch * k] = top_k_sequence.u / temperature;
      indices_out[ite + batch * k] = top_k_sequence.p;

      // set the max value to -MAX_T_VAL so that the value doesn't get picked again
      scores_in[batch * vocab_size + top_k_sequence.p] = -MAX_T_VAL;
    }

    // Attention: Add a block-level memory fence here.
    // This ensures that the write to global memory by thread 0 is visible
    // to all other threads in the block before the next iteration begins.
    // Without this, other threads might read the old (pre-modification)
    // score in the next iteration, leading to the same item being picked again.
    __threadfence_block();

    __syncthreads();
  }
}

// Launcher for the improved Top-K kernel.
void LaunchImprovedGetTopK(cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  dim3 grid(batch_size, 1, 1);
  // Use a larger block size for better hardware utilization, as in the improved file.
  dim3 block(1024, 1, 1);
  GetTopKKernel<1024><<<grid, block, 0, stream>>>(indices_out, scores_in, scores_out, batch_size, vocab_size, k, temperature);
  CUDA_CHECK(cudaGetLastError());
}
// END of improved Top-K kernel and helpers

// Sets up random thresholds for top p or top k sampling
__global__ void RandomThresholdKernel(curandState* curand_states, float* thresholds, int batch_size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < batch_size) {
    // For numerical stability, we use 0.9999999f not 1.0f to avoid zero probabilities.
    thresholds[index] = 0.9999999f * curand_uniform(&curand_states[index]);
  }
}

template <int kBlockSize>
__global__ void SampleKernel(float* prefix_sums, int* indices, int* index_out, int sample_range, int indices_stride, float* thresholds) {
  int batch = blockIdx.x;
  int index = threadIdx.x;

  __shared__ int first_index;
  if (threadIdx.x == 0) {
    first_index = sample_range - 1;
  }
  __syncthreads();

  for (; index < sample_range - 1; index += blockDim.x) {
    float sum = prefix_sums[batch * sample_range + index];
    // TOP P or K
    if (sum >= thresholds[batch]) {
      atomicMin(&first_index, index);
      break;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    index_out[batch] = indices[batch * indices_stride + first_index];
  }
}

void LaunchSampleKernel(SamplingData* data, cudaStream_t stream, float* scores, int* indices, int* index_out, int sample_range, int batch_size, int indices_stride, float p = 0.0, int k = -1) {
  dim3 grid(batch_size, 1, 1);
  dim3 block(256, 1, 1);
  // Prefix Sums
  FilterOnTopP<256><<<grid, block, 0, stream>>>(scores, data->prefix_sums.get(), data->scores_temp.get(), data->scores_buffer.get(), sample_range, batch_size, p);
  CUDA_CHECK(cudaGetLastError());
  DispatchBlockwiseSoftmaxForward<false>(stream, data->scores_adjusted.get(), const_cast<const float*>(data->prefix_sums.get()), k, indices_stride, k, batch_size);
  PrefixSumKernel<256><<<grid, block, 0, stream>>>(data->scores_adjusted.get(), data->prefix_sums_adjusted.get(), sample_range, batch_size);
  CUDA_CHECK(cudaGetLastError());
  // Random Thresholds for Top P or Top K Sampling
  RandomThresholdKernel<<<int(batch_size / 128) + 1, 128, 0, stream>>>(data->curand_states.get(), data->thresholds.get(), batch_size);
  CUDA_CHECK(cudaGetLastError());
  SampleKernel<256><<<grid, block, 0, stream>>>(data->prefix_sums_adjusted.get(), indices, index_out, sample_range, indices_stride, data->thresholds.get());
  CUDA_CHECK(cudaGetLastError());
}

void LaunchSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size) {
  // Sort indices and scores
  LaunchPopulateOffsets(data->offsets.get(), vocab_size, batch_size, stream);
  LaunchPopulateIndices(data->indices_in.get(), vocab_size, batch_size, stream);
  LaunchSortPairs<float>(data->temp_buffer.get(), data->temp_storage_bytes, scores_in, scores_out,
                         data->indices_in.get(), indices_out, vocab_size * batch_size, batch_size, data->offsets.get(),
                         stream, /*is_descending*/ true);
}

// Enum to represent the chosen algorithm for a given Top-K task
enum class TopKAlgorithm {
  SELECTION_SORT,
  BITONIC_SORT,
  FULL_SORT
};

struct TopKConfig {
  TopKAlgorithm algorithm = TopKAlgorithm::FULL_SORT;
  int block_size = 256;
  int num_partitions = 0;  // Only relevant for bitonic
  int sort_size = 0; // Only relevant for bitonic
};

// Cache key: a tuple of (vocab_size, batch_size, k)
using BenchmarkingCacheKey = std::tuple<int, int, int>;

// The cache stores the best algorithm configuration for a given key.
static std::map<BenchmarkingCacheKey, TopKConfig> algorithm_cache;
static std::mutex cache_mutex;  // Mutex to make cache access thread-safe

// Kernel to fill an array with random data for benchmarking
__global__ void FillRandom(float* array, curandState* states, int n, int batch_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int batch_idx = (static_cast<long long>(i) * batch_size) / n;
    array[i] = curand_uniform(&states[batch_idx]);
  }
}

// --- Softmax with Temperature ---
// This is a modified version of the Softmax logic that integrates temperature scaling
// to avoid a separate kernel launch, specifically for the Full Sort path.

template <typename T, typename AccumT>
struct MaxFloatWithTemp {
  __device__ __forceinline__ MaxFloatWithTemp(AccumT t) : temp(t) {}
  __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
    return ::max(max, (AccumT)v / temp);
  }
  const AccumT temp;
};

template <typename T, typename AccumT>
struct SumExpFloatWithTemp {
  __device__ __forceinline__ SumExpFloatWithTemp(AccumT v, AccumT t) : max_k(v), temp(t) {}
  __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
    return sum + exp(((AccumT)v / temp) - max_k);
  }
  const AccumT max_k;
  const AccumT temp;
};

template <typename T, typename AccumT, typename OutT>
struct SoftmaxForwardEpilogueWithTemp {
  __device__ __forceinline__ SoftmaxForwardEpilogueWithTemp(AccumT max_input, AccumT sum, AccumT t)
      : max_input(max_input), sum(sum), temp(t) {}

  __device__ __forceinline__ OutT operator()(T input) const {
    return static_cast<OutT>(exp(((AccumT)input / temp) - max_input) / sum);
  }

  const AccumT max_input;
  const AccumT sum;
  const AccumT temp;
};

template <int ILP, typename scalar_t, typename accscalar_t, typename outscalar_t>
__global__ void SoftmaxBlockForwardWithTemperature(outscalar_t* output, scalar_t* input, int classes,
                                                   int input_stride, int output_stride, float temperature) {
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accscalar_t*>(smem);
  input += blockIdx.x * input_stride;
  output += blockIdx.x * output_stride;

  const int shift = 0;  // Simplified for this use case, assuming aligned data

  accscalar_t threadMax = IlpReduce<MaxFloatWithTemp, ILP, scalar_t, accscalar_t>(
      shift, input, classes, MaxFloatWithTemp<scalar_t, accscalar_t>(temperature), -std::numeric_limits<accscalar_t>::max());
  accscalar_t max_k = SoftmaxReduce<Max, accscalar_t>(
      sdata, threadMax, Max<accscalar_t>(), -std::numeric_limits<accscalar_t>::max());

  accscalar_t threadExp = IlpReduce<SumExpFloatWithTemp, ILP, scalar_t, accscalar_t>(
      shift, input, classes, SumExpFloatWithTemp<scalar_t, accscalar_t>(max_k, temperature), static_cast<accscalar_t>(0));
  accscalar_t sumAll = SoftmaxReduce<Add, accscalar_t>(
      sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

  SoftmaxForwardEpilogueWithTemp<scalar_t, accscalar_t, outscalar_t> epilogue(max_k, sumAll, temperature);
  WriteFpropResults<ILP, scalar_t, accscalar_t, outscalar_t, SoftmaxForwardEpilogueWithTemp>(classes, input, output, epilogue);
}

void DispatchBlockwiseSoftmaxForwardWithTemperature(cudaStream_t stream, float* output, const float* input, int softmax_elements,
                                                    int input_stride, int output_stride, int batch_count, float temperature) {
  dim3 grid(batch_count);
  constexpr int ILP = sizeof(float4) / sizeof(float);
  dim3 block = SoftmaxGetBlockSize(ILP, softmax_elements);
  SoftmaxBlockForwardWithTemperature<ILP, float, float, float>
      <<<grid, block, block.x * sizeof(float), stream>>>(output, const_cast<float*>(input),
                                                         softmax_elements, input_stride, output_stride, temperature);
  CUDA_CHECK(cudaGetLastError());
}
// --- End of Softmax with Temperature ---

void RunTopKViaSelectionSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  // The output of the kernel will be the temperature-scaled scores. We'll store
  // these in another pre-allocated buffer, `scores_buffer`.
  float* scaled_scores = data->scores_buffer.get();

  // Attention: The kernel modifies the `scores_in` tensor in-place.
  // This might have unintended side effects on the original `scores_in` tensor if it is used elsewhere later.
  LaunchImprovedGetTopK(stream, scores_in, scaled_scores, indices_out, vocab_size, batch_size, k, temperature);

  // Finally, apply softmax to the scaled scores to get the final probabilities,
  // writing the result to the `scores_out` buffer.
  DispatchBlockwiseSoftmaxForward<false>(stream, scores_out, const_cast<const float*>(scaled_scores), k, k, k, batch_size);
}

// ------------------------------------------------------------------
// START of Map-Reduce implementations
// ------------------------------------------------------------------

template <int kBlockSize, bool use_cub = false>
__global__ void CopyAndSoftmaxKernel(int* final_indices, float* final_scores,
                                     const int* sorted_indices, const float* sorted_scores,
                                     int k, float temperature, int input_stride) {
  const int batch_idx = blockIdx.x;
  const int* batch_sorted_indices = sorted_indices + batch_idx * input_stride;
  const float* batch_sorted_scores = sorted_scores + batch_idx * input_stride;

  if constexpr (use_cub) {
    // STEP 1: All threads cooperatively copy the final indices.
    for (int i = threadIdx.x; i < k; i += kBlockSize) {
      final_indices[batch_idx * k + i] = batch_sorted_indices[i];
    }

    // STEP 2: Perform softmax using a parallel reduction to match the reference implementation's math.
    typedef cub::BlockReduce<float, kBlockSize> BlockReduce;
    __shared__ union {
      typename BlockReduce::TempStorage reduce_max;
      typename BlockReduce::TempStorage reduce_sum;
    } temp_storage;

    // Each thread loads its score and applies temperature. Pad with a large negative for non-participating threads.
    float thread_score = -std::numeric_limits<float>::max();
    if (threadIdx.x < k) {
      thread_score = batch_sorted_scores[threadIdx.x] / temperature;
    }

    // Parallel reduction to find the maximum score.
    float max_val = BlockReduce(temp_storage.reduce_max).Reduce(thread_score, cub::Max());
    __syncthreads();

    // Calculate `exp(score - max)` for each thread's score.
    float thread_exp = 0.0f;
    if (threadIdx.x < k) {
      thread_exp = expf(thread_score - max_val);
    }

    // Parallel reduction to find the sum of the exponentials.
    float sum_exp = BlockReduce(temp_storage.reduce_sum).Reduce(thread_exp, cub::Sum());
    __syncthreads();

    // STEP 3: All threads write the final softmax probability.
    for (int i = threadIdx.x; i < k; i += kBlockSize) {
      float scaled_score = batch_sorted_scores[i] / temperature;
      final_scores[batch_idx * k + i] = expf(scaled_score - max_val) / sum_exp;
    }
  } else {
    __shared__ float top_k_scores_smem[64];  // max_k

    // Cooperatively load the top k scores into shared memory
    if (threadIdx.x < k) {
      top_k_scores_smem[threadIdx.x] = batch_sorted_scores[threadIdx.x] / temperature;
    }
    __syncthreads();

    // Thread 0 computes max and sum_exp for softmax
    __shared__ float max_val;
    __shared__ float sum_exp;
    if (threadIdx.x == 0) {
      max_val = -std::numeric_limits<float>::max();
      for (int i = 0; i < k; i++) {
        if (top_k_scores_smem[i] > max_val) {
          max_val = top_k_scores_smem[i];
        }
      }
      sum_exp = 0.0f;
      for (int i = 0; i < k; i++) {
        sum_exp += expf(top_k_scores_smem[i] - max_val);
      }
    }
    __syncthreads();

    // All threads write final results
    for (int i = threadIdx.x; i < k; i += kBlockSize) {
      final_indices[batch_idx * k + i] = batch_sorted_indices[i];
      final_scores[batch_idx * k + i] = expf(top_k_scores_smem[i] - max_val) / sum_exp;
    }
  }
}

template <int kBlockSize, int kSortSize>
__global__ void FindBlockTopK_BitonicSort(const float* scores_in,
                                          int* intermediate_indices,
                                          float* intermediate_scores,
                                          int vocab_size,
                                          int num_partitions) {
  // Shared memory for sorting one partition. Its size must be a power of 2.
  __shared__ float smem_scores[kSortSize];
  __shared__ int smem_indices[kSortSize];

  const int batch_idx = blockIdx.y;
  const int partition_idx = blockIdx.x;

  const float* batch_scores_in = scores_in + batch_idx * vocab_size;
  const int partition_size = (vocab_size + num_partitions - 1) / num_partitions;
  const int partition_start = partition_idx * partition_size;

  // Load data from global to shared memory
  for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
    int global_idx = partition_start + i;
    if (i < partition_size && global_idx < vocab_size) {
      smem_scores[i] = batch_scores_in[global_idx];
      smem_indices[i] = global_idx;
    } else {
      // Pad with minimum values to ensure they are sorted to the end
      smem_scores[i] = -std::numeric_limits<float>::max();
      smem_indices[i] = -1;
    }
  }
  __syncthreads();

  // --- In-place Bitonic Sort (descending) ---
  for (int k = 2; k <= kSortSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < kSortSize; i += kBlockSize) {
        int ixj = i ^ j;
        if (ixj > i) {
          if ((i & k) == 0) {  // Sort ascending
            if (smem_scores[i] > smem_scores[ixj]) {
              float temp_s = smem_scores[i];
              smem_scores[i] = smem_scores[ixj];
              smem_scores[ixj] = temp_s;
              int temp_i = smem_indices[i];
              smem_indices[i] = smem_indices[ixj];
              smem_indices[ixj] = temp_i;
            }
          } else {  // Sort descending
            if (smem_scores[i] < smem_scores[ixj]) {
              float temp_s = smem_scores[i];
              smem_scores[i] = smem_scores[ixj];
              smem_scores[ixj] = temp_s;
              int temp_i = smem_indices[i];
              smem_indices[i] = smem_indices[ixj];
              smem_indices[ixj] = temp_i;
            }
          }
        }
      }
      __syncthreads();
    }
  }
  // Final pass to make the whole array descending
  for (int i = threadIdx.x; i < kSortSize / 2; i += kBlockSize) {
    if (smem_scores[i] < smem_scores[kSortSize - 1 - i]) {
      float temp_s = smem_scores[i];
      smem_scores[i] = smem_scores[kSortSize - 1 - i];
      smem_scores[kSortSize - 1 - i] = temp_s;
      int temp_i = smem_indices[i];
      smem_indices[i] = smem_indices[kSortSize - 1 - i];
      smem_indices[kSortSize - 1 - i] = temp_i;
    }
  }
  __syncthreads();

  // Have the first `max_k` threads write out the top results
  if (threadIdx.x < kBitonicSortMaxK) {
    int offset = (batch_idx * num_partitions + partition_idx) * kBitonicSortMaxK;
    intermediate_scores[offset + threadIdx.x] = smem_scores[threadIdx.x];
    intermediate_indices[offset + threadIdx.x] = smem_indices[threadIdx.x];
  }
}

void RunTopKViaMapReduceBitonicSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature, int num_partitions, int sort_size) {
  constexpr int block_size = 256;

  float* intermediate_scores = data->scores_buffer.get();
  int* intermediate_indices = data->indices_in.get();

  // Stage 1: Map using the new Bitonic Sort kernel
  dim3 grid_stage1(num_partitions, batch_size);
  dim3 block_stage1(block_size);

  switch (sort_size) {
    case 512:
      FindBlockTopK_BitonicSort<block_size, 512><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions);
      break;
    case 1024:
      FindBlockTopK_BitonicSort<block_size, 1024><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions);
      break;
    case 2048:
      FindBlockTopK_BitonicSort<block_size, 2048><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions);
      break;
    case 4096:
      FindBlockTopK_BitonicSort<block_size, 4096><<<grid_stage1, block_stage1, 0, stream>>>(
          scores_in, intermediate_indices, intermediate_scores, vocab_size, num_partitions);
      break;
    default:
      assert(false && "Unsupported sort_size");
      break;
  }
  CUDA_CHECK(cudaGetLastError());

  // Stage 2: Sort the small intermediate buffer using CUB Segmented Sort
  int num_intermediate_results_per_batch = num_partitions * kBitonicSortMaxK;
  int total_intermediate_results = batch_size * num_intermediate_results_per_batch;
  float* sorted_scores = data->scores_temp.get();
  int* sorted_indices = data->indices_sorted.get();

  LaunchPopulateOffsets(data->offsets.get(), num_intermediate_results_per_batch, batch_size, stream);

  size_t temp_storage_bytes_needed = 0;
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_storage_bytes_needed, intermediate_scores, sorted_scores, intermediate_indices, sorted_indices, total_intermediate_results, batch_size, data->offsets.get(), data->offsets.get() + 1, 0, sizeof(float) * 8, stream));

  if (data->temp_storage_bytes < temp_storage_bytes_needed) {
    std::cerr << "FATAL ERROR in RunTopKViaMapReduceBitonicSort: Pre-allocated temp_buffer is too small." << std::endl;
    return;
  }

  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(data->temp_buffer.get(), temp_storage_bytes_needed, intermediate_scores, sorted_scores, intermediate_indices, sorted_indices, total_intermediate_results, batch_size, data->offsets.get(), data->offsets.get() + 1, 0, sizeof(float) * 8, stream));

  // Stage 3: Launch the (known good) kernel to copy the top k and apply softmax
  dim3 grid_stage3(batch_size);
  dim3 block_stage3(256);
  CopyAndSoftmaxKernel<256><<<grid_stage3, block_stage3, 0, stream>>>(indices_out, scores_out, sorted_indices, sorted_scores, k, temperature, num_intermediate_results_per_batch);
  CUDA_CHECK(cudaGetLastError());
}
// ------------------------------------------------------------------
// END of Map-Reduce implementations
// ------------------------------------------------------------------

void RunTopKViaFullSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  // Step 1: Perform a full, segmented sort on the input scores.
  // The results (sorted scores and indices) are stored in temporary buffers.
  float* sorted_scores = data->scores_buffer.get();
  int* sorted_indices = data->indices_in.get();
  LaunchSort(data, stream, scores_in, sorted_scores, sorted_indices, vocab_size, batch_size);

  // Step 2: Launch a specialized kernel to handle the final steps.
  // This kernel will:
  //   a) Copy the top 'k' indices for each batch item from the sorted temporary buffer to the final output buffer.
  //   b) Read the corresponding top 'k' scores.
  //   c) Apply the temperature scaling to these scores.
  //   d) Compute the softmax on the scaled scores.
  //   e) Write the final softmax probabilities to the output scores buffer.
  // The 'vocab_size' is passed as the input_stride to correctly index into the sorted results for each batch item.
  dim3 grid(batch_size);
  dim3 block(256);
  CopyAndSoftmaxKernel<256><<<grid, block, 0, stream>>>(indices_out, scores_out, sorted_indices, sorted_scores, k, temperature, vocab_size);
  CUDA_CHECK(cudaGetLastError());
}

void RandomTopkInput(cudaStream_t stream, float* data, curandState* batch_state, int total_size, int batch_size) {
  FillRandom<<<(total_size + 255) / 256, 256, 0, stream>>>(data, batch_state, total_size, batch_size);
  CUDA_CHECK(cudaGetLastError());
}

const char* AlgorithmToString(TopKAlgorithm algo) {
  switch (algo) {
    case TopKAlgorithm::SELECTION_SORT:
      return "SELECTION_SORT";
    case TopKAlgorithm::BITONIC_SORT:
      return "BITONIC_SORT";
    case TopKAlgorithm::FULL_SORT:
      return "FULL_SORT";
    default:
      return "UNKNOWN";
  }
}

#define CUDA_CHECK_WITH_CONFIG(call, config)                                    \
  do {                                                                          \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "\n--- CUDA Error during benchmark ---\n");               \
      fprintf(stderr, "Algorithm:  %s\n", AlgorithmToString(config.algorithm)); \
      fprintf(stderr, "Partitions: %d\n", config.num_partitions);               \
      fprintf(stderr, "SortSize:  %d\n", config.sort_size);                    \
      fprintf(stderr, "Block Size: %d\n", config.block_size);                   \
      fprintf(stderr, "Error:      %s\n", cudaGetErrorString(err));             \
      fprintf(stderr, "Location:   %s:%d\n", __FILE__, __LINE__);               \
      fprintf(stderr, "-------------------------------------\n");               \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

// Performs a one-time benchmark to find the fastest Top-K algorithm for a given configuration.
TopKConfig BenchmarkAndGetBestAlgorithm(SamplingData* data, cudaStream_t stream, int vocab_size, int batch_size, int k) {
  assert(k <= 64);
  BenchmarkingCacheKey key = {vocab_size, batch_size, k};
  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = algorithm_cache.find(key);
  if (it != algorithm_cache.end()) return it->second;

  auto d_rand_scores = CudaMallocArray<float>(vocab_size * batch_size);
  auto d_rand_indices = CudaMallocArray<int>(k * batch_size);
  auto d_rand_out = CudaMallocArray<float>(k * batch_size);
  int total_size = vocab_size * batch_size;
  RandomTopkInput(stream, d_rand_scores.get(), data->curand_states.get(), total_size, batch_size);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  constexpr int warmup_runs = 2, timing_runs = 5;
  float temperature = 1.0f;

  struct Result {
    TopKConfig config;
    float time;
  };

  std::vector<Result> results;

  auto benchmark_algorithm = [&](TopKConfig config, auto func) {
    for (int i = 0; i < warmup_runs; ++i) {
      func();
    }
    // Check for any async errors from the warmup runs. cudaGetLastError resets the error state.
    CUDA_CHECK_WITH_CONFIG(cudaGetLastError(), config);

    CUDA_CHECK_WITH_CONFIG(cudaEventRecord(start, stream), config);
    for (int i = 0; i < timing_runs; ++i) {
      func();
    }
    // Check for async errors from the timed runs before stopping the timer.
    CUDA_CHECK_WITH_CONFIG(cudaGetLastError(), config);

    CUDA_CHECK_WITH_CONFIG(cudaEventRecord(stop, stream), config);
    CUDA_CHECK_WITH_CONFIG(cudaEventSynchronize(stop), config);
    float ms = 0.0f;
    CUDA_CHECK_WITH_CONFIG(cudaEventElapsedTime(&ms, start, stop), config);
    results.push_back({config, ms / timing_runs});
  };

  benchmark_algorithm({TopKAlgorithm::SELECTION_SORT}, [&]() { RunTopKViaSelectionSort(data, stream, d_rand_scores.get(), d_rand_out.get(), d_rand_indices.get(), vocab_size, batch_size, k, temperature); });

  for (int sort_size : {512, 1024, 2048, 4096}) {
    for (int num_partitions : {32, 64, 128, 256}) {
      assert(num_partitions <= kBitonicSortMaxPartitions);
      // Check if the partition size is valid for the given sort_size
      if (vocab_size <= sort_size * num_partitions && vocab_size > sort_size * num_partitions / 2) {
        benchmark_algorithm({TopKAlgorithm::BITONIC_SORT, num_partitions, 256, sort_size}, [&]() {
          RunTopKViaMapReduceBitonicSort(data, stream, d_rand_scores.get(), d_rand_out.get(), d_rand_indices.get(), vocab_size, batch_size, k, temperature, num_partitions, sort_size);
        });
      }
    }
  }

  benchmark_algorithm({TopKAlgorithm::FULL_SORT}, [&]() { RunTopKViaFullSort(data, stream, d_rand_scores.get(), d_rand_out.get(), d_rand_indices.get(), vocab_size, batch_size, k, temperature); });

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  auto best_it = std::min_element(results.begin(), results.end(), [](const auto& a, const auto& b) { return a.time < b.time; });
  TopKConfig winner = best_it->config;
  algorithm_cache[key] = winner;
  return winner;
}

void GetTopKSubset(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  TopKConfig chosen_config;
  if (k <= 8) {
    chosen_config.algorithm = TopKAlgorithm::SELECTION_SORT;
  } else if (k <= 64) {
    // TODO: replace it by a lookup table from offline benchmark.
    chosen_config = BenchmarkAndGetBestAlgorithm(data, stream, vocab_size, batch_size, k);
  }

  switch (chosen_config.algorithm) {
    case TopKAlgorithm::SELECTION_SORT:
      RunTopKViaSelectionSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature);
      break;
    case TopKAlgorithm::BITONIC_SORT:
      RunTopKViaMapReduceBitonicSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature, chosen_config.num_partitions, chosen_config.sort_size);
      break;
    default:
      RunTopKViaFullSort(data, stream, scores_in, scores_out, indices_out, vocab_size, batch_size, k, temperature);
      break;
  }
}

// Kernel launcher for combined (or separate) top k and top p sampling; where k is the max number of tokens to sample and p is the probability threshold
void GetSample(SamplingData* data, cudaStream_t stream, int32_t* next_token_out, float* scores_in, int vocab_size, int batch_size, int k, float p, float temperature) {
  if (k <= 0 || k > vocab_size) {
    k = vocab_size;
  }
  GetTopKSubset(data, stream, scores_in, data->scores_sorted.get(), data->indices_sorted.get(), vocab_size, batch_size, k, temperature);
  // Sample kernel
  int sample_range = k;
  // All Top-K algorithm paths produce a packed output buffer where the results for each
  // batch item are contiguous. Therefore, the stride between batches is simply k.
  int indices_stride = k;
  LaunchSampleKernel(data, stream, data->scores_sorted.get(), data->indices_sorted.get(), next_token_out, sample_range, batch_size, indices_stride, p, k);
}

}  // namespace cuda
}  // namespace Generators
