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
  indices_sorted = CudaMallocArray<int>(vocab_size * batch_size);
  scores_sorted = CudaMallocArray<float>(vocab_size * batch_size);
  scores_softmaxed = CudaMallocArray<float>(vocab_size * batch_size);
  prefix_sums = CudaMallocArray<float>(vocab_size * batch_size);
  thresholds = CudaMallocArray<float>(batch_size);
  indices_in = CudaMallocArray<int>(vocab_size * batch_size);
  offsets = CudaMallocArray<int>(batch_size + 1);
  curand_states = CudaMallocArray<curandState>(batch_size);
  temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, (float*)nullptr, (float*)nullptr,
                                                     (int*)nullptr, (int*)nullptr, vocab_size * batch_size, batch_size, (int*)nullptr, (int*)nullptr, 0, sizeof(float) * 8, stream);
  temp_buffer = CudaMallocArray<float>(temp_storage_bytes / sizeof(float));

  InitCurandStates<<<int(batch_size / 128) + 1, 128, 0, stream>>>(random_seed, curand_states.get(), batch_size);
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
                                    int input_stride, int output_stride, accscalar_t temperature) {
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
      shift, input, classes, SumExpFloat<scalar_t, accscalar_t>(max_k / temperature), static_cast<accscalar_t>(0));
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
                                     int input_stride, int output_stride, int batch_count, float temperature) {
  dim3 grid(batch_count);
  constexpr int ILP = sizeof(float4) / sizeof(float);
  dim3 block = SoftmaxGetBlockSize(ILP, softmax_elements);
  if (is_log_softmax) {
    SoftmaxBlockForward<ILP, float, float, float, LogSoftmaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(float), stream>>>(output, const_cast<float*>(input),
                                                            softmax_elements, input_stride, output_stride, temperature);
  } else {
    SoftmaxBlockForward<ILP, float, float, float, SoftmaxForwardEpilogue>
        <<<grid, block, block.x * sizeof(float), stream>>>(output, const_cast<float*>(input),
                                                            softmax_elements, input_stride, output_stride, temperature);
  }
}
template void DispatchBlockwiseSoftmaxForward<true>(cudaStream_t, float*, const float*, int, int, int, int, float);

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
    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
                                                       d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream);
  } else {
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
                                             d_values_in, d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream);
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
    cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_storage_bytes, d_keys_in, (T*)nullptr,
                                                       d_values_in, (int*)nullptr, num_items, num_segments, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream);
  } else {
    cub::DeviceSegmentedRadixSort::SortPairs(nullptr, temp_storage_bytes, d_keys_in, (T*)nullptr,
                                             d_values_in, (int*)nullptr, num_items, num_segments, d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream);
  }
}

// Sampling Kernels and Launchers

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
      prefix_sums[local_index + batch * sample_range] = prefix_sum;
    }
  }
}

// Get top k indices and scores from unsorted input
template <int max_k, int kBlockSize>
__global__ void GetTopKKernel(int* indices_out, float* scores_in, float* scores_out, int batch_size, int vocab_size, int k) {
  TopK<float, max_k> thread_top_k;
  thread_top_k.Init();
  int batch = blockIdx.x;

  for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    thread_top_k.Insert(scores_in[i + batch * vocab_size], i);
  }

  // reduce in thread block
  typedef cub::BlockReduce<TopK<float, max_k>, kBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  TopK<float, max_k> top_k_sequence = BlockReduce(temp_storage).Reduce(thread_top_k, reduce_topk_op<float, max_k>);
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int i = 0; i < k; i++) {
      scores_out[i + batch * k] = top_k_sequence.value[i];
      indices_out[i + batch * k] = top_k_sequence.key[i];
    }
  }
}

// Gets all top K indices and scores from unsorted input
template <int max_k>
void LaunchGetTopKSubset(cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k) {
  dim3 grid(batch_size, 1, 1);
  dim3 block(256, 1, 1);
  GetTopKKernel<max_k, 256><<<grid, block, 0, stream>>>(indices_out, scores_in, scores_out, batch_size, vocab_size, k);
}

// Sets up random thresholds for top p or top k sampling
__global__ void RandomThresholdKernelTopPAndK(curandState* curand_states, float* thresholds, float* prefix_sums, int batch_size, float p, int k) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  float k_prob = prefix_sums[k - 1];
  if (index < batch_size) {
    float min_p = fminf(p, k_prob);
    thresholds[index] = min_p * curand_uniform(&curand_states[index]);
  }
}

// Sets up random thresholds for top p or top k sampling
__global__ void RandomThresholdKernelTopP(curandState* curand_states, float* thresholds, float* prefix_sums, int batch_size, float p) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < batch_size) {
    thresholds[index] = p * curand_uniform(&curand_states[index]);
  }
}

// Sets up random thresholds for top p or top k sampling
__global__ void RandomThresholdKernelTopK(curandState* curand_states, float* thresholds, float* prefix_sums, int batch_size, int k) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < batch_size) {
    thresholds[index] = prefix_sums[k - 1] * curand_uniform(&curand_states[index]);
  }
}

template <int kBlockSize>
__global__ void SampleKernel(float* prefix_sums, int* indices, int* index_out, int sample_range, float* thresholds) {
  int batch = blockIdx.x;
  int index = threadIdx.x;

  __shared__ int first_index;
  if (threadIdx.x == 0) {
    first_index = sample_range - 1;
  }
  __syncthreads();

  for (; index < sample_range; index += blockDim.x) {
    if (index + batch * sample_range < blockDim.x * sample_range) {
      float sum = prefix_sums[index + batch * sample_range];
      // TOP P or K
      if (sum >= thresholds[batch] || index == sample_range - 1) {
        atomicMin(&first_index, index);
        break;
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    index_out[batch] = indices[first_index + batch * sample_range];
  }
}

void LaunchSampleKernel(SamplingData* data, cudaStream_t stream, float* scores, int* indices, int* index_out, int sample_range, int batch_size, float p = 0.0, int k = -1) {
  dim3 grid(batch_size, 1, 1);
  dim3 block(256, 1, 1);
  // Prefix Sums
  std::span<float> prefix_sums{data->prefix_sums.get(), static_cast<size_t>(sample_range * batch_size)};
  PrefixSumKernel<256><<<grid, block, 0, stream>>>(scores, prefix_sums.data(), sample_range, batch_size);
  // Random Thresholds for Top P or Top K Sampling
  std::span<float> thresholds{data->thresholds.get(), static_cast<size_t>(batch_size)};
  if (p > 0.0 && k > 1) {
    RandomThresholdKernelTopPAndK<<<int(batch_size / 128) + 1, 128, 0, stream>>>(data->curand_states.get(), thresholds.data(), prefix_sums.data(), batch_size, p, k);
  } else if (p > 0.0) {
    RandomThresholdKernelTopP<<<int(batch_size / 128) + 1, 128, 0, stream>>>(data->curand_states.get(), thresholds.data(), prefix_sums.data(), batch_size, p);
  } else if (k > 1) {
    RandomThresholdKernelTopK<<<int(batch_size / 128) + 1, 128, 0, stream>>>(data->curand_states.get(), thresholds.data(), prefix_sums.data(), batch_size, k);
  }
  SampleKernel<256><<<grid, block, 0, stream>>>(prefix_sums.data(), indices, index_out, sample_range, thresholds.data());
}

// Top P+K Kernel Launchers

// Outputs sorted scores and corresponding indices... scores_out and indices_out should already be allocated
void SoftmaxAndSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, float temperature) {
  // Softmax scores
  std::span<float> scores{data->scores_softmaxed.get(), static_cast<size_t>(vocab_size * batch_size)};
  DispatchBlockwiseSoftmaxForward<false>(stream, scores.data(), const_cast<const float*>(scores_in), vocab_size, vocab_size, vocab_size, batch_size, temperature);
  // Sort indices by scores
  std::span<int> offsets_gpu{data->offsets.get(), static_cast<size_t>(batch_size + 1)};
  LaunchPopulateOffsets(offsets_gpu.data(), vocab_size, batch_size, stream);
  std::span<int32_t> indices_in{data->indices_in.get(), static_cast<size_t>(vocab_size * batch_size)};
  LaunchPopulateIndices(indices_in.data(), vocab_size, batch_size, stream);
  std::span<float> temp_span{data->temp_buffer.get(), data->temp_storage_bytes / sizeof(float)};
  LaunchSortPairs<float>(temp_span.data(), data->temp_storage_bytes, scores.data(), scores_out,
                         indices_in.data(), indices_out, vocab_size * batch_size, batch_size, offsets_gpu.data(),
                         stream, /*is_descending*/ true);
}

void LaunchGetTopKSubsetFullSort(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k) {
  // Sort indices and scores
  std::span<float> scores_sorted{data->scores_sorted.get(), static_cast<size_t>(vocab_size * batch_size)};
  std::span<int> indices_sorted{data->indices_sorted.get(), static_cast<size_t>(vocab_size * batch_size)};
  std::span<int> offsets_gpu{data->offsets.get(), static_cast<size_t>(batch_size + 1)};
  LaunchPopulateOffsets(offsets_gpu.data(), vocab_size, batch_size, stream);
  std::span<int32_t> indices_in{data->indices_in.get(), static_cast<size_t>(vocab_size * batch_size)};
  LaunchPopulateIndices(indices_in.data(), vocab_size, batch_size, stream);
  std::span<float> temp_span{data->temp_buffer.get(), data->temp_storage_bytes / sizeof(float)};
  LaunchSortPairs<float>(temp_span.data(), data->temp_storage_bytes, scores_in, scores_sorted.data(),
                         indices_in.data(), indices_sorted.data(), vocab_size * batch_size, batch_size, offsets_gpu.data(),
                         stream, /*is_descending*/ true);
}

void GetTopKSubset(SamplingData* data, cudaStream_t stream, float* scores_in, float* scores_out, int* indices_out, int vocab_size, int batch_size, int k, float temperature) {
  // Softmax scores
  std::span<float> scores_softmaxed{data->scores_softmaxed.get(), static_cast<size_t>(vocab_size * batch_size)};
  DispatchBlockwiseSoftmaxForward<false>(stream, scores_softmaxed.data(), const_cast<const float*>(scores_in), vocab_size, vocab_size, vocab_size, batch_size, temperature);
// Get top k subset
#define GetTopK(max_k)                                \
  LaunchGetTopKSubset<max_k>(stream,                  \
                             scores_softmaxed.data(), \
                             scores_out,              \
                             indices_out,             \
                             vocab_size,              \
                             batch_size,              \
                             k)

  if (k <= 4) {
    GetTopK(4);
  } else if (k <= 8) {
    GetTopK(8);
  } else if (k <= 16) {
    GetTopK(16);
  } else if (k <= 32) {
    GetTopK(32);
  } else if (k <= 64) {
    GetTopK(64);
  } else {
    LaunchGetTopKSubsetFullSort(data, stream, scores_softmaxed.data(), scores_out, indices_out, vocab_size, batch_size, k);
  }
}

// Kernel launcher for combined (or seperate) top k and top p sampling; where k is the max number of tokens to sample and p is the probability threshold
void GetSample(SamplingData* data, cudaStream_t stream, int32_t* next_token_out, float* scores_in, int vocab_size, int batch_size, int k, float p, float temperature) {
  int sample_range = (k > 0 && k <= 64) ? k : vocab_size;
  std::span<float> scores_sorted(data->scores_sorted.get(), static_cast<size_t>(sample_range * batch_size));
  std::span<int> indices_sorted(data->indices_sorted.get(), static_cast<size_t>(sample_range * batch_size));
  if (k > 0 && k < vocab_size) {
    GetTopKSubset(data, stream, scores_in, scores_sorted.data(), indices_sorted.data(), vocab_size, batch_size, k, temperature);
  } else {
    SoftmaxAndSort(data, stream, scores_in, scores_sorted.data(), indices_sorted.data(), vocab_size, batch_size, temperature);
  }
  // Sample kernel
  LaunchSampleKernel(data, stream, scores_sorted.data(), indices_sorted.data(), next_token_out, sample_range, batch_size, p, k);
}

}  // namespace cuda
}  // namespace Generators
