// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/device/device_segmented_radix_sort.cuh>
#include <algorithm>  // For std::max
#include <cassert>

#include "cuda_topk.h"
#include "cuda_topk_benchmark_cache.h"
#include "cuda_topk_benchmark.cuh"
#include "cuda_topk_common.cuh"
#include "cuda_topk_full_sort.cuh"
#include "cuda_topk_per_batch_radix_sort.cuh"
#include "cuda_topk_flash_convergent.cuh"
#include "cuda_topk_distributed_select_sort.cuh"
#include "cuda_topk_hybrid_sort.cuh"
#include "cuda_topk_iterative_sort.cuh"
#include "cuda_topk_cascaded_sort.cuh"
#include "cuda_topk_select_sort.cuh"
#include "cuda_topk_sort_benchmark_cache.h"

namespace Generators {
namespace cuda {

// Constructor for the host-side parameter planning struct.
TopkDataDetail::TopkDataDetail(int batch_size, int vocab_size, cudaStream_t stream) {
  // Partition sizes are now calculated just-in-time in the benchmark/run functions
  // based on the specific `k` value for the operation.
  hybrid_sort_partition_size = 0;
  iterative_sort_partition_size = 0;
  cascaded_sort_partition_size = 0;
  flash_convergent_partition_size = 0;
  flash_convergent_partition_size_k = 0;

  // Calculate the maximum possible size for intermediate buffers. This is determined
  // by the algorithm that requires the most space. The formula is:
  // batch_size * CeilDiv(vocab_size, min_partition_size) * max_k
  constexpr int kFastSortMaxK = std::max({kHybridSortMaxK, kIterativeSortMaxK, kCascadedSortMaxK, kConvergentSortMaxK});
  int max_fast_sort_intermediate_size = batch_size * CeilDiv(vocab_size, kFastSortMinPartitionSize) * kFastSortMaxK;

  // The intermediate buffer must be large enough to hold either the full vocabulary (for full_sort)
  // or the maximum number of candidates from the partition-based sorts.
  size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;
  intermediate_buffer_elements = std::max(vocab_batch_size, static_cast<size_t>(max_fast_sort_intermediate_size));

  // Determine the CUB temporary storage size, which is the maximum required by either
  // the batched radix sort or the full segmented sort.
  auto per_batch_radix_sort_temp_storage_bytes = per_batch_radix_sort::GetTempStorageBytes(vocab_size, stream);
  auto full_sort_temp_storage_bytes = full_sort::GetTempStorageBytes(static_cast<int>(vocab_batch_size), batch_size, stream);
  cub_temp_storage_bytes = std::max(per_batch_radix_sort_temp_storage_bytes, full_sort_temp_storage_bytes);
}

// Calculates the total size of the single device memory buffer needed.
size_t TopkData::CalculateTotalSize(int batch_size, int vocab_size, cudaStream_t stream) {
  TopkDataDetail detail(batch_size, vocab_size, stream);

  size_t total_size = 0;
  // Two pairs of ping-pong buffers for scores and indices.
  total_size += AlignUp(detail.intermediate_buffer_elements * sizeof(int), kGpuBufferAlignment);
  total_size += AlignUp(detail.intermediate_buffer_elements * sizeof(int), kGpuBufferAlignment);
  total_size += AlignUp(detail.intermediate_buffer_elements * sizeof(float), kGpuBufferAlignment);
  total_size += AlignUp(detail.intermediate_buffer_elements * sizeof(float), kGpuBufferAlignment);
  // Buffer for batch offsets for CUB's segmented sort.
  total_size += AlignUp((batch_size + 1) * sizeof(int), kGpuBufferAlignment);
  // Temporary storage for CUB.
  total_size += AlignUp(detail.cub_temp_storage_bytes, kGpuBufferAlignment);
  // Buffers for distributed selection sort.
  total_size += AlignUp(topk_impl_details::kTopKDistributedSelectSortMaxBatchSize * sizeof(int), kGpuBufferAlignment);
  total_size += AlignUp(topk_impl_details::kTopKDistributedSelectSortMaxShards * topk_impl_details::kTopKDistributedSelectSortMaxTopK * sizeof(int), kGpuBufferAlignment);
  total_size += AlignUp(topk_impl_details::kTopKDistributedSelectSortMaxShards * topk_impl_details::kTopKDistributedSelectSortMaxTopK * sizeof(float), kGpuBufferAlignment);

  return total_size;
}

// Partitions the single memory buffer into individual pointers.
void TopkData::InitializeBuffers(int batch_size, int /*vocab_size*/, cudaStream_t /*stream*/) {
  uint8_t* current_ptr = memory_buffer_span_.data();

  intermediate_indices_1 = reinterpret_cast<int*>(current_ptr);
  current_ptr += AlignUp(intermediate_buffer_elements * sizeof(int), kGpuBufferAlignment);

  intermediate_indices_2 = reinterpret_cast<int*>(current_ptr);
  current_ptr += AlignUp(intermediate_buffer_elements * sizeof(int), kGpuBufferAlignment);

  intermediate_scores_1 = reinterpret_cast<float*>(current_ptr);
  current_ptr += AlignUp(intermediate_buffer_elements * sizeof(float), kGpuBufferAlignment);

  intermediate_scores_2 = reinterpret_cast<float*>(current_ptr);
  current_ptr += AlignUp(intermediate_buffer_elements * sizeof(float), kGpuBufferAlignment);

  batch_offsets = reinterpret_cast<int*>(current_ptr);
  current_ptr += AlignUp((batch_size + 1) * sizeof(int), kGpuBufferAlignment);

  cub_temp_storage = reinterpret_cast<unsigned char*>(current_ptr);
  current_ptr += AlignUp(cub_temp_storage_bytes, kGpuBufferAlignment);

  top_k_distributed_select_sort_lock = reinterpret_cast<int*>(current_ptr);
  current_ptr += AlignUp(topk_impl_details::kTopKDistributedSelectSortMaxBatchSize * sizeof(int), kGpuBufferAlignment);

  top_k_distributed_select_sort_keys = reinterpret_cast<int*>(current_ptr);
  current_ptr += AlignUp(topk_impl_details::kTopKDistributedSelectSortMaxShards * topk_impl_details::kTopKDistributedSelectSortMaxTopK * sizeof(int), kGpuBufferAlignment);

  top_k_distributed_select_sort_values = reinterpret_cast<float*>(current_ptr);
  current_ptr += AlignUp(topk_impl_details::kTopKDistributedSelectSortMaxShards * topk_impl_details::kTopKDistributedSelectSortMaxTopK * sizeof(float), kGpuBufferAlignment);
}

// Constructor for the main data management struct.
TopkData::TopkData(int batch_size, int vocab_size, cudaStream_t stream, void* buffer, size_t buffer_size)
    : TopkDataDetail(batch_size, vocab_size, stream) {
  CUDA_CHECK(cudaGetDevice(&device_id));

  local_algo_cache_.fill(TopkAlgo::UNKNOWN);

  if (buffer) {
    // Wrap an externally provided buffer. The caller is responsible for ensuring the size is sufficient.
    assert(buffer_size >= CalculateTotalSize(batch_size, vocab_size, stream));
    memory_buffer_span_ = std::span<uint8_t>(static_cast<uint8_t*>(buffer), buffer_size);
  } else {
    // Self-allocate. If buffer_size is provided by a derived class, use it.
    // Otherwise, calculate the size needed for this base class.
    size_t self_size = (buffer_size > 0) ? buffer_size : CalculateTotalSize(batch_size, vocab_size, stream);
    memory_buffer_owner_ = CudaMallocArray<uint8_t>(self_size);
    memory_buffer_span_ = std::span<uint8_t>(memory_buffer_owner_.get(), self_size);
  }

  InitializeBuffers(batch_size, vocab_size, stream);

  // TODO: we shall cache deviceProp with inference session so that we need not query device property every time.
  cudaDeviceProp deviceProp;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device_id));

  // Initialize metadata for distributed selection sort.
  top_k_distributed_select_sort_shards = std::min(topk_impl_details::kTopKDistributedSelectSortMaxShards, deviceProp.multiProcessorCount);
  CUDA_CHECK(cudaMemsetAsync(top_k_distributed_select_sort_lock, 0, sizeof(int), stream));
}

// A kernel to compact output data from a strided layout to a contiguous one.
// This is useful for testing when an algorithm produces a non-contiguous output.
template <typename T>
__global__ void CompactStridedData(const T* input, T* output, int k, int batch_size, int input_stride) {
  const int batch_idx = blockIdx.x;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    size_t in_idx = static_cast<size_t>(batch_idx) * input_stride + i;
    size_t out_idx = static_cast<size_t>(batch_idx) * k + i;
    output[out_idx] = input[in_idx];
  }
}

void TopkDataCompact::CompactOutput(int batch_size, int k, cudaStream_t stream) {
  topk_scores_compact = CudaMallocArray<float>(static_cast<size_t>(batch_size) * k);
  topk_indices_compact = CudaMallocArray<int>(static_cast<size_t>(batch_size) * k);
  dim3 grid(batch_size);
  dim3 block(256);
  CompactStridedData<float><<<grid, block, 0, stream>>>(topk_scores, topk_scores_compact.get(), k, batch_size, topk_stride);
  CompactStridedData<int><<<grid, block, 0, stream>>>(topk_indices, topk_indices_compact.get(), k, batch_size, topk_stride);
}

// Main dispatcher for Top-K. It implements the caching and benchmarking logic to select and run the best algorithm.
void RunTopK(TopkData* topk_data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(topk_data != nullptr);
  assert(vocab_size > 0);
  assert(batch_size > 0);
  assert(k > 0 && k <= vocab_size);

  // 1. Check the fast, local cache first.
  TopkAlgo algo = (k <= kMaxBenchmarkLocalCache) ? topk_data->local_algo_cache_[k] : TopkAlgo::UNKNOWN;

  if (algo == TopkAlgo::UNKNOWN) {
    // 2. Local cache miss, check the persistent global cache.
    algo = GetTopkBenchmarkCache(topk_data->device_id, batch_size, vocab_size, k);

    if (algo == TopkAlgo::UNKNOWN) {
      // 3. Global cache also misses, run the online benchmark to find the best algorithm.
      algo = BenchmarkAndSelectBestAlgo(topk_data, stream, scores_in, vocab_size, batch_size, k);
    }

    // Update the local cache for subsequent calls within this session.
    if (k <= kMaxBenchmarkLocalCache) {
      topk_data->local_algo_cache_[k] = algo;
    }
  }

  // Dispatch to the selected algorithm.
  switch (algo) {
    case TopkAlgo::SELECTION:
      select_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::HYBRID:
      hybrid_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::ITERATIVE:
      iterative_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::CASCADED:
      cascaded_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::CONVERGENT:
      flash_convergent::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::DISTRIBUTED_SELECT:
      distributed_select_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::PER_BATCH_RADIX:
      per_batch_radix_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::FULL:
      full_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    default:
      // Fallback if something went wrong during benchmarking.
      break;
  }

  // `full_sort` is the ultimate fallback to guarantee correctness.
  full_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
}

}  // namespace cuda
}  // namespace Generators
