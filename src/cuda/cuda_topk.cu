// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/device/device_segmented_radix_sort.cuh>

#include "cuda_topk.h"
#include "cuda_topk_benchmark.cuh"
#include "cuda_topk_full_sort.cuh"
#include "cuda_topk_radix_sort.cuh"
#include "cuda_topk_hybrid_sort.cuh"
#include "cuda_topk_select_sort.cuh"
#include "cuda_topk_select_sort_distributed.cuh"

namespace Generators {
namespace cuda {

TopkData::TopkData(int batch_size, int vocab_size, cudaStream_t stream) {
  hybrid_sort_partition_size = EstimateHybridSortBestPartitionSize(vocab_size);

  // Initialize the benchmark cache with UNKNOWN.
  for (int i = 0; i <= kMaxBenchmarkK; ++i) {
    best_algo_cache[i] = TopkAlgo::UNKNOWN;
  }

  // --- Buffer Size Calculation ---
  // We must calculate the maximum potential buffer sizes across all algorithms that use the intermediate buffers.

  // Size needed for intermediate reduction buffers in HybridSort.
  size_t hybrid_sort_buffer_elements = GetHybridSortIntermediateSize(batch_size, vocab_size, hybrid_sort_partition_size);

  // Size needed for other algorithms (like RadixSort, FullSort).
  size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;

  // The intermediate buffers must be large enough for any of these scenarios.
  size_t max_buffer_elements = std::max({vocab_batch_size, hybrid_sort_buffer_elements});

  // Allocate all necessary device memory
  // TODO: we shall consider allocating from a pre-allocated memory pool instead of separate cudaMalloc calls.
  intermediate_indices_1 = CudaMallocArray<int>(max_buffer_elements);
  intermediate_indices_2 = CudaMallocArray<int>(max_buffer_elements);
  intermediate_scores_1 = CudaMallocArray<float>(max_buffer_elements);
  intermediate_scores_2 = CudaMallocArray<float>(max_buffer_elements);
  batch_offsets = CudaMallocArray<int>(batch_size + 1);

  auto radix_sort_temp_storage_bytes = GetRadixSortCubTempStorageBytes(vocab_size, stream);
  auto full_sort_temp_storage_bytes = GetFullSortCubTempStorageBytes(vocab_batch_size, batch_size, stream);
  cub_temp_storage_bytes = std::max(radix_sort_temp_storage_bytes, full_sort_temp_storage_bytes);
  cub_temp_storage = CudaMallocArray<unsigned char>(this->cub_temp_storage_bytes);

  // Allocate buffers for distributed sort
  top_k_distributed_lock = CudaMallocArray<int>(batch_size);
  cudaMemset(top_k_distributed_lock.get(), 0, batch_size * sizeof(int));

  int device = -1;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp device_prop;
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device));
  top_k_shards = std::min(top_k_shards, device_prop.multiProcessorCount);

  // Allocate a sparse buffer for intermediate candidates: batch_size * num_shards * max_k
  size_t dist_buffer_size = static_cast<size_t>(batch_size) * top_k_shards * kDistributedSortMaxK;
  top_k_distributed_keys = CudaMallocArray<int>(dist_buffer_size);
  top_k_distributed_values = CudaMallocArray<float>(dist_buffer_size);
}

// Kernel to compact strided data into a dense layout.
// Used to convert data from a [batch, stride] layout to a dense [batch, k] layout.
template <typename T>
__global__ void CompactStridedData(const T* input, T* output, int k, int batch_size, int input_stride) {
  const int batch_idx = blockIdx.x;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    int in_idx = batch_idx * input_stride + i;
    int out_idx = batch_idx * k + i;
    output[out_idx] = input[in_idx];
  }
}

void TopkDataCompact::CompactOutput(int batch_size, int vocab_size, cudaStream_t stream, int k) {
  topk_scores_compact = CudaMallocArray<float>(static_cast<size_t>(batch_size) * k);
  topk_indices_compact = CudaMallocArray<int>(static_cast<size_t>(batch_size) * k);
  dim3 grid(batch_size);
  dim3 block(256);
  CompactStridedData<float><<<grid, block, 0, stream>>>(topk_scores, topk_scores_compact.get(), k, batch_size, topk_stride);
  CompactStridedData<int><<<grid, block, 0, stream>>>(topk_indices, topk_indices_compact.get(), k, batch_size, topk_stride);
}

void RunTopK(TopkData* topk_data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(topk_data != nullptr);
  assert(vocab_size > 0);
  assert(batch_size > 0);
  assert(k > 0 && k <= vocab_size);

  // For small k, use online benchmarking to find the best algorithm and cache the result.
  if (k <= kMaxBenchmarkK) {
    TopkAlgo algo = topk_data->best_algo_cache[k];
    if (algo == TopkAlgo::UNKNOWN) {
      // First time for this k, run benchmark.
      algo = BenchmarkAndSelectBestAlgo(topk_data, stream, scores_in, vocab_size, batch_size, k);
    }

    switch (algo) {
      case TopkAlgo::SELECTION:
        RunTopKViaSelectionSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
        return;
      case TopkAlgo::DISTRIBUTED:
        RunTopKViaDistributedSelectionSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
        return;
      case TopkAlgo::HYBRID:
        RunTopKViaHybridSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
        return;
      default:
        // Fallback to heuristics if something went wrong during benchmarking.
        break;
    }
  }

  if (k <= kHybridSortMaxK) {
    RunTopKViaHybridSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
    return;
  }

  // For very large k, CUB-based segmented full sort or per-batch radix sort are the fallbacks.
  if (batch_size <= 2) {
    RunTopKViaRadixSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
  } else {
    RunTopKViaFullSort(topk_data, stream, scores_in, vocab_size, batch_size, k);
  }
}

}  // namespace cuda
}  // namespace Generators
