// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/device/device_segmented_radix_sort.cuh>

#include "cuda_topk.h"
#include "cuda_topk_benchmark_cache.h"
#include "cuda_topk_benchmark.cuh"
#include "cuda_topk_full_sort.cuh"
#include "cuda_topk_radix_sort.cuh"
#include "cuda_topk_partition_sort.cuh"
#include "cuda_topk_distributed_select_sort.cuh"
#include "cuda_topk_hybrid_sort.cuh"
#include "cuda_topk_flash_sort.cuh"
#include "cuda_topk_llm_sort.cuh"
#include "cuda_topk_select_sort.cuh"
#include <cassert>

namespace Generators {
namespace cuda {

TopkDataDetail::TopkDataDetail(int batch_size, int vocab_size, cudaStream_t stream) {
  hybrid_sort_partition_size = hybrid_sort::EstimateBestPartitionSize(vocab_size);
  flash_sort_partition_size = flash_sort::EstimateBestPartitionSize(vocab_size);
  llm_sort_partition_size = llm_sort::EstimateBestPartitionSize(vocab_size);
  partition_sort_partition_size = radix_partition_sort::EstimateBestPartitionSize(vocab_size);

  size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;
  intermediate_buffer_elements = std::max(
      {vocab_batch_size,
       hybrid_sort::GetIntermediateSize(batch_size, vocab_size, hybrid_sort_partition_size),
       flash_sort::GetIntermediateSize(batch_size, vocab_size, flash_sort_partition_size),
       llm_sort::GetIntermediateSize(batch_size, vocab_size, llm_sort_partition_size),
       radix_partition_sort::GetIntermediateSize(batch_size, vocab_size, partition_sort_partition_size)});

  auto radix_sort_temp_storage_bytes = radix_sort::GetTempStorageBytes(vocab_size, stream);
  auto full_sort_temp_storage_bytes = full_sort::GetTempStorageBytes(static_cast<int>(vocab_batch_size), batch_size, stream);
  cub_temp_storage_bytes = std::max(radix_sort_temp_storage_bytes, full_sort_temp_storage_bytes);
}

size_t TopkData::CalculateTotalSize(int batch_size, int vocab_size, cudaStream_t stream) {
  TopkDataDetail detail(batch_size, vocab_size, stream);

  size_t total_size = 0;
  total_size += AlignUp(detail.intermediate_buffer_elements * sizeof(int), kGpuBufferAlignment);
  total_size += AlignUp(detail.intermediate_buffer_elements * sizeof(int), kGpuBufferAlignment);
  total_size += AlignUp(detail.intermediate_buffer_elements * sizeof(float), kGpuBufferAlignment);
  total_size += AlignUp(detail.intermediate_buffer_elements * sizeof(float), kGpuBufferAlignment);
  total_size += AlignUp((batch_size + 1) * sizeof(int), kGpuBufferAlignment);
  total_size += AlignUp(detail.cub_temp_storage_bytes, kGpuBufferAlignment);

  // Space for distributed selection sort for the lock and sorted keys/values
  total_size += AlignUp(topk_impl_details::kTopKDistributedSelectSortMaxBatchSize * sizeof(int), kGpuBufferAlignment);
  total_size += AlignUp(topk_impl_details::kTopKDistributedSelectSortMaxShards * topk_impl_details::kTopKDistributedSelectSortMaxTopK * sizeof(int), kGpuBufferAlignment);
  total_size += AlignUp(topk_impl_details::kTopKDistributedSelectSortMaxShards * topk_impl_details::kTopKDistributedSelectSortMaxTopK * sizeof(float), kGpuBufferAlignment);

  return total_size;
}

void TopkData::InitializeBuffers(int batch_size, int vocab_size, cudaStream_t stream) {
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

TopkData::TopkData(int batch_size, int vocab_size, cudaStream_t stream, void* buffer, size_t buffer_size)
    : TopkDataDetail(batch_size, vocab_size, stream) {
  // Get and cache the device ID once during initialization.
  CUDA_CHECK(cudaGetDevice(&device_id));

  // Initialize the local cache.
  local_algo_cache_.fill(TopkAlgo::UNKNOWN);

  if (buffer) {
    // Wrap an externally provided buffer. The caller is responsible for the size.
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
  cudaMemset(top_k_distributed_select_sort_lock, 0, sizeof(int));
}

template <typename T>
__global__ void CompactStridedData(const T* input, T* output, int k, int batch_size, int input_stride) {
  const int batch_idx = blockIdx.x;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    int in_idx = batch_idx * input_stride + i;
    int out_idx = batch_idx * k + i;
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

void RunTopK(TopkData* topk_data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k) {
  assert(topk_data != nullptr);
  assert(vocab_size > 0);
  assert(batch_size > 0);
  assert(k > 0 && k <= vocab_size);

  // Check the local cache first.
  TopkAlgo algo = (k <= kMaxBenchmarkLocalCache) ? topk_data->local_algo_cache_.at(k) : TopkAlgo::UNKNOWN;
  if (algo == TopkAlgo::UNKNOWN) {
    // Local cache miss, check the global persistent cache.
    algo = GetTopkBenchmarkCache(topk_data->device_id, batch_size, vocab_size, k);

    if (algo == TopkAlgo::UNKNOWN) {
      // Global cache also miss, run benchmark.
      algo = BenchmarkAndSelectBestAlgo(topk_data, stream, scores_in, vocab_size, batch_size, k);
    }

    // Update the local cache for subsequent calls.
    if (k <= kMaxBenchmarkLocalCache) {
      topk_data->local_algo_cache_[k] = algo;
    }
  }

  switch (algo) {
    case TopkAlgo::SELECTION:
      selection_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::HYBRID:
      hybrid_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::FLASH:
      flash_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::LLM:
      llm_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::PARTITION:
      radix_partition_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::DISTRIBUTED:
      distributed_select_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::RADIX:
      radix_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    case TopkAlgo::FULL:
      full_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
      return;
    default:
      // Fallback if something went wrong during benchmarking.
      break;
  }

  // Full sort is the fallback.
  full_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
}

}  // namespace cuda
}  // namespace Generators
