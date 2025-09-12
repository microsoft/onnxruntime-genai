// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../generators.h"
#include "cuda_topk.h"
#include "cuda_topk_benchmark_cache.h"
#include "cuda_topk_benchmark.cuh"
#include "cuda_topk_full_sort.cuh"
#include "cuda_topk_radix_sort.cuh"
#include "cuda_topk_hybrid_sort.cuh"
#include "cuda_topk_flash_sort.cuh"
#include "cuda_topk_select_sort.cuh"
#include <cassert>
#include <cstdio> // For printf

namespace Generators {
namespace cuda {

size_t TopkData::CalculateTotalSize(int batch_size, int vocab_size, cudaStream_t stream) {
  size_t total_size = 0;

  int partition_size = hybrid_sort::EstimateBestPartitionSize(vocab_size);
  size_t hybrid_sort_buffer_elements = hybrid_sort::GetIntermediateSize(batch_size, vocab_size, partition_size);
  size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;
  size_t max_buffer_elements = std::max({vocab_batch_size, hybrid_sort_buffer_elements});

  total_size += AlignUp(max_buffer_elements * sizeof(int), kGpuBufferAlignment);
  total_size += AlignUp(max_buffer_elements * sizeof(int), kGpuBufferAlignment);
  total_size += AlignUp(max_buffer_elements * sizeof(float), kGpuBufferAlignment);
  total_size += AlignUp(max_buffer_elements * sizeof(float), kGpuBufferAlignment);
  total_size += AlignUp((batch_size + 1) * sizeof(int), kGpuBufferAlignment);

  auto radix_sort_temp_storage_bytes = radix_sort::GetTempStorageBytes(vocab_size, stream);
  auto full_sort_temp_storage_bytes = full_sort::GetTempStorageBytes(vocab_batch_size, batch_size, stream);
  size_t temp_storage_bytes = std::max(radix_sort_temp_storage_bytes, full_sort_temp_storage_bytes);
  total_size += AlignUp(temp_storage_bytes, kGpuBufferAlignment);

  return total_size;
}

void TopkData::InitializeBuffers(int batch_size, int vocab_size, cudaStream_t stream) {
  uint8_t* current_ptr = memory_buffer_span_.data();
  size_t max_buffer_elements = std::max({static_cast<size_t>(vocab_size) * batch_size,
                                         hybrid_sort::GetIntermediateSize(batch_size, vocab_size, hybrid_sort_partition_size)});

  intermediate_indices_1 = reinterpret_cast<int*>(current_ptr);
  current_ptr += AlignUp(max_buffer_elements * sizeof(int), kGpuBufferAlignment);

  intermediate_indices_2 = reinterpret_cast<int*>(current_ptr);
  current_ptr += AlignUp(max_buffer_elements * sizeof(int), kGpuBufferAlignment);

  intermediate_scores_1 = reinterpret_cast<float*>(current_ptr);
  current_ptr += AlignUp(max_buffer_elements * sizeof(float), kGpuBufferAlignment);

  intermediate_scores_2 = reinterpret_cast<float*>(current_ptr);
  current_ptr += AlignUp(max_buffer_elements * sizeof(float), kGpuBufferAlignment);

  batch_offsets = reinterpret_cast<int*>(current_ptr);
  current_ptr += AlignUp((batch_size + 1) * sizeof(int), kGpuBufferAlignment);

  size_t vocab_batch_size = static_cast<size_t>(vocab_size) * batch_size;
  auto radix_sort_temp_storage_bytes = radix_sort::GetTempStorageBytes(vocab_size, stream);
  auto full_sort_temp_storage_bytes = full_sort::GetTempStorageBytes(vocab_batch_size, batch_size, stream);
  cub_temp_storage_bytes = std::max(radix_sort_temp_storage_bytes, full_sort_temp_storage_bytes);

  cub_temp_storage = reinterpret_cast<unsigned char*>(current_ptr);
  current_ptr += AlignUp(cub_temp_storage_bytes, kGpuBufferAlignment);
}

TopkData::TopkData(int batch_size, int vocab_size, cudaStream_t stream, void* buffer, size_t buffer_size) {
  hybrid_sort_partition_size = hybrid_sort::EstimateBestPartitionSize(vocab_size);

  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  best_algo_cache_ = GetTopkBenchmarkCache(device_id, batch_size, vocab_size);

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

  // For small k, use online benchmarking to find the best algorithm and cache the result.
  if (k <= kMaxBenchmarkK) {
    TopkAlgo algo = topk_data->best_algo_cache_->at(k);
    if (algo == TopkAlgo::UNKNOWN) {
      // First time for this k, run benchmark.
      algo = BenchmarkAndSelectBestAlgo(topk_data->best_algo_cache_, topk_data, stream, scores_in, vocab_size, batch_size, k);
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
      default:
        // Fallback to below algorithms if something went wrong during benchmarking.
        break;
    }
  }

  if (k <= kHybridSortMaxK) {
    hybrid_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
    return;
  }

  // For very large k, CUB-based segmented full sort or per-batch radix sort are the fallbacks.
  if (batch_size <= 2) {
    radix_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
  } else {
    full_sort::RunTopK(topk_data, stream, scores_in, vocab_size, batch_size, k);
  }
}

}  // namespace cuda
}  // namespace Generators

