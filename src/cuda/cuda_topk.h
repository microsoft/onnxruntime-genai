// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <curand_kernel.h>

#include "cuda_common.h"

namespace Generators {
namespace cuda {

constexpr int kHybridSortMaxK = 256;  // The maximum k allowed for hybrid sort.

// This struct holds all the device memory buffers and other data required for Top-K operations.
struct TopkData {
  TopkData(int batch_size, int vocab_size, cudaStream_t stream);
  TopkData() = delete;
  TopkData(const TopkData&) = delete;
  TopkData& operator=(const TopkData&) = delete;
  
  // The estimated best partition size for hybrid sort.
  int hybrid_sort_partition_size;

  // The estimated threshold to use selection sort instead of other algorithm when k <= threshold
  int selection_sort_k_threshold;

  // --- Intermediate Buffers for Top-K Algorithms ---

  // - Full sort - Holds top-k indices for output
  // - Selection sort: Holds top-k indices for output
  // - Hybrid sort: A "ping-pong" buffer for indices during the reduction phase.
  cuda_unique_ptr<int> intermediate_indices_1;

  // - Full sort - Holds the initial vocabulary indices before sorting.
  // - Hybrid sort - A "ping-pong" buffer for indices during the reduction phase.
  cuda_unique_ptr<int> intermediate_indices_2;

  // - Full sort: Holds the fully sorted raw scores.
  // - Selection sort: Holds the top-k scores for selection sort.
  // - Hybrid sort: A "ping-pong" buffer for raw scores during the reduction phase.
  cuda_unique_ptr<float> intermediate_scores_1;

  // - Selection sort: Holds a copy of input scores. Will be updated in place by selection sort kernel.
  // - Hybrid sort: A "ping-pong" buffer for raw scores during the reduction phase.
  cuda_unique_ptr<float> intermediate_scores_2;

  // - Full sort: General-purpose temporary storage for CUB's DeviceSegmentedRadixSort
  cuda_unique_ptr<unsigned char> cub_temp_storage;
  size_t cub_temp_storage_bytes = 0;

  // - Full sort: Stores the start offset of each batch segment for CUB's segmented sort
  cuda_unique_ptr<int> batch_offsets;

  // --- Information of Final Output (Input to Sampling Stage) ---
  const float* topk_scores = nullptr;  // pointer to the top-k scores data (in either intermediate_scores_1 or intermediate_scores_2)
  const int* topk_indices = nullptr;   // pointer to the top-k indices data (in either intermediate_indices_1 or intermediate_indices_2)
  int topk_stride = 0;                 // stride of the top-k output data: k for selection sort, vocab_size for full sort, kHybridSortMaxK for hybrid sort
};

// For parity test, a derived struct to help compact output buffers.
struct TopkDataCompact : public TopkData {
  TopkDataCompact(int batch_size, int vocab_size, cudaStream_t stream)
      : TopkData(batch_size, vocab_size, stream) {}
  TopkDataCompact() = delete;
  TopkDataCompact(const TopkDataCompact&) = delete;
  TopkDataCompact& operator=(const TopkDataCompact&) = delete;

  void CompactOutput(int batch_size, int vocab_size, cudaStream_t stream, int k);

  cuda_unique_ptr<float> topk_scores_compact;  // compact [batch_size, k] output scores
  cuda_unique_ptr<int> topk_indices_compact;   // compact [batch_size, k] output indices
};

// Main dispatcher for Top-K. Used by the sampling logic. The topk_data will be updated for output pointers and stride.
void GetTopK(TopkData* topk_data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);

// Top-K algorithm implementations. These are exposed for testing and benchmarking.
void RunTopKViaSelectionSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
void RunTopKViaFullSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
void RunTopKViaHybridSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);

void RunTopKViaBitonicSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);

void RunTopKViaRadixSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);

}  // namespace cuda
}  // namespace Generators
