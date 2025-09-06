// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <curand_kernel.h>

#include "cuda_common.h"

namespace Generators {
namespace cuda {

constexpr int kHybridSortMaxK = 64;  // up to 256.

// This struct holds all the device memory buffers required for Top-K operations.
// The user of this struct is responsible for allocating and managing the memory.
struct TopkData {
  TopkData(int batch_size, int vocab_size, cudaStream_t stream);

  // --- Intermediate Buffers for Top-K Algorithms ---

  // Used to hold initial vocabulary indices for full sort, and intermediate
  // indices during the reduction phase of hybrid sort.
  cuda_unique_ptr<int> intermediate_indices_1;

  // A dedicated "ping-pong" buffer for the hybrid sort index reduction.
  cuda_unique_ptr<int> intermediate_indices_2;

  // Primary buffer for holding raw scores.
  // - Full sort: Holds the fully sorted raw scores.
  // - Selection sort: Not used directly for output, but reserved.
  // - Hybrid sort: Holds intermediate and final reduced raw scores.
  cuda_unique_ptr<float> intermediate_scores_1;

  // A secondary "ping-pong" buffer used by the hybrid sort's score reduction phase.
  cuda_unique_ptr<float> intermediate_scores_2;

  // General-purpose temporary storage for CUB's DeviceSegmentedRadixSort (for full sort only).
  cuda_unique_ptr<unsigned char> cub_temp_storage;
  size_t cub_temp_storage_bytes = 0;

  // Stores the start offset of each batch segment for CUB's segmented sort (for full sort only).
  cuda_unique_ptr<int> batch_offsets;

  // --- Information of Final Output (Input to Sampling Stage) ---
  const float* topk_scores = nullptr;  // pointer to the top-k scores data (in intermediate_scores_1 or intermediate_scores_2)
  const int* topk_indices = nullptr;   // pointer to the top-k indices data (in intermediate_indices_1 or intermediate_indices_2)
  int topk_stride = 0;                 // stride of the top-k output data: k for selection sort, vocab_size for full sort, max_k (kHybridSortMaxK) for hybrid sort
};

// For parity test, a derived struct to help compact output buffers.
struct TopkDataCompact : public TopkData {
  TopkDataCompact(int batch_size, int vocab_size, cudaStream_t stream)
      : TopkData(batch_size, vocab_size, stream) {}
  void CompactOutput(int batch_size, int vocab_size, cudaStream_t stream, int k);
  cuda_unique_ptr<float> topk_scores_compact;  // compact [batch_size, k] output scores
  cuda_unique_ptr<int> topk_indices_compact;   // compact [batch_size, k] output indices
};

// Main dispatcher for Top-K. Used by the sampling logic. The topk_data will be updated for output pointers and stride.
void GetTopK(TopkData* topk_data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);

// The specific Top-K algorithm implementations. These are exposed for testing and benchmarking.
// They all adhere to the same contract: find the top `k` raw logits and indices and write them
// to `scores_out` and `indices_out` in a compact [batch_size, k] layout.
void RunTopKViaSelectionSort(TopkData* data, cudaStream_t stream, float* scores_in, int vocab_size, int batch_size, int k);
void RunTopKViaFullSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
void RunTopKViaHybridSort(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k, int partition_size);

}  // namespace cuda
}  // namespace Generators
