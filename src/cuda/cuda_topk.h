// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <curand_kernel.h>
#include <memory>
#include <array>

#include "cuda_common.h"
#include "../smartptrs.h"

namespace Generators {
namespace cuda {

// To enable stable Top-K for kernels that support it, define STABLE_TOPK during compilation.
// By default, a faster, unstable sort is used for the initial partitioning stage.
#ifdef STABLE_TOPK
constexpr bool kStableTopK = true;
#else
constexpr bool kStableTopK = false;
#endif

// Helper to align memory addresses.
__host__ __device__ inline size_t AlignUp(size_t size, size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

constexpr int kGpuBufferAlignment = 256;

// Most common K value used in top-k sampling for Large Language Models typically falls in the range of 20 to 50.
// So max k values shall be 64 or higher.
// We enable higher k values in some algorithms for testing. The drawback is larger buffer sizes and longer compilation time.

// Redix sort uses registers in stage 2 so it could support no more than 64 due to limit of registers.
constexpr int kRadixSortMaxK = 64;           // The maximum k (up to 64) allowed for radix sort. Must be power of 2.
constexpr int kHybridSortMaxK = 256;         // The maximum k (up to 256) allowed for hybrid sort. Must be power of 2.
constexpr int kFlashSortMaxK = 128;          // The maximum k (up to 256) allowed for flash sort. Must be power of 2.
constexpr int kLlmSortMaxK = 64;             // The maximum k (up to 256) allowed for LLM sort. Must be power of 2.
constexpr int kMaxBenchmarkLocalCache = 64;  // The maximum local cache of online benchmarking results.

namespace topk_impl_details {
constexpr int kTopKDistributedSelectSortMaxShards = 32;
constexpr int kTopKDistributedSelectSortMaxBatchSize = 1;
constexpr int kTopKDistributedSelectSortMaxTopK = 64;
constexpr int kTopKDistributedSelectSortMinVocabSize = 100000;
}  // namespace topk_impl_details

// Enum for the different Top-K algorithms used in online benchmarking.
enum class TopkAlgo { SELECTION,
                      HYBRID,
                      FLASH,
                      LLM,
                      PARTITION,
                      DISTRIBUTED,
                      RADIX,
                      FULL,
                      UNKNOWN = -1 };

struct TopkDataDetail {
  TopkDataDetail(int batch_size, int vocab_size, cudaStream_t stream);
  virtual ~TopkDataDetail() = default;

  // The partition size can be used to test different parition sizes in benchmark tests.
  // If not for that purpose, there is no need to cache the parition size here.

  // The estimated best partition size for hybrid sort
  int hybrid_sort_partition_size = 0;

  // The estimated best partition size for flash sort
  int flash_sort_partition_size = 0;

  // The estimated best partition size for llm sort
  int llm_sort_partition_size = 0;

  // The estimated best partition size for radix partition sort
  int partition_sort_partition_size = 0;

  size_t intermediate_buffer_elements = 0;

  size_t cub_temp_storage_bytes = 0;
};

// This struct holds all the device memory buffers and other data required for Top-K operations.
struct TopkData : public TopkDataDetail {
  TopkData(int batch_size, int vocab_size, cudaStream_t stream, void* buffer = nullptr, size_t buffer_size = 0);
  virtual ~TopkData() = default;
  TopkData(const TopkData&) = delete;
  TopkData& operator=(const TopkData&) = delete;

  // Calculates the total memory required for all buffers.
  static size_t CalculateTotalSize(int batch_size, int vocab_size, cudaStream_t stream);

  // Caching the device_id to avoid repeated calls to cudaGetDevice
  int device_id;

  // A local, lock-free cache for the best algorithm for each k.
  std::array<TopkAlgo, kMaxBenchmarkLocalCache + 1> local_algo_cache_;

  // --- Intermediate Buffers for Top-K Algorithms (Pointers into memory_buffer_span_) ---

  // - Full sort - Holds top-k indices for output
  // - Selection sort: Holds top-k indices for output
  // - Hybrid sort: A "ping-pong" buffer for indices during the reduction phase.
  int* intermediate_indices_1;

  // - Full sort - Holds the initial vocabulary indices before sorting.
  // - Hybrid sort - A "ping-pong" buffer for indices during the reduction phase.
  int* intermediate_indices_2;

  // - Full sort: Holds the fully sorted raw scores.
  // - Selection sort: Holds the top-k scores for selection sort.
  // - Hybrid sort: A "ping-pong" buffer for raw scores during the reduction phase.
  float* intermediate_scores_1;

  // - Selection sort: Holds a copy of input scores. Will be updated in place by selection sort kernel.
  // - Hybrid sort: A "ping-pong" buffer for raw scores during the reduction phase.
  float* intermediate_scores_2;

  // - Full sort: General-purpose temporary storage for CUB's DeviceSegmentedRadixSort
  unsigned char* cub_temp_storage;

  // - Full sort: Stores the start offset of each batch segment for CUB's segmented sort
  int* batch_offsets;

  // Distributed Selection sort: Following are metadata and buffers associated
  // with the distributed selection sort Top k implementation
  int top_k_distributed_select_sort_shards;
  int* top_k_distributed_select_sort_lock;
  int* top_k_distributed_select_sort_keys;
  float* top_k_distributed_select_sort_values;

  // --- Information of Final Output (Input to Sampling Stage) ---
  const float* topk_scores = nullptr;
  const int* topk_indices = nullptr;
  int topk_stride = 0;

 protected:
  // Assigns pointers based on offsets into the single allocated buffer.
  virtual void InitializeBuffers(int batch_size, int vocab_size, cudaStream_t stream);

  // If buffer is provided externally, this unique_ptr will be null. Otherwise it will own the allocated memory.
  cuda_unique_ptr<uint8_t> memory_buffer_owner_;

  // A view of the allocated memory buffer or the externally provided buffer.
  std::span<uint8_t> memory_buffer_span_;
};

// For parity test, a derived struct to help compact output buffers.
struct TopkDataCompact : public TopkData {
  TopkDataCompact(int batch_size, int vocab_size, cudaStream_t stream, void* buffer = nullptr, size_t buffer_size = 0)
      : TopkData(batch_size, vocab_size, stream, buffer, buffer_size) {}

  void CompactOutput(int batch_size, int k, cudaStream_t stream);

  cuda_unique_ptr<float> topk_scores_compact;  // compact [batch_size, k] output scores
  cuda_unique_ptr<int> topk_indices_compact;   // compact [batch_size, k] output indices
};

// Main dispatcher for Top-K. It will automatically choose the best algorithm based on problem size.
void RunTopK(TopkData* topk_data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);

// Below are NOT public APIs. They are exposed for testing purpose.

/**
 * @brief Finds the top-k elements from a batch of scores using a basic selection sort algorithm on the GPU.
 * Primarily intended for baseline performance comparison.
 */
namespace selection_sort {
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace selection_sort

/**
 * @brief A distributed selection sort algorithm that partitions the input data across multiple thread blocks.
 * Each block independently finds local top-k candidates, which are then merged to determine the global top-k.
 * This approach is particularly effective for very large vocabularies and small batch sizes.
 */
namespace distributed_select_sort {
bool IsSupported(int batch_size, int vocab_size, int k);
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace distributed_select_sort

/**
 * @brief Fully sorts the entire input array using CUB's device-wide fragmented radix sort,
 * then extracts the top-k elements from the sorted result.
 */
namespace full_sort {
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace full_sort

/**
 * @brief Sequentially sorts each item in the batch using CUB's device radix sort. This is an effective strategy for smaller batch sizes
 * where launching separate, independent sorts is efficient.
 */
namespace radix_sort {
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace radix_sort

/**
 * @brief A high-performance two-stage, partition-and-reduce sort using block-wide radix sort (cub::BlockRadixSort).
 * In Stage 1, the input is divided into partitions, and a kernel finds the top candidates within each partition.
 * In Stage 2, a second kernel performs a final, fast reduction on these candidates, operating on data held in registers.
 * This algorithm is very efficient for small to medium k values (up to 64).
 */
namespace radix_partition_sort {
bool IsSupported(int batch_size, int vocab_size, int k);
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace radix_partition_sort

/**
 * @brief Implements a hybrid, multi-stage approach. Data is first partitioned and sorted locally within thread blocks with radix sort.
 * A final reduction stage using bitonic sort merges these partitions to find the global top-k.
 */
namespace hybrid_sort {
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace hybrid_sort

/**
 * @brief A high-performance, single-kernel cooperative sort algorithm with iterative reduction.
 */
namespace flash_sort {
bool IsSupported(int batch_size, int vocab_size, int k);
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace flash_sort

/**
 * @brief A high-performance, single-kernel cooperative cascaded sort algorithm optimized for popular LLM.
 */
namespace llm_sort {
bool IsSupported(int batch_size, int vocab_size, int k);
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace llm_sort

}  // namespace cuda
}  // namespace Generators
