// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <curand_kernel.h>
#include <memory>
#include <array>
#include <algorithm>

#include "cuda_common.h"

namespace Generators {
namespace cuda {

enum class SortAlgo {
  WARP_BITONIC = 0,
  CUB_WARP_MERGE,
  CUB_BLOCK_MERGE,
  CUB_BLOCK_RADIX,
  COUNT  // Keep this last
};

// To enable stable Top-K for kernels that support it, define STABLE_TOPK during compilation.
// A stable sort preserves the original relative order of elements with equal scores.
// By default, a faster, unstable sort is used if STABLE_TOPK is not defined.
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

// The maximum supported k value for each specialized algorithm.
// These are tuned based on algorithm design and performance benchmarks.
// For instance, a larger k requires more shared memory or registers in the reduction phase.
constexpr int kHybridSortMaxK = 256;
constexpr int kIterativeSortMaxK = 64;
constexpr int kCascadedSortMaxK = 64;
constexpr int kConvergentSortMaxK = 64;

// The minimum partition size used across the fast sort algorithms. This is used to
// calculate a safe upper bound for intermediate buffer allocation.
constexpr int kFastSortMinPartitionSize = 1024;

// The maximum k value for which benchmark results are stored in the fast, lock-free local cache.
constexpr int kMaxBenchmarkLocalCache = 64;

namespace topk_impl_details {
constexpr int kTopKDistributedSelectSortMaxShards = 32;
constexpr int kTopKDistributedSelectSortMaxBatchSize = 1;
constexpr int kTopKDistributedSelectSortMaxTopK = 64;
constexpr int kTopKDistributedSelectSortMinVocabSize = 100000;
}  // namespace topk_impl_details

// Enum for the different Top-K algorithms available for online benchmarking.
enum class TopkAlgo { SELECTION,
                      HYBRID,
                      ITERATIVE,
                      CASCADED,
                      CONVERGENT,
                      DISTRIBUTED_SELECT,
                      PER_BATCH_RADIX,
                      FULL,
                      UNKNOWN = -1 };

// This struct holds pre-calculated, host-side parameters for the Top-K operation.
// It is separated from TopkData to keep device-side memory management distinct from host-side planning.
struct TopkDataDetail {
  TopkDataDetail(int batch_size, int vocab_size, cudaStream_t stream);
  virtual ~TopkDataDetail() = default;

  // The estimated best partition size for each algorithm, calculated on the host.
  int hybrid_sort_partition_size = 0;
  int iterative_sort_partition_size = 0;
  int cascaded_sort_partition_size = 0;
  int flash_convergent_partition_size = 0;
  int flash_convergent_partition_size_k = 0;  // The k value used in estimating the partition size for flash_convergent.

  // The number of elements required for intermediate buffers, sized to accommodate the worst-case scenario.
  size_t intermediate_buffer_elements = 0;

  // The size of the temporary storage buffer required by CUB routines.
  size_t cub_temp_storage_bytes = 0;
};

// This struct manages all device memory buffers and operational data required for the Top-K algorithms.
// It allocates a single contiguous block of memory and partitions it among several pointers
// to minimize allocation overhead and improve memory locality.
struct TopkData : public TopkDataDetail {
  TopkData(int batch_size, int vocab_size, cudaStream_t stream, void* buffer = nullptr, size_t buffer_size = 0);
  virtual ~TopkData() = default;
  TopkData(const TopkData&) = delete;
  TopkData& operator=(const TopkData&) = delete;

  // Calculates the total device memory required for all internal buffers.
  static size_t CalculateTotalSize(int batch_size, int vocab_size, cudaStream_t stream);

  // Caching the device_id to avoid repeated calls to cudaGetDevice.
  int device_id;

  // A local, lock-free cache for the best algorithm for each k. This provides a fast path
  // to algorithm selection within a single TopkData instance.
  std::array<TopkAlgo, kMaxBenchmarkLocalCache + 1> local_algo_cache_;

  // --- Intermediate Buffers for Top-K Algorithms (Pointers into memory_buffer_span_) ---

  // Buffers used as "ping-pong" or intermediate storage during reduction phases.
  // Their specific roles vary depending on the algorithm being executed.
  int* intermediate_indices_1;
  int* intermediate_indices_2;
  float* intermediate_scores_1;
  float* intermediate_scores_2;

  // Temporary storage for CUB's device-wide sorting routines (`full_sort`, `per_batch_radix_sort`).
  unsigned char* cub_temp_storage;

  // Stores the start offset of each batch segment for CUB's segmented sort (`full_sort`).
  int* batch_offsets;

  // Buffers and metadata for the `distributed_select_sort` algorithm.
  int top_k_distributed_select_sort_shards;
  int* top_k_distributed_select_sort_lock;
  int* top_k_distributed_select_sort_keys;
  float* top_k_distributed_select_sort_values;

  // --- Final Output Pointers ---
  // After a Top-K algorithm runs, these pointers will point to the memory location
  // within the intermediate buffers that holds the final result.
  const float* topk_scores = nullptr;
  const int* topk_indices = nullptr;
  // The stride between the start of each batch's results in the output buffers.
  // A stride > k indicates the results are not contiguously packed and may need a compaction step.
  int topk_stride = 0;

 protected:
  // Assigns pointers based on offsets into the single allocated buffer.
  virtual void InitializeBuffers(int batch_size, int vocab_size, cudaStream_t stream);

  // If the memory buffer is provided externally, this unique_ptr will be null.
  // Otherwise, it owns the allocated device memory.
  cuda_unique_ptr<uint8_t> memory_buffer_owner_;

  // A span representing the memory buffer, whether owned or externally provided.
  std::span<uint8_t> memory_buffer_span_;
};

// A derived struct for testing that adds functionality to compact a strided output into a contiguous one.
struct TopkDataCompact : public TopkData {
  TopkDataCompact(int batch_size, int vocab_size, cudaStream_t stream, void* buffer = nullptr, size_t buffer_size = 0)
      : TopkData(batch_size, vocab_size, stream, buffer, buffer_size) {}

  void CompactOutput(int batch_size, int k, cudaStream_t stream);

  cuda_unique_ptr<float> topk_scores_compact;
  cuda_unique_ptr<int> topk_indices_compact;
};

// Main dispatcher for Top-K. It automatically chooses the best algorithm based on problem size
// and caches the decision for subsequent calls.
void RunTopK(TopkData* topk_data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);

// The following namespaces expose the individual algorithms for direct calling, primarily for testing and debugging.

/**
 * @brief Finds the top-k elements using a basic GPU-based selection sort.
 * This is primarily intended for correctness validation and as a performance baseline. It is only efficient for very small k.
 */
namespace select_sort {
constexpr const char* kAlgorithmName = "Select";
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace select_sort

/**
 * @brief A distributed selection sort that partitions the input across SMs.
 * Each SM finds local candidates, which are then merged atomically.
 * This approach is particularly effective for very large vocabularies and small batch sizes, and is unbeatable for k=1.
 */
namespace distributed_select_sort {
constexpr const char* kAlgorithmName = "Distributed_Select";
bool IsSupported(int batch_size, int vocab_size, int k);
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace distributed_select_sort

/**
 * @brief Fully sorts the entire input array using CUB's device-wide segmented radix sort,
 * then extracts the top-k elements. This is a robust but inefficient fallback.
 */
namespace full_sort {
constexpr const char* kAlgorithmName = "Segmented_Radix";
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace full_sort

/**
 * @brief Sequentially sorts each batch item independently using CUB's device radix sort.
 * This is an effective strategy for smaller batch sizes where parallelism between batches is high.
 */
namespace per_batch_radix_sort {
constexpr const char* kAlgorithmName = "Per_Batch_Radix";
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace per_batch_radix_sort

/**
 * @brief A two-stage cooperative algorithm specialized for **small k (2-16)**.
 * Its single-step reduction phase has minimal overhead, making it ideal when the core
 * sorting computation is small. It uses a single block to merge all partition candidates,
 * choosing the fastest CUB block-level sort internally.
 */
namespace flash_convergent {
constexpr const char* kAlgorithmName = "Flash_Convergent";
bool IsSupported(int batch_size, int vocab_size, int k);
int EstimateBestPartitionSize(int vocab_size, int k);
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace flash_convergent

/**
 * @brief A two-kernel "cooperative hybrid" sort, the champion for **very large k (>64)** or **very large vocabularies**.
 * Stage 1: A standard, highly scalable kernel finds top candidates in partitions.
 * Stage 2: A single cooperative kernel performs an efficient, multi-step reduction cascade based on a pre-computed optimal plan.
 * This design combines scalability with low reduction overhead.
 */
namespace hybrid_sort {
constexpr const char* kAlgorithmName = "Hybrid_Sort";
bool IsSupported(int batch_size, int vocab_size, int k);
int EstimateBestPartitionSize(int vocab_size);
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace hybrid_sort

/**
 * @brief A single-kernel cooperative sort specialized for **mid-to-large k (32-64) on large vocabularies (>250k)**.
 * It uses a smart host-side planner to choose a single, optimal reduction factor that minimizes workload
 * imbalance. This balances the low overhead of a single-kernel iterative reduction with an efficient, adaptive plan.
 */
namespace iterative_sort {
constexpr const char* kAlgorithmName = "Iterative_Sort";
bool IsSupported(int batch_size, int vocab_size, int k);
int EstimateBestPartitionSize(int vocab_size, int k_padded);
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace iterative_sort

/**
 * @brief A high-performance, single-kernel cooperative "cascaded" sort, the all-around champion for **mid-range k (16-64)**.
 * It uses an adaptive, two-step reduction strategy based on a pre-computed optimal plan from H200
 * benchmarks, providing the best performance for the most common LLM workloads.
 */
namespace cascaded_sort {
constexpr const char* kAlgorithmName = "Cascaded_Sort";
bool IsSupported(int batch_size, int vocab_size, int k);
int EstimateBestPartitionSize(int vocab_size);
void RunTopK(TopkData* data, cudaStream_t stream, const float* scores_in, int vocab_size, int batch_size, int k);
}  // namespace cascaded_sort

}  // namespace cuda
}  // namespace Generators
