// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

namespace Generators {
namespace cuda {
namespace topk_common {

// --- Stable Sort Key Packing/Unpacking ---
// Helper functions to manage 64-bit keys for stable Top-K sorting.
// The key is composed of a transformed float score and an inverted index,
// designed to be sorted in DESCENDING order to achieve the desired result.

/**
 * @brief Creates a sortable 64-bit key from a score and an index for stable descending sort.
 * @param score The float score.
 * @param index The integer index.
 * @return A 64-bit unsigned integer key.
 */
__device__ __forceinline__ uint64_t PackStableSortKey(float score, int index) {
  // Transform float bits to sortable integer bits. For positive floats, this flips the
  // sign bit to make them larger than negative floats. For negative floats, it inverts
  // all bits, maintaining their relative order.
  uint32_t score_bits = __float_as_uint(score);
  uint32_t sortable_score = (score_bits & 0x80000000) ? (~score_bits) : (score_bits | 0x80000000);

  // Invert the index to ensure that for ties in score, a smaller original index
  // results in a larger key value. When sorted descending, this achieves the tie-break.
  uint32_t inverted_index = UINT_MAX - static_cast<uint32_t>(index);

  return (static_cast<uint64_t>(sortable_score) << 32) | inverted_index;
}

/**
 * @brief Unpacks the original float score from a 64-bit stable sort key.
 * @param key The 64-bit key created by PackStableSortKey.
 * @return The original float score.
 */
__device__ __forceinline__ float UnpackStableSortScore(uint64_t key) {
  uint32_t sortable_score = static_cast<uint32_t>(key >> 32);

  // Reverse the transformation from sortable integer bits back to original float bits.
  uint32_t score_bits;
  if (sortable_score & 0x80000000) {
    // Was originally a positive float
    score_bits = sortable_score & 0x7FFFFFFF;
  } else {
    // Was originally a negative float
    score_bits = ~sortable_score;
  }
  return __uint_as_float(score_bits);
}

/**
 * @brief Unpacks the original integer index from a 64-bit stable sort key.
 * @param key The 64-bit key created by PackStableSortKey.
 * @return The original integer index.
 */
__device__ __forceinline__ int UnpackStableSortIndex(uint64_t key) {
  uint32_t inverted_index = static_cast<uint32_t>(key & 0xFFFFFFFF);
  return static_cast<int>(UINT_MAX - inverted_index);
}

}  // namespace topk_common
}  // namespace cuda
}  // namespace Generators
