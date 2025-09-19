// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <float.h>
#include "cuda_topk.h"

namespace Generators {
namespace cuda {
namespace warp_merge {

/**
 * @brief Performs an in-place, warp-wide bitonic sort on data held in registers.
 * Sorts `warpSize` elements distributed across the threads of a single warp.
 * The result is sorted in descending order by score.
 */
__device__ inline void WarpBitonicSort(float& score, int& index) {
  const int lane_id = threadIdx.x % warpSize;

  // The bitonic sort network is constructed in stages.
  for (int k = 2; k <= warpSize; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      int paired_lane = lane_id ^ j;
      float paired_score = __shfl_sync(0xFFFFFFFF, score, paired_lane);
      int paired_index = __shfl_sync(0xFFFFFFFF, index, paired_lane);

      // Determine the sort direction for this stage of the bitonic network.
      bool direction = ((lane_id & k) == 0);

#ifdef STABLE_TOPK
      // For stable sort, include tie-breaking logic (smaller index wins for equal scores).
      bool is_mine_greater = (score > paired_score) || (score == paired_score && index < paired_index);
#else
      // For unstable sort, no tie-breaking is needed for performance.
      bool is_mine_greater = score > paired_score;
#endif

      // Determine the min and max values of the pair.
      float s_max = is_mine_greater ? score : paired_score;
      int i_max = is_mine_greater ? index : paired_index;
      float s_min = is_mine_greater ? paired_score : score;
      int i_min = is_mine_greater ? paired_index : index;

      // Redistribute the min/max values based on the sort direction for this stage
      // to achieve an overall descending sort.
      if (direction) {  // "Descending" part of the bitonic sequence
        score = (lane_id < paired_lane) ? s_max : s_min;
        index = (lane_id < paired_lane) ? i_max : i_min;
      } else {  // "Ascending" part of the bitonic sequence
        score = (lane_id < paired_lane) ? s_min : s_max;
        index = (lane_id < paired_lane) ? i_min : i_max;
      }
    }
  }
}

}  // namespace warp_merge
}  // namespace cuda
}  // namespace Generators
