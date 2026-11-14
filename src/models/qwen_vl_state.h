// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>
#include <stdexcept>

#include "multi_modal.h"

namespace Generators {

inline int64_t CheckedQwenVisionProduct(int64_t left, int64_t right) {
  if (left < 0 || right < 0 ||
      (right != 0 && left > std::numeric_limits<int64_t>::max() / right)) {
    throw std::runtime_error("Qwen vision size product is negative or exceeds int64 range");
  }
  return left * right;
}

inline int64_t CheckedQwenVisionSum(int64_t left, int64_t right) {
  if (left < 0 || right < 0 || left > std::numeric_limits<int64_t>::max() - right) {
    throw std::runtime_error("Qwen vision size sum is negative or exceeds int64 range");
  }
  return left + right;
}

inline int64_t GetQwenGridTokenCount(int64_t temporal, int64_t height, int64_t width,
                                     int64_t spatial_merge_size = 1) {
  if (temporal <= 0 || height <= 0 || width <= 0) {
    throw std::runtime_error("Qwen image_grid_thw values must be positive");
  }
  if (spatial_merge_size <= 0) {
    throw std::runtime_error("Qwen spatial_merge_size must be positive");
  }
  const auto merge_square = CheckedQwenVisionProduct(spatial_merge_size, spatial_merge_size);
  return CheckedQwenVisionProduct(temporal, CheckedQwenVisionProduct(height, width)) / merge_square;
}

struct QwenVisionState : VisionState {
  using VisionState::VisionState;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;
};

int64_t GetQwenImageCount(const std::vector<ExtraInput>& extra_inputs);

struct QwenPatchLayout {
  int64_t padded_image_stride{};
  int64_t temporal_multiplier{};

  int64_t ImagePatchCount(int64_t grid_tokens, int64_t height, int64_t width) const {
    return temporal_multiplier > 0
               ? CheckedQwenVisionProduct(temporal_multiplier, CheckedQwenVisionProduct(height, width))
               : grid_tokens;
  }

  int64_t ImagePatchOffset(int64_t image_index, int64_t packed_offset) const {
    return padded_image_stride > 0 ? CheckedQwenVisionProduct(image_index, padded_image_stride) : packed_offset;
  }
};

inline QwenPatchLayout ResolveQwenPatchLayout(int64_t total_patches,
                                              int64_t total_grid_tokens,
                                              int64_t total_hw,
                                              int64_t max_grid_tokens,
                                              int64_t num_images,
                                              bool all_temporal_dims_one = true) {
  if (total_patches == total_grid_tokens) {
    return {};
  }

  const bool temporal_padded = total_patches > 0 && total_hw > 0 && total_patches % total_hw == 0;
  const int64_t candidate_stride =
      num_images > 0 && total_patches % num_images == 0 ? total_patches / num_images : 0;
  const bool stride_padded = all_temporal_dims_one && candidate_stride > 0 &&
                             candidate_stride == max_grid_tokens;

  if (stride_padded) {
    return {.padded_image_stride = candidate_stride};
  }
  if (temporal_padded) {
    return {.temporal_multiplier = total_patches / total_hw};
  }

  throw std::runtime_error("pixel_values patch count (" + std::to_string(total_patches) +
                           ") does not match image_grid_thw patch count (" +
                           std::to_string(total_grid_tokens) + ")");
}

}  // namespace Generators