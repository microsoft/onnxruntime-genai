// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "multi_modal.h"

namespace Generators {

// Qwen vision graphs exported with a single-image dummy input require one run
// per image. This state slices flattened pixel values and concatenates outputs.
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
    return temporal_multiplier > 0 ? temporal_multiplier * height * width : grid_tokens;
  }

  int64_t ImagePatchOffset(int64_t image_index, int64_t packed_offset) const {
    return padded_image_stride > 0 ? image_index * padded_image_stride : packed_offset;
  }
};

inline QwenPatchLayout ResolveQwenPatchLayout(int64_t total_patches,
                                              int64_t total_grid_tokens,
                                              int64_t total_hw,
                                              int64_t max_grid_tokens,
                                              int64_t num_images) {
  if (total_patches == total_grid_tokens) {
    return {};
  }

  const bool temporal_padded = total_patches > 0 && total_hw > 0 && total_patches % total_hw == 0;
  const int64_t candidate_stride =
      num_images > 0 && total_patches % num_images == 0 ? total_patches / num_images : 0;
  const bool stride_padded = candidate_stride > 0 && candidate_stride == max_grid_tokens;

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
