// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "multi_modal.h"

namespace Generators {

// Per-image vision loop for Gemma 4 models whose vision graph has a static
// batch dimension of one.
struct Gemma4VisionState : VisionState {
  using VisionState::VisionState;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_images, const int64_t num_image_tokens) override;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 private:
  std::vector<int64_t> image_token_counts_;
  size_t pixel_values_index_{SIZE_MAX};
  size_t position_ids_index_{SIZE_MAX};
};

}  // namespace Generators