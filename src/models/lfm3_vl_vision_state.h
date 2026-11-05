// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "multi_modal.h"

namespace Generators {

// LFM3-VL's vision graph takes one image per run: [num_patches, row_width] pixel rows and a [1, 3]
// image_grid_thw. The processor concatenates the rows of all images, so with several images this state
// runs the graph on each image's rows and copies its features into the shared buffer in image order.
struct Lfm3VlVisionState : VisionState {
  Lfm3VlVisionState(const MultiModalLanguageModel& model, const GeneratorParams& params);

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_images,
                      const int64_t num_image_tokens) override;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

 private:
  std::vector<int64_t> grid_thw_;  // [num_images * 3], (1, patch rows, patch cols) per image
};

}  // namespace Generators
