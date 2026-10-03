// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/vision/multi_modal_vision.h"

namespace Generators {

struct PixtralVisionState : VisionState {
  using VisionState::VisionState;

  int64_t GetImageFeatureBatchSize(const std::vector<ExtraInput>& extra_inputs) const override;
  int64_t GetNumImageTokens(const std::vector<ExtraInput>& extra_inputs) const override;
  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_images,
                      const int64_t num_image_tokens) override;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

 private:
  std::vector<int64_t> image_heights_;
  std::vector<int64_t> image_widths_;
};

}  // namespace Generators