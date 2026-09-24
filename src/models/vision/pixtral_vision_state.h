// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/vision/multi_modal_vision.h"

namespace Generators {

// PixtralVisionState: per-image vision loop for Pixtral / Mistral3.
//
// Each image is independently smart_resize'd to a different resolution.
// The preprocessor zero-pads all images to max(H) × max(W) and provides
// image_sizes[N, 2] with per-image (H, W).  This subclass slices
// pixel_values[i, :, :H_i, :W_i] for each image, runs vision.onnx with
// [1, C, H_i, W_i], and concatenates the resulting features.
struct PixtralVisionState : VisionState {
  using VisionState::VisionState;  // inherit constructor

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_images, const int64_t num_image_tokens) override;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 private:
  std::vector<int64_t> image_heights_;
  std::vector<int64_t> image_widths_;
};

}  // namespace Generators
