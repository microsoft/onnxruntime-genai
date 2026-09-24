// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/multi_modal_vision.h"

namespace Generators {

// QwenVisionState: per-image slicing loop for Qwen2.5-VL / Qwen3-VL.
//
// vision.onnx is exported for exactly one image (Dynamo unrolls Python
// for-loops at trace time, so an N-image dummy produces a graph that only
// works for that exact N).  This subclass iterates over images in C++,
// creating zero-copy sub-tensor views of pixel_values / image_grid_thw and
// writing each result into the correct offset of the pre-allocated
// image_features output buffer.
struct QwenVisionState : VisionState {
  using VisionState::VisionState;  // inherit constructor

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;
};

}  // namespace Generators
