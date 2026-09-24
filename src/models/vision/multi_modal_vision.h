// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/model.h"
#include "models/io/multi_modal_features.h"
#include "models/io/extra_inputs.h"

namespace Generators {

struct MultiModalLanguageModel;
struct MultiModalPipelineState;

// Base VisionState: runs vision.onnx with a single State::Run() call.
// Works for models whose vision encoder accepts batched input (Phi, Gemma).
struct VisionState : State {
  VisionState(const MultiModalLanguageModel& model, const GeneratorParams& params);
  VisionState(const VisionState&) = delete;
  VisionState& operator=(const VisionState&) = delete;

  virtual void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_images, const int64_t num_image_tokens);
  virtual int64_t GetImageFeatureBatchSize(const std::vector<ExtraInput>& extra_inputs) const;
  virtual int64_t GetNumImageTokens(const std::vector<ExtraInput>& extra_inputs) const;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 protected:
  friend struct MultiModalPipelineState;

  const MultiModalLanguageModel& model_;
  int64_t num_image_tokens_;
  int64_t num_images_{};
  ExtraInputs extra_inputs_{*this};  // Model inputs
  std::unique_ptr<MultiModalFeatures> image_features_;
};

inline void ValidateImageGridThwLayoutAndCount(const std::vector<int64_t>& shape,
                                               size_t elem_count,
                                               int64_t num_images,
                                               const char* tensor_name) {
  if (num_images < 0) {
    throw std::runtime_error(std::string(tensor_name) + " num_images must be non-negative");
  }

  if (shape.size() != 2) {
    throw std::runtime_error(std::string(tensor_name) + " must have rank 2 [num_images, 3]");
  }

  if (shape[0] < 0 || shape[1] < 0) {
    throw std::runtime_error(std::string(tensor_name) + " dimensions must be non-negative");
  }

  if (shape[1] != 3) {
    throw std::runtime_error(std::string(tensor_name) + " second dimension must be 3");
  }

  const size_t shape_image_count = static_cast<size_t>(shape[0]);
  const size_t expected_image_count = static_cast<size_t>(num_images);
  if (shape_image_count < expected_image_count) {
    throw std::runtime_error(std::string(tensor_name) + " shape[0] (" + std::to_string(shape_image_count) +
                             ") is less than required image count (" + std::to_string(expected_image_count) + ")");
  }

  if (elem_count % 3 != 0 || elem_count / 3 < expected_image_count) {
    throw std::runtime_error(std::string(tensor_name) + " element count (" + std::to_string(elem_count) +
                             ") is less than required for " + std::to_string(num_images) +
                             " images (need at least 3 values per image)");
  }
}

// Returns the number of images in the current batch. Two strategies are tried in order:
//
// 1. pixel_values rank-3 path (Phi, Gemma, and legacy Qwen processors):
//    The extension returns pixel_values as [N, patches_per_image, patch_dim], so
//    the batch size is simply shape[0].
//
// 2. image_grid_thw fallback (Qwen2.5-VL / Qwen3-VL after the multi-image flatten fix):
//    Some processors flatten pixel_values to rank 2 [total_patches, patch_dim] so that
//    vision.onnx—which is exported for a single image and loops per-image in
//    VisionState::Run—always receives a 2-D input regardless of image count.
//    Rank-2 pixel_values carries no image-count information, so we fall through and
//    read num_images from image_grid_thw.shape[0] ([num_images, 3]).
// Factory: pick the right VisionState subclass based on model type.
std::unique_ptr<VisionState> CreateVisionState(const MultiModalLanguageModel& model, const GeneratorParams& params);

}  // namespace Generators
