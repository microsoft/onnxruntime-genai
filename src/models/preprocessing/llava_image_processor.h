// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "models/preprocessing/processor.h"

namespace Generators {

// Replaces each image_token_id in input_ids with tokens_per_image copies of it, the LLaVA-style
// placeholder expansion. Throws unless the prompt holds exactly one image token per image, so the
// image features can never be misaligned with, or silently dropped from, the decoder positions.
std::vector<int32_t> ExpandLlavaImageTokens(const std::vector<int32_t>& input_ids, int32_t image_token_id,
                                            int64_t tokens_per_image, int64_t num_images);

// Image processor for "llava" models: a fixed-resolution vision encoder whose output, a fixed number
// of feature vectors per image, replaces the image placeholder token in the prompt.
//
//  - The image transforms (decode, optional pad to square, resize, rescale, normalize, ...) come from
//    the ort-extensions pipeline in model.vision.config_filename and must produce [N, C, H, W].
//  - The prompt holds one model.image_token_id token per image. Each one is expanded to
//    model.vision.num_visual_tokens copies, or, when that is not set, to the static token dimension
//    of the vision model's [num_images, num_tokens, hidden] output.
//  - The embedding model receives the expanded input_ids and the image features, and puts the
//    feature rows at the image token positions, in order.
struct LlavaImageProcessor : Processor {
  LlavaImageProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor_;

  ONNXTensorElementDataType pixel_values_type_;
  // The vision input's static [C, H, W]; a dynamic dimension is <= 0 and not checked.
  std::array<int64_t, 3> pixel_values_chw_{-1, -1, -1};
  int32_t image_token_id_{};
  int64_t tokens_per_image_{};
  // Whether the vision model emits [num_images, num_tokens, hidden] (true) or [total_tokens, hidden].
  bool batched_features_{true};
};

}  // namespace Generators
