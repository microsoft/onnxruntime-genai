// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>

#include "models/preprocessing/processor.h"

namespace Generators {

// LFM2-VL special tokens. The image token is repeated once per projected vision feature and the
// whole run is wrapped in the start/end markers, matching `Lfm2VlProcessor._build_image_tokens`.
inline constexpr char kLfm2VlImageToken[] = "<image>";
inline constexpr char kLfm2VlImageStartToken[] = "<|image_start|>";
inline constexpr char kLfm2VlImageEndToken[] = "<|image_end|>";

// Patch/token geometry of a single image after the ort-extensions smart resize.
//
// The SigLIP2 NaViT encoder consumes a flat sequence of `encoder_patch_size` squared patches, and
// the multi-modal projector then pixel-unshuffles that grid by `downsample_factor` in both
// directions, so one decoder token covers a downsample_factor x downsample_factor block of patches.
struct Lfm2VlImageGeometry {
  int64_t patch_rows{};   // image height in encoder patches
  int64_t patch_cols{};   // image width in encoder patches
  int64_t num_patches{};  // patch_rows * patch_cols: sequence length seen by the vision encoder
  int64_t num_tokens{};   // decoder tokens the projector emits for this image
};

// Throws if the resized image is not a whole number of patches; smart resize always rounds to a
// multiple of encoder_patch_size * downsample_factor, so a violation means the preprocessing
// pipeline in processor_config.json disagrees with the vision config in genai_config.json.
Lfm2VlImageGeometry ComputeLfm2VlImageGeometry(int64_t image_height, int64_t image_width,
                                               int64_t encoder_patch_size, int64_t downsample_factor);

// "<|image_start|>" + "<image>" * num_tokens + "<|image_end|>"
std::string BuildLfm2VlImagePlaceholder(int64_t num_tokens);

// Replaces the i-th "<image>" in the prompt with the placeholder run for image i. Images without a
// matching "<image>" in the prompt are prepended, which keeps prompts that were not built from the
// chat template working. Throws if the prompt asks for more images than were supplied.
std::string ExpandLfm2VlImageTokens(const std::string& prompt, const std::vector<int64_t>& tokens_per_image);

struct Lfm2VlImageProcessor : Processor {
  Lfm2VlImageProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor_;

  ONNXTensorElementDataType pixel_values_type_;

  int64_t encoder_patch_size_{};
  int64_t downsample_factor_{};
  int64_t max_num_patches_{};
};

}  // namespace Generators
