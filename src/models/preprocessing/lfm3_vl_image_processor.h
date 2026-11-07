// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>

#include "models/preprocessing/lfm2_vl_image_processor.h"

namespace Generators {

// The chat template writes "<|vision_start|><|image|><|vision_end|>" per image, and the processor
// only repeats "<|image|>", once per vision feature, as `Lfm3VlProcessor` does.
inline constexpr char kLfm3VlImageToken[] = "<|image|>";
inline constexpr char kLfm3VlVideoToken[] = "<|video|>";

// Copies each image's rows out of the zero-padded [N, longest_image_rows, row_width] batch to sit one
// after another in `destination`, which holds sum(image_rows) rows.
void CompactLfm3VlImageRows(ThreadPool* thread_pool, const float* batch, int64_t longest_image_rows, int64_t row_width,
                            const std::vector<int64_t>& image_rows, float* destination);

// Replaces the i-th "<|image|>" with tokens_per_image[i] copies of it. Throws unless the prompt holds
// exactly one "<|image|>" per image, and on any "<|video|>", since video is not supported.
std::string ExpandLfm3VlImageTokens(const std::string& prompt, const std::vector<int64_t>& tokens_per_image);

// The processor_config.json pipeline ends in ort-extensions' PatchImage, which writes each image the
// way the checkpoint's Lfm3VlImageProcessor does: rows go 2x2 block by block, and a row holds a patch as
// (channel, time, y, x) with both time steps equal.
struct Lfm3VlImageProcessor : Processor {
  Lfm3VlImageProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor_;

  ONNXTensorElementDataType pixel_values_type_;

  int64_t row_width_{};
  int64_t merge_size_{};
};

}  // namespace Generators
