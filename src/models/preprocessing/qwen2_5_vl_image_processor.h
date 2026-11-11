// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/model.h"
#include "models/preprocessing/processor.h"
#include "ortx_processor.h"

namespace Generators {

void ExtractQwenImagePatches(ThreadPool* thread_pool, const float* source, float* destination,
                             int64_t height, int64_t width, int64_t channels,
                             int64_t patch_size, int64_t temporal_patch_size);

struct QwenImageProcessor : Processor {
  QwenImageProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;
  std::vector<int32_t> ExpandImagePrompt(const Tokenizer& tokenizer, const std::string& prompt,
                                         std::span<const int64_t> grids) const override;
  ImagePositionLayout GetImagePositionLayout(const std::array<int64_t, 3>& grid) const override;

 protected:
  int64_t spatial_merge_size_;

 private:
  ort_extensions::OrtxObjectPtr<OrtxProcessor> processor_;

  ONNXTensorElementDataType pixel_values_type_;
  int64_t patch_size_{14};  // Qwen2.5-VL uses 14, Qwen3-VL uses 16
};

struct LingImageProcessor : QwenImageProcessor {
  using QwenImageProcessor::QwenImageProcessor;
  std::vector<int32_t> ExpandImagePrompt(const Tokenizer& tokenizer, const std::string& prompt,
                                         std::span<const int64_t> grids) const override;
  ImagePositionLayout GetImagePositionLayout(const std::array<int64_t, 3>& grid) const override;
};

}  // namespace Generators
