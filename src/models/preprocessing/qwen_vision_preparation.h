// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "processor.h"
#include "vision_preparation.h"

namespace Generators {

struct QwenImageProcessor;

struct QwenVisionPreparation : VisionPreparation {
  QwenVisionPreparation(const Config& config, const SessionInfo& info,
                        const Tokenizer& tokenizer, const QwenImageProcessor& images, ThreadPool* thread_pool);
  size_t Estimate(const VisionPreparationInput& input, size_t host_limit) const override;
  std::unique_ptr<const VisionPlan> Plan(const Tokenizer& tokenizer,
                                         const VisionPreparationInput& input, size_t host_limit) const override;
  std::vector<NamedTensors> Materialize(const VisionPlan& plan) const override;
  void ValidateTextTokens(std::span<const int32_t> tokens) const override;

 private:
  const Config& config_;
  const QwenImageProcessor& images_;
  ThreadPool* thread_pool_;
  int64_t min_pixels_{}, max_pixels_{};
  int64_t video_min_pixels_{}, video_max_pixels_{};
  size_t max_video_tokens_{};
  int64_t pixel_width_{};
  ONNXTensorElementDataType pixel_type_{};
  int32_t vision_end_token_{}, video_token_{};
  ort_extensions::OrtxObjectPtr<OrtxProcessor> videos_;
};

}  // namespace Generators
