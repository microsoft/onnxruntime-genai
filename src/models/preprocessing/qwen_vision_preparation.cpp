// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qwen_vision_preparation.h"
#include "qwen_vision_position.h"
#include "qwen_vision_processing.h"

namespace Generators {

QwenVisionPreparation::QwenVisionPreparation(const Config& config, const SessionInfo& info,
                                             const Tokenizer& tokenizer, const QwenImageProcessor& images, ThreadPool* thread_pool)
    : PatchVisionPreparation{config, info, tokenizer, images, thread_pool} {
  const auto& vision = config.model.vision;
  if (!vision.video_config_filename) return;
  if (config.model.type != "qwen3_5" && config.model.type != "qwen3_5_moe" && config.model.type != "qwen4_exp")
    throw std::runtime_error("Packed video requires a verified Qwen3.5 or Qwen4Exp position contract.");
  PatchProcessingLimits video;
  const auto definition = ReadPatchProcessor(config, *vision.video_config_filename, video);
  if (video.operations != std::vector<std::string>{"Resize", "Rescale", "Normalize", "PatchImage"} ||
      video.min_pixels <= 0 || video.max_pixels < video.min_pixels || video.patch_size != vision.patch_size ||
      video.temporal_patch_size != 2 || video.merge_size != vision.spatial_merge_size ||
      video.smart_resize != 0 || video.coefficient_type != "int16" || video.max_video_tokens <= 0 ||
      video.bgr_normalization)
    throw std::runtime_error("Packed video requires the bounded Qwen3-VL temporal preprocessing pipeline.");
  video_min_pixels_ = video.min_pixels;
  video_max_pixels_ = video.max_pixels;
  max_video_tokens_ = static_cast<size_t>(video.max_video_tokens);
  InitializeVideo(definition);
}

std::array<int64_t, 3> QwenVisionPreparation::ImageGrid(uint32_t width, uint32_t height) const {
  return QwenImageGrid(width, height, config_.model.vision.patch_size, config_.model.vision.spatial_merge_size,
                       min_pixels_, max_pixels_);
}

std::array<int64_t, 3> QwenVisionPreparation::VideoGrid(uint32_t width, uint32_t height, size_t frames) const {
  return QwenVideoGrid(width, height, frames, config_.model.vision.patch_size, config_.model.vision.spatial_merge_size,
                       video_min_pixels_, video_max_pixels_, max_video_tokens_);
}

size_t QwenVisionPreparation::PromptBytes(std::string_view prompt, const VisionMediaPromptSource& source, size_t maximum_bytes) const {
  return QwenMediaPromptBytes(prompt, source, config_.model.vision.spatial_merge_size, maximum_bytes);
}

std::string QwenVisionPreparation::ExpandPrompt(std::string_view prompt, std::span<const std::array<int64_t, 3>> images,
                                                std::span<const VisionVideoPrompt> videos, size_t maximum_bytes,
                                                std::vector<size_t>&) const {
  return ExpandQwenMediaPrompt(prompt, images, videos, config_.model.vision.spatial_merge_size, maximum_bytes);
}

std::shared_ptr<const VisionPosition> QwenVisionPreparation::Position(std::array<int64_t, 3> grid) const {
  return MakeQwenVisionPosition(grid, config_.model.vision.spatial_merge_size);
}

std::array<int64_t, 2> QwenVisionPreparation::CoordinateRange() const {
  if (config_.model.context_length <= 0)
    throw std::runtime_error("Packed Qwen positions require a positive exported context length.");
  return {0, static_cast<int64_t>(config_.model.context_length) - 1};
}

}  // namespace Generators
