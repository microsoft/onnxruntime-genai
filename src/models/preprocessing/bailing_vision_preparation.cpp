// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bailing_vision_preparation.h"
#include "bailing_vision_processing.h"
#include "bailing_vision_position.h"
#include "qwen2_5_vl_image_processor.h"
#include "genai_tokenizer.h"

#include <algorithm>

namespace Generators {

BailingVisionPreparation::BailingVisionPreparation(const Config& config, const SessionInfo& info,
                                                   const Tokenizer& tokenizer, const QwenImageProcessor& images,
                                                   ThreadPool* thread_pool)
    : PatchVisionPreparation{config, info, tokenizer, images, thread_pool} {
  const auto& vision = config.model.vision;
  if (config.model.type != "bailing_moe_v3_vl")
    throw std::runtime_error("Ling preparation requires its Bailing VL model contract.");
  size_t index = 0;
  for (const auto* marker : {"<|video_start|>", "<|video_end|>"}) {
    const auto tokens = tokenizer.Encode(marker);
    if (tokens.size() != 1 || tokens[0] <= 0)
      throw std::runtime_error("Ling preparation requires the source video boundary tokens.");
    video_boundaries_[index++] = tokens[0];
  }
  if (!vision.video_config_filename) return;
  PatchProcessingLimits video;
  const auto definition = ReadPatchProcessor(config, *vision.video_config_filename, video);
  if (video.operations != std::vector<std::string>{"Resize", "Rescale", "Normalize", "PatchImage"} ||
      video.min_pixels <= 0 || video.total_pixels <= 0 || video.patch_size != vision.patch_size ||
      video.temporal_patch_size != 2 || video.merge_size != vision.spatial_merge_size ||
      video.smart_resize != 0 || video.coefficient_type != "int16" || video.interpolation != "CUBIC_OPENCV" ||
      video.bgr_normalization)
    throw std::runtime_error("Ling video requires the source cubic temporal preprocessing pipeline.");
  video_min_pixels_ = video.min_pixels;
  video_total_pixels_ = video.total_pixels;
  InitializeVideo(definition);
}

std::array<int64_t, 3> BailingVisionPreparation::ImageGrid(uint32_t width, uint32_t height) const {
  return BailingImageGrid(width, height, config_.model.vision.patch_size, config_.model.vision.spatial_merge_size,
                          min_pixels_, max_pixels_);
}

std::array<int64_t, 3> BailingVisionPreparation::VideoGrid(uint32_t width, uint32_t height, size_t frames) const {
  return BailingVideoGrid(width, height, frames, config_.model.vision.patch_size, config_.model.vision.spatial_merge_size,
                          video_min_pixels_, video_total_pixels_);
}

size_t BailingVisionPreparation::PromptBytes(std::string_view prompt, const VisionMediaPromptSource& source, size_t maximum_bytes) const {
  return BailingMediaPromptBytes(prompt, source, config_.model.vision.spatial_merge_size, maximum_bytes);
}

std::string BailingVisionPreparation::ExpandPrompt(std::string_view prompt, std::span<const std::array<int64_t, 3>> images,
                                                   std::span<const VisionVideoPrompt> videos, size_t maximum_bytes,
                                                   std::vector<size_t>& unit_order) const {
  return ExpandBailingMediaPrompt(prompt, images, videos, config_.model.vision.spatial_merge_size, maximum_bytes, &unit_order);
}

std::shared_ptr<const VisionPosition> BailingVisionPreparation::Position(std::array<int64_t, 3> grid) const {
  return MakeBailingVisionPosition(grid, config_.model.vision.spatial_merge_size);
}

std::array<int64_t, 2> BailingVisionPreparation::CoordinateRange() const {
  if (config_.model.context_length <= 0)
    throw std::runtime_error("Ling positions require a positive exported context length.");
  return {-static_cast<int64_t>(config_.model.context_length), static_cast<int64_t>(config_.model.context_length) - 1};
}

void BailingVisionPreparation::ValidateTextTokens(std::span<const int32_t> tokens) const {
  PatchVisionPreparation::ValidateTextTokens(tokens);
  for (const auto boundary : video_boundaries_)
    if (std::find(tokens.begin(), tokens.end(), boundary) != tokens.end())
      throw std::invalid_argument("Ling video markers require the formatted prompt and videos overload.");
}

void BailingVisionPreparation::ValidateDecoderMetadata(const OrtModelMetadata& metadata) const {
  const char* key = "com.microsoft.genai.vision_position_range";
  const auto keys = metadata.GetCustomMetadataMapKeysAllocated();
  const auto range = CoordinateRange();
  const auto expected = std::to_string(range[0]) + "," + std::to_string(range[1]);
  if (std::find(keys.begin(), keys.end(), key) == keys.end() || metadata.LookupCustomMetadataMap(key) != expected)
    throw std::runtime_error("Ling vision requires a matching signed rotary-cache export. Rebuild the complete model package.");
}

std::shared_ptr<Tensor> BailingVisionPreparation::ConvertVideoPixels(OrtxTensor& pixels) const {
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  const auto source = ProcessTensor<float>(thread_pool_, &pixels, allocator);
  return std::make_shared<Tensor>(ConvertVisionPixels(thread_pool_, *source, pixel_type_, allocator));
}

}  // namespace Generators
