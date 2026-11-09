// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "patch_vision_preparation.h"

namespace Generators {

struct QwenVisionPreparation : PatchVisionPreparation {
  QwenVisionPreparation(const Config& config, const SessionInfo& info,
                        const Tokenizer& tokenizer, const QwenImageProcessor& images, ThreadPool* thread_pool);
  std::array<int64_t, 3> ImageGrid(uint32_t width, uint32_t height) const override;
  std::array<int64_t, 3> VideoGrid(uint32_t width, uint32_t height, size_t frames) const override;
  size_t PromptBytes(std::string_view prompt, const VisionMediaPromptSource& source, size_t maximum_bytes) const override;
  std::string ExpandPrompt(std::string_view prompt, std::span<const std::array<int64_t, 3>> images,
                           std::span<const VisionVideoPrompt> videos, size_t maximum_bytes,
                           std::vector<size_t>& unit_order) const override;
  std::shared_ptr<const VisionPosition> Position(std::array<int64_t, 3> grid) const override;
  std::array<int64_t, 2> CoordinateRange() const override;

 private:
  int64_t video_min_pixels_{}, video_max_pixels_{};
  size_t max_video_tokens_{};
};

}  // namespace Generators
