// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "vision_media_prompt.h"
#include <string>
#include <string_view>
#include <vector>

namespace Generators {

std::array<int64_t, 3> BailingImageGrid(uint32_t width, uint32_t height, size_t patch, size_t merge,
                                        int64_t min_pixels, int64_t max_pixels);
std::array<int64_t, 3> BailingVideoGrid(uint32_t width, uint32_t height, size_t frames, size_t patch, size_t merge,
                                        int64_t min_pixels, int64_t total_pixels);
size_t BailingMediaPromptBytes(std::string_view prompt, const VisionMediaPromptSource& source,
                               size_t merge, size_t maximum_bytes);
std::string ExpandBailingMediaPrompt(std::string_view prompt, std::span<const std::array<int64_t, 3>> images,
                                     std::span<const VisionVideoPrompt> videos, size_t merge, size_t maximum_bytes,
                                     std::vector<size_t>* unit_order = nullptr);

}  // namespace Generators
