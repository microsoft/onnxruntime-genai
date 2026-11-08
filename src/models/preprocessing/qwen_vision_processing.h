// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "span.h"

namespace Generators {

struct QwenVideoPrompt {
  std::array<int64_t, 3> grid;
  std::span<const double> timestamps;
};

size_t QwenMediaPromptBytes(std::string_view prompt,
                            std::span<const std::array<int64_t, 3>> image_grids,
                            std::span<const QwenVideoPrompt> videos, size_t merge_size,
                            size_t maximum_bytes);

std::string ExpandQwenMediaPrompt(std::string_view prompt,
                                  std::span<const std::array<int64_t, 3>> image_grids,
                                  std::span<const QwenVideoPrompt> videos, size_t merge_size,
                                  size_t maximum_bytes);

std::array<int64_t, 3> QwenImageGrid(uint32_t width, uint32_t height, size_t patch, size_t merge,
                                     int64_t min_pixels, int64_t max_pixels);
std::array<int64_t, 3> QwenVideoGrid(uint32_t width, uint32_t height, size_t frames,
                                     size_t patch, size_t merge, int64_t min_pixels,
                                     int64_t max_pixels, size_t max_video_tokens);

}  // namespace Generators
