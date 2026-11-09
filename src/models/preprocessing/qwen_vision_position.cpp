// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qwen_vision_position.h"

#include <algorithm>

namespace Generators {
namespace {

struct QwenVisionPosition : VisionPosition {
  QwenVisionPosition(size_t height, size_t width)
      : VisionPosition{height * width, static_cast<int64_t>(std::max(height, width))},
        height{height},
        width{width} {}

  void Fill(size_t offset, int64_t base,
            const std::array<std::span<int64_t>, 3>& output) const override {
    const size_t count = output[0].size();
    if (output[1].size() != count || output[2].size() != count ||
        offset > rows || count > rows - offset)
      throw std::invalid_argument("Vision coordinate range exceeds its descriptor.");
    if (base < 0 || base > std::numeric_limits<int64_t>::max() - text_advance)
      throw std::overflow_error("Vision coordinate range exceeds int64.");
    for (size_t index = 0; index < count; ++index) {
      output[0][index] = base;
      output[1][index] = base + static_cast<int64_t>((offset + index) / width);
      output[2][index] = base + static_cast<int64_t>((offset + index) % width);
    }
  }

  bool Equals(const VisionPosition& other) const override {
    const auto* value = dynamic_cast<const QwenVisionPosition*>(&other);
    return value && value->height == height && value->width == width;
  }

  size_t SizeBytes() const override { return sizeof(*this); }

  const size_t height;
  const size_t width;
};

}  // namespace

std::shared_ptr<const VisionPosition> MakeQwenVisionPosition(
    const std::array<int64_t, 3>& grid, size_t merge_size) {
  if (merge_size == 0 || merge_size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      grid[0] != 1 || grid[1] <= 0 || grid[2] <= 0 ||
      grid[1] % static_cast<int64_t>(merge_size) != 0 ||
      grid[2] % static_cast<int64_t>(merge_size) != 0)
    throw std::invalid_argument("Qwen positions require a complete single-group merged grid.");
  const auto height = static_cast<uint64_t>(grid[1]) / merge_size;
  const auto width = static_cast<uint64_t>(grid[2]) / merge_size;
  if (height > std::numeric_limits<size_t>::max() / width ||
      height > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / width)
    throw std::overflow_error("Vision coordinate grid exceeds its row limit.");
  return std::make_shared<QwenVisionPosition>(static_cast<size_t>(height), static_cast<size_t>(width));
}

}  // namespace Generators
