// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bailing_vision_position.h"

#include <algorithm>

namespace Generators {
namespace {

struct BailingVisionPosition : VisionPosition {
  BailingVisionPosition(size_t temporal, size_t height, size_t width)
      : VisionPosition{temporal * height * width, static_cast<int64_t>(temporal - 1) * 2 + 1},
        temporal{temporal},
        height{height},
        width{width} {}

  void Fill(size_t offset, int64_t base,
            const std::array<std::span<int64_t>, 3>& output) const override {
    const size_t count = output[0].size();
    if (output[1].size() != count || output[2].size() != count ||
        offset > rows || count > rows - offset)
      throw std::invalid_argument("Vision coordinate range exceeds its descriptor.");
    const auto spatial = static_cast<int64_t>(std::max(height, width) / 2);
    if (base < 0 || spatial > std::numeric_limits<int64_t>::max() - (text_advance - 1) ||
        base > std::numeric_limits<int64_t>::max() - (text_advance - 1) - spatial)
      throw std::overflow_error("Vision coordinate range exceeds int64.");
    const size_t spatial_rows = height * width;
    for (size_t index = 0; index < count; ++index) {
      const size_t row = offset + index;
      const auto time = base + static_cast<int64_t>(row / spatial_rows) * 2;
      output[0][index] = time;
      output[1][index] = time + (static_cast<int64_t>((row % spatial_rows) / width) -
                                 static_cast<int64_t>((height - 1) / 2));
      output[2][index] = time + (static_cast<int64_t>(row % width) -
                                 static_cast<int64_t>((width - 1) / 2));
    }
  }

  bool Equals(const VisionPosition& other) const override {
    const auto* value = dynamic_cast<const BailingVisionPosition*>(&other);
    return value && value->temporal == temporal && value->height == height && value->width == width;
  }

  size_t SizeBytes() const override { return sizeof(*this); }

  const size_t temporal;
  const size_t height;
  const size_t width;
};

}  // namespace

std::shared_ptr<const VisionPosition> MakeBailingVisionPosition(
    const std::array<int64_t, 3>& grid, size_t merge_size) {
  if (merge_size == 0 || merge_size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      grid[0] <= 0 || grid[1] <= 0 || grid[2] <= 0 ||
      grid[1] % static_cast<int64_t>(merge_size) != 0 ||
      grid[2] % static_cast<int64_t>(merge_size) != 0)
    throw std::invalid_argument("Bailing positions require a complete temporally merged grid.");
  const auto temporal = static_cast<uint64_t>(grid[0]);
  const auto height = static_cast<uint64_t>(grid[1]) / merge_size;
  const auto width = static_cast<uint64_t>(grid[2]) / merge_size;
  if (temporal > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 2 + 1 ||
      height > std::numeric_limits<size_t>::max() / width ||
      temporal > std::numeric_limits<size_t>::max() / (height * width) ||
      temporal > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / (height * width))
    throw std::overflow_error("Vision coordinate grid exceeds its row limit.");
  return std::make_shared<BailingVisionPosition>(static_cast<size_t>(temporal),
                                                 static_cast<size_t>(height), static_cast<size_t>(width));
}

}  // namespace Generators
