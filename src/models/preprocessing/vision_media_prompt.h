// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include "span.h"

namespace Generators {

struct VisionVideoPrompt {
  std::array<int64_t, 3> grid;
  std::span<const double> timestamps;
};

// Estimates read bounded metadata without allocating grid or timestamp arrays.
struct VisionMediaPromptSource {
  virtual ~VisionMediaPromptSource() = default;
  virtual size_t ImageCount() const = 0;
  virtual size_t VideoCount() const = 0;
  virtual std::array<int64_t, 3> ImageGrid(size_t image) const = 0;
  virtual std::array<int64_t, 3> VideoGrid(size_t video) const = 0;
  virtual size_t FrameCount(size_t video) const = 0;
  virtual double Timestamp(size_t video, size_t frame) const = 0;
};

}  // namespace Generators
