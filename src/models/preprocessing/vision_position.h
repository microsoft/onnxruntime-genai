// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "span.h"

namespace Generators {

struct VisionPosition {
  VisionPosition(size_t rows, int64_t text_advance) : rows{rows}, text_advance{text_advance} {
    if (rows == 0 || rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) || text_advance < 0)
      throw std::invalid_argument("Vision position descriptor exceeds its coordinate limits.");
  }
  virtual ~VisionPosition() = default;

  virtual void Fill(size_t offset, int64_t base,
                    const std::array<std::span<int64_t>, 3>& output) const = 0;
  virtual bool Equals(const VisionPosition& other) const = 0;
  virtual size_t SizeBytes() const = 0;

  const size_t rows;
  const int64_t text_advance;
};

}  // namespace Generators
