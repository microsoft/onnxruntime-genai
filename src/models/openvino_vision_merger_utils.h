// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace Generators {

struct QwenImageSlice {
  int64_t patch_offset{};
  int64_t num_patches{};
  int64_t temporal{};
  int64_t height{};
  int64_t width{};
};

std::vector<QwenImageSlice> PlanQwenImageSlices(std::span<const int64_t> grid_values,
                                                int64_t total_patches,
                                                int64_t merge_size);

}  // namespace Generators
