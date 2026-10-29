// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace Generators {

template <typename T>
struct DeviceSpan;

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

// Returns the prompt positions holding image_token_id, reading the tokens' current device contents.
std::vector<int64_t> FindQwenImageTokenRows(DeviceSpan<int32_t> tokens, int32_t image_token_id);

}  // namespace Generators
