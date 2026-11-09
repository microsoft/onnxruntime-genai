// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>

#include "vision_position.h"

namespace Generators {

std::shared_ptr<const VisionPosition> MakeBailingVisionPosition(
    const std::array<int64_t, 3>& grid, size_t merge_size);

}  // namespace Generators
