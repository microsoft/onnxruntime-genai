// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>

namespace Generators {

void ExtractQwenPatchesSequential(const float* source, float* destination,
                                  int64_t height, int64_t width, int64_t channels,
                                  int64_t patch_size);
void TransposeVideoSequential(const float* source, float* destination,
                              int64_t images, int64_t channels, int64_t height, int64_t width);
void WriteLfm2PatchesSequential(const float* image, float* destination, int64_t channels,
                                int64_t height, int64_t width, int64_t patch_size);

}  // namespace Generators
