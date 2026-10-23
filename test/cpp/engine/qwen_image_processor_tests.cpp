// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "models/preprocessing/qwen2_5_vl_image_processor.h"

namespace Generators::test {
namespace {

TEST(QwenPatchedPixelValuesShapeTest, MatchesRankTwoVisionInput) {
  EXPECT_EQ(GetQwenPatchedPixelValuesShape(12, 1536, 2),
            (std::vector<int64_t>{12, 1536}));
}

TEST(QwenPatchedPixelValuesShapeTest, MatchesRankThreeVisionInput) {
  EXPECT_EQ(GetQwenPatchedPixelValuesShape(12, 1536, 3),
            (std::vector<int64_t>{1, 12, 1536}));
}

TEST(QwenPatchedPixelValuesShapeTest, DefaultsToRankTwoWithoutVisionSession) {
  EXPECT_EQ(GetQwenPatchedPixelValuesShape(12, 1536, std::nullopt),
            (std::vector<int64_t>{12, 1536}));
}

TEST(QwenPatchedPixelValuesShapeTest, RejectsUnsupportedVisionInputRank) {
  EXPECT_THROW(GetQwenPatchedPixelValuesShape(12, 1536, 4), std::runtime_error);
}

}  // namespace
}  // namespace Generators::test
