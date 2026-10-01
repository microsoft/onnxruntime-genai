// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "generator/generators.h"
#include "models/qwen_vl_state.h"
#include "telemetry_test_environment.h"

namespace {

template <typename Fn>
std::string CaptureRuntimeErrorMessage(Fn&& fn) {
  try {
    fn();
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return {};
}

}  // namespace

TEST(QwenPatchLayoutTest, UsesPaddedStrideForDifferentImageGridSizes) {
  const auto layout = Generators::ResolveQwenPatchLayout(
      /*total_patches=*/27520,
      /*total_grid_tokens=*/20004,
      /*total_hw=*/20004,
      /*max_grid_tokens=*/6880,
      /*num_images=*/4);

  EXPECT_EQ(layout.padded_image_stride, 6880);
  EXPECT_EQ(layout.temporal_multiplier, 0);
  EXPECT_EQ(layout.ImagePatchOffset(2, 12504), 13760);
  EXPECT_EQ(layout.ImagePatchCount(1900, 38, 50), 1900);
}

TEST(QwenPatchLayoutTest, KeepsPackedLayoutWhenPatchCountsMatchGrid) {
  const auto layout = Generators::ResolveQwenPatchLayout(20004, 20004, 20004, 6880, 4);
  EXPECT_EQ(layout.padded_image_stride, 0);
  EXPECT_EQ(layout.temporal_multiplier, 0);
  EXPECT_EQ(layout.ImagePatchOffset(2, 12504), 12504);
}

TEST(QwenPatchLayoutTest, RejectsUnsupportedPatchLayout) {
  const std::string message = CaptureRuntimeErrorMessage([] {
    Generators::ResolveQwenPatchLayout(
        /*total_patches=*/20005,
        /*total_grid_tokens=*/20004,
        /*total_hw=*/20004,
        /*max_grid_tokens=*/6880,
        /*num_images=*/4);
  });

  EXPECT_EQ(message, "pixel_values patch count (20005) does not match image_grid_thw patch count (20004)");
}

TEST(QwenPatchLayoutTest, PreservesUnambiguousTemporalPadding) {
  const auto layout = Generators::ResolveQwenPatchLayout(
      /*total_patches=*/27,
      /*total_grid_tokens=*/18,
      /*total_hw=*/9,
      /*max_grid_tokens=*/10,
      /*num_images=*/2);

  EXPECT_EQ(layout.padded_image_stride, 0);
  EXPECT_EQ(layout.temporal_multiplier, 3);
  EXPECT_EQ(layout.ImagePatchOffset(1, 12), 12);
  EXPECT_EQ(layout.ImagePatchCount(10, 1, 5), 15);
}

TEST(QwenPatchLayoutTest, UsesPaddedStrideWhenTotalIsAlsoDivisibleByTotalHw) {
  const auto layout = Generators::ResolveQwenPatchLayout(
      /*total_patches=*/768,
      /*total_grid_tokens=*/384,
      /*total_hw=*/384,
      /*max_grid_tokens=*/256,
      /*num_images=*/3);

  EXPECT_EQ(layout.padded_image_stride, 256);
  EXPECT_EQ(layout.temporal_multiplier, 0);
  EXPECT_EQ(layout.ImagePatchOffset(2, 320), 512);
  EXPECT_EQ(layout.ImagePatchCount(64, 8, 8), 64);
}

TEST(QwenPatchLayoutTest, PreservesTemporalPaddingWhenStrideDoesNotMatchMaximumGrid) {
  const auto layout = Generators::ResolveQwenPatchLayout(
      /*total_patches=*/24,
      /*total_grid_tokens=*/12,
      /*total_hw=*/12,
      /*max_grid_tokens=*/8,
      /*num_images=*/2);

  EXPECT_EQ(layout.padded_image_stride, 0);
  EXPECT_EQ(layout.temporal_multiplier, 2);
  EXPECT_EQ(layout.ImagePatchOffset(1, 8), 8);
  EXPECT_EQ(layout.ImagePatchCount(8, 2, 4), 16);
}

TEST(QwenPatchLayoutTest, PreservesTemporalPaddingWhenStrideMatchesMaximumGrid) {
  const auto layout = Generators::ResolveQwenPatchLayout(
      /*total_patches=*/16,
      /*total_grid_tokens=*/12,
      /*total_hw=*/8,
      /*max_grid_tokens=*/8,
      /*num_images=*/2,
      /*all_temporal_dims_one=*/false);

  EXPECT_EQ(layout.padded_image_stride, 0);
  EXPECT_EQ(layout.temporal_multiplier, 2);
  EXPECT_EQ(layout.ImagePatchOffset(0, 0), 0);
  EXPECT_EQ(layout.ImagePatchCount(4, 2, 2), 8);
  EXPECT_EQ(layout.ImagePatchOffset(1, 8), 8);
  EXPECT_EQ(layout.ImagePatchCount(8, 2, 2), 8);
}

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
