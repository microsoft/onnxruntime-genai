// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "models/preprocessing/lfm2_vl_image_processor.h"

// LFM2-VL splits an image into 16x16 encoder patches and the projector then pixel-unshuffles that
// grid by 2, so the decoder sees one token per 32x32 pixel block. These tests pin the two things
// that have to agree for the pipeline to line up: how many tokens an image is worth, and how many
// placeholder tokens the prompt gets in return.

namespace Generators::test {
namespace {

constexpr int64_t kEncoderPatchSize = 16;
constexpr int64_t kDownsampleFactor = 2;

Lfm2VlImageGeometry Geometry(int64_t height, int64_t width) {
  return ComputeLfm2VlImageGeometry(height, width, kEncoderPatchSize, kDownsampleFactor);
}

template <typename Fn>
std::string CaptureThrowMessage(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}

}  // namespace

TEST(Lfm2VlImageGeometryTest, TileSizedImageIsWorth256Tokens) {
  // A 512x512 tile is the upper bound the Hugging Face processor targets: 32x32 patches -> 16x16
  // tokens. This is the number `_compute_tokens_per_tile` returns for the shipped config.
  const auto geometry = Geometry(512, 512);
  EXPECT_EQ(geometry.patch_rows, 32);
  EXPECT_EQ(geometry.patch_cols, 32);
  EXPECT_EQ(geometry.num_patches, 1024);
  EXPECT_EQ(geometry.num_tokens, 256);
}

TEST(Lfm2VlImageGeometryTest, NonSquareImageKeepsPerAxisGrid) {
  const auto geometry = Geometry(256, 384);
  EXPECT_EQ(geometry.patch_rows, 16);
  EXPECT_EQ(geometry.patch_cols, 24);
  EXPECT_EQ(geometry.num_patches, 384);
  EXPECT_EQ(geometry.num_tokens, 8 * 12);
}

TEST(Lfm2VlImageGeometryTest, OddPatchGridRoundsTokensUp) {
  // Smart resize normally snaps to a multiple of patch_size * downsample_factor, but the token
  // count still has to round up the way `_compute_tokens_for_image` does when it does not.
  const auto geometry = Geometry(16 * 5, 16 * 3);
  EXPECT_EQ(geometry.patch_rows, 5);
  EXPECT_EQ(geometry.patch_cols, 3);
  EXPECT_EQ(geometry.num_tokens, 3 * 2);
}

TEST(Lfm2VlImageGeometryTest, SmallestImageIsOneToken) {
  const auto geometry = Geometry(32, 32);
  EXPECT_EQ(geometry.num_patches, 4);
  EXPECT_EQ(geometry.num_tokens, 1);
}

TEST(Lfm2VlImageGeometryTest, RejectsSizeThatIsNotAWholeNumberOfPatches) {
  const std::string message = CaptureThrowMessage([] { Geometry(100, 512); });
  EXPECT_NE(message.find("whole number of 16-pixel patches"), std::string::npos) << message;
}

TEST(Lfm2VlImageGeometryTest, RejectsNonPositiveDimensions) {
  EXPECT_FALSE(CaptureThrowMessage([] { Geometry(0, 512); }).empty());
  EXPECT_FALSE(CaptureThrowMessage([] { Geometry(512, -16); }).empty());
}

TEST(Lfm2VlImageTokensTest, PlaceholderWrapsImageTokensInStartAndEndMarkers) {
  EXPECT_EQ(BuildLfm2VlImagePlaceholder(3), "<|image_start|><image><image><image><|image_end|>");
}

TEST(Lfm2VlImageTokensTest, ExpandsEachImageTokenWithItsOwnTokenCount) {
  const std::string expanded = ExpandLfm2VlImageTokens("a<image>b<image>c", {2, 1});
  EXPECT_EQ(expanded,
            "a<|image_start|><image><image><|image_end|>b<|image_start|><image><|image_end|>c");
}

TEST(Lfm2VlImageTokensTest, LeavesPromptsWithoutImagesAlone) {
  EXPECT_EQ(ExpandLfm2VlImageTokens("describe this", {}), "describe this");
}

TEST(Lfm2VlImageTokensTest, PrependsImagesThePromptNeverReferenced) {
  // Prompts that were not built from the chat template still have to consume every image, otherwise
  // the vision features would outnumber the placeholder positions in the decoder.
  EXPECT_EQ(ExpandLfm2VlImageTokens("describe this", {1}),
            "<|image_start|><image><|image_end|>describe this");
}

TEST(Lfm2VlImageTokensTest, RejectsPromptAskingForMoreImagesThanProvided) {
  const std::string message =
      CaptureThrowMessage([] { ExpandLfm2VlImageTokens("<image><image>", {4}); });
  EXPECT_NE(message.find("more <image> tokens than the 1 images"), std::string::npos) << message;
}

}  // namespace Generators::test
