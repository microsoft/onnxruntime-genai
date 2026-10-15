// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "models/preprocessing/lfm2_vl_image_processor.h"
#include "models/threadpool.h"

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
  // Smart resize normally yields an even patch grid; for an odd one the token count must round up,
  // as `_compute_tokens_for_image` does.
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
  for (const auto [height, width] : {std::pair{0, 512}, std::pair{512, -16}}) {
    const std::string message = CaptureThrowMessage([=] { Geometry(height, width); });
    EXPECT_NE(message.find("must be positive"), std::string::npos) << message;
  }
}

// The vision encoder's patch embedding is a plain linear layer over the flattened patch, so the
// element order inside each patch is the whole contract. Pin it to `convert_image_to_patches`:
// patch (row, col), element (y, x, c) <- image[c][row * p + y][col * p + x].
TEST(Lfm2VlImagePatchesTest, FlattensPatchesInYXChannelOrderFromPaddedBatch) {
  constexpr int64_t channels = 3, padded_height = 8, padded_width = 12, patch = 2;
  const Lfm2VlImageGeometry geometry = ComputeLfm2VlImageGeometry(4, 8, patch, kDownsampleFactor);

  std::vector<float> image(channels * padded_height * padded_width);
  for (size_t i = 0; i < image.size(); ++i) image[i] = static_cast<float>(i);

  std::vector<float> patches(geometry.num_patches * patch * patch * channels, -1.0f);
  WriteLfm2VlImagePatches(nullptr, image.data(), channels, padded_height, padded_width, geometry, patch, patches.data());

  for (int64_t row = 0; row < geometry.patch_rows; ++row) {
    for (int64_t col = 0; col < geometry.patch_cols; ++col) {
      for (int64_t y = 0; y < patch; ++y) {
        for (int64_t x = 0; x < patch; ++x) {
          for (int64_t c = 0; c < channels; ++c) {
            const int64_t index = ((row * geometry.patch_cols + col) * patch * patch + y * patch + x) * channels + c;
            const int64_t source = c * padded_height * padded_width + (row * patch + y) * padded_width + col * patch + x;
            EXPECT_EQ(patches[index], static_cast<float>(source)) << "row " << row << " col " << col << " y " << y << " x " << x << " c " << c;
          }
        }
      }
    }
  }
}

TEST(Lfm2VlImagePatchesTest, ParallelWorkersMatchSequentialOutput) {
  constexpr int64_t channels = 3, height = 512, width = 512, patch = 16;
  const Lfm2VlImageGeometry geometry =
      ComputeLfm2VlImageGeometry(height, width, patch, kDownsampleFactor);

  std::vector<float> image(static_cast<size_t>(channels * height * width));
  for (size_t i = 0; i < image.size(); ++i) {
    image[i] = static_cast<float>(i % 251);
  }

  const size_t output_size =
      static_cast<size_t>(geometry.num_patches * patch * patch * channels);
  std::vector<float> expected(output_size);
  WriteLfm2VlImagePatches(nullptr, image.data(), channels, height, width, geometry,
                          patch, expected.data());

  for (size_t worker_count : {1U, 3U}) {
    ThreadPool pool{worker_count};
    std::vector<float> actual(output_size);
    WriteLfm2VlImagePatches(&pool, image.data(), channels, height, width, geometry,
                            patch, actual.data());
    EXPECT_EQ(actual, expected);
  }
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

TEST(Lfm2VlImageTokensTest, RejectsPromptAskingForMoreImagesThanProvided) {
  const std::string message =
      CaptureThrowMessage([] { ExpandLfm2VlImageTokens("<image><image>", {4}); });
  EXPECT_NE(message.find("contains 2 <image> tokens but 1 images were provided"), std::string::npos) << message;
}

TEST(Lfm2VlImageTokensTest, RejectsPromptWithFewerImageTokensThanImages) {
  // Prepending or appending the unreferenced images would have to guess their order relative to the
  // referenced ones; the Hugging Face processor rejects the mismatch, and so does this one.
  const std::string message =
      CaptureThrowMessage([] { ExpandLfm2VlImageTokens("describe <image>", {64, 256}); });
  EXPECT_NE(message.find("contains 1 <image> tokens but 2 images were provided"), std::string::npos) << message;
}

TEST(Lfm2VlImageTokensTest, RejectsImageTokenWhenNoImagesWereProvided) {
  const std::string message = CaptureThrowMessage([] { ExpandLfm2VlImageTokens("<image>describe", {}); });
  EXPECT_NE(message.find("contains 1 <image> tokens but 0 images were provided"), std::string::npos) << message;
}

}  // namespace Generators::test
