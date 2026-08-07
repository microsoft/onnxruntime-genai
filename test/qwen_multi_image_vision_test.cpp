// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "models/multi_modal.h"

namespace Generators::test {
namespace {

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

TEST(QwenVisionMultiImageTest, RejectsInsufficientGridElementsViaGeneratorInputs) {
  const std::vector<int64_t> shape{2, 3};
  const std::string message = CaptureThrowMessage([&] {
    ValidateImageGridThwLayoutAndCount(shape, 4, 2, "image_grid_thw");
  });
  EXPECT_NE(message.find("need at least 3 values per image"), std::string::npos) << message;
}

TEST(QwenVisionMultiImageTest, RejectsRank1GridTensorViaGeneratorInputs) {
  const std::vector<int64_t> shape{6};
  const std::string message = CaptureThrowMessage([&] {
    ValidateImageGridThwLayoutAndCount(shape, 6, 2, "image_grid_thw");
  });
  EXPECT_NE(message.find("must have rank 2"), std::string::npos) << message;
}

TEST(QwenVisionMultiImageTest, RejectsGridWithUnexpectedSecondDimension) {
  const std::vector<int64_t> shape{2, 2};
  const std::string message = CaptureThrowMessage([&] {
    ValidateImageGridThwLayoutAndCount(shape, 4, 2, "image_grid_thw");
  });
  EXPECT_NE(message.find("second dimension must be 3"), std::string::npos) << message;
}

TEST(QwenVisionMultiImageTest, RejectsNegativeImageCount) {
  const std::vector<int64_t> shape{2, 3};
  const std::string message = CaptureThrowMessage([&] {
    ValidateImageGridThwLayoutAndCount(shape, 6, -1, "image_grid_thw");
  });
  EXPECT_NE(message.find("num_images must be non-negative"), std::string::npos) << message;
}

TEST(QwenVisionMultiImageTest, AcceptsValidGridShapeViaGeneratorInputs) {
  const std::vector<int64_t> shape{2, 3};
  EXPECT_NO_THROW(ValidateImageGridThwLayoutAndCount(shape, 6, 2, "image_grid_thw"));
}

}  // namespace Generators::test
