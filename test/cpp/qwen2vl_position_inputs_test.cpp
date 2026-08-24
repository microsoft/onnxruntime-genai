// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <vector>

#include "models/io/qwen_vl_position_inputs.h"

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

TEST(Qwen2VLPositionInputsTest, RejectsNegativeGridDimensions) {
  const std::vector<int64_t> grid{1, -1, 2};
  const std::string message = CaptureThrowMessage([&] {
    ValidateQwen2VLGridTensorValues(grid.data(), grid.size(), "image_grid_thw");
  });
  EXPECT_NE(message.find("non-negative"), std::string::npos) << message;
}

TEST(Qwen2VLPositionInputsTest, RejectsExcessiveGridDimensions) {
  const std::vector<int64_t> grid{1, 1, 20000};
  const std::string message = CaptureThrowMessage([&] {
    ValidateQwen2VLGridTensorValues(grid.data(), grid.size(), "video_grid_thw");
  });
  EXPECT_NE(message.find("<= 16384"), std::string::npos) << message;
}

TEST(Qwen2VLPositionInputsTest, RejectsVisionLenExceedingSequenceLength) {
  const std::string message = CaptureThrowMessage([&] {
    ValidateQwen2VLVisionLengthFitsSequence(8, 8, 8, 5, 100);
  });
  EXPECT_NE(message.find("positions available in sequence"), std::string::npos) << message;
}

TEST(Qwen2VLPositionInputsTest, RejectsGridWithIncorrectElementCount) {
  const std::vector<int64_t> grid{1, 2};
  const std::string message = CaptureThrowMessage([&] {
    ValidateQwen2VLGridTensorValues(grid.data(), grid.size(), "image_grid_thw");
  });
  EXPECT_NE(message.find("divisible by 3"), std::string::npos) << message;
}

TEST(Qwen2VLPositionInputsTest, AcceptsValidGridDimensions) {
  const std::vector<int64_t> grid{1, 8, 8, 2, 4, 4};
  EXPECT_NO_THROW(ValidateQwen2VLGridTensorValues(grid.data(), grid.size(), "image_grid_thw"));
  EXPECT_NO_THROW(ValidateQwen2VLVisionLengthFitsSequence(2, 2, 2, 10, 100));
}

}  // namespace Generators::test
