// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <vector>

#include "models/model_type.h"
#include "models/preprocessing/lfm3_vl_image_processor.h"
#include "models/threadpool.h"

// These pin the two places where the processor and the checkpoint's Lfm3VlProcessor have to agree on image
// boundaries: which rows belong to which image, and how many markers each image turns into.

namespace Generators::test {
namespace {

std::string Repeat(const std::string& text, int64_t count) {
  std::string out;
  for (int64_t i = 0; i < count; ++i) {
    out += text;
  }
  return out;
}

}  // namespace

// Two images of 1 and 3 rows, 2 values wide, with -1 in the padded row.
TEST(Lfm3VlCompactImageRowsTest, DropsThePaddingBetweenImages) {
  const std::vector<float> batch{1, 2, -1, -1, -1, -1,  //
                                 3, 4, 5, 6, 7, 8};
  std::vector<float> rows(8);
  CompactLfm3VlImageRows(nullptr, batch.data(), 3, 2, {1, 3}, rows.data());
  EXPECT_EQ(rows, (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(Lfm3VlCompactImageRowsTest, ThreadPoolWritesTheSameRows) {
  // More values than the parallel cost gate, so the pool really runs.
  constexpr int64_t kRows = 256;
  constexpr int64_t kWidth = 1536;
  std::vector<float> batch(static_cast<size_t>(2 * kRows * kWidth));
  std::iota(batch.begin(), batch.end(), 0.0f);
  ASSERT_GE(static_cast<double>(batch.size()), ThreadPool::kMinimumParallelCost);
  const std::vector<int64_t> image_rows{kRows - 16, kRows};
  std::vector<float> serial(static_cast<size_t>((2 * kRows - 16) * kWidth));
  std::vector<float> threaded(serial.size());
  ThreadPool pool{4};
  CompactLfm3VlImageRows(nullptr, batch.data(), kRows, kWidth, image_rows, serial.data());
  CompactLfm3VlImageRows(&pool, batch.data(), kRows, kWidth, image_rows, threaded.data());
  EXPECT_EQ(serial, threaded);
  // The second image starts right after the first one's rows, read from the start of its padded slot.
  EXPECT_EQ(serial[static_cast<size_t>((kRows - 16) * kWidth)], batch[static_cast<size_t>(kRows * kWidth)]);
}

TEST(Lfm3VlExpandImageTokensTest, EachMarkerGetsItsImagesCountInOrder) {
  const std::string token{kLfm3VlImageToken};
  EXPECT_EQ(ExpandLfm3VlImageTokens("<|vision_start|>" + token + "<|vision_end|>a <|vision_start|>" + token + "<|vision_end|>b", {2, 3}),
            "<|vision_start|>" + Repeat(token, 2) + "<|vision_end|>a <|vision_start|>" + Repeat(token, 3) + "<|vision_end|>b");
}

TEST(Lfm3VlModelTypeTest, RunsThroughTheVisionPipelineNotThePlainDecoder) {
  EXPECT_TRUE(ModelType::IsVLM("lfm3_vl"));
  EXPECT_FALSE(ModelType::IsLLM("lfm3_vl"));
  EXPECT_FALSE(ModelType::IsLFM2("lfm3_vl"));
  EXPECT_FALSE(ModelType::IsQwenVLFamily("lfm3_vl"));
}

}  // namespace Generators::test
