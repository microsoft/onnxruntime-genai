// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "models/model.h"
#include "models/io/multi_modal_features.h"

namespace Generators::test {
namespace {

template <size_t ShapeSize, size_t ValueSize>
std::unique_ptr<OrtValue> FloatTensor(const std::array<int64_t, ShapeSize>& shape,
                                      const std::array<float, ValueSize>& values) {
  GetOrtEnv();
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  auto tensor = OrtValue::CreateTensor<float>(allocator, shape);
  std::copy(values.begin(), values.end(), tensor->GetTensorMutableData<float>());
  return tensor;
}

std::vector<float> TensorValues(const OrtValue& tensor) {
  const size_t count = tensor.GetTensorTypeAndShapeInfo()->GetElementCount();
  const float* data = tensor.GetTensorData<float>();
  return {data, data + count};
}

TEST(MergeImageFeaturesIntoEmbeddingsTest, CopiesRowsInOrderForTwoAndThreeDimensionalEmbeddings) {
  auto embeddings_2d = FloatTensor(std::array<int64_t, 2>{3, 2}, std::array<float, 6>{0, 1, 2, 3, 4, 5});
  auto embeddings_3d = FloatTensor(std::array<int64_t, 3>{1, 3, 2}, std::array<float, 6>{0, 1, 2, 3, 4, 5});
  auto features = FloatTensor(std::array<int64_t, 2>{2, 2}, std::array<float, 4>{10, 11, 20, 21});
  const std::array<int64_t, 2> target_rows{0, 2};

  EXPECT_EQ(MergeImageFeaturesIntoEmbeddings(*embeddings_2d, *features, target_rows), 2u);
  EXPECT_EQ(MergeImageFeaturesIntoEmbeddings(*embeddings_3d, *features, target_rows), 2u);
  const std::vector<float> expected{10, 11, 2, 3, 20, 21};
  EXPECT_EQ(TensorValues(*embeddings_2d), expected);
  EXPECT_EQ(TensorValues(*embeddings_3d), expected);
}

TEST(MergeImageFeaturesIntoEmbeddingsTest, RejectsInvalidRowsBeforeWriting) {
  auto embeddings = FloatTensor(std::array<int64_t, 2>{3, 2}, std::array<float, 6>{0, 1, 2, 3, 4, 5});
  auto features = FloatTensor(std::array<int64_t, 2>{2, 2}, std::array<float, 4>{10, 11, 20, 21});
  const std::vector<float> original = TensorValues(*embeddings);
  const std::array<int64_t, 2> out_of_range{0, 3};
  const std::array<int64_t, 1> negative{-1};

  EXPECT_THROW(MergeImageFeaturesIntoEmbeddings(*embeddings, *features, out_of_range), std::runtime_error);
  EXPECT_EQ(TensorValues(*embeddings), original);
  EXPECT_THROW(MergeImageFeaturesIntoEmbeddings(*embeddings, *features, negative), std::runtime_error);
  EXPECT_EQ(TensorValues(*embeddings), original);
}

TEST(MergeImageFeaturesIntoEmbeddingsTest, RejectsNonFloatTensorsAndHiddenDimensionMismatch) {
  GetOrtEnv();
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  auto embeddings = FloatTensor(std::array<int64_t, 2>{2, 2}, std::array<float, 4>{0, 1, 2, 3});
  auto features = FloatTensor(std::array<int64_t, 2>{1, 2}, std::array<float, 2>{10, 11});
  auto float16_embeddings =
      OrtValue::CreateTensor<Ort::Float16_t>(allocator, std::array<int64_t, 2>{2, 2});
  auto float16_features =
      OrtValue::CreateTensor<Ort::Float16_t>(allocator, std::array<int64_t, 2>{1, 2});
  auto mismatched_features = FloatTensor(std::array<int64_t, 2>{1, 3}, std::array<float, 3>{10, 11, 12});
  const std::array<int64_t, 1> target_rows{0};

  EXPECT_THROW(MergeImageFeaturesIntoEmbeddings(*float16_embeddings, *features, target_rows), std::runtime_error);
  EXPECT_THROW(MergeImageFeaturesIntoEmbeddings(*embeddings, *float16_features, target_rows), std::runtime_error);
  EXPECT_THROW(MergeImageFeaturesIntoEmbeddings(*embeddings, *mismatched_features, target_rows), std::runtime_error);
}

}  // namespace
}  // namespace Generators::test
