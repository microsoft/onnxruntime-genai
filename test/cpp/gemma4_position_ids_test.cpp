// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <gtest/gtest.h>

#include "models/preprocessing/gemma4_multimodal_processor.h"
#include "telemetry_test_environment.h"

namespace {

template <typename T>
void CheckPositionIds(ONNXTensorElementDataType type) {
  constexpr std::array<int64_t, 16> source{
      0, 0, 7, 0, 3, 1, -1, -1,
      0, 0, 2, 0, 1, 1, -1, -1};
  constexpr std::array<int64_t, 3> shape{2, 4, 2};
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  for (int64_t target_patches : {3, 4, 5}) {
    auto output = Generators::ConvertAndResizeGemma4PositionIds(source.data(), shape, target_patches, type, allocator);
    EXPECT_EQ(output->GetTensorTypeAndShapeInfo()->GetElementType(), type);
    EXPECT_EQ(output->GetTensorTypeAndShapeInfo()->GetShape(),
              (std::vector<int64_t>{2, target_patches, 2}));
    const T* result = output->GetTensorData<T>();
    for (int64_t batch = 0; batch < 2; ++batch) {
      for (int64_t patch = 0; patch < target_patches; ++patch) {
        for (int64_t coordinate = 0; coordinate < 2; ++coordinate) {
          const T expected = patch < 4
                                 ? static_cast<T>(source[batch * 8 + patch * 2 + coordinate])
                                 : static_cast<T>(-1);
          EXPECT_EQ(result[batch * target_patches * 2 + patch * 2 + coordinate], expected);
        }
      }
    }
  }
}

TEST(Gemma4PositionIdsTests, TrimsCopiesAndPadsInt32AndInt64) {
  CheckPositionIds<int32_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  CheckPositionIds<int64_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
}

TEST(Gemma4PositionIdsTests, PreservesRankTwoInputsWhenTrimming) {
  constexpr std::array<int64_t, 8> source{0, 0, 7, 0, 3, 1, -1, -1};
  constexpr std::array<int64_t, 2> shape{4, 2};
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  auto output = Generators::ConvertAndResizeGemma4PositionIds(
      source.data(), shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, allocator);
  EXPECT_EQ(output->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{3, 2}));
  EXPECT_EQ(std::vector<int32_t>(output->GetTensorData<int32_t>(),
                                 output->GetTensorData<int32_t>() + 6),
            (std::vector<int32_t>{0, 0, 7, 0, 3, 1}));
}

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
