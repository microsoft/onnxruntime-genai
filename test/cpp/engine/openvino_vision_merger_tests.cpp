// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "config.h"
#include "models/model.h"
#include "models/openvino_multi_modal.h"
#include "models/openvino_vision_merger.h"
#include "models/openvino_vision_merger_utils.h"

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

TEST(PlanQwenImageSlicesTest, PlansOnePackedImage) {
  const std::array<int64_t, 3> grids{1, 2, 4};
  const auto slices = PlanQwenImageSlices(grids, 8, 2);

  ASSERT_EQ(slices.size(), 1u);
  EXPECT_EQ(slices[0].patch_offset, 0);
  EXPECT_EQ(slices[0].num_patches, 8);
  EXPECT_EQ(slices[0].temporal, 1);
  EXPECT_EQ(slices[0].height, 2);
  EXPECT_EQ(slices[0].width, 4);
}

TEST(PlanQwenImageSlicesTest, PlansDifferentImagesInPackedAndPaddedLayouts) {
  const std::array<int64_t, 6> grids{1, 2, 2, 1, 4, 2};
  const auto packed = PlanQwenImageSlices(grids, 12, 2);
  ASSERT_EQ(packed.size(), 2u);
  EXPECT_EQ(packed[0].patch_offset, 0);
  EXPECT_EQ(packed[0].num_patches, 4);
  EXPECT_EQ(packed[1].patch_offset, 4);
  EXPECT_EQ(packed[1].num_patches, 8);

  const auto padded = PlanQwenImageSlices(grids, 16, 2);
  ASSERT_EQ(padded.size(), 2u);
  EXPECT_EQ(padded[0].patch_offset, 0);
  EXPECT_EQ(padded[0].num_patches, 4);
  EXPECT_EQ(padded[1].patch_offset, 8);
  EXPECT_EQ(padded[1].num_patches, 8);
}

TEST(PlanQwenImageSlicesTest, EqualImagesUsePackedOffsets) {
  const std::array<int64_t, 6> grids{1, 2, 2, 1, 2, 2};
  const auto slices = PlanQwenImageSlices(grids, 8, 2);

  ASSERT_EQ(slices.size(), 2u);
  EXPECT_EQ(slices[0].patch_offset, 0);
  EXPECT_EQ(slices[1].patch_offset, 4);
}

TEST(PlanQwenImageSlicesTest, RejectsTemporalMultiplierAndMismatchedPatchCounts) {
  const std::array<int64_t, 3> temporal_grid{1, 2, 2};
  EXPECT_THROW(PlanQwenImageSlices(temporal_grid, 8, 2), std::runtime_error);

  const std::array<int64_t, 3> reviewer_grid{1, 4, 4};
  EXPECT_THROW(PlanQwenImageSlices(reviewer_grid, 4, 2), std::runtime_error);
  EXPECT_NO_THROW(PlanQwenImageSlices(reviewer_grid, 16, 2));
}

TEST(PlanQwenImageSlicesTest, RejectsInvalidGridDimensionsAndTripletCount) {
  const std::array<int64_t, 3> non_divisible_grid{1, 3, 4};
  const std::array<int64_t, 3> zero_grid{1, 0, 4};
  const std::array<int64_t, 3> negative_grid{1, 4, -2};
  const std::array<int64_t, 2> incomplete_grid{1, 4};

  EXPECT_THROW(PlanQwenImageSlices(non_divisible_grid, 12, 2), std::runtime_error);
  EXPECT_THROW(PlanQwenImageSlices(zero_grid, 0, 2), std::runtime_error);
  EXPECT_THROW(PlanQwenImageSlices(negative_grid, 0, 2), std::runtime_error);
  EXPECT_THROW(PlanQwenImageSlices(incomplete_grid, 2, 2), std::runtime_error);
}

TEST(OpenVINOVisionStageSessionOptionsTest, PrefersStageThenVisionThenDecoderOptions) {
  Config config;
  OverlayConfig(config, R"({"model":{
    "decoder":{"session_options":{"log_id":"decoder"}},
    "vision":{"pipeline":[{
      "patch_embed":{"filename":"patch_embed.onnx","session_options":{"log_id":"stage"}},
      "merger":{"filename":"merger.onnx"}
    }]}
  }})");
  ASSERT_EQ(config.model.vision.pipeline.size(), 2u);
  EXPECT_EQ(GetOpenVINOVisionStageSessionOptions(config, config.model.vision.pipeline[0]).log_id, "stage");
  EXPECT_EQ(GetOpenVINOVisionStageSessionOptions(config, config.model.vision.pipeline[1]).log_id, "decoder");

  // OverlayConfig swaps in a parsed copy, so index the pipeline again afterwards.
  OverlayConfig(config, R"({"model":{"vision":{"session_options":{"log_id":"vision"}}}})");
  EXPECT_EQ(GetOpenVINOVisionStageSessionOptions(config, config.model.vision.pipeline[0]).log_id, "stage");
  EXPECT_EQ(GetOpenVINOVisionStageSessionOptions(config, config.model.vision.pipeline[1]).log_id, "vision");
}

TEST(OpenVINOVlmInputsDeviceTest, RequiresCpuMemoryInputs) {
  // OpenVINO (any device_type) and the CPU EP both keep GenAI inputs in CPU memory.
  EXPECT_NO_THROW(ValidateOpenVINOVlmInputsDevice(DeviceType::CPU, DeviceType::OpenVINO));
  EXPECT_NO_THROW(ValidateOpenVINOVlmInputsDevice(DeviceType::CPU, DeviceType::CPU));

  for (auto device : {DeviceType::CUDA, DeviceType::DML, DeviceType::NvTensorRtRtx, DeviceType::RyzenAI,
                      DeviceType::WEBGPU, DeviceType::AMDGPU}) {
    EXPECT_THROW(ValidateOpenVINOVlmInputsDevice(device, device), std::runtime_error);
  }
}

TEST(OpenVINOVlmBeamSearchTest, RejectsMultipleBeams) {
  EXPECT_THROW(ValidateOpenVINOVlmBeamSearch(2), std::runtime_error);
  EXPECT_NO_THROW(ValidateOpenVINOVlmBeamSearch(1));
  EXPECT_NO_THROW(ValidateOpenVINOVlmBeamSearch(0));
  EXPECT_NO_THROW(ValidateOpenVINOVlmBeamSearch(-1));
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
