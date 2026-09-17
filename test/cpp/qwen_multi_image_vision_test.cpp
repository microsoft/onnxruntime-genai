// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "models/qwen_vl_state.h"

#define OGA_USE_SPAN 1
#include "ort_genai.h"

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

struct QwenVisionHarness {
  std::unique_ptr<OgaModel> model;
  std::unique_ptr<OgaMultiModalProcessor> processor;
  std::unique_ptr<OgaGeneratorParams> params;
  std::unique_ptr<OgaGenerator> generator;
  std::unique_ptr<OgaNamedTensors> inputs;

  static QwenVisionHarness Create(const std::string& model_path = std::string(MODEL_PATH) + "qwen3-vl",
                                  int64_t batch_size = 1) {
    const std::array<std::string, 2> image_path_storage{
        std::string(MODEL_PATH) + "../images/australia.jpg",
        std::string(MODEL_PATH) + "../images/landscape.jpg",
    };
    const std::array<const char*, 2> image_paths{
        image_path_storage[0].c_str(),
        image_path_storage[1].c_str(),
    };

    QwenVisionHarness harness;
    harness.model = OgaModel::Create(model_path.c_str());
    harness.processor = OgaMultiModalProcessor::Create(*harness.model);
    auto images = OgaImages::Load(image_paths);
    harness.inputs = harness.processor->ProcessImages(
        "<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Describe these images",
        images.get());
    harness.params = OgaGeneratorParams::Create(*harness.model);
    harness.params->SetSearchOption("batch_size", static_cast<double>(batch_size));
    if (batch_size > 1) {
      harness.params->SetSearchOption("max_length", 2.0);
    }
    harness.generator = OgaGenerator::Create(*harness.model, *harness.params);
    return harness;
  }
};

std::filesystem::path CreateQwenStrideTestModel() {
  static constexpr std::array<uint8_t, 410> model_data{
      0x08, 0x07, 0x3a, 0x8f, 0x03, 0x0a, 0x39, 0x0a, 0x0c, 0x70, 0x69, 0x78, 0x65, 0x6c, 0x5f,
      0x76, 0x61, 0x6c, 0x75, 0x65, 0x73, 0x0a, 0x06, 0x73, 0x74, 0x61, 0x72, 0x74, 0x73, 0x0a,
      0x04, 0x65, 0x6e, 0x64, 0x73, 0x0a, 0x04, 0x61, 0x78, 0x65, 0x73, 0x0a, 0x05, 0x73, 0x74,
      0x65, 0x70, 0x73, 0x12, 0x07, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x64, 0x22, 0x05, 0x53,
      0x6c, 0x69, 0x63, 0x65, 0x0a, 0x2a, 0x0a, 0x07, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x64,
      0x0a, 0x04, 0x70, 0x61, 0x64, 0x73, 0x0a, 0x04, 0x7a, 0x65, 0x72, 0x6f, 0x12, 0x0e, 0x69,
      0x6d, 0x61, 0x67, 0x65, 0x5f, 0x66, 0x65, 0x61, 0x74, 0x75, 0x72, 0x65, 0x73, 0x22, 0x03,
      0x50, 0x61, 0x64, 0x12, 0x10, 0x71, 0x77, 0x65, 0x6e, 0x5f, 0x73, 0x74, 0x72, 0x69, 0x64,
      0x65, 0x5f, 0x74, 0x65, 0x73, 0x74, 0x2a, 0x16, 0x08, 0x01, 0x10, 0x07, 0x42, 0x06, 0x73,
      0x74, 0x61, 0x72, 0x74, 0x73, 0x4a, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x2a, 0x14, 0x08, 0x01, 0x10, 0x07, 0x42, 0x04, 0x65, 0x6e, 0x64, 0x73, 0x4a, 0x08, 0xff,
      0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0x2a, 0x14, 0x08, 0x01, 0x10, 0x07, 0x42, 0x04,
      0x61, 0x78, 0x65, 0x73, 0x4a, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2a,
      0x15, 0x08, 0x01, 0x10, 0x07, 0x42, 0x05, 0x73, 0x74, 0x65, 0x70, 0x73, 0x4a, 0x08, 0x04,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2a, 0x2c, 0x08, 0x04, 0x10, 0x07, 0x42, 0x04,
      0x70, 0x61, 0x64, 0x73, 0x4a, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2a, 0x0e, 0x10, 0x01, 0x42, 0x04, 0x7a,
      0x65, 0x72, 0x6f, 0x4a, 0x04, 0x00, 0x00, 0x00, 0x00, 0x5a, 0x2a, 0x0a, 0x0c, 0x70, 0x69,
      0x78, 0x65, 0x6c, 0x5f, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x73, 0x12, 0x1a, 0x0a, 0x18, 0x08,
      0x01, 0x12, 0x14, 0x0a, 0x0d, 0x12, 0x0b, 0x6e, 0x75, 0x6d, 0x5f, 0x70, 0x61, 0x74, 0x63,
      0x68, 0x65, 0x73, 0x0a, 0x03, 0x08, 0x80, 0x0c, 0x5a, 0x20, 0x0a, 0x0e, 0x69, 0x6d, 0x61,
      0x67, 0x65, 0x5f, 0x67, 0x72, 0x69, 0x64, 0x5f, 0x74, 0x68, 0x77, 0x12, 0x0e, 0x0a, 0x0c,
      0x08, 0x07, 0x12, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x03, 0x62, 0x2d, 0x0a,
      0x0e, 0x69, 0x6d, 0x61, 0x67, 0x65, 0x5f, 0x66, 0x65, 0x61, 0x74, 0x75, 0x72, 0x65, 0x73,
      0x12, 0x1b, 0x0a, 0x19, 0x08, 0x01, 0x12, 0x15, 0x0a, 0x0e, 0x12, 0x0c, 0x6e, 0x75, 0x6d,
      0x5f, 0x66, 0x65, 0x61, 0x74, 0x75, 0x72, 0x65, 0x73, 0x0a, 0x03, 0x08, 0x80, 0x10, 0x42,
      0x04, 0x0a, 0x00, 0x10, 0x0e};

  const auto model_path = std::filesystem::path(::testing::TempDir()) / "qwen_stride_test";
  std::filesystem::remove_all(model_path);
  std::filesystem::copy(std::filesystem::path(MODEL_PATH) / "qwen3-vl", model_path,
                        std::filesystem::copy_options::recursive);
  std::ofstream vision_model(model_path / "dummy_vision.onnx", std::ios::binary | std::ios::trunc);
  vision_model.write(reinterpret_cast<const char*>(model_data.data()), model_data.size());
  return model_path;
}

}  // namespace

TEST(QwenVisionMultiImageTest, RejectsMalformedGridInBatchSizeDetection) {
  auto harness = QwenVisionHarness::Create();
  std::array<int64_t, 4> malformed_grid{1, 32, 32, 1};
  auto malformed_tensor = OgaTensor::Create(
      malformed_grid.data(), std::array<int64_t, 2>{2, 2});
  harness.inputs->Set("image_grid_thw", *malformed_tensor);

  const std::string message = CaptureRuntimeErrorMessage([&] {
    harness.generator->SetInputs(*harness.inputs);
  });
  EXPECT_NE(message.find("image_grid_thw second dimension must be 3"), std::string::npos) << message;
}

TEST(QwenVisionMultiImageTest, RejectsMalformedGridInMultiImageVisionRun) {
  auto harness = QwenVisionHarness::Create();

  std::array<float, 2> pixel_values{};
  auto rank_three_pixel_values = OgaTensor::Create(
      pixel_values.data(), std::array<int64_t, 3>{2, 1, 1});
  harness.inputs->Set("pixel_values", *rank_three_pixel_values);

  // The values form two valid flat triplets for position-id processing,
  // while the [3, 2] layout is rejected specifically by QwenVisionState::Run.
  std::array<int64_t, 6> malformed_grid{1, 1, 1, 1, 1, 1};
  auto malformed_tensor = OgaTensor::Create(
      malformed_grid.data(), std::array<int64_t, 2>{3, 2});
  harness.inputs->Set("image_grid_thw", *malformed_tensor);

  const std::string message = CaptureRuntimeErrorMessage([&] {
    harness.generator->SetInputs(*harness.inputs);
  });
  EXPECT_NE(message.find("image_grid_thw second dimension must be 3"), std::string::npos) << message;
}

TEST(QwenVisionMultiImageTest, AcceptsValidGridThroughPublicInputValidation) {
  auto harness = QwenVisionHarness::Create();
  harness.inputs->Delete("input_ids");
  EXPECT_NO_THROW(harness.generator->SetInputs(*harness.inputs));
}

TEST(QwenVisionMultiImageTest, UsesEachImagesPaddedInputSegment) {
  auto harness = QwenVisionHarness::Create(CreateQwenStrideTestModel().string(), 2);

  std::array<int32_t, 4> input_ids{1, 2, 3, 4};
  auto input_ids_tensor = OgaTensor::Create(input_ids.data(), std::array<int64_t, 2>{2, 2});
  harness.inputs->Delete("input_ids");
  harness.inputs->Set("input_ids", *input_ids_tensor);

  std::vector<float> pixel_values(16 * 1536, -1.0f);
  std::fill_n(pixel_values.begin(), 1536, 11.0f);
  std::fill_n(pixel_values.begin() + 8 * 1536, 1536, 22.0f);
  std::fill_n(pixel_values.begin() + 12 * 1536, 1536, 23.0f);
  auto pixel_tensor = OgaTensor::Create(pixel_values.data(), std::array<int64_t, 2>{16, 1536});
  harness.inputs->Set("pixel_values", *pixel_tensor);

  std::array<int64_t, 6> image_grid_thw{1, 1, 4, 1, 2, 4};
  auto grid_tensor = OgaTensor::Create(image_grid_thw.data(), std::array<int64_t, 2>{2, 3});
  harness.inputs->Set("image_grid_thw", *grid_tensor);

  std::array<int64_t, 2> image_token_counts{1, 2};
  auto token_count_tensor = OgaTensor::Create(image_token_counts.data(), std::array<int64_t, 1>{2});
  harness.inputs->Set("num_image_tokens", *token_count_tensor);

  ASSERT_NO_THROW(harness.generator->SetInputs(*harness.inputs));
  auto image_features = harness.generator->GetInput("image_features");
  ASSERT_EQ(image_features->Shape(), (std::vector<int64_t>{3, 2048}));
  const auto* feature_data = static_cast<const float*>(image_features->Data());
  EXPECT_FLOAT_EQ(feature_data[0], 11.0f);
  EXPECT_FLOAT_EQ(feature_data[2048], 22.0f);
  EXPECT_FLOAT_EQ(feature_data[2 * 2048], 23.0f);
}

TEST(QwenVisionMultiImageTest, UsesPaddedStrideForDifferentImageGridSizes) {
  constexpr int64_t total_patches = 27520;
  constexpr int64_t total_grid_tokens = 20004;
  constexpr int64_t total_hw = 20004;
  constexpr int64_t max_grid_tokens = 6880;
  constexpr int64_t num_images = 4;

  const auto layout = Generators::ResolveQwenPatchLayout(
      total_patches, total_grid_tokens, total_hw, max_grid_tokens, num_images);

  EXPECT_EQ(layout.padded_image_stride, 6880);
  EXPECT_EQ(layout.temporal_multiplier, 0);
  EXPECT_EQ(layout.ImagePatchOffset(2, 5624 + 6880), 13760);
  EXPECT_EQ(layout.ImagePatchCount(1900, 38, 50), 1900);
}

TEST(QwenVisionMultiImageTest, KeepsPackedLayoutWhenPatchCountsMatchGrid) {
  const auto layout = Generators::ResolveQwenPatchLayout(20004, 20004, 20004, 6880, 4);
  EXPECT_EQ(layout.padded_image_stride, 0);
  EXPECT_EQ(layout.temporal_multiplier, 0);
  EXPECT_EQ(layout.ImagePatchOffset(2, 12504), 12504);
}

TEST(QwenVisionMultiImageTest, RejectsUnsupportedPatchLayout) {
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

TEST(QwenVisionMultiImageTest, PreservesUnambiguousTemporalPadding) {
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

TEST(QwenVisionMultiImageTest, UsesPaddedStrideWhenTotalIsAlsoDivisibleByTotalHw) {
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

TEST(QwenVisionMultiImageTest, PreservesTemporalPaddingWhenStrideDoesNotMatchMaximumGrid) {
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

TEST(QwenVisionMultiImageTest, PreservesTemporalPaddingWhenStrideMatchesMaximumGrid) {
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
