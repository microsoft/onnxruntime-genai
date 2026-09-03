// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <memory>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "models/multi_modal.h"

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

  static QwenVisionHarness Create() {
    const std::string model_path = std::string(MODEL_PATH) + "qwen3-vl";
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
    harness.generator = OgaGenerator::Create(*harness.model, *harness.params);
    return harness;
  }
};

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

TEST(QwenVisionMultiImageTest, RejectsAmbiguousStrideAndTemporalPadding) {
  const std::string message = CaptureRuntimeErrorMessage([] {
    Generators::ResolveQwenPatchLayout(
        /*total_patches=*/24,
        /*total_grid_tokens=*/12,
        /*total_hw=*/12,
        /*max_grid_tokens=*/8,
        /*num_images=*/2);
  });

  EXPECT_EQ(message, "pixel_values patch layout is ambiguous between per-image stride padding and temporal padding");
}
