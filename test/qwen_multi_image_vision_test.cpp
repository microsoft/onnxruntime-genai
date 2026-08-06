// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#define OGA_USE_SPAN 1
#include "ort_genai.h"
#include "test_utils.h"

namespace {

std::string GetQwen3VlModelPath() {
  return std::string(MODEL_PATH) + "qwen3-vl";
}

std::vector<std::string> GetQwenVisionImagePaths() {
  return {
      std::string(MODEL_PATH) + "../images/australia.jpg",
      std::string(MODEL_PATH) + "../images/landscape.jpg",
  };
}

std::string BuildMultiImagePrompt(size_t image_count) {
  std::string prompt;
  for (size_t i = 0; i < image_count; ++i) {
    prompt += "<|vision_start|><|image_pad|><|vision_end|>";
  }
  prompt += "Describe these images";
  return prompt;
}

struct QwenVisionHarness {
  std::unique_ptr<OgaModel> model;
  std::unique_ptr<OgaMultiModalProcessor> processor;
  std::unique_ptr<OgaGeneratorParams> params;
  std::unique_ptr<OgaGenerator> generator;
  std::unique_ptr<OgaNamedTensors> inputs;

  static QwenVisionHarness Create() {
    const std::string model_path = GetQwen3VlModelPath();
    if (!std::filesystem::exists(std::filesystem::path(model_path) / "genai_config.json")) {
      throw std::runtime_error("qwen3-vl test model not found");
    }

    const auto image_paths_storage = GetQwenVisionImagePaths();
    for (const auto& image_path : image_paths_storage) {
      if (!std::filesystem::exists(image_path)) {
        throw std::runtime_error("Qwen vision test image not found: " + image_path);
      }
    }

    std::vector<const char*> image_paths;
    image_paths.reserve(image_paths_storage.size());
    for (const auto& image_path : image_paths_storage) {
      image_paths.push_back(image_path.c_str());
    }

    QwenVisionHarness harness;
    harness.model = OgaModel::Create(model_path.c_str());
    harness.processor = OgaMultiModalProcessor::Create(*harness.model);
    auto images = OgaImages::Load(image_paths);
    const std::string prompt = BuildMultiImagePrompt(image_paths.size());
    harness.inputs = harness.processor->ProcessImages(prompt.c_str(), images.get());
    harness.params = OgaGeneratorParams::Create(*harness.model);
    harness.generator = OgaGenerator::Create(*harness.model, *harness.params);
    return harness;
  }
};

void SkipIfModelUnavailable() {
  const std::string model_path = GetQwen3VlModelPath();
  if (!std::filesystem::exists(std::filesystem::path(model_path) / "genai_config.json")) {
    GTEST_SKIP() << "qwen3-vl test model not found at " << model_path;
  }

  for (const auto& image_path : GetQwenVisionImagePaths()) {
    if (!std::filesystem::exists(image_path)) {
      GTEST_SKIP() << "Qwen vision test image not found at " << image_path;
    }
  }
}

}  // namespace

TEST(QwenVisionMultiImageTest, RejectsInsufficientGridElementsViaGeneratorInputs) {
  SkipIfModelUnavailable();

  auto harness = QwenVisionHarness::Create();

  std::vector<int64_t> malformed_grid{1, 32, 32, 1};
  auto malformed_tensor = OgaTensor::Create(
      malformed_grid.data(),
      std::array<int64_t, 2>{2, 2},
      OgaElementType_int64);
  harness.inputs->Set("image_grid_thw", *malformed_tensor);

  EXPECT_THROW(harness.generator->SetInputs(*harness.inputs), std::runtime_error);
}

TEST(QwenVisionMultiImageTest, RejectsRank1GridTensorViaGeneratorInputs) {
  SkipIfModelUnavailable();

  auto harness = QwenVisionHarness::Create();

  std::vector<int64_t> malformed_grid{1, 32};
  auto malformed_tensor = OgaTensor::Create(
      malformed_grid.data(),
      std::array<int64_t, 1>{2},
      OgaElementType_int64);
  harness.inputs->Set("image_grid_thw", *malformed_tensor);

  EXPECT_THROW(harness.generator->SetInputs(*harness.inputs), std::runtime_error);
}

TEST(QwenVisionMultiImageTest, AcceptsValidGridShapeViaGeneratorInputs) {
  SkipIfModelUnavailable();

  auto harness = QwenVisionHarness::Create();

  EXPECT_NO_THROW(harness.generator->SetInputs(*harness.inputs));
}
