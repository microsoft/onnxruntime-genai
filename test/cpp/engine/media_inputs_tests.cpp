// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <tuple>

#include <gtest/gtest.h>

#include "models/preprocessing/processor.h"
#include "engine/media.h"
#include "engine_test_helpers.h"
#include "engine_test_doubles.h"

namespace Generators::test {

TEST(MediaInputsTest, RetainsPngAndJpegDimensionsWithoutPreprocessing) {
  for (const auto& [filename, width, height] :
       {std::tuple{"sheet.png", 487u, 206u}, std::tuple{"cars.jpg", 467u, 311u}}) {
    const std::string path = std::string{MODEL_PATH} + "../images/" + filename;
    const std::array paths{path.c_str()};
    auto file_images = LoadImages(paths);
    ASSERT_EQ(file_images->info_.size(), 1u);
    EXPECT_EQ(file_images->info_[0].width, width);
    EXPECT_EQ(file_images->info_[0].height, height);

    std::ifstream input(path, std::ios::binary);
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    std::array<const void*, 1> buffers{bytes.data()};
    const std::array sizes{bytes.size()};
    auto buffer_images = LoadImagesFromBuffers(buffers, sizes);
    ASSERT_EQ(buffer_images->info_.size(), 1u);
    EXPECT_EQ(buffer_images->info_[0].width, width);
    EXPECT_EQ(buffer_images->info_[0].height, height);
    EXPECT_EQ(buffer_images->info_[0].encoded_bytes, bytes.size());
    EXPECT_EQ(file_images->info_[0].encoded_bytes, bytes.size());
  }
}

TEST(MediaInputsTest, RejectsInvalidImageBufferBeforeCopying) {
  std::array<const void*, 1> buffers{nullptr};
  const std::array<size_t, 1> sizes{1};
  EXPECT_THROW(LoadImagesFromBuffers(buffers, sizes), std::runtime_error);
}

TEST(MediaInputsTest, RejectsOversizedImageCollectionsBeforeReadingPayloads) {
  const auto path = std::filesystem::temp_directory_path() /
                    ("ortgenai_image_limit_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  {
    std::ofstream output(path, std::ios::binary);
    output.seekp(300 * 1024 * 1024 - 1);
    output.put('\0');
    ASSERT_TRUE(output);
  }
  const std::string filename = path.string();
  const std::array paths{filename.c_str(), filename.c_str()};
  try {
    LoadImages(paths);
    ADD_FAILURE() << "Accepted more than 512 MiB of encoded images";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("512 MiB loading limit"), std::string::npos);
  }
  EXPECT_TRUE(std::filesystem::remove(path));
}

TEST(MediaInputsTest, RejectsCombinedBufferBytesBeforeInspectingEitherBuffer) {
  const uint8_t unused = 0;
  std::array<const void*, 2> buffers{&unused, &unused};
  const std::array<size_t, 2> sizes{256 * 1024 * 1024, 256 * 1024 * 1024 + 1};
  EXPECT_THROW(LoadImagesFromBuffers(buffers, sizes), std::runtime_error);
}

TEST(MediaInputsTest, TextOnlyModelTypesRejectVisionComponents) {
  for (const char* type : {"qwen3_5_text", "qwen3_5_moe_text", "gpt2"}) {
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
    config->model.type = type;
    config->model.vision.filename = "vision.onnx";
    try {
      CreateModel(GetOrtEnv(), std::move(config));
      FAIL() << type << " accepted a vision encoder";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string{error.what()}.find("Text-only model type"), std::string::npos);
    }
  }
}

TEST(MediaInputsTest, VisionModelTypeWithoutEncoderDoesNotEnableVision) {
  for (const char* type : {"qwen3_5", "qwen3_5_moe"}) {
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
    config->model.type = type;
    config->engine.dynamic_batching.emplace();
    try {
      CreateModel(GetOrtEnv(), std::move(config));
      FAIL() << type << " accepted an incomplete vision package";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string{error.what()}.find("exported vision encoder"), std::string::npos);
    }
  }
}

TEST(MediaInputsTest, TextOnlyImageRejectionPreservesTheNextTurn) {
  auto model = LoadDummyDecoderModel();
  auto engine = MakeDoublesEngine(model, 1024, EosToken(*model)).engine;
  auto request = engine->CreateRequest();
  const std::string path = std::string{MODEL_PATH} + "../images/cars.jpg";
  const std::array paths{path.c_str()};
  auto images = LoadImages(paths);
  const TurnOptions options;
  EXPECT_THROW(request->BeginTurn("Describe the image.", options, images.get()), std::invalid_argument);
  EXPECT_TRUE(request->IsAwaitingFirstTurn());
  const std::array<int32_t, 3> tokens{2, 3, 4};
  EXPECT_EQ(request->BeginTurn(tokens), 1u);
}

TEST(MediaInputsTest, PackedVisionRequiresTheCpuEmbeddingComponent) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  config->model.type = "qwen3_5";
  config->model.vision.filename = "vision.onnx";
  config->engine.dynamic_batching.emplace();
  try {
    CreateModel(GetOrtEnv(), std::move(config));
    FAIL() << "Packed vision accepted a package without a CPU embedding component";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("pure CPU embeddings"), std::string::npos);
  }
}

TEST(MediaInputsTest, LeavesUnknownHeaderDimensionsUnavailable) {
  const std::array<uint8_t, 24> invalid{};
  std::array<const void*, 1> buffers{invalid.data()};
  const std::array sizes{invalid.size()};
  auto images = LoadImagesFromBuffers(buffers, sizes);
  ASSERT_EQ(images->info_.size(), 1u);
  EXPECT_EQ(images->info_[0].width, 0u);
  EXPECT_EQ(images->info_[0].height, 0u);
}

TEST(MediaInputsTest, PositionsUseGeometryRatherThanExpandedRowCount) {
  auto first = std::make_shared<MediaItem>();
  first->grid = {1, 4, 6};
  first->rows = 6;
  auto second = std::make_shared<MediaItem>();
  second->grid = {1, 4, 4};
  second->rows = 4;
  const std::array<MediaSpan, 2> media{{{3, first}, {12, second}}};
  EXPECT_EQ(MediaPosition(media, 2, 2), (std::array<int64_t, 3>{2, 2, 2}));
  EXPECT_EQ(MediaPosition(media, 3, 2), (std::array<int64_t, 3>{3, 3, 3}));
  EXPECT_EQ(MediaPosition(media, 7, 2), (std::array<int64_t, 3>{3, 4, 4}));
  EXPECT_EQ(MediaPosition(media, 9, 2), (std::array<int64_t, 3>{6, 6, 6}));
  EXPECT_EQ(MediaPosition(media, 13, 2), (std::array<int64_t, 3>{9, 9, 10}));
  EXPECT_EQ(MediaPosition(media, 16, 2), (std::array<int64_t, 3>{11, 11, 11}));
  EXPECT_EQ(MediaPosition(media, 7, 2), (std::array<int64_t, 3>{3, 4, 4}));
}

TEST(MediaInputsTest, ResizePreflightMatchesNativeTieAndMinimumPixelRules) {
  EXPECT_EQ(MediaImageGrid(500, 400, 16, 2, 65536, 262144), (std::array<int64_t, 3>{1, 24, 32}));
  EXPECT_EQ(MediaImageGrid(30, 17, 16, 2, 65536, 262144), (std::array<int64_t, 3>{1, 14, 22}));
  EXPECT_EQ(MediaImageGrid(381, 137, 16, 2, 65536, 16777216), (std::array<int64_t, 3>{1, 10, 28}));
  EXPECT_THROW(MediaImageGrid(0, 100, 16, 2, 65536, 16777216), std::invalid_argument);
  EXPECT_THROW(MediaImageGrid(1, 201, 16, 2, 65536, 16777216), std::invalid_argument);
  EXPECT_THROW(MediaImageGrid(UINT32_MAX, UINT32_MAX, 16, 2, 65536, 16777216), std::invalid_argument);
}

TEST(MediaInputsTest, PositionRangesPreserveImageBoundariesAndLaterTextOffsets) {
  auto first = std::make_shared<MediaItem>();
  first->grid = {1, 4, 6};
  first->rows = 6;
  auto second = std::make_shared<MediaItem>();
  second->grid = {1, 4, 4};
  second->rows = 4;
  const std::array<MediaSpan, 2> media{{{3, first}, {12, second}}};
  const std::array<std::array<int64_t, 20>, 3> expected{{
      {0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7, 8, 9, 9, 9, 9, 11, 12, 13, 14},
      {0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7, 8, 9, 9, 10, 10, 11, 12, 13, 14},
      {0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 10, 9, 10, 11, 12, 13, 14},
  }};
  for (const auto& [start, count] : {std::pair{0u, 20u}, std::pair{4u, 10u},
                                     std::pair{14u, 4u}, std::pair{20u, 0u}}) {
    std::array<std::vector<int64_t>, 3> actual;
    for (auto& plane : actual) plane.resize(count);
    MediaPositions(media, start, 2, {actual[0], actual[1], actual[2]});
    for (size_t plane = 0; plane < 3; ++plane)
      EXPECT_TRUE(std::equal(actual[plane].begin(), actual[plane].end(), expected[plane].begin() + start));
  }
}

TEST(MediaInputsTest, PrefixIdentityDoesNotReplaceVocabularyInputs) {
  auto model = LoadDummyDecoderModel();
  auto engine = MakeDoublesEngine(model, 1024, EosToken(*model)).engine;
  const std::array<int32_t, 5> tokens{2, 3, 4, 4, 5};
  auto first = CreateRequestWithPrompt(engine, tokens);
  auto second = CreateRequestWithPrompt(engine, tokens);
  auto image_a = std::make_shared<MediaItem>();
  image_a->rows = 2;
  image_a->prefix_base = 1;
  auto image_b = std::make_shared<MediaItem>();
  image_b->rows = 2;
  image_b->prefix_base = 3;
  first->ReserveMedia(1);
  second->ReserveMedia(1);
  const std::array<MediaSpan, 1> first_media{{{2, image_a}}}, second_media{{{2, image_b}}};
  first->AppendMedia(first_media);
  second->AppendMedia(second_media);
  EXPECT_TRUE(std::equal(first->TokensCpu().begin(), first->TokensCpu().end(), tokens.begin()));
  EXPECT_TRUE(std::equal(second->TokensCpu().begin(), second->TokensCpu().end(), tokens.begin()));
  EXPECT_EQ(first->PrefixTokensCpu()[1], second->PrefixTokensCpu()[1]);
  EXPECT_NE(first->PrefixTokensCpu()[2], second->PrefixTokensCpu()[2]);
  EXPECT_NE(first->PrefixTokensCpu()[3], second->PrefixTokensCpu()[3]);
  auto invalid = engine->CreateRequest();
  const std::array<int32_t, 1> private_id{-1};
  EXPECT_THROW(invalid->BeginTurn(private_id), std::invalid_argument);
  EXPECT_TRUE(invalid->IsAwaitingFirstTurn());
}

TEST(MediaInputsTest, EncoderFailureBudgetIsRequestLocalAndClearedAtTurnBoundaries) {
  auto model = LoadDummyDecoderModel();
  auto engine = MakeDoublesEngine(model, 1024, EosToken(*model)).engine;
  const std::array<int32_t, 3> tokens{2, 3, 4};
  auto first = CreateRequestWithPrompt(engine, tokens);
  auto second = CreateRequestWithPrompt(engine, tokens);
  auto shared_image = std::make_shared<MediaItem>();
  EXPECT_TRUE(first->RetryMediaPreparation(shared_image));
  EXPECT_FALSE(first->RetryMediaPreparation(shared_image));
  EXPECT_TRUE(second->RetryMediaPreparation(shared_image));
  EXPECT_FALSE(second->RetryMediaPreparation(shared_image));

  first->ClearMediaPreparationFailure();
  EXPECT_TRUE(first->RetryMediaPreparation(shared_image));
  EXPECT_FALSE(first->RetryMediaPreparation(shared_image));
  auto other_image = std::make_shared<MediaItem>();
  EXPECT_TRUE(first->RetryMediaPreparation(other_image));
  EXPECT_FALSE(first->RetryMediaPreparation(other_image));

  first->CompleteCancelFromEngine(*engine, first->CurrentTurnId());
  EXPECT_TRUE(first->RetryMediaPreparation(other_image));
}

}  // namespace Generators::test
