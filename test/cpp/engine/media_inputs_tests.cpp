// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <chrono>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <tuple>

#include <gtest/gtest.h>

#include "models/preprocessing/processor.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "models/preprocessing/qwen2_5_vl_image_processor.h"
#include "models/preprocessing/multi_modal_processor.h"
#include "models/decoder_only.h"
#include "engine/media.h"
#include "engine_test_helpers.h"
#include "engine_test_doubles.h"

namespace Generators::test {

TEST(BFloat16ConversionTest, NearestEvenPreservesSpecialValuesAndRoundsTies) {
  for (const auto& [bits, expected] : std::array<std::pair<uint32_t, uint16_t>, 19>{
           {{0x00000000, 0x0000}, {0x80000000, 0x8000}, {0x3f807fff, 0x3f80}, {0x3f808000, 0x3f80}, {0x3f808001, 0x3f81}, {0x3f818000, 0x3f82}, {0xbf808000, 0xbf80}, {0xbf818000, 0xbf82}, {0x00008000, 0x0000}, {0x00008001, 0x0001}, {0x00018000, 0x0002}, {0x007fffff, 0x0080}, {0x7f7fffff, 0x7f80}, {0xff7fffff, 0xff80}, {0x7f800000, 0x7f80}, {0xff800000, 0xff80}, {0x7f800001, 0x7fc0}, {0x7fc12345, 0x7fc1}, {0xff800001, 0xffc0}}}) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    EXPECT_EQ(Float32ToBFloat16RoundNearestEven(value), expected) << bits;
  }
}

TEST(MediaInputsTest, PromptPlanningPreservesProcessorSpecificWhitespace) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "qwen3-5");
  const Tokenizer tokenizer{*config};
  const std::string marker = "<|vision_start|><|image_pad|><|vision_end|>";
  const std::string expanded = "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>";
  const std::array<int64_t, 6> grids{1, 2, 4, 1, 2, 4};
  for (const auto& [type, separator] : {std::pair{"qwen3_5", ""}, {"qwen4_exp", ""}, {"bailing_moe_v3_vl", "\n"}}) {
    config->model.type = type;
    const MultiModalProcessor processor{*config, {}, nullptr, nullptr};
    for (const std::string suffix : {"", " ", "\n\n  code", "\tcode"}) {
      const auto prompt = "before " + marker + marker + suffix;
      EXPECT_EQ(processor.processor_->ExpandImagePrompt(tokenizer, prompt, grids),
                tokenizer.Encode(("before " + expanded + separator + expanded + separator + suffix).c_str()));
    }
  }
}

TEST(MediaInputsTest, QwenImagePromptPreservesLegacyWrapperFormatting) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "qwen3-5");
  const QwenImageProcessor processor{*config, {}};
  const Tokenizer tokenizer{*config};
  const std::string expanded = "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>";
  const std::array<int64_t, 6> grids{1, 2, 4, 1, 2, 4};
  EXPECT_EQ(processor.ExpandImagePrompt(tokenizer, "", grids),
            tokenizer.Encode((expanded + " " + expanded).c_str()));
  EXPECT_EQ(processor.ExpandImagePrompt(tokenizer, "before <|vision_start|> <|vision_end|> after",
                                        std::span<const int64_t>{grids}.first(3)),
            tokenizer.Encode(("before " + expanded + " after").c_str()));
}

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

TEST(MediaInputsTest, LingImageExpansionRetokenizesOrderedImagesAndWhitespace) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "qwen3-5");
  const LingImageProcessor processor{*config, {}};
  const Tokenizer tokenizer{*config};
  const std::string marker = "<|vision_start|><|image_pad|><|vision_end|>";
  const std::string expanded = "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>\n";
  const std::array<int64_t, 3> grid{1, 2, 4};
  const std::array<int64_t, 6> two_grids{1, 2, 4, 1, 2, 4};
  for (const std::string suffix : {"", " ", "\n", "\n\n  code", "\tcode"}) {
    EXPECT_EQ(processor.ExpandImagePrompt(tokenizer, marker + suffix, grid),
              tokenizer.Encode((expanded + suffix).c_str()));
    EXPECT_EQ(processor.ExpandImagePrompt(tokenizer, "code " + marker + marker + suffix, two_grids),
              tokenizer.Encode(("code " + expanded + expanded + suffix).c_str()));
  }
  EXPECT_EQ(processor.ExpandImagePrompt(tokenizer, "text only\n", {}), tokenizer.Encode("text only\n"));
  const auto processed = processor.Process(tokenizer, Payload{"text only\n", {}, nullptr, nullptr});
  const auto ids = processed->at(std::string{Config::Defaults::InputIdsName});
  EXPECT_EQ((std::vector<int32_t>{ids->GetData<int32_t>(), ids->GetData<int32_t>() + ids->GetElementCount()}),
            tokenizer.Encode("text only\n"));
}

TEST(MediaInputsTest, LingImageExpansionRejectsInvalidMarkersAndGrids) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "qwen3-5");
  const LingImageProcessor processor{*config, {}};
  const Tokenizer tokenizer{*config};
  const std::string marker = "<|vision_start|><|image_pad|><|vision_end|>";
  const std::array<int64_t, 3> grid{1, 2, 4};
  EXPECT_THROW(processor.ExpandImagePrompt(tokenizer, "", grid), std::invalid_argument);
  EXPECT_THROW(processor.ExpandImagePrompt(tokenizer, marker + marker, grid), std::invalid_argument);
  config->model.vision.spatial_merge_size = 0;
  const LingImageProcessor invalid_processor{*config, {}};
  EXPECT_THROW(invalid_processor.ExpandImagePrompt(tokenizer, marker, grid), std::invalid_argument);
  for (const std::array<int64_t, 3> invalid : {std::array<int64_t, 3>{2, 2, 4}, {1, 3, 4}, {1, 0, 4}})
    EXPECT_THROW(processor.ExpandImagePrompt(tokenizer, marker, invalid), std::invalid_argument);
  const std::array<int64_t, 2> incomplete{1, 2};
  EXPECT_THROW(processor.ExpandImagePrompt(tokenizer, marker, incomplete), std::invalid_argument);
  const std::array<int64_t, 3> overflowing{1, INT64_MAX - 1, INT64_MAX - 1};
  EXPECT_THROW(processor.ExpandImagePrompt(tokenizer, marker, overflowing), std::overflow_error);
}

TEST(MediaInputsTest, LingAdmissionRejectsExactCapacityBeforePixelDecoding) {
  const char* model_path = std::getenv("GENAI_NATIVE_CAPTURE_MODEL");
  if (!model_path || !*model_path)
    GTEST_SKIP() << "Set GENAI_NATIVE_CAPTURE_MODEL to a real Ling vision package.";
  auto config = CreateConfig(GetOrtEnv(), model_path);
  if (config->model.type != "bailing_moe_v3_vl")
    GTEST_SKIP() << "This admission regression requires Ling newline retokenization.";
  const char* overlay = std::getenv("GENAI_NATIVE_CAPTURE_CONFIG_OVERLAY");
  if (overlay && *overlay)
    OverlayConfig(*config, overlay);
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  auto* decoder = dynamic_cast<DecoderOnly_Model*>(model.get());
  ASSERT_NE(decoder, nullptr);
  Tokenizer tokenizer{*model->config_};
  auto engine = std::make_shared<Engine>(model);
  const std::string png_path = std::string{MODEL_PATH} + "../images/sheet.png";
  const std::string jpeg_path = std::string{MODEL_PATH} + "../images/cars.jpg";
  const std::array paths{png_path.c_str(), jpeg_path.c_str()};
  auto images = LoadImages(paths);
  std::ifstream input(png_path, std::ios::binary);
  const std::vector<uint8_t> png{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  std::ifstream jpeg_input(jpeg_path, std::ios::binary);
  const std::vector<uint8_t> jpeg{std::istreambuf_iterator<char>{jpeg_input}, std::istreambuf_iterator<char>{}};
  ASSERT_GE(png.size(), 33u);
  // Keep the PNG header, but no pixel payload: reaching preprocessing must fail.
  std::array<const void*, 2> buffers{png.data(), jpeg.data()};
  const std::array<size_t, 2> sizes{33, jpeg.size()};
  auto header_only = LoadImagesFromBuffers(buffers, sizes);
  const auto prefix = tokenizer.Encode("retained context");
  const std::string marker = "<|vision_start|><|image_pad|><|vision_end|>";
  TurnOptions options;
  options.do_sample = false;
  options.min_generated_tokens = 1;
  options.max_generated_tokens = 1;
  for (const std::string suffix : {"", " ", "\n", "\n\n  code", "\tcode"}) {
    SCOPED_TRACE(suffix);
    const std::string prompt = marker + marker + suffix;
    const auto processed = decoder->vision_->processor->Process(prompt, images.get(), nullptr);
    const auto ids = processed->at(std::string{Config::Defaults::InputIdsName});
    const std::vector<int32_t> expected{ids->GetData<int32_t>(), ids->GetData<int32_t>() + ids->GetElementCount()};
    for (size_t spare : {0u, 1u}) {
      auto request = CreateEngineRequest(engine, prefix.size() + 1 + expected.size() + spare);
      ASSERT_EQ(request->BeginTurn(prefix, options), 1u);
      size_t runs = 0;
      while (engine->HasPendingRequests()) {
        ASSERT_LT(runs++, 2048u);
        const auto event = RunOne(*engine);
        ASSERT_FALSE(event.flags & EngineEventFlagFailed);
      }
      const auto before = request->TokensCpu();
      const std::vector<int32_t> retained{before.begin(), before.end()};
      ASSERT_EQ(retained.size(), prefix.size() + 1);
      try {
        request->BeginTurn(prompt, options, header_only.get());
        FAIL() << "Header-only image reached successful admission.";
      } catch (const std::exception& error) {
        const std::string message{error.what()};
        if (spare == 0)
          EXPECT_NE(message.find("Expanded image rows exceed"), std::string::npos) << message;
        else
          EXPECT_NE(message.find("[ImageDecoder]"), std::string::npos) << message;
      }
      EXPECT_EQ(request->CurrentTurnId(), 1u);
      EXPECT_EQ((std::vector<int32_t>{request->TokensCpu().begin(), request->TokensCpu().end()}), retained);
      if (spare) {
        EXPECT_EQ(request->BeginTurn(prompt, options, images.get()), 2u);
        const auto tokens = request->TokensCpu();
        ASSERT_EQ(tokens.size(), retained.size() + expected.size());
        EXPECT_TRUE(std::equal(expected.begin(), expected.end(), tokens.begin() + retained.size()));
      } else {
        EXPECT_EQ(request->BeginTurn("x", options, nullptr), 2u);
      }
      engine->CloseRequest(request);
    }
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
  for (const char* type : {"bailing_hybrid", "qwen3_5_text", "qwen3_5_moe_text", "qwen4_exp_text", "qwen38_flash_graph", "gpt2"}) {
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
  for (const char* type : {"bailing_moe_v3_vl", "qwen3_5", "qwen3_5_moe", "qwen4_exp", "phi3v", "gemma3"}) {
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
  first->positions = {{0, 0, 0}, 3, 3};
  first->rows = 6;
  auto second = std::make_shared<MediaItem>();
  second->grid = {1, 4, 4};
  second->positions = {{0, 0, 0}, 2, 2};
  second->rows = 4;
  const std::array<MediaSpan, 2> media{{{3, first}, {12, second}}};
  EXPECT_EQ(MediaPosition(media, 2), (std::array<int64_t, 3>{2, 2, 2}));
  EXPECT_EQ(MediaPosition(media, 3), (std::array<int64_t, 3>{3, 3, 3}));
  EXPECT_EQ(MediaPosition(media, 7), (std::array<int64_t, 3>{3, 4, 4}));
  EXPECT_EQ(MediaPosition(media, 9), (std::array<int64_t, 3>{6, 6, 6}));
  EXPECT_EQ(MediaPosition(media, 13), (std::array<int64_t, 3>{9, 9, 10}));
  EXPECT_EQ(MediaPosition(media, 16), (std::array<int64_t, 3>{11, 11, 11}));
  EXPECT_EQ(MediaPosition(media, 7), (std::array<int64_t, 3>{3, 4, 4}));
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
  first->positions = {{0, 0, 0}, 3, 3};
  first->rows = 6;
  auto second = std::make_shared<MediaItem>();
  second->grid = {1, 4, 4};
  second->positions = {{0, 0, 0}, 2, 2};
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
    MediaPositions(media, start, {actual[0], actual[1], actual[2]});
    for (size_t plane = 0; plane < 3; ++plane)
      EXPECT_TRUE(std::equal(actual[plane].begin(), actual[plane].end(), expected[plane].begin() + start));
  }
}

TEST(MediaInputsTest, LingPositionsCenterEvenAndOddGridsAndContinueFromTime) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "qwen3-5");
  const LingImageProcessor processor{*config, {}};
  auto first = std::make_shared<MediaItem>();
  first->grid = {1, 4, 4};
  first->rows = 4;
  first->positions = processor.GetImagePositionLayout(first->grid);
  auto second = std::make_shared<MediaItem>();
  second->grid = {1, 6, 10};
  second->rows = 15;
  second->positions = processor.GetImagePositionLayout(second->grid);
  const std::array<MediaSpan, 2> media{{{1, first}, {7, second}}};
  const std::array<std::array<int64_t, 24>, 3> expected{{
      {0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6},
      {0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6},
      {0, 1, 2, 1, 2, 2, 3, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 5, 6},
  }};
  for (size_t start = 0; start < expected[0].size(); ++start) {
    for (size_t count : {size_t{1}, std::min(size_t{7}, expected[0].size() - start)}) {
      std::array<std::vector<int64_t>, 3> actual;
      for (auto& plane : actual) plane.resize(count);
      MediaPositions(media, start, {actual[0], actual[1], actual[2]});
      for (size_t plane = 0; plane < 3; ++plane)
        EXPECT_TRUE(std::equal(actual[plane].begin(), actual[plane].end(), expected[plane].begin() + start));
    }
  }
  second->positions = processor.GetImagePositionLayout({1, 6, 10});
  const std::array<MediaSpan, 1> early{{{1, second}}};
  EXPECT_EQ(MediaPosition(early, 1), (std::array<int64_t, 3>{1, 0, -1}));
  EXPECT_EQ(MediaPosition(early, 16), (std::array<int64_t, 3>{2, 2, 2}));
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
