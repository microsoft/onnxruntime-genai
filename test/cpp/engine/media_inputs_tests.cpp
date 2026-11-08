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
#include "models/preprocessing/genai_tokenizer.h"
#include "models/decoder_only.h"
#include "models/env_utils.h"
#include "engine/media.h"
#include "engine_test_helpers.h"
#include "engine_test_doubles.h"

namespace Generators::test {

TEST(MediaInputsTest, RealQwenVideoCanonicalOrderAndEncoderInputs) {
  const auto path = GetEnv("ORTGENAI_ENGINE_VIDEO_MODEL");
  if (path.empty()) {
    bool required = false;
    GetEnv("ORTGENAI_REQUIRE_ENGINE_VIDEO", required);
    if (required) FAIL() << "ORTGENAI_ENGINE_VIDEO_MODEL must name the complete Qwen video package.";
    GTEST_SKIP() << "Set ORTGENAI_ENGINE_VIDEO_MODEL for actual-weight canonical video validation.";
  }
  auto config = CreateConfig(GetOrtEnv(), path.c_str());
  ASSERT_EQ(config->model.decoder.num_hidden_layers, 64);
  auto model = std::dynamic_pointer_cast<DecoderOnly_Model>(CreateModel(GetOrtEnv(), std::move(config)));
  ASSERT_NE(model, nullptr);
  MediaStore store{model, 512 * 1024 * 1024, 2 * 1024 * 1024};
  Tokenizer tokenizer{*model->config_};
  std::vector<uint8_t> rgb(4 * 64 * 96 * 3, 0);
  std::array<OrtxFrame, 4> frames{};
  for (size_t frame = 0; frame < frames.size(); ++frame) {
    auto* data = rgb.data() + frame * 64 * 96 * 3;
    for (size_t pixel = 0; pixel < 64 * 96; ++pixel) data[pixel * 3 + (frame < 2 ? 0 : 2)] = 255;
    frames[frame] = {data, 64 * 96 * 3, 96, 64, 96 * 3, static_cast<double>(frame)};
  }
  const std::array<std::span<const OrtxFrame>, 1> clips{frames};
  auto videos = LoadVideosFromFrames(clips);
  std::fill(rgb.begin(), rgb.end(), 0);
  const std::string prompt = "<|vision_start|><|video_pad|><|vision_end|>Which color appears first?";
  const auto raw = tokenizer.Encode(prompt.c_str());
  auto prepared = store.Prepare(prompt, raw, nullptr, 0, 4096, videos.get());
  ASSERT_EQ(prepared.media.size(), 2u);
  EXPECT_LT(prepared.media[0].start, prepared.media[1].start);
  EXPECT_NE(prepared.media[0].item, prepared.media[1].item);
  const auto output = GetEnv("ORTGENAI_ENGINE_VIDEO_REFERENCE_OUTPUT");
  for (size_t group = 0; group < prepared.media.size(); ++group) {
    const auto& item = prepared.media[group].item;
    EXPECT_EQ(item->grid, (std::array<int64_t, 3>{1, 4, 6}));
    EXPECT_EQ(item->rows, 6u);
    EXPECT_EQ(item->marker_token, model->config_->model.video_token_id);
    ASSERT_EQ(item->pixels->GetType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    ASSERT_EQ(item->pixels->GetElementCount(), 24 * 1536u);
    const auto* data = static_cast<const uint16_t*>(item->pixels->GetRawData());
    for (size_t value = 0; value < item->pixels->GetElementCount(); ++value) {
      const auto channel = value % 1536 / 512;
      ASSERT_EQ(data[value], channel == (group == 0 ? 0u : 2u) ? 0x3c00u : 0xbc00u)
          << "group=" << group << "value=" << value;
    }
    store.Encode(item);
    if (!output.empty()) {
      for (const auto& [name, tensor] :
           {std::pair{"patches", item->pixels}, std::pair{"features", item->features}}) {
        auto cpu = tensor->GetByteSpan().CopyDeviceToCpu();
        std::ofstream file(std::filesystem::path(output) / (std::string{name} + "-" + std::to_string(group) + ".bin"),
                           std::ios::binary);
        ASSERT_TRUE(file);
        file.write(reinterpret_cast<const char*>(cpu.data()), static_cast<std::streamsize>(cpu.size()));
        ASSERT_TRUE(file);
      }
    }
  }
  store.Retain(prepared.media);
  std::array<OrtxFrame, 4> forward{}, reverse{};
  for (size_t frame = 0; frame < frames.size(); ++frame) {
    CheckResult(OrtxGetRawFrame(videos->clips_[0].get(), frame, &forward[frame]));
    CheckResult(OrtxGetRawFrame(videos->clips_[0].get(), frames.size() - 1 - frame, &reverse[frame]));
    reverse[frame].timestamp_seconds = 10.0 + static_cast<double>(frame);
  }
  const std::array<std::span<const OrtxFrame>, 2> ordered_clips{forward, reverse};
  auto ordered = LoadVideosFromFrames(ordered_clips);
  const std::string mixed_prompt =
      "<|vision_start|><|video_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>";
  auto mixed = store.Prepare(mixed_prompt, tokenizer.Encode(mixed_prompt.c_str()), nullptr, 0, 4096, ordered.get());
  ASSERT_EQ(mixed.media.size(), 4u);
  EXPECT_EQ(mixed.media[0].item, prepared.media[0].item);
  EXPECT_EQ(mixed.media[1].item, prepared.media[1].item);
  EXPECT_EQ(mixed.media[2].item, prepared.media[1].item);
  EXPECT_EQ(mixed.media[3].item, prepared.media[0].item);

  std::string long_prompt = prompt;
  for (size_t index = 0; index < 500; ++index) long_prompt += " abc";
  const auto long_tokens = tokenizer.Encode(long_prompt.c_str());
  MediaStore empty_budget{model, 2500000, 2 * 1024 * 1024};
  EXPECT_NO_THROW(empty_budget.Prepare(long_prompt, long_tokens, nullptr, 0, 4096, videos.get()));
  MediaStore retained_budget{model, 2500000, 2 * 1024 * 1024};
  auto retained = retained_budget.Prepare(prompt, raw, nullptr, 0, 4096, videos.get());
  retained_budget.Retain(retained.media);
  EXPECT_THROW(retained_budget.Prepare(long_prompt, long_tokens, nullptr, 0, 4096, videos.get()),
               std::runtime_error);
  auto recovered = retained_budget.Prepare(prompt, raw, nullptr, 0, 4096, videos.get());
  ASSERT_EQ(recovered.media.size(), retained.media.size());
  EXPECT_EQ(recovered.media[0].item, retained.media[0].item);
  EXPECT_EQ(recovered.media[1].item, retained.media[1].item);

  const auto video_filename = model->config_->model.vision.video_config_filename;
  auto video_definition = model->config_->ResolvePath(*video_filename).open(std::ios::binary);
  std::string definition{std::istreambuf_iterator<char>{video_definition}, std::istreambuf_iterator<char>{}};
  constexpr std::string_view normalization = "\"qwen3_vl\": 0";
  const auto normalization_offset = definition.find(normalization);
  ASSERT_NE(normalization_offset, std::string::npos);
  definition[normalization_offset + normalization.size() - 1] = '1';
  const auto invalid_path = std::filesystem::temp_directory_path() /
                            ("ortgenai_video_rgb_" +
                             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
  {
    std::ofstream invalid_definition(invalid_path);
    invalid_definition << definition;
    ASSERT_TRUE(invalid_definition);
  }
  auto resolver = std::move(model->config_->package_resolver);
  model->config_->package_resolver = [&](const fs::path& base, std::string_view value) {
    if (value == "invalid-rgb.json") return fs::path{invalid_path.string()};
    return resolver ? resolver(base, value) : base / std::string{value};
  };
  model->config_->model.vision.video_config_filename = "invalid-rgb.json";
  std::string error;
  try {
    MediaStore invalid_rgb{model, 512 * 1024 * 1024, 2 * 1024 * 1024};
  } catch (const std::runtime_error& failure) {
    error = failure.what();
  }
  model->config_->package_resolver = std::move(resolver);
  std::filesystem::remove(invalid_path);
  EXPECT_NE(error.find("bounded Qwen3-VL temporal preprocessing pipeline"), std::string::npos) << error;

  model->config_->model.vision.video_config_filename = video_filename;
  const auto type = model->config_->model.type;
  model->config_->model.type = "qwen2_5_vl";
  EXPECT_THROW((MediaStore{model, 512 * 1024 * 1024, 2 * 1024 * 1024}), std::runtime_error);
  model->config_->model.type = type;
  model->config_->model.vision.video_config_filename.reset();
  model->config_->model.video_token_id = 0;
  MediaStore image_only{model, 512 * 1024 * 1024, 2 * 1024 * 1024};
  const auto video_marker = tokenizer.Encode("<|video_pad|>");
  EXPECT_THROW(image_only.ValidateTextTokens(video_marker), std::invalid_argument);
  EXPECT_THROW(image_only.Prepare("<|video_pad|>", video_marker, nullptr, 0, 4096), std::invalid_argument);
  const std::array<int32_t, 1> ordinary_zero{0};
  EXPECT_NO_THROW(image_only.ValidateTextTokens(ordinary_zero));
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

TEST(MediaInputsTest, VideoFramesOwnOrderedClipsAndPresentationTimes) {
  std::array<uint8_t, 9> padded{1, 2, 3, 0, 0, 0, 4, 5, 6};
  std::array<uint8_t, 3> other{7, 8, 9};
  const std::array<OrtxFrame, 2> first{{
      {padded.data(), padded.size(), 1, 2, 6, 0.25},
      {other.data(), other.size(), 1, 1, 3, 1.25},
  }};
  const std::array<OrtxFrame, 1> second{first[1]};
  const std::array<std::span<const OrtxFrame>, 2> clips{first, second};
  auto videos = LoadVideosFromFrames(clips);
  padded.fill(0);
  other.fill(0);
  ASSERT_EQ(videos->clips_.size(), 2u);
  EXPECT_EQ(videos->packed_bytes_, 12u);
  size_t count{};
  ASSERT_EQ(OrtxGetRawFrameCount(videos->clips_[0].get(), &count), kOrtxOK);
  EXPECT_EQ(count, 2u);
  OrtxFrame frame{};
  ASSERT_EQ(OrtxGetRawFrame(videos->clips_[0].get(), 0, &frame), kOrtxOK);
  EXPECT_EQ(frame.row_stride, 3u);
  EXPECT_EQ(frame.data_size, 6u);
  EXPECT_DOUBLE_EQ(frame.timestamp_seconds, 0.25);
  const std::array<uint8_t, 6> expected{1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), static_cast<const uint8_t*>(frame.data)));
  ASSERT_EQ(OrtxGetRawFrame(videos->clips_[1].get(), 0, &frame), kOrtxOK);
  EXPECT_DOUBLE_EQ(frame.timestamp_seconds, 1.25);
  EXPECT_EQ(static_cast<const uint8_t*>(frame.data)[0], 7);
}

TEST(MediaInputsTest, VideoFramesRejectLateInvalidClipsAndCombinedQuotasBeforeCopying) {
  const uint8_t unused = 0;
  const std::array<OrtxFrame, 1> first{{{&unused, 6, 2, 1, 6, 0}}};
  auto second = first;
  second[0].data_size = 5;
  const std::array<std::span<const OrtxFrame>, 2> clips{first, second};
  EXPECT_THROW(LoadVideosFromFrames(clips), std::runtime_error);
  second[0].data_size = 6;
  VideoFrameLimits limits;
  limits.max_bytes = 11;
  EXPECT_THROW(LoadVideosFromFrames(clips, limits), std::runtime_error);
}

TEST(MediaInputsTest, VideoFramesEnforceLoadingLimitsWithoutImposingModelMinimums) {
  const std::array<uint8_t, 3> pixels{1, 2, 3};
  const std::array<OrtxFrame, 1> frames{{{pixels.data(), pixels.size(), 1, 1, 3, 0}}};
  const std::array<std::span<const OrtxFrame>, 1> clips{frames};
  auto videos = LoadVideosFromFrames(clips);
  ASSERT_EQ(videos->clips_.size(), 1u);
  VideoFrameLimits limits;
  limits.max_clips = 0;
  EXPECT_THROW(LoadVideosFromFrames(clips, limits), std::runtime_error);
  limits.max_clips = 1;
  limits.max_frames = 0;
  EXPECT_THROW(LoadVideosFromFrames(clips, limits), std::runtime_error);
  limits.max_frames = 1;
  limits.max_bytes = 2;
  EXPECT_THROW(LoadVideosFromFrames(clips, limits), std::runtime_error);
  limits.max_bytes = 3;
  limits.max_dimension = 0;
  EXPECT_THROW(LoadVideosFromFrames(clips, limits), std::runtime_error);
  EXPECT_THROW(LoadVideosFromFrames({}), std::runtime_error);
}

TEST(MediaInputsTest, VideoPromptPreservesOuterWrapperAndSourceTimestampRounding) {
  const std::array times{0.0, 0.5, 1.0, 1.5, 2.0};
  const std::array<QwenVideoPrompt, 1> clips{{{{3, 2, 2}, times}}};
  const auto expanded = ExpandQwenMediaPrompt(
      "before<|vision_start|><|video_pad|><|vision_end|>after", {}, clips, 2, 1024);
  EXPECT_EQ(expanded,
            "before<|vision_start|>"
            "<0.2 seconds><|vision_start|><|video_pad|><|vision_end|>"
            "<1.2 seconds><|vision_start|><|video_pad|><|vision_end|>"
            "<2.0 seconds><|vision_start|><|video_pad|><|vision_end|>"
            "<|vision_end|>after");
  EXPECT_EQ(ExpandQwenMediaPrompt("plain text", {}, {}, 2, 10), "plain text");
  EXPECT_THROW(ExpandQwenMediaPrompt("plain text", {}, {}, 2, 9), std::invalid_argument);
  EXPECT_EQ(ExpandQwenMediaPrompt(
                "before<|vision_start|><|video_pad|><|vision_end|>after", {}, clips, 2, expanded.size()),
            expanded);
  EXPECT_THROW(ExpandQwenMediaPrompt(
                   "before<|vision_start|><|video_pad|><|vision_end|>after", {}, clips, 2, expanded.size() - 1),
               std::invalid_argument);
}

TEST(MediaInputsTest, VideoPromptKeepsIndependentImageAndClipOrdering) {
  const std::array<std::array<int64_t, 3>, 1> images{{{1, 2, 4}}};
  const std::array first_times{4.0, 4.0};
  const std::array second_times{8.0, 9.0};
  const std::array<QwenVideoPrompt, 2> clips{{{{1, 2, 2}, first_times}, {{1, 2, 4}, second_times}}};
  EXPECT_EQ(ExpandQwenMediaPrompt(
                "<|vision_start|><|video_pad|><|vision_end|> \n"
                "<|vision_start|><|image_pad|><|vision_end|> "
                "<|vision_start|><|video_pad|><|vision_end|>",
                images, clips, 2, 1024),
            "<|vision_start|><4.0 seconds><|vision_start|><|video_pad|><|vision_end|><|vision_end|> \n"
            "<|vision_start|><|image_pad|><|image_pad|><|vision_end|> "
            "<|vision_start|><8.5 seconds><|vision_start|><|video_pad|><|video_pad|><|vision_end|><|vision_end|>");
}

TEST(MediaInputsTest, VideoPromptRejectsMalformedMarkersAndTemporalMetadata) {
  std::array times{0.0, 1.0};
  std::array<QwenVideoPrompt, 1> clips{{{{1, 2, 2}, times}}};
  constexpr std::string_view marker = "<|vision_start|><|video_pad|><|vision_end|>";
  for (std::string_view invalid :
       {"<|video_pad|>", "<|image_pad|>", "<|vision_end|>", "<|vision_start|>", "<|vision_start|><|video_pad|>"})
    EXPECT_THROW(ExpandQwenMediaPrompt(invalid, {}, clips, 2, 1024), std::invalid_argument);
  EXPECT_THROW(ExpandQwenMediaPrompt("", {}, clips, 2, 1024), std::invalid_argument);
  EXPECT_THROW(ExpandQwenMediaPrompt(marker, {}, {}, 2, 1024), std::invalid_argument);
  EXPECT_THROW(ExpandQwenMediaPrompt(marker, {}, clips, 0, 1024), std::invalid_argument);
  clips[0].grid[0] = 2;
  EXPECT_THROW(ExpandQwenMediaPrompt(marker, {}, clips, 2, 1024), std::invalid_argument);
  clips[0].grid[0] = 1;
  clips[0].grid[1] = 3;
  EXPECT_THROW(ExpandQwenMediaPrompt(marker, {}, clips, 2, 1024), std::invalid_argument);
  clips[0].grid[1] = 2;
  times[0] = 2.0;
  EXPECT_THROW(ExpandQwenMediaPrompt(marker, {}, clips, 2, 1024), std::invalid_argument);
  times[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(ExpandQwenMediaPrompt(marker, {}, clips, 2, 1024), std::invalid_argument);
  times.fill(std::numeric_limits<double>::max());
  EXPECT_THROW(ExpandQwenMediaPrompt(marker, {}, clips, 2, 1024), std::invalid_argument);
}

TEST(MediaInputsTest, VideoProcessorAssetBindingIsOptionalAndParsedAtomically) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  EXPECT_FALSE(config->model.vision.video_config_filename.has_value());
  OverlayConfig(*config, R"({"model":{"vision":{"video_config_filename":"video_processor_config.json"}}})");
  EXPECT_EQ(config->model.vision.video_config_filename, "video_processor_config.json");
  EXPECT_THROW(OverlayConfig(*config, R"({"model":{"vision":{"video_config_filename":42}}})"), std::exception);
  EXPECT_EQ(config->model.vision.video_config_filename, "video_processor_config.json");
}

TEST(MediaInputsTest, TextOnlyModelTypesRejectVisionComponents) {
  for (const char* type : {"qwen3_5_text", "qwen3_5_moe_text", "qwen4_exp_text", "qwen38_flash_graph", "gpt2"}) {
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
  for (const char* type : {"qwen3_5", "qwen3_5_moe", "qwen4_exp"}) {
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

TEST(MediaInputsTest, VideoResizeMatchesSourceSpatialAndTemporalRounding) {
  EXPECT_EQ(MediaVideoGrid(29, 17, 4, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{2, 2, 4}));
  EXPECT_EQ(MediaVideoGrid(112, 80, 5, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{3, 4, 8}));
  EXPECT_EQ(MediaVideoGrid(1500, 1000, 4, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{2, 44, 66}));
  EXPECT_EQ(MediaVideoGrid(896, 896, 5, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{3, 56, 56}));
  EXPECT_EQ(MediaVideoGrid(896, 896, 3, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{2, 54, 54}));
}

TEST(MediaInputsTest, VideoResizeUsesWholeClipRatherThanEncoderGroupBudget) {
  EXPECT_EQ(MediaVideoGrid(1500, 1000, 64, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{32, 32, 48}));
  EXPECT_EQ(MediaVideoGrid(1500, 1000, 2, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{1, 44, 66}));
  EXPECT_EQ(MediaVideoGrid(32, 32, 2, 16, 2, 65536, 25165824, 768), (std::array<int64_t, 3>{1, 12, 12}));
}

TEST(MediaInputsTest, VideoResizeRejectsUnsupportedAndOverflowingGeometry) {
  EXPECT_THROW(MediaVideoGrid(0, 100, 4, 16, 2, 4096, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(MediaVideoGrid(UINT32_MAX, 100, 4, 16, 2, 4096, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(MediaVideoGrid(1, 201, 4, 16, 2, 4096, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(MediaVideoGrid(100, 100, 1, 16, 2, 4096, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(MediaVideoGrid(100, 100, 4, 0, 2, 4096, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(MediaVideoGrid(100, 100, 4, 16, 0, 4096, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(MediaVideoGrid(100, 100, 4, 16, 2, 0, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(MediaVideoGrid(100, 100, 4, 16, 2, 4096, 4095, 768), std::invalid_argument);
  EXPECT_THROW(MediaVideoGrid(100, 100, 4, 16, 2, 4096, 25165824, 0), std::invalid_argument);
  EXPECT_THROW(MediaVideoGrid(100, 100, std::numeric_limits<size_t>::max(), 16, 2, 4096, 25165824, 768), std::overflow_error);
  EXPECT_THROW(MediaVideoGrid(100, 100, 4, std::numeric_limits<size_t>::max(), 2, 4096, 25165824, 768), std::overflow_error);
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
