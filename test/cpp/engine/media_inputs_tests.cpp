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
#include "models/preprocessing/multi_modal_processor.h"
#include "models/preprocessing/qwen_vision_position.h"
#include "models/preprocessing/qwen_vision_processing.h"
#include "models/preprocessing/qwen2_5_vl_image_processor.h"
#include "models/preprocessing/bailing_vision_position.h"
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
  {
    auto image_processor = std::dynamic_pointer_cast<QwenImageProcessor>(model->vision_->processor->processor_);
    ASSERT_NE(image_processor, nullptr);
    const std::array<const char*, 2> paths{MODEL_PATH "../images/sheet.png", MODEL_PATH "../images/cars.jpg"};
    auto images = LoadImages(paths);
    auto pixels_only = image_processor->ProcessImages(*images);
    EXPECT_EQ(pixels_only->count(std::string{Config::Defaults::InputIdsName}), 0u);
    EXPECT_EQ(pixels_only->count(std::string{Config::Defaults::NumImageTokens}), 0u);
    const std::string image_prompt =
        "before<|vision_start|><|image_pad|><|vision_end|>between<|vision_start|><|image_pad|><|vision_end|>after";
    auto combined = model->vision_->processor->Process(image_prompt, images.get(), nullptr);
    for (const auto name : {Config::Defaults::PixelValuesName, Config::Defaults::ImageGridThwName}) {
      const auto& pixels = pixels_only->at(std::string{name});
      const auto& together = combined->at(std::string{name});
      ASSERT_EQ(pixels->GetType(), together->GetType());
      ASSERT_EQ(pixels->GetShape(), together->GetShape());
      EXPECT_EQ(std::memcmp(pixels->GetRawData(), together->GetRawData(),
                            pixels->GetElementCount() * Ort::SizeOf(pixels->GetType())),
                0);
    }
    const auto& grids = pixels_only->at(std::string{Config::Defaults::ImageGridThwName});
    ASSERT_EQ(grids->GetElementCount(), 6u);
    std::array<std::array<int64_t, 3>, 2> image_grids{};
    for (size_t image = 0; image < image_grids.size(); ++image)
      std::copy_n(grids->GetData<int64_t>() + image * 3, 3, image_grids[image].begin());
    const auto expanded = ExpandQwenMediaPrompt(
        image_prompt, image_grids, {}, static_cast<size_t>(model->config_->model.vision.spatial_merge_size), 1024 * 1024);
    const auto expected = model->vision_->processor->tokenizer_->Encode(expanded.c_str());
    const auto& ids = combined->at(std::string{Config::Defaults::InputIdsName});
    ASSERT_EQ(ids->GetElementCount(), expected.size());
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), ids->GetData<int32_t>()));
  }
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
  store.Identify(prepared, 0, {});
  auto identical = prepared;
  identical.media[0].positions = MakeQwenVisionPosition({1, 4, 6}, 2);
  store.Identify(identical, 0, {});
  EXPECT_EQ(identical.media[0].prefix, prepared.media[0].prefix);
  auto changed_positions = prepared;
  changed_positions.media[0].positions = MakeQwenVisionPosition({1, 6, 4}, 2);
  store.Identify(changed_positions, 0, {});
  EXPECT_EQ(changed_positions.media[0].item, prepared.media[0].item);
  EXPECT_NE(changed_positions.media[0].prefix, prepared.media[0].prefix);
  EXPECT_EQ(changed_positions.media[1].prefix, prepared.media[1].prefix);
  EXPECT_LT(prepared.media[0].start, prepared.media[1].start);
  EXPECT_NE(prepared.media[0].item, prepared.media[1].item);
  const auto output = GetEnv("ORTGENAI_ENGINE_VIDEO_REFERENCE_OUTPUT");
  for (size_t group = 0; group < prepared.media.size(); ++group) {
    const auto& item = prepared.media[group].item;
    const auto& encoder = model->config_->model.vision;
    const auto& pixels = item->inputs.at(encoder.inputs.pixel_values);
    const auto& grid = item->inputs.at(encoder.inputs.image_grid_thw);
    ASSERT_EQ(grid->GetType(), Ort::TypeToTensorType<int64_t>);
    ASSERT_EQ(grid->GetElementCount(), 3u);
    const std::array<int64_t, 3> expected_grid{1, 4, 6};
    EXPECT_TRUE(std::equal(expected_grid.begin(), expected_grid.end(), grid->GetData<int64_t>()));
    EXPECT_EQ(item->rows, 6u);
    EXPECT_EQ(prepared.tokens[prepared.media[group].start], model->config_->model.video_token_id);
    ASSERT_EQ(pixels->GetType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    ASSERT_EQ(pixels->GetElementCount(), 24 * 1536u);
    const auto* data = static_cast<const uint16_t*>(pixels->GetRawData());
    for (size_t value = 0; value < pixels->GetElementCount(); ++value) {
      const auto channel = value % 1536 / 512;
      ASSERT_EQ(data[value], channel == (group == 0 ? 0u : 2u) ? 0x3c00u : 0xbc00u)
          << "group=" << group << "value=" << value;
    }
    store.Encode(item);
    if (!output.empty()) {
      for (const auto& [name, tensor] :
           {std::pair{"patches", pixels}, std::pair{"features", item->features}}) {
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
  const auto item = prepared.media[0].item;
  {
    auto invalid = std::make_shared<MediaItem>();
    invalid->rows = item->rows;
    invalid->feature_bytes = std::numeric_limits<size_t>::max();
    invalid->inputs = item->inputs;
    const auto& name = model->config_->model.vision.inputs.pixel_values;
    auto renamed = invalid->inputs.extract(name);
    renamed.key() = "unknown_encoder_input";
    invalid->inputs.insert(std::move(renamed));
    EXPECT_THROW(store.Encode(invalid), std::invalid_argument);
    EXPECT_EQ(invalid->features, nullptr);
    invalid->inputs = item->inputs;
    auto wrong_type = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    wrong_type->CreateTensor(item->inputs.at(name)->GetShape());
    invalid->inputs[name] = std::move(wrong_type);
    EXPECT_THROW(store.Encode(invalid), std::invalid_argument);
    EXPECT_EQ(invalid->features, nullptr);
    auto wrong_shape = std::make_shared<Tensor>(GetDeviceInterface(DeviceType::CPU), item->inputs.at(name)->GetType());
    wrong_shape->CreateTensor(std::array<int64_t, 2>{1, item->inputs.at(name)->GetShape().back() + 1});
    invalid->inputs[name] = std::move(wrong_shape);
    EXPECT_THROW(store.Encode(invalid), std::invalid_argument);
    invalid->inputs.erase(name);
    EXPECT_THROW(store.Encode(invalid), std::invalid_argument);
    EXPECT_EQ(invalid->features, nullptr);
    EXPECT_NE(item->features, nullptr);
  }
  const std::array<MediaSpan, 2> slices{{
      {1, item, MakeQwenVisionPosition({1, 2, 4}, 2), 2, 1},
      {5, item, MakeQwenVisionPosition({1, 2, 4}, 2), 2, 4},
  }};
  const std::array<int64_t, 8> lookup_tokens{raw[0], raw[0], raw[0], raw[0], raw[0], raw[0], raw[0], raw[0]};
  const auto features = item->features->GetByteSpan().CopyDeviceToCpu();
  const auto row_bytes = static_cast<size_t>(model->cpu_embedding_->hidden_size_) * Ort::SizeOf(model->cpu_embedding_->type_);
  for (const auto& [first_row, rows] : {std::pair{0u, 8u}, std::pair{2u, 4u}}) {
    Tensor embeddings{GetDeviceInterface(DeviceType::CPU), model->cpu_embedding_->type_};
    embeddings.CreateTensor(std::array<int64_t, 2>{rows, model->cpu_embedding_->hidden_size_});
    CpuEmbedding::Workspace workspace;
    model->cpu_embedding_->Run(std::span<const int64_t>{lookup_tokens}.first(rows), embeddings, workspace);
    const auto original_view = embeddings.GetByteSpan().CopyDeviceToCpu();
    const std::vector<uint8_t> original{original_view.begin(), original_view.end()};
    ApplyMediaFeatures(slices, first_row, rows, embeddings, 0);
    const auto replaced = embeddings.GetByteSpan().CopyDeviceToCpu();
    for (size_t row = 0; row < rows; ++row) {
      const auto absolute = first_row + row;
      const auto source = absolute >= 1 && absolute < 3 ? absolute : absolute >= 5 && absolute < 7 ? absolute - 1
                                                                                                   : item->rows;
      const auto* expected = source < item->rows ? features.data() + source * row_bytes : original.data() + row * row_bytes;
      EXPECT_EQ(std::memcmp(replaced.data() + row * row_bytes, expected, row_bytes), 0);
    }
  }
  std::array<OrtxFrame, 4> forward{}, reverse{};
  for (size_t frame = 0; frame < frames.size(); ++frame) {
    CheckResult(OrtxGetRawFrame(videos->clips_[0].get(), frame, &forward[frame]));
    CheckResult(OrtxGetRawFrame(videos->clips_[0].get(), frames.size() - 1 - frame, &reverse[frame]));
    reverse[frame].timestamp_seconds = 10.0 + static_cast<double>(frame);
  }
  {
    // Uniform colors preserve patch bytes while changing the grid feed.
    auto swapped = forward;
    for (auto& frame : swapped) {
      frame.width = 64;
      frame.height = 96;
      frame.row_stride = 64 * 3;
    }
    const std::array<std::span<const OrtxFrame>, 1> swapped_clips{swapped};
    auto swapped_videos = LoadVideosFromFrames(swapped_clips);
    auto swapped_turn = store.Prepare(prompt, raw, nullptr, 0, 4096, swapped_videos.get());
    ASSERT_EQ(swapped_turn.media.size(), prepared.media.size());
    const auto& name = model->config_->model.vision.inputs.pixel_values;
    for (size_t group = 0; group < prepared.media.size(); ++group) {
      const auto& original = prepared.media[group].item;
      const auto& changed = swapped_turn.media[group].item;
      EXPECT_NE(original, changed);
      const auto& original_pixels = original->inputs.at(name);
      const auto& changed_pixels = changed->inputs.at(name);
      ASSERT_EQ(original_pixels->GetShape(), changed_pixels->GetShape());
      ASSERT_EQ(original_pixels->GetType(), changed_pixels->GetType());
      EXPECT_EQ(std::memcmp(original_pixels->GetRawData(), changed_pixels->GetRawData(),
                            original_pixels->GetElementCount() * Ort::SizeOf(original_pixels->GetType())),
                0);
    }
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
  EXPECT_EQ(QwenMediaPromptBytes(
                "before<|vision_start|><|video_pad|><|vision_end|>after", {}, clips, 2, expanded.size()),
            expanded.size());
  EXPECT_THROW(QwenMediaPromptBytes(
                   "before<|vision_start|><|video_pad|><|vision_end|>after", {}, clips, 2, expanded.size() - 1),
               std::invalid_argument);
}

TEST(MediaInputsTest, PromptEstimateCountsLargeImageExpansionWithoutMaterialization) {
  constexpr std::string_view marker = "<|vision_start|><|image_pad|><|vision_end|>";
  const std::array<std::array<int64_t, 3>, 1> images{{{1, 8192, 8192}}};
  constexpr size_t expected_bytes = size_t{4096} * 4096 * 13 + 30;
  EXPECT_EQ(QwenMediaPromptBytes(marker, images, {}, 2, expected_bytes), expected_bytes);
  EXPECT_THROW(QwenMediaPromptBytes(marker, images, {}, 2, expected_bytes - 1), std::invalid_argument);
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
  first->rows = 6;
  auto second = std::make_shared<MediaItem>();
  second->rows = 4;
  const std::array<MediaSpan, 2> media{{
      {3, first, MakeQwenVisionPosition({1, 4, 6}, 2)},
      {12, second, MakeQwenVisionPosition({1, 4, 4}, 2)},
  }};
  EXPECT_EQ(MediaPosition(media, 2), (std::array<int64_t, 3>{2, 2, 2}));
  EXPECT_EQ(MediaPosition(media, 3), (std::array<int64_t, 3>{3, 3, 3}));
  EXPECT_EQ(MediaPosition(media, 7), (std::array<int64_t, 3>{3, 4, 4}));
  EXPECT_EQ(MediaPosition(media, 9), (std::array<int64_t, 3>{6, 6, 6}));
  EXPECT_EQ(MediaPosition(media, 13), (std::array<int64_t, 3>{9, 9, 10}));
  EXPECT_EQ(MediaPosition(media, 16), (std::array<int64_t, 3>{11, 11, 11}));
  EXPECT_EQ(MediaPosition(media, 7), (std::array<int64_t, 3>{3, 4, 4}));
}

TEST(MediaInputsTest, ResizePreflightMatchesNativeTieAndMinimumPixelRules) {
  EXPECT_EQ(QwenImageGrid(500, 400, 16, 2, 65536, 262144), (std::array<int64_t, 3>{1, 24, 32}));
  EXPECT_EQ(QwenImageGrid(30, 17, 16, 2, 65536, 262144), (std::array<int64_t, 3>{1, 14, 22}));
  EXPECT_EQ(QwenImageGrid(381, 137, 16, 2, 65536, 16777216), (std::array<int64_t, 3>{1, 10, 28}));
  EXPECT_THROW(QwenImageGrid(0, 100, 16, 2, 65536, 16777216), std::invalid_argument);
  EXPECT_THROW(QwenImageGrid(1, 201, 16, 2, 65536, 16777216), std::invalid_argument);
  EXPECT_THROW(QwenImageGrid(UINT32_MAX, UINT32_MAX, 16, 2, 65536, 16777216), std::invalid_argument);
}

TEST(MediaInputsTest, VideoResizeMatchesSourceSpatialAndTemporalRounding) {
  EXPECT_EQ(QwenVideoGrid(29, 17, 4, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{2, 2, 4}));
  EXPECT_EQ(QwenVideoGrid(112, 80, 5, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{3, 4, 8}));
  EXPECT_EQ(QwenVideoGrid(1500, 1000, 4, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{2, 44, 66}));
  EXPECT_EQ(QwenVideoGrid(896, 896, 5, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{3, 56, 56}));
  EXPECT_EQ(QwenVideoGrid(896, 896, 3, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{2, 54, 54}));
}

TEST(MediaInputsTest, VideoResizeUsesWholeClipRatherThanEncoderGroupBudget) {
  EXPECT_EQ(QwenVideoGrid(1500, 1000, 64, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{32, 32, 48}));
  EXPECT_EQ(QwenVideoGrid(1500, 1000, 2, 16, 2, 4096, 25165824, 768), (std::array<int64_t, 3>{1, 44, 66}));
  EXPECT_EQ(QwenVideoGrid(32, 32, 2, 16, 2, 65536, 25165824, 768), (std::array<int64_t, 3>{1, 12, 12}));
}

TEST(MediaInputsTest, VideoResizeRejectsUnsupportedAndOverflowingGeometry) {
  EXPECT_THROW(QwenVideoGrid(0, 100, 4, 16, 2, 4096, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(QwenVideoGrid(UINT32_MAX, 100, 4, 16, 2, 4096, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(QwenVideoGrid(1, 201, 4, 16, 2, 4096, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(QwenVideoGrid(100, 100, 1, 16, 2, 4096, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(QwenVideoGrid(100, 100, 4, 0, 2, 4096, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(QwenVideoGrid(100, 100, 4, 16, 0, 4096, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(QwenVideoGrid(100, 100, 4, 16, 2, 0, 25165824, 768), std::invalid_argument);
  EXPECT_THROW(QwenVideoGrid(100, 100, 4, 16, 2, 4096, 4095, 768), std::invalid_argument);
  EXPECT_THROW(QwenVideoGrid(100, 100, 4, 16, 2, 4096, 25165824, 0), std::invalid_argument);
  EXPECT_THROW(QwenVideoGrid(100, 100, std::numeric_limits<size_t>::max(), 16, 2, 4096, 25165824, 768), std::overflow_error);
  EXPECT_THROW(QwenVideoGrid(100, 100, 4, std::numeric_limits<size_t>::max(), 2, 4096, 25165824, 768), std::overflow_error);
}

TEST(MediaInputsTest, PositionRangesPreserveImageBoundariesAndLaterTextOffsets) {
  auto first = std::make_shared<MediaItem>();
  first->rows = 6;
  auto second = std::make_shared<MediaItem>();
  second->rows = 4;
  const std::array<MediaSpan, 2> media{{
      {3, first, MakeQwenVisionPosition({1, 4, 6}, 2)},
      {12, second, MakeQwenVisionPosition({1, 4, 4}, 2)},
  }};
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

TEST(MediaInputsTest, NoncontiguousFeatureSlicesHaveIndependentPositionRanges) {
  auto item = std::make_shared<MediaItem>();
  item->rows = 6;
  const auto positions = MakeQwenVisionPosition({1, 2, 4}, 2);
  const std::array<MediaSpan, 2> media{{
      {1, item, positions, 2, 1},
      {5, item, positions, 2, 4},
  }};
  EXPECT_EQ(MediaPosition(media, 2), (std::array<int64_t, 3>{1, 1, 2}));
  EXPECT_EQ(MediaPosition(media, 3), (std::array<int64_t, 3>{3, 3, 3}));
  EXPECT_EQ(MediaPosition(media, 6), (std::array<int64_t, 3>{5, 5, 6}));
  EXPECT_EQ(MediaPosition(media, 7), (std::array<int64_t, 3>{7, 7, 7}));
  EXPECT_THROW((MediaSpan{0, item, positions, 3, 0}), std::invalid_argument);
  EXPECT_THROW((MediaSpan{0, item, positions, 2, 5}), std::invalid_argument);
  EXPECT_THROW((MediaSpan{std::numeric_limits<size_t>::max(), item}), std::invalid_argument);
}

TEST(MediaInputsTest, PositionRangesRejectOverflowAndIncompatiblePlanes) {
  std::array<int64_t, 2> first{}, second{};
  std::array<int64_t, 1> short_plane{};
  EXPECT_THROW(MediaPositions({}, 0, {first, second, short_plane}), std::invalid_argument);
  EXPECT_THROW(MediaPositions({}, std::numeric_limits<size_t>::max(), {first, second, first}),
               std::overflow_error);
  const auto position = MakeQwenVisionPosition({1, 4, 6}, 2);
  EXPECT_THROW(position->Fill(5, 0, {first, second, first}), std::invalid_argument);
  EXPECT_THROW(position->Fill(0, std::numeric_limits<int64_t>::max(), {first, second, first}),
               std::overflow_error);
  auto item = std::make_shared<MediaItem>();
  item->rows = 2;
  const std::array<MediaSpan, 2> unordered{{{3, item}, {1, item}}};
  const std::array<MediaSpan, 2> overlapping{{{1, item}, {2, item}}};
  EXPECT_THROW(MediaPositions(unordered, 0, {first, second, first}), std::invalid_argument);
  EXPECT_THROW(MediaPositions(overlapping, 0, {first, second, first}), std::invalid_argument);
}

TEST(MediaInputsTest, BailingPositionsUseSignedCentersAndTemporalTextAdvance) {
  for (const size_t temporal : {1u, 2u}) {
    auto item = std::make_shared<MediaItem>();
    item->rows = temporal * 24;
    const auto positions = MakeBailingVisionPosition({static_cast<int64_t>(temporal), 8, 12}, 2);
    const std::array<MediaSpan, 1> media{{{1, item, positions}}};
    EXPECT_EQ(MediaPosition(media, 1), (std::array<int64_t, 3>{1, 0, -1}));
    EXPECT_EQ(MediaPosition(media, 6), (std::array<int64_t, 3>{1, 0, 4}));
    EXPECT_EQ(MediaPosition(media, 7), (std::array<int64_t, 3>{1, 1, -1}));
    if (temporal == 2)
      EXPECT_EQ(MediaPosition(media, 25), (std::array<int64_t, 3>{3, 2, 1}));
    const auto next = static_cast<int64_t>((temporal - 1) * 2 + 2);
    EXPECT_EQ(MediaPosition(media, 1 + item->rows), (std::array<int64_t, 3>{next, next, next}));
    EXPECT_TRUE(positions->Equals(*MakeBailingVisionPosition({static_cast<int64_t>(temporal), 8, 12}, 2)));
    EXPECT_FALSE(positions->Equals(*MakeQwenVisionPosition({1, 8, 12}, 2)));
  }
}

TEST(MediaInputsTest, BailingCoordinateLimitsDoNotOverflowIntermediateSums) {
  const auto positions = MakeBailingVisionPosition({1, 8, 12}, 2);
  std::array<int64_t, 1> time{}, height{}, width{};
  const auto limit = std::numeric_limits<int64_t>::max();
  positions->Fill(23, limit - 3, {time, height, width});
  EXPECT_EQ(time[0], limit - 3);
  EXPECT_EQ(height[0], limit - 1);
  EXPECT_EQ(width[0], limit);
  EXPECT_THROW(positions->Fill(23, limit - 2, {time, height, width}), std::overflow_error);
  EXPECT_THROW(MakeBailingVisionPosition({0, 8, 12}, 2), std::invalid_argument);
  EXPECT_THROW(MakeBailingVisionPosition({1, 7, 12}, 2), std::invalid_argument);
  EXPECT_THROW(MakeBailingVisionPosition({limit, 8, 12}, 2), std::overflow_error);
}

TEST(MediaInputsTest, PrefixIdentityDoesNotReplaceVocabularyInputs) {
  auto model = LoadDummyDecoderModel();
  auto engine = MakeDoublesEngine(model, 1024, EosToken(*model)).engine;
  const std::array<int32_t, 5> tokens{2, 3, 4, 4, 5};
  auto first = CreateRequestWithPrompt(engine, tokens);
  auto second = CreateRequestWithPrompt(engine, tokens);
  auto image_a = std::make_shared<MediaItem>();
  image_a->rows = 2;
  auto image_b = std::make_shared<MediaItem>();
  image_b->rows = 2;
  first->ReserveMedia(1);
  second->ReserveMedia(1);
  std::array<MediaSpan, 1> first_media{{{2, image_a}}}, second_media{{{2, image_b}}};
  auto first_prefix = std::make_shared<MediaPrefix>();
  first_prefix->first_id = 1;
  auto second_prefix = std::make_shared<MediaPrefix>();
  second_prefix->first_id = 3;
  first_media[0].prefix = first_prefix;
  second_media[0].prefix = second_prefix;
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

TEST(MediaInputsTest, DecoderIdentityIncludesSlicesTokensAndCoordinateSemantics) {
  auto item = std::make_shared<MediaItem>();
  item->rows = 8;
  const std::array<int32_t, 6> tokens{2, 2, 2, 2, 2, 2};
  MediaSpan span{3, item, MakeQwenVisionPosition({1, 4, 6}, 2), 6, 1};
  MediaPrefix identity;
  identity.item = item;
  identity.positions = span.positions;
  identity.tokens.assign(tokens.begin(), tokens.end());
  identity.feature_offset = span.feature_offset;
  identity.position_base = 3;
  EXPECT_TRUE(identity.Matches(span, tokens, 3));
  span.positions = MakeQwenVisionPosition({1, 4, 6}, 2);
  EXPECT_TRUE(identity.Matches(span, tokens, 3));
  span.positions = MakeQwenVisionPosition({1, 6, 4}, 2);
  EXPECT_FALSE(identity.Matches(span, tokens, 3));
  span.positions = MakeBailingVisionPosition({1, 4, 6}, 2);
  EXPECT_FALSE(identity.Matches(span, tokens, 3));
  span.positions = identity.positions;
  EXPECT_FALSE(identity.Matches(span, tokens, 4));
  span.feature_offset = 2;
  EXPECT_FALSE(identity.Matches(span, tokens, 3));
  span.feature_offset = 1;
  auto different_tokens = tokens;
  different_tokens[2] = 3;
  EXPECT_FALSE(identity.Matches(span, different_tokens, 3));
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
