// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#define OGA_USE_SPAN 1
#include "ort_genai.h"
#include "models/io/conv_kv_cache.h"
#include "models/io/static_kv_cache.h"
#include "models/multi_modal.h"
#include "telemetry_test_environment.h"
#include "engine/engine.h"

namespace {

std::unique_ptr<Generators::Config> MakeConfig(bool is_pipeline) {
  auto config = std::make_unique<Generators::Config>();
  config->model.decoder.sliding_window.emplace();
  config->model.decoder.sliding_window->window_size = 64;
  if (is_pipeline)
    config->model.decoder.pipeline.emplace_back();
  return config;
}

struct CacheTestModel : Generators::Model {
  explicit CacheTestModel(bool is_pipeline) : Model{MakeConfig(is_pipeline)} {}

  std::unique_ptr<Generators::State> CreateState(
      Generators::DeviceSpan<int32_t>, const Generators::GeneratorParams&) const override {
    return nullptr;
  }
};

TEST(KvCacheTests, UsesDeviceWindowSizeForSingleSessionModel) {
  CacheTestModel model{false};
  Generators::Config::Search search;
  EXPECT_TRUE(Generators::UsesNonRewindableWindowedKeyValueCache(
      model, model.config_->model.decoder));
  EXPECT_EQ(Generators::GetWindowedKeyValueCacheSize(model, search, 4096), 80);
}

TEST(KvCacheTests, IgnoresTopLevelDeviceWindowSizeForPipelineModel) {
  CacheTestModel model{true};
  Generators::Config::Search search;
  EXPECT_FALSE(Generators::UsesNonRewindableWindowedKeyValueCache(
      model, model.config_->model.decoder));
  EXPECT_EQ(Generators::GetWindowedKeyValueCacheSize(model, search, 4096), 0);
}

class CacheDispatchTest : public ::testing::TestWithParam<const char*> {};

std::unique_ptr<OgaModel> LoadCacheFixture(const std::string& path, bool cuda) {
  auto config = OgaConfig::Create(path.c_str());
  if (cuda) config->AppendProvider("cuda");
  auto model = OgaModel::Create(*config);
  EXPECT_STREQ(model->GetDeviceType(), cuda ? "CUDA" : "CPU");
  return model;
}

float ReadCacheElement(OgaTensor& tensor, size_t index) {
  if (tensor.Type() == OgaElementType_bfloat16) {
    const auto bits = static_cast<const uint16_t*>(tensor.Data())[index];
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
  }
  return static_cast<const float*>(tensor.Data())[index];
}

TEST_P(CacheDispatchTest, PrefillDecodeAndFreshStateCarrySparseStates) {
  const std::string_view parameter{GetParam()};
  const bool cuda = parameter.ends_with("-cuda");
  const auto fixture = cuda ? parameter.substr(0, parameter.size() - 5) : parameter;
  const bool recurrent = fixture.starts_with("recurrent");
  const bool bf16 = fixture == "recurrent-bf16";
  const std::string model_path = std::string{MODEL_PATH} + "cache-dispatch/" + std::string{fixture};
  auto internal_model = Generators::CreateModel(Generators::GetOrtEnv(), model_path.c_str());
  EXPECT_EQ(Generators::HasConvKeyValueCache(*internal_model), !recurrent);
  // Do not strip the metadata to make recurrent auto-discovery work.
  EXPECT_EQ(internal_model->config_->model.decoder.layer_types.size(), recurrent ? 4u : 3u);

  auto model = LoadCacheFixture(model_path, cuda);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 16);
  auto generator = OgaGenerator::Create(*model, *params);
  const int attention_index = recurrent ? 2 : 1;
  const int final_fixed_index = recurrent ? 3 : 2;
  const int score_scale = recurrent ? 13 : 7;

  auto expect_state = [&](int layer, const char* kind, float expected) {
    const auto name = "present." + std::to_string(layer) + "." + kind;
    auto tensor = generator->GetOutput(name.c_str());
    ASSERT_EQ(tensor->Type(), bf16 && std::string_view{kind} == "conv_state"
                                  ? OgaElementType_bfloat16
                                  : OgaElementType_float32);
    ASSERT_EQ(tensor->Shape(), (std::string_view{kind} == "conv_state"
                                    ? std::vector<int64_t>{1, 2, 3}
                                    : std::vector<int64_t>{1, 2, 2, 2}));
    size_t count = 1;
    for (auto dim : tensor->Shape()) count *= static_cast<size_t>(dim);
    for (size_t i = 0; i < count; ++i) EXPECT_FLOAT_EQ(ReadCacheElement(*tensor, i), expected) << name;
  };
  auto expect_outputs = [&](int total, const std::vector<int32_t>& tokens, int64_t query_length) {
    expect_state(0, "conv_state", static_cast<float>(total));
    expect_state(final_fixed_index, "conv_state", static_cast<float>(3 * total));
    if (recurrent) {
      expect_state(0, "recurrent_state", static_cast<float>(2 * total));
      expect_state(final_fixed_index, "recurrent_state", static_cast<float>(4 * total));
    }
    for (const auto* kind : {"key", "value"}) {
      const auto name = "present." + std::to_string(attention_index) + "." + kind;
      auto tensor = generator->GetOutput(name.c_str());
      ASSERT_EQ(tensor->Type(), bf16 ? OgaElementType_bfloat16 : OgaElementType_float32);
      ASSERT_EQ(tensor->Shape(), (std::vector<int64_t>{1, 1, static_cast<int64_t>(tokens.size()), 1}));
      const float scale = std::string_view{kind} == "key" ? 1.0f : 2.0f;
      for (size_t i = 0; i < tokens.size(); ++i) EXPECT_FLOAT_EQ(ReadCacheElement(*tensor, i), scale * tokens[i]);
    }
    // GetLogits() lazily executes another forward after token selection; read
    // the actual completed graph output instead so assertions never advance state.
    auto logits = generator->GetOutput("logits");
    ASSERT_EQ(logits->Type(), OgaElementType_float32);
    ASSERT_EQ(logits->Shape(), (std::vector<int64_t>{1, query_length, 16}));
    const auto* data = static_cast<const float*>(logits->Data());
    size_t count = 1;
    for (auto dim : logits->Shape()) count *= static_cast<size_t>(dim);
    for (size_t i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ(data[i], i % 16 == static_cast<size_t>(1 + (score_scale * total) % 14) ? 1.0f : 0.0f)
          << "total=" << total << ", logit=" << i;
    }
  };

  std::vector<int32_t> tokens{1, 2, 3};
  generator->AppendTokens(tokens);
  int total = 6;
  expect_outputs(total, tokens, 3);
  int32_t previous_token = 0;
  for (int step = 0; step < 4; ++step) {
    if (step > 0) {
      tokens.push_back(previous_token);
      total += previous_token;
    }
    const int32_t expected = 1 + (score_scale * total) % 14;
    generator->GenerateNextToken();
    ASSERT_EQ(generator->GetSequenceCount(0), tokens.size() + 1);
    EXPECT_EQ(generator->GetSequence(0).back(), expected);
    // The first selection consumes prefill logits; subsequent forwards consume
    // the preceding generated token.
    expect_outputs(total, tokens, step == 0 ? 3 : 1);
    previous_token = expected;
  }
  if (recurrent) {
    generator->RewindTo(0);
  } else {
    EXPECT_THROW(generator->RewindTo(0), std::runtime_error);
    generator = OgaGenerator::Create(*model, *params);
  }
  tokens = {2};
  generator->AppendTokens(tokens);
  expect_outputs(2, tokens, 1);
}

INSTANTIATE_TEST_SUITE_P(LegacyGenerator, CacheDispatchTest,
                         ::testing::Values("recurrent", "conv", "lfm2"));

#if USE_CUDA
INSTANTIATE_TEST_SUITE_P(CudaLegacyGenerator, CacheDispatchTest,
                         ::testing::Values("recurrent-cuda", "recurrent-bf16-cuda", "conv-cuda", "lfm2-cuda"));
#endif

class MultiModalCacheTest : public ::testing::TestWithParam<bool> {};

TEST_P(MultiModalCacheTest, PreparedImageAndEmptyMediaDecodeCarrySparseStates) {
  const std::string path = std::string{MODEL_PATH} + "cache-dispatch/multimodal";
  auto model = LoadCacheFixture(path, GetParam());
  EXPECT_STREQ(model->GetType(), "nemotron_h_omni");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 16);

  for (const bool image : {false, true}) {
    SCOPED_TRACE(image ? "prepared image" : "text only");
    auto generator = OgaGenerator::Create(*model, *params);
    std::array<int32_t, 3> ids{1, image ? 18 : 2, 3};
    std::array<float, 12> pixels;
    pixels.fill(4.0f);
    std::array<int64_t, 1> image_tokens{image ? 1 : 0};
    auto ids_tensor = OgaTensor::Create(ids.data(), std::array<int64_t, 2>{1, 3});
    auto pixels_tensor = OgaTensor::Create(pixels.data(), std::array<int64_t, 4>{1, 3, 2, 2});
    auto count_tensor = OgaTensor::Create(image_tokens.data(), std::array<int64_t, 1>{1});
    auto inputs = OgaNamedTensors::Create();
    inputs->Set("input_ids", *ids_tensor);
    inputs->Set("pixel_values", *pixels_tensor);
    inputs->Set("num_image_tokens", *count_tensor);
    generator->SetInputs(*inputs);

    // The decoder uses the first embedding channel plus absolute positions for
    // its KV rows, and also adds mask length to each forward's fixed-state delta.
    std::vector<float> rows{1.0f, image ? 5.0f : 3.0f, 5.0f};
    int kv_total = image ? 11 : 9;
    int state_total = kv_total + 3;
    for (int step = 0; step < 4; ++step) {
      SCOPED_TRACE(step);
      for (const auto& [layer, scale] : {std::pair{0, 1}, std::pair{3, 3}}) {
        for (const auto* kind : {"conv_state", "ssm_state"}) {
          const auto name = "present." + std::to_string(layer) + "." + kind;
          auto tensor = generator->GetOutput(name.c_str());
          const bool ssm = std::string_view{kind} == "ssm_state";
          ASSERT_EQ(tensor->Type(), OgaElementType_float32);
          ASSERT_EQ(tensor->Shape(), (ssm ? std::vector<int64_t>{1, 2, 2, 2} : std::vector<int64_t>{1, 2, 3}));
          const auto* data = static_cast<const float*>(tensor->Data());
          for (size_t i = 0; i < (ssm ? 8u : 6u); ++i)
            EXPECT_FLOAT_EQ(data[i], static_cast<float>((scale + (ssm ? 1 : 0)) * state_total));
        }
      }
      for (const auto* kind : {"key", "value"}) {
        const auto name = std::string{"present.2."} + kind;
        auto tensor = generator->GetOutput(name.c_str());
        ASSERT_EQ(tensor->Type(), OgaElementType_float32);
        ASSERT_EQ(tensor->Shape(), (std::vector<int64_t>{1, 1, static_cast<int64_t>(rows.size()), 1}));
        const auto* data = static_cast<const float*>(tensor->Data());
        const float scale = std::string_view{kind} == "key" ? 1.0f : 2.0f;
        for (size_t i = 0; i < rows.size(); ++i) EXPECT_FLOAT_EQ(data[i], scale * rows[i]);
      }
      const int32_t expected = 1 + (10 * state_total + 3 * kv_total) % 14;
      auto logits = generator->GetOutput("logits");
      ASSERT_EQ(logits->Type(), OgaElementType_float32);
      ASSERT_EQ(logits->Shape(), (std::vector<int64_t>{1, step == 0 ? 3 : 1, 16}));
      const auto* data = static_cast<const float*>(logits->Data());
      for (int i = 0; i < (step == 0 ? 48 : 16); ++i)
        EXPECT_FLOAT_EQ(data[i], i % 16 == expected ? 1.0f : 0.0f);
      generator->GenerateNextToken();
      ASSERT_EQ(generator->GetSequenceCount(0), rows.size() + 1);
      EXPECT_EQ(generator->GetSequence(0).back(), expected);
      if (step < 3) {
        // GetLogits intentionally advances the selected token here. Media is
        // empty on decode, so embedding must not gather from vanished features.
        generator->GetLogits();
        const auto row = static_cast<int>(expected + rows.size());
        rows.push_back(static_cast<float>(row));
        kv_total += row;
        state_total += row + static_cast<int>(rows.size());
      }
    }
  }
}

TEST_P(MultiModalCacheTest, PreparedFeaturesUseCompactedPromptWithoutRunningVision) {
  const std::string path = std::string{MODEL_PATH} + "cache-dispatch/multimodal-retention";
  auto config = Generators::CreateConfig(Generators::GetOrtEnv(), path.c_str());
  if (GetParam()) Generators::SetProviderOption(*config, "cuda", {}, {});
  EXPECT_EQ(config->model.vision.inputs.is_video, "is_video");
  EXPECT_EQ(config->model.vision.outputs.retention_mask, "retention_mask");
  EXPECT_EQ(config->model.vision.outputs.kept_indices, "kept_indices");
  auto native_model = Generators::CreateModel(Generators::GetOrtEnv(), std::move(config));
  auto* vision_model = dynamic_cast<Generators::MultiModalLanguageModel*>(native_model.get());
  ASSERT_NE(vision_model, nullptr);
  ASSERT_EQ(vision_model->vision_device_->GetType(), GetParam() ? Generators::DeviceType::CUDA : Generators::DeviceType::CPU);
  std::vector<float> features;
  std::vector<int64_t> counts;
  std::vector<bool> retention;
  // Real ONNX If runs: all image rows survive, while the arithmetic video
  // branch keeps values > 4. This is dynamic-K plumbing, not EVS mathematics.
  for (const auto& means : {std::vector<float>{4}, std::vector<float>{3, 8, 10}, std::vector<float>{12, 2}}) {
    const bool video = means.size() != 1;
    std::vector<int64_t> expected_indices;
    for (size_t i = 0; i < means.size(); ++i)
      if (!video || means[i] > 4) expected_indices.push_back(static_cast<int64_t>(i));
    auto pixels = OrtValue::CreateTensor<float>(Ort::Allocator::GetWithDefaultOptions(),
                                                std::vector<int64_t>{static_cast<int64_t>(means.size()), 3, 2, 2});
    for (size_t i = 0; i < means.size(); ++i)
      std::fill_n(pixels->GetTensorMutableData<float>() + 12 * i, 12, means[i]);
    auto flag = OrtValue::CreateTensor<bool>(Ort::Allocator::GetWithDefaultOptions(), std::vector<int64_t>{});
    *flag->GetTensorMutableData<bool>() = video;
    std::array<const char*, 2> input_names{"pixel_values", "is_video"};
    std::array<OrtValue*, 2> values{pixels.get(), flag.get()};
    std::array<const char*, 3> output_names{"image_features", "retention_mask", "kept_indices"};
    std::array<OrtValue*, 3> outputs{};
    vision_model->vision_session_->Run(nullptr, input_names.data(), values.data(), values.size(),
                                       output_names.data(), outputs.data(), output_names.size());
    std::array<std::unique_ptr<OrtValue>, 3> owned{
        std::unique_ptr<OrtValue>{outputs[0]}, std::unique_ptr<OrtValue>{outputs[1]}, std::unique_ptr<OrtValue>{outputs[2]}};
    ASSERT_EQ(owned[0]->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{static_cast<int64_t>(expected_indices.size()), 2}));
    ASSERT_EQ(owned[1]->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{static_cast<int64_t>(means.size())}));
    ASSERT_EQ(owned[2]->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{static_cast<int64_t>(expected_indices.size())}));
    ASSERT_EQ(owned[0]->GetTensorTypeAndShapeInfo()->GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ASSERT_EQ(owned[1]->GetTensorTypeAndShapeInfo()->GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
    ASSERT_EQ(owned[2]->GetTensorTypeAndShapeInfo()->GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    counts.push_back(owned[0]->GetTensorTypeAndShapeInfo()->GetShape()[0]);
    for (size_t output = 0; output < owned.size(); ++output) {
      auto& tensor = *owned[output];
      auto buffer = Generators::ByteWrapTensor(Generators::DeviceForTensor(tensor, *vision_model->vision_device_), tensor);
      const auto bytes = buffer.CopyDeviceToCpu();
      if (output == 0) {
        const auto previous = features.size();
        features.resize(previous + bytes.size() / sizeof(float));
        std::memcpy(features.data() + previous, bytes.data(), bytes.size());
      } else if (output == 1) {
        for (size_t i = 0; i < means.size(); ++i) {
          bool keep;
          std::memcpy(&keep, bytes.data() + i * sizeof(bool), sizeof(bool));
          EXPECT_EQ(keep, !video || means[i] > 4);
          retention.push_back(keep);
        }
      } else {
        for (size_t i = 0; i < expected_indices.size(); ++i) {
          int64_t index;
          std::memcpy(&index, bytes.data() + i * sizeof(index), sizeof(index));
          EXPECT_EQ(index, expected_indices[i]);
        }
      }
    }
  }
  ASSERT_EQ(features, (std::vector<float>{4, 4, 8, 8, 10, 10, 12, 12}));
  ASSERT_EQ(counts, (std::vector<int64_t>{1, 2, 1}));
  std::vector<int32_t> ids;
  size_t media_position = 0;
  for (const int32_t token : {1, 18, 2, 18, 18, 18, 18, 18, 3}) {
    if (token != 18 || retention.at(media_position++)) ids.push_back(token);
  }
  ASSERT_EQ(ids, (std::vector<int32_t>{1, 18, 2, 18, 18, 18, 3}));
  native_model.reset();
  auto model = LoadCacheFixture(path, GetParam());
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 16);
  auto generator = OgaGenerator::Create(*model, *params);
  // The host removed discarded placeholders before search saw the prompt. The
  // generator gets no raw pixel_values/is_video and must not rerun vision.
  std::array<int64_t, 7> mask{1, 1, 1, 1, 1, 1, 1};
  auto ids_tensor = OgaTensor::Create(ids.data(), std::array<int64_t, 2>{1, 7});
  auto features_tensor = OgaTensor::Create(features.data(), std::array<int64_t, 2>{4, 2});
  auto mask_tensor = OgaTensor::Create(mask.data(), std::array<int64_t, 2>{1, 7});
  auto count_tensor = OgaTensor::Create(counts.data(), std::array<int64_t, 1>{3});
  auto inputs = OgaNamedTensors::Create();
  inputs->Set("input_ids", *ids_tensor);
  inputs->Set("image_features", *features_tensor);
  inputs->Set("attention_mask", *mask_tensor);
  inputs->Set("num_image_tokens", *count_tensor);
  generator->SetInputs(*inputs);

  std::vector<float> rows{1, 5, 4, 11, 14, 17, 9};
  int kv_total = 61;
  int state_total = 68;
  for (int step = 0; step < 4; ++step) {
    SCOPED_TRACE(step);
    for (const auto& [layer, scale] : {std::pair{0, 1}, std::pair{3, 3}}) {
      for (const auto* kind : {"conv_state", "ssm_state"}) {
        const bool ssm = std::string_view{kind} == "ssm_state";
        const auto name = "present." + std::to_string(layer) + "." + kind;
        auto tensor = generator->GetOutput(name.c_str());
        ASSERT_EQ(tensor->Type(), OgaElementType_float32);
        ASSERT_EQ(tensor->Shape(), (ssm ? std::vector<int64_t>{1, 2, 2, 2} : std::vector<int64_t>{1, 2, 3}));
        for (size_t i = 0; i < (ssm ? 8u : 6u); ++i)
          EXPECT_FLOAT_EQ(ReadCacheElement(*tensor, i), static_cast<float>((scale + (ssm ? 1 : 0)) * state_total));
      }
    }
    for (const auto* kind : {"key", "value"}) {
      auto tensor = generator->GetOutput((std::string{"present.2."} + kind).c_str());
      ASSERT_EQ(tensor->Type(), OgaElementType_float32);
      ASSERT_EQ(tensor->Shape(), (std::vector<int64_t>{1, 1, static_cast<int64_t>(rows.size()), 1}));
      for (size_t i = 0; i < rows.size(); ++i)
        EXPECT_FLOAT_EQ(ReadCacheElement(*tensor, i), rows[i] * (std::string_view{kind} == "key" ? 1 : 2));
    }
    const int32_t expected = 1 + (10 * state_total + 3 * kv_total) % 14;
    auto logits = generator->GetOutput("logits");
    ASSERT_EQ(logits->Type(), OgaElementType_float32);
    ASSERT_EQ(logits->Shape(), (std::vector<int64_t>{1, step == 0 ? 7 : 1, 16}));
    for (int i = 0; i < (step == 0 ? 112 : 16); ++i)
      EXPECT_FLOAT_EQ(ReadCacheElement(*logits, i), i % 16 == expected ? 1.0f : 0.0f);
    generator->GenerateNextToken();
    ASSERT_EQ(generator->GetSequenceCount(0), rows.size() + 1);
    EXPECT_EQ(generator->GetSequence(0).back(), expected);
    if (step < 3) {
      generator->GetLogits();
      const auto row = static_cast<int>(expected + rows.size());
      rows.push_back(static_cast<float>(row));
      kv_total += row;
      state_total += row + static_cast<int>(rows.size());
    }
  }
}

TEST_P(MultiModalCacheTest, PreparedFeaturesInferCountsIncludingEmptyMedia) {
  const std::string path = std::string{MODEL_PATH} + "cache-dispatch/multimodal-retention";
  auto model = LoadCacheFixture(path, GetParam());
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 12);
  for (const bool media : {false, true}) {
    auto generator = OgaGenerator::Create(*model, *params);
    std::array<int32_t, 3> ids{1, media ? 18 : 2, 3};
    std::array<float, 2> features{4, 4};
    std::array<int32_t, 3> mask{1, 1, 1};
    auto ids_tensor = OgaTensor::Create(ids.data(), std::array<int64_t, 2>{1, 3});
    auto features_tensor = OgaTensor::Create(features.data(), std::array<int64_t, 2>{media ? 1 : 0, 2});
    auto mask_tensor = OgaTensor::Create(mask.data(), std::array<int64_t, 2>{1, 3});
    auto inputs = OgaNamedTensors::Create();
    inputs->Set("input_ids", *ids_tensor);
    inputs->Set("image_features", *features_tensor);
    if (media) inputs->Set("attention_mask", *mask_tensor);
    generator->SetInputs(*inputs);
    const int expected = media ? 6 : 8;
    auto logits = generator->GetOutput("logits");
    ASSERT_EQ(logits->Shape(), (std::vector<int64_t>{1, 3, 16}));
    for (int i = 0; i < 48; ++i)
      EXPECT_FLOAT_EQ(ReadCacheElement(*logits, i), i % 16 == expected ? 1.0f : 0.0f);
    generator->GenerateNextToken();
    EXPECT_EQ(generator->GetSequenceCount(0), 4u);
    EXPECT_EQ(generator->GetSequence(0).back(), expected);
  }
}

TEST_P(MultiModalCacheTest, PreparedFeaturesRejectUncompactedOrAmbiguousInputs) {
  const std::string path = std::string{MODEL_PATH} + "cache-dispatch/multimodal-retention";
  auto model = LoadCacheFixture(path, GetParam());
  for (const auto& [invalid, message] : {
           std::pair{"width", "embedding input's shape and element type"},
           std::pair{"rank", "shape [kept_tokens, hidden_size]"},
           std::pair{"feature_type", "embedding input's shape and element type"},
           std::pair{"counts", "does not match"},
           std::pair{"negative_counts", "does not match"},
           std::pair{"count_type", "int64 vector"},
           std::pair{"placeholders", "exactly one image placeholder"},
           std::pair{"raw_media", "do not mix raw media"},
           std::pair{"dispatch", "do not mix raw media"},
           std::pair{"retention_mask", "do not mix raw media"},
           std::pair{"kept_indices", "do not mix raw media"},
           std::pair{"padding", "must be unpadded"},
           std::pair{"mask_length", "compacted input_ids length"},
           std::pair{"mask_value", "must be all ones"},
           std::pair{"mask_type", "metadata must use int32 or int64"},
           std::pair{"positions", "generated from compacted input_ids"},
           std::pair{"beams", "initial batch-1, beam-1"},
           std::pair{"batch", "shape [1, sequence_length]"},
           std::pair{"id_type", "CPU int32 input_ids"}}) {
    SCOPED_TRACE(invalid);
    const std::string_view error{invalid};
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 12);
    if (error == "beams") params->SetSearchOption("num_beams", 2);
    auto generator = OgaGenerator::Create(*model, *params);
    std::array<int32_t, 4> ids{error == "padding" ? 0 : 1, 18, 18, error == "placeholders" ? 18 : 3};
    std::array<int64_t, 4> wide_ids{1, 18, 18, 3};
    std::array<float, 4> features{4, 4, 8, 8};
    std::array<int64_t, 4> wide_features{4, 4, 8, 8};
    std::array<int64_t, 4> mask{1, 1, 1, error == "mask_value" ? 0 : 1};
    std::array<int64_t, 2> counts{error == "negative_counts" ? -1 : 1, error == "counts" ? 2 : 1};
    std::array<float, 12> pixels{};
    std::array<bool, 1> flag{true};
    auto ids_tensor = error == "id_type"
                          ? OgaTensor::Create(wide_ids.data(), std::array<int64_t, 2>{1, 4})
                          : OgaTensor::Create(ids.data(), error == "batch" ? std::array<int64_t, 2>{2, 2} : std::array<int64_t, 2>{1, 4});
    const auto feature_shape = error == "rank" ? std::vector<int64_t>{1, 2, 2}
                                               : std::vector<int64_t>{2, error == "width" ? 1 : 2};
    auto features_tensor = error == "feature_type" ? OgaTensor::Create(wide_features.data(), feature_shape)
                                                   : OgaTensor::Create(features.data(), feature_shape);
    auto count_tensor = error == "count_type" ? OgaTensor::Create(features.data(), std::array<int64_t, 1>{2})
                                              : OgaTensor::Create(counts.data(), std::array<int64_t, 1>{2});
    auto mask_tensor = error == "mask_type"
                           ? OgaTensor::Create(features.data(), std::array<int64_t, 2>{1, 4})
                           : OgaTensor::Create(mask.data(), std::array<int64_t, 2>{1, error == "mask_length" ? 3 : 4});
    auto pixels_tensor = OgaTensor::Create(pixels.data(), std::array<int64_t, 4>{1, 3, 2, 2});
    auto flag_tensor = OgaTensor::Create(flag.data(), std::array<int64_t, 0>{});
    auto inputs = OgaNamedTensors::Create();
    inputs->Set("input_ids", *ids_tensor);
    inputs->Set("image_features", *features_tensor);
    inputs->Set("num_image_tokens", *count_tensor);
    inputs->Set("attention_mask", *mask_tensor);
    if (error == "raw_media") inputs->Set("pixel_values", *pixels_tensor);
    if (error == "dispatch") inputs->Set("is_video", *flag_tensor);
    if (error == "retention_mask") inputs->Set("retention_mask", *flag_tensor);
    if (error == "kept_indices") inputs->Set("kept_indices", *count_tensor);
    if (error == "positions") inputs->Set("position_ids", *mask_tensor);
    try {
      generator->SetInputs(*inputs);
      FAIL() << "Invalid prepared inputs unexpectedly executed";
    } catch (const std::exception& e) {
      EXPECT_NE(std::string_view{e.what()}.find(message), std::string_view::npos) << e.what();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(CpuPreparedInputs, MultiModalCacheTest, ::testing::Values(false));
#if USE_CUDA
INSTANTIATE_TEST_SUITE_P(CudaPreparedInputs, MultiModalCacheTest, ::testing::Values(true));
#endif

std::shared_ptr<Generators::Model> LoadPhaseModel(const std::string& fixture, bool cuda,
                                                  std::string_view overlay = {}) {
  auto config = std::make_unique<Generators::Config>(
      std::string{MODEL_PATH} + "cache-dispatch/" + fixture, overlay);
  if (cuda) Generators::SetProviderOption(*config, "cuda", {}, {});
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), std::move(config));
  EXPECT_EQ(model->p_device_->GetType(), cuda ? Generators::DeviceType::CUDA : Generators::DeviceType::CPU);
  return model;
}

std::shared_ptr<Generators::Tensor> PhaseOffset(const Generators::Model& model, int64_t offset) {
  auto value = OrtValue::CreateTensor<int64_t>(model.allocator_cpu_, std::array<int64_t, 0>{});
  *value->GetTensorMutableData<int64_t>() = offset;
  return std::make_shared<Generators::Tensor>(std::move(value));
}

std::vector<uint8_t> PhaseOutputBytes(Generators::Generator& generator, const char* name) {
  auto* value = generator.state_->GetOutput(name);
  auto span = Generators::ByteWrapTensor(
      Generators::DeviceForTensor(*value, *generator.model_->p_device_), *value);
  const auto data = span.CopyDeviceToCpu();
  return {data.begin(), data.end()};
}

void ExpectPhase(Generators::Generator& generator, bool expected) {
  const auto bytes = PhaseOutputBytes(generator, "observed_phase");
  ASSERT_EQ(bytes.size(), sizeof(bool));
  bool actual;
  std::memcpy(&actual, bytes.data(), sizeof(actual));
  EXPECT_EQ(actual, expected);
}

void AppendPhaseTokens(Generators::Generator& generator, std::vector<int32_t> tokens) {
  generator.AppendTokens(Generators::cpu_span<const int32_t>{tokens.data(), tokens.size()});
}

class CachePhaseTest : public ::testing::TestWithParam<bool> {};

Generators::NamedTensors PhasePreparedText(const Generators::Model& model, const std::vector<int32_t>& tokens) {
  auto ids = OrtValue::CreateTensor<int32_t>(model.allocator_cpu_, std::array<int64_t, 2>{1, static_cast<int64_t>(tokens.size())});
  std::copy(tokens.begin(), tokens.end(), ids->GetTensorMutableData<int32_t>());
  auto features = OrtValue::CreateTensor(model.allocator_cpu_, std::array<int64_t, 2>{0, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  return {{"input_ids", std::make_shared<Generators::Tensor>(std::move(ids))},
          {"image_features", std::make_shared<Generators::Tensor>(std::move(features))}};
}

template <typename Action>
void ExpectCachePhaseError(Action action, std::string_view message) {
  try {
    action();
    ADD_FAILURE() << "Unsupported cache-phase operation was accepted";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string_view{e.what()}.find(message), std::string_view::npos) << e.what();
  }
}

TEST_P(CachePhaseTest, ScalarPhaseCarriesAllCachesAndIgnoresDirtyFreshStateAndOffsets) {
  for (const auto* fixture : {"phase", "phase-alias"}) {
    auto model = LoadPhaseModel(fixture, GetParam());
    auto params = Generators::CreateGeneratorParams(*model);
    auto generator = Generators::CreateGenerator(*model, *params);
    const auto& phase_name = model->config_->model.decoder.inputs.has_previous_state;
    const auto* phase_storage = generator->state_->GetInput(phase_name.c_str())->GetTensorData<bool>();
    EXPECT_FALSE(*phase_storage);
    auto offset = PhaseOffset(*model, 100);
    generator->extra_inputs_.push_back({"cache_offset", offset});
    for (size_t i = 0; i < generator->state_->inputs_.size(); ++i) {
      if (std::string_view{generator->state_->input_names_[i]}.find("state") == std::string_view::npos)
        continue;
      auto* input = generator->state_->inputs_[i];
      if (input->GetTensorTypeAndShapeInfo()->GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        continue;
      auto span = Generators::ByteWrapTensor(*model->p_device_kvcache_, *input);
      auto bytes = span.CopyDeviceToCpu();
      for (size_t j = 0; j < bytes.size(); j += sizeof(float)) {
        const float dirty = 123.0f;
        std::memcpy(bytes.data() + j, &dirty, sizeof(dirty));
      }
      span.CopyCpuToDevice();
    }
    // The first Update swaps nonshared buffers. Dirty both directions so the
    // actual first decoder input, not just an unconsumed allocation, is nonzero.
    for (size_t i = 0; i < generator->state_->outputs_.size(); ++i) {
      if (std::string_view{generator->state_->output_names_[i]}.find("state") == std::string_view::npos)
        continue;
      auto* output = generator->state_->outputs_[i];
      auto span = Generators::ByteWrapTensor(*model->p_device_kvcache_, *output);
      auto bytes = span.CopyDeviceToCpu();
      for (size_t j = 0; j < bytes.size(); j += sizeof(float)) {
        const float dirty = 123.0f;
        std::memcpy(bytes.data() + j, &dirty, sizeof(dirty));
      }
      span.CopyCpuToDevice();
    }
    AppendPhaseTokens(*generator, {2});
    ExpectPhase(*generator, false);
    auto bytes = PhaseOutputBytes(*generator, "observed_offset");
    int64_t observed_offset;
    std::memcpy(&observed_offset, bytes.data(), sizeof(observed_offset));
    EXPECT_EQ(observed_offset, 100);
    auto expect_states = [&](float total, size_t length) {
      for (const auto& [layer, scale] : {std::pair{0, 1}, std::pair{3, 3}}) {
        for (const auto* kind : {"conv_state", "recurrent_state"}) {
          const auto name = "present." + std::to_string(layer) + "." + kind;
          const auto data = PhaseOutputBytes(*generator, name.c_str());
          const float expected = total * static_cast<float>(scale + (std::string_view{kind} == "recurrent_state" ? 1 : 0));
          for (size_t i = 0; i < data.size(); i += sizeof(float)) {
            float actual;
            std::memcpy(&actual, data.data() + i, sizeof(actual));
            EXPECT_FLOAT_EQ(actual, expected);
          }
        }
      }
      for (const auto* kind : {"key", "value"}) {
        const auto data = PhaseOutputBytes(*generator, (std::string{"present.2."} + kind).c_str());
        ASSERT_EQ(data.size(), length * sizeof(float));
        for (size_t i = 0; i < length; ++i) {
          float actual;
          std::memcpy(&actual, data.data() + i * sizeof(float), sizeof(actual));
          EXPECT_FLOAT_EQ(actual, static_cast<float>(i + 2) * (std::string_view{kind} == "value" ? 2.0f : 1.0f));
        }
      }
      const auto data = PhaseOutputBytes(*generator, "logits");
      ASSERT_EQ(data.size(), 16 * sizeof(float));
      const auto selected = 1 + (13 * static_cast<int>(total)) % 14;
      for (size_t i = 0; i < 16; ++i) {
        float actual;
        std::memcpy(&actual, data.data() + i * sizeof(float), sizeof(actual));
        EXPECT_FLOAT_EQ(actual, i == static_cast<size_t>(selected) ? 1.0f : 0.0f);
      }
    };
    expect_states(2, 1);
    *offset->ort_tensor_->GetTensorMutableData<int64_t>() = 0;
    AppendPhaseTokens(*generator, {3});
    ExpectPhase(*generator, true);
    EXPECT_EQ(generator->state_->GetInput(phase_name.c_str())->GetTensorData<bool>(), phase_storage);
    expect_states(5, 2);
    bytes = PhaseOutputBytes(*generator, "observed_offset");
    std::memcpy(&observed_offset, bytes.data(), sizeof(observed_offset));
    EXPECT_EQ(observed_offset, 0);

    auto fresh = Generators::CreateGenerator(*model, *params);
    fresh->extra_inputs_.push_back({"cache_offset", offset});
    AppendPhaseTokens(*fresh, {1, 2, 3});
    ExpectPhase(*fresh, false);
  }
}

TEST_P(CachePhaseTest, FailedExecutionPoisonsUntilNewGenerator) {
  auto model = LoadPhaseModel("phase", GetParam());
  auto params = Generators::CreateGeneratorParams(*model);
  for (const bool previously_valid : {false, true}) {
    auto generator = Generators::CreateGenerator(*model, *params);
    auto offset = PhaseOffset(*model, 0);
    generator->extra_inputs_.push_back({"cache_offset", offset});
    if (previously_valid) {
      AppendPhaseTokens(*generator, {2});
      ExpectPhase(*generator, false);
    }
    *offset->ort_tensor_->GetTensorMutableData<int64_t>() = -1;
    EXPECT_THROW(AppendPhaseTokens(*generator, {2}), std::exception);
    EXPECT_EQ(*generator->state_->GetInput("has_previous_state")->GetTensorData<bool>(), previously_valid);
    *offset->ort_tensor_->GetTensorMutableData<int64_t>() = 0;
    try {
      AppendPhaseTokens(*generator, {2});
      FAIL() << "A failed managed execution was retried";
    } catch (const std::exception& e) {
      EXPECT_NE(std::string_view{e.what()}.find("create a new Generator"), std::string_view::npos) << e.what();
    }
    auto fresh = Generators::CreateGenerator(*model, *params);
    fresh->extra_inputs_.push_back({"cache_offset", offset});
    AppendPhaseTokens(*fresh, {2});
    ExpectPhase(*fresh, false);
  }
}

TEST_P(CachePhaseTest, PreparedBfloat16FeaturesUseDecoderSuccessPhase) {
  auto model = LoadPhaseModel("phase-multimodal", GetParam());
  auto params = Generators::CreateGeneratorParams(*model);
  for (const int64_t length : {1, 3}) {
    for (const bool image : {false, true}) {
      auto generator = Generators::CreateGenerator(*model, *params);
      auto ids = OrtValue::CreateTensor<int32_t>(model->allocator_cpu_, std::array<int64_t, 2>{1, length});
      std::fill_n(ids->GetTensorMutableData<int32_t>(), static_cast<size_t>(length), 2);
      if (image) ids->GetTensorMutableData<int32_t>()[0] = 18;
      auto features = OrtValue::CreateTensor(model->allocator_cpu_, std::array<int64_t, 2>{image ? 1 : 0, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
      if (image) {
        const std::array<uint16_t, 2> bits{0x4080, 0x4080};  // Native BF16 4.0 features.
        std::memcpy(features->GetTensorMutableData<void>(), bits.data(), sizeof(bits));
      }
      Generators::NamedTensors inputs{
          {"input_ids", std::make_shared<Generators::Tensor>(std::move(ids))},
          {"image_features", std::make_shared<Generators::Tensor>(std::move(features))}};
      generator->SetInputs(inputs);
      ExpectPhase(*generator, false);
      int kv_total = 2 * static_cast<int>(length) + static_cast<int>(length * (length - 1) / 2) + (image ? 2 : 0);
      int state_total = kv_total + static_cast<int>(length);
      std::vector<float> rows;
      for (int64_t i = 0; i < length; ++i)
        rows.push_back(static_cast<float>((image && i == 0 ? 4 : 2) + i));
      auto expect_outputs = [&](int64_t query_length) {
        for (const auto& [layer, scale] : {std::pair{0, 1}, std::pair{3, 3}}) {
          for (const auto* kind : {"conv_state", "ssm_state"}) {
            const auto data = PhaseOutputBytes(*generator, ("present." + std::to_string(layer) + "." + kind).c_str());
            const auto expected = static_cast<float>((scale + (std::string_view{kind} == "ssm_state" ? 1 : 0)) * state_total);
            for (size_t i = 0; i < data.size(); i += sizeof(float)) {
              float actual;
              std::memcpy(&actual, data.data() + i, sizeof(actual));
              EXPECT_FLOAT_EQ(actual, expected);
            }
          }
        }
        for (const auto* kind : {"key", "value"}) {
          const auto data = PhaseOutputBytes(*generator, (std::string{"present.2."} + kind).c_str());
          ASSERT_EQ(data.size(), rows.size() * sizeof(float));
          for (size_t i = 0; i < rows.size(); ++i) {
            float actual;
            std::memcpy(&actual, data.data() + i * sizeof(float), sizeof(actual));
            EXPECT_FLOAT_EQ(actual, rows[i] * (std::string_view{kind} == "value" ? 2.0f : 1.0f));
          }
        }
        const int selected = 1 + (10 * state_total + 3 * kv_total) % 14;
        const auto logits = PhaseOutputBytes(*generator, "logits");
        ASSERT_EQ(logits.size(), static_cast<size_t>(query_length) * 16 * sizeof(float));
        for (size_t i = 0; i < logits.size() / sizeof(float); ++i) {
          float actual;
          std::memcpy(&actual, logits.data() + i * sizeof(float), sizeof(actual));
          EXPECT_FLOAT_EQ(actual, i % 16 == static_cast<size_t>(selected) ? 1.0f : 0.0f);
        }
      };
      expect_outputs(length);
      const int first_token = 1 + (10 * state_total + 3 * kv_total) % 14;
      generator->GenerateNextToken();
      generator->GenerateNextToken();
      ExpectPhase(*generator, true);
      rows.push_back(static_cast<float>(first_token + length));
      kv_total += first_token + static_cast<int>(length);
      state_total += first_token + 2 * static_cast<int>(length) + 1;
      expect_outputs(1);
      const auto first = PhaseOutputBytes(*generator, "present.0.ssm_state");
      ASSERT_EQ(first.size(), 8 * sizeof(float));
      EXPECT_EQ(generator->state_->GetOutput("present.0.ssm_state")->GetTensorTypeAndShapeInfo()->GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      EXPECT_THROW(generator->RewindToLength(0), std::exception);
    }
  }
}

TEST_P(CachePhaseTest, PreparedPromptValidationFailurePoisonsBeforeDecoderPreparation) {
  auto model = LoadPhaseModel("phase-multimodal", GetParam());
  auto params = Generators::CreateGeneratorParams(*model);
  for (const bool device_input : {false, true}) {
    auto generator = Generators::CreateGenerator(*model, *params);
    auto rejected = PhasePreparedText(*model, {model->config_->model.pad_token_id});
    auto device_tokens = params->p_device->Allocate<int32_t>(1);
    if (device_input) {
      rejected.erase("input_ids");
      generator->SetInputs(rejected);
      device_tokens.CpuSpan()[0] = model->config_->model.pad_token_id;
      device_tokens.CopyCpuToDevice();
      ExpectCachePhaseError([&] { generator->AppendTokens(device_tokens); }, "must be unpadded");
    } else {
      ExpectCachePhaseError([&] { generator->SetInputs(rejected); }, "must be unpadded");
    }
    // Search history was already mutated, even though the decoder never started.
    ASSERT_EQ(generator->TokenCount(), size_t{1});
    EXPECT_FALSE(*generator->state_->GetInput("has_previous_state")->GetTensorData<bool>());
    auto corrected = PhasePreparedText(*model, {2});
    if (device_input) {
      device_tokens.CpuSpan()[0] = 2;
      device_tokens.CopyCpuToDevice();
      ExpectCachePhaseError([&] { generator->AppendTokens(device_tokens); }, "create a new Generator");
    } else {
      ExpectCachePhaseError([&] { generator->SetInputs(corrected); }, "create a new Generator");
    }
    EXPECT_EQ(generator->TokenCount(), size_t{1});
    ExpectCachePhaseError([&] { AppendPhaseTokens(*generator, {2}); }, "create a new Generator");
    ExpectCachePhaseError([&] { generator->GenerateNextToken(); }, "create a new Generator");
    ExpectCachePhaseError([&] { generator->GetLogits(); }, "create a new Generator");
    EXPECT_EQ(generator->TokenCount(), size_t{1});
    auto fresh = Generators::CreateGenerator(*model, *params);
    fresh->SetInputs(corrected);
    ExpectPhase(*fresh, false);
    EXPECT_EQ(fresh->TokenCount(), size_t{1});
    const auto logits = PhaseOutputBytes(*fresh, "logits");
    ASSERT_EQ(logits.size(), 16 * sizeof(float));
    for (size_t i = 0; i < 16; ++i) {
      float actual;
      std::memcpy(&actual, logits.data() + i * sizeof(float), sizeof(actual));
      EXPECT_FLOAT_EQ(actual, i == 9 ? 1.0f : 0.0f);
    }
    AppendPhaseTokens(*fresh, {3});
    ExpectPhase(*fresh, true);
  }
}

TEST_P(CachePhaseTest, ActualEmbeddingFailurePoisonsInitialAndCachedManagedOperations) {
  auto model = LoadPhaseModel("phase-multimodal-embedding-failure", GetParam());
  auto params = Generators::CreateGeneratorParams(*model);
  for (const bool previously_valid : {false, true}) {
    auto generator = Generators::CreateGenerator(*model, *params);
    if (previously_valid) {
      generator->SetInputs(PhasePreparedText(*model, {2}));
      ExpectPhase(*generator, false);
      AppendPhaseTokens(*generator, {3});
      ExpectPhase(*generator, true);
      ExpectCachePhaseError([&] { AppendPhaseTokens(*generator, {13}); }, "embedding_failure_reshape");
    } else {
      ExpectCachePhaseError([&] { generator->SetInputs(PhasePreparedText(*model, {13})); }, "embedding_failure_reshape");
    }
    const size_t rejected_length = previously_valid ? 3 : 1;
    ASSERT_EQ(generator->TokenCount(), rejected_length);
    EXPECT_EQ(*generator->state_->GetInput("has_previous_state")->GetTensorData<bool>(), previously_valid);
    ExpectCachePhaseError([&] { AppendPhaseTokens(*generator, {2}); }, "create a new Generator");
    ExpectCachePhaseError([&] { generator->GenerateNextToken(); }, "create a new Generator");
    ExpectCachePhaseError([&] { generator->GetLogits(); }, "create a new Generator");
    EXPECT_EQ(generator->TokenCount(), rejected_length);
    auto fresh = Generators::CreateGenerator(*model, *params);
    fresh->SetInputs(PhasePreparedText(*model, {2}));
    ExpectPhase(*fresh, false);
    AppendPhaseTokens(*fresh, {3});
    ExpectPhase(*fresh, true);
  }
}

TEST_P(CachePhaseTest, UnsupportedOperationsAndDuplicateOwnershipReject) {
  auto model = LoadPhaseModel("phase", GetParam());
  ExpectCachePhaseError([&] { Generators::Engine::CreateDependencies(model); }, "does not support Engine");
  ExpectCachePhaseError([&] { Generators::Engine engine(model, {}); }, "does not support Engine");
  for (const auto* unsupported : {"capture", "chunking", "batch", "beams", "ngram"}) {
    auto params = Generators::CreateGeneratorParams(*model);
    if (std::string_view{unsupported} == "capture") params->use_graph_capture = true;
    if (std::string_view{unsupported} == "chunking") params->search.chunk_size = 1;
    if (std::string_view{unsupported} == "batch") params->search.batch_size = 2;
    if (std::string_view{unsupported} == "beams") params->search.num_beams = 2;
    if (std::string_view{unsupported} == "ngram") params->speculative.ngram_size = 2;
    EXPECT_THROW(Generators::CreateGenerator(*model, *params), std::exception);
  }
  auto params = Generators::CreateGeneratorParams(*model);
  auto generator = Generators::CreateGenerator(*model, *params);
  auto offset = PhaseOffset(*model, 0);
  generator->extra_inputs_.push_back({"cache_offset", offset});
  AppendPhaseTokens(*generator, {2});
  EXPECT_THROW(generator->RewindToLength(0), std::exception);
  EXPECT_THROW(generator->RewindToLength(generator->TokenCount()), std::exception);
  EXPECT_THROW(generator->SnapshotState(), std::exception);
  EXPECT_THROW(generator->state_->CropToAccepted(1, 0), std::exception);
  EXPECT_THROW(generator->CropToAccepted(1, 0), std::exception);
  ExpectCachePhaseError([&] { AppendPhaseTokens(*generator, {2, 3}); }, "cached decoding requires S==1");
  auto duplicate = Generators::CreateGenerator(*model, *params);
  auto flag = OrtValue::CreateTensor<bool>(model->allocator_cpu_, std::array<int64_t, 0>{});
  duplicate->extra_inputs_.push_back({"has_previous_state", std::make_shared<Generators::Tensor>(std::move(flag))});
  ExpectCachePhaseError([&] { AppendPhaseTokens(*duplicate, {2}); }, "runtime-owned");
  auto multimodal = LoadPhaseModel("phase-multimodal", GetParam());
  auto mm_params = Generators::CreateGeneratorParams(*multimodal);
  auto mm_generator = Generators::CreateGenerator(*multimodal, *mm_params);
  EXPECT_THROW(mm_generator->RewindToLength(0), std::exception);
  ExpectCachePhaseError([&] { mm_generator->state_->RewindTo(0); }, "does not support rewind");
  ExpectCachePhaseError([&] { AppendPhaseTokens(*mm_generator, {2}); }, "requires prepared image_features");
  auto bool_value = OrtValue::CreateTensor<bool>(multimodal->allocator_cpu_, std::array<int64_t, 0>{});
  Generators::NamedTensors duplicate_prepared{
      {"has_previous_state", std::make_shared<Generators::Tensor>(std::move(bool_value))}};
  auto ids = OrtValue::CreateTensor<int32_t>(multimodal->allocator_cpu_, std::array<int64_t, 2>{1, 1});
  *ids->GetTensorMutableData<int32_t>() = 2;
  duplicate_prepared["input_ids"] = std::make_shared<Generators::Tensor>(std::move(ids));
  auto features = OrtValue::CreateTensor(multimodal->allocator_cpu_, std::array<int64_t, 2>{0, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  duplicate_prepared["image_features"] = std::make_shared<Generators::Tensor>(std::move(features));
  auto mm_duplicate = Generators::CreateGenerator(*multimodal, *mm_params);
  ExpectCachePhaseError([&] { mm_duplicate->SetInputs(duplicate_prepared); }, "runtime-owned");
}

TEST_P(CachePhaseTest, ParserAndActualDecoderMetadataValidation) {
  Generators::Config defaults;
  EXPECT_TRUE(defaults.model.decoder.inputs.has_previous_state.empty());
  for (const auto* invalid : {"true", "1", "[]", "{}"})
    EXPECT_THROW(Generators::OverlayConfig(defaults,
                                           std::string{R"({"model":{"decoder":{"inputs":{"has_previous_state":)"} + invalid + "}}}}"),
                 std::exception);
  for (const auto* alias : {"missing", "cache_offset"}) {
    auto model = LoadPhaseModel("phase", GetParam(),
                                std::string{R"({"model":{"decoder":{"inputs":{"has_previous_state":")"} + alias + R"("}}}})");
    auto params = Generators::CreateGeneratorParams(*model);
    ExpectCachePhaseError([&] { Generators::CreateGenerator(*model, *params); },
                          std::string_view{alias} == "missing" ? "decoder input not found" : "must be a BOOL scalar []");
  }
  auto wrong_rank = LoadPhaseModel("phase-rank", GetParam());
  ExpectCachePhaseError([&] { Generators::CreateGenerator(*wrong_rank, *Generators::CreateGeneratorParams(*wrong_rank)); }, "must be a BOOL scalar []");
  auto vision_only = LoadPhaseModel("phase-multimodal", GetParam(),
                                    R"({"model":{"decoder":{"inputs":{"has_previous_state":"is_video"}}}})");
  EXPECT_TRUE(vision_only->session_info_.HasInput("is_video"));
  ExpectCachePhaseError([&] { Generators::CreateGenerator(*vision_only, *Generators::CreateGeneratorParams(*vision_only)); }, "decoder input not found");
}

INSTANTIATE_TEST_SUITE_P(CpuCachePhase, CachePhaseTest, ::testing::Values(false));
#if USE_CUDA
INSTANTIATE_TEST_SUITE_P(CudaCachePhase, CachePhaseTest, ::testing::Values(true));
#endif

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
