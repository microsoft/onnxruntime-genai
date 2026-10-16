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

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
