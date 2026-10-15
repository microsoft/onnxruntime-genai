// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

#define OGA_USE_SPAN 1
#include "ort_genai.h"
#include "models/io/conv_kv_cache.h"
#include "models/io/static_kv_cache.h"
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

TEST_P(CacheDispatchTest, PrefillDecodeAndFreshStateCarrySparseStates) {
  const bool recurrent = std::string_view{GetParam()} == "recurrent";
  const std::string model_path = std::string{MODEL_PATH} + "cache-dispatch/" + GetParam();
  auto internal_model = Generators::CreateModel(Generators::GetOrtEnv(), model_path.c_str());
  EXPECT_EQ(Generators::HasConvKeyValueCache(*internal_model), !recurrent);
  // Do not strip the metadata to make recurrent auto-discovery work.
  EXPECT_EQ(internal_model->config_->model.decoder.layer_types.size(), recurrent ? 4u : 3u);

  auto model = OgaModel::Create(model_path.c_str());
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 16);
  auto generator = OgaGenerator::Create(*model, *params);
  const int attention_index = recurrent ? 2 : 1;
  const int final_fixed_index = recurrent ? 3 : 2;
  const int score_scale = recurrent ? 13 : 7;

  auto expect_state = [&](int layer, const char* kind, float expected) {
    const auto name = "present." + std::to_string(layer) + "." + kind;
    auto tensor = generator->GetOutput(name.c_str());
    ASSERT_EQ(tensor->Type(), OgaElementType_float32);
    ASSERT_EQ(tensor->Shape(), (std::string_view{kind} == "conv_state"
                                    ? std::vector<int64_t>{1, 2, 3}
                                    : std::vector<int64_t>{1, 2, 2, 2}));
    size_t count = 1;
    for (auto dim : tensor->Shape()) count *= static_cast<size_t>(dim);
    const auto* data = static_cast<const float*>(tensor->Data());
    for (size_t i = 0; i < count; ++i) EXPECT_FLOAT_EQ(data[i], expected) << name;
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
      ASSERT_EQ(tensor->Type(), OgaElementType_float32);
      ASSERT_EQ(tensor->Shape(), (std::vector<int64_t>{1, 1, static_cast<int64_t>(tokens.size()), 1}));
      const auto* data = static_cast<const float*>(tensor->Data());
      const float scale = std::string_view{kind} == "key" ? 1.0f : 2.0f;
      for (size_t i = 0; i < tokens.size(); ++i) EXPECT_FLOAT_EQ(data[i], scale * tokens[i]);
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

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
