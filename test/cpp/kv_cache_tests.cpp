// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>

#include <gtest/gtest.h>

#include "models/io/shared_kv_cache.h"
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

struct CacheTestState : Generators::State {
  CacheTestState(const Generators::GeneratorParams& params, const Generators::Model& model)
      : State{params, model} {}

  Generators::DeviceSpan<float> Run(int, Generators::DeviceSpan<int32_t>&,
                                    Generators::DeviceSpan<int32_t>) override {
    return {};
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

TEST(KvCacheTests, KeepsFullWindowWhenMaxLengthIsShorter) {
  CacheTestModel model{false};
  Generators::Config::Search search;
  for (const int max_length : {16, 63, 64}) {
    EXPECT_EQ(Generators::GetWindowedKeyValueCacheSize(model, search, max_length), 64);
  }
  EXPECT_EQ(Generators::GetWindowedKeyValueCacheSize(model, search, 70), 70);
  EXPECT_EQ(Generators::GetWindowedKeyValueCacheSize(model, search, 80), 80);
}

TEST(KvCacheTests, LimitsSlackWithoutTruncatingWindow) {
  CacheTestModel model{false};
  Generators::Config::Search search;
  search.chunk_size = 33;
  EXPECT_EQ(Generators::GetWindowedKeyValueCacheSize(model, search, 16), 64);
  EXPECT_EQ(Generators::GetWindowedKeyValueCacheSize(model, search, 80), 80);
  EXPECT_EQ(Generators::GetWindowedKeyValueCacheSize(model, search, 4096), 96);

  model.config_->model.decoder.sliding_window->cache_slack = 48;
  EXPECT_EQ(Generators::GetWindowedKeyValueCacheSize(model, search, 16), 64);
  EXPECT_EQ(Generators::GetWindowedKeyValueCacheSize(model, search, 100), 100);
  EXPECT_EQ(Generators::GetWindowedKeyValueCacheSize(model, search, 4096), 112);
}

TEST(KvCacheTests, PreservesWindowCapacityWithGraphDerivedHeadShapes) {
  for (const bool explicit_layers : {false, true}) {
    CacheTestModel model{false};
    auto& decoder = model.config_->model.decoder;
    decoder.num_hidden_layers = 1;
    decoder.num_key_value_heads = 1;
    decoder.head_size = 128;
    if (explicit_layers)
      decoder.sliding_window->layers = {0};
    auto session = OrtSession::Create(
        Generators::GetOrtEnv(), std::filesystem::path{MODEL_PATH "gemma4/dummy_text.onnx"}.c_str(),
        model.session_options_.get());
    model.session_info_.Add(*session);
    auto params = std::make_shared<Generators::GeneratorParams>(*model.config_);
    params->search.max_length = 16;
    params->search.past_present_share_buffer = true;
    CacheTestState state{*params, model};
    Generators::SharedKeyValueCache cache{state};

    ASSERT_EQ(cache.GetPresents().size(), 2U);
    for (const auto& present : cache.GetPresents()) {
      EXPECT_EQ(present->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{1, 4, 64, 256}));
    }
    cache.Update({}, 16);
    EXPECT_NO_THROW(cache.RewindTo(4));
  }
}

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
