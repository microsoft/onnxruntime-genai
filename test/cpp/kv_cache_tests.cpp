// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "generator/generators.h"
#include "models/io/model_managed_kv_cache.h"
#include "models/io/static_kv_cache.h"
#include "models/io/windowed_kv_cache.h"
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

struct ModelManagedTestState : Generators::State {
  using Generators::State::State;

  Generators::DeviceSpan<float> Run(int, Generators::DeviceSpan<int32_t>&, Generators::DeviceSpan<int32_t>) override {
    return {};
  }
};

TEST(KvCacheTests, ModelManagedCacheQueuesRewindOption) {
  CacheTestModel model{false};
  auto params = std::make_shared<Generators::GeneratorParams>(*model.config_);
  ModelManagedTestState state{*params, model};

  // Constructing the cache resets the session-held state for the new generator.
  Generators::ModelManagedKeyValueCache cache{state};
  ASSERT_EQ(state.ep_dynamic_options_next_run_.size(), 1u);
  EXPECT_EQ(state.ep_dynamic_options_next_run_[0].first, "kvcache_rewind");
  EXPECT_EQ(state.ep_dynamic_options_next_run_[0].second, "0");

  cache.RewindTo(7);
  ASSERT_EQ(state.ep_dynamic_options_next_run_.size(), 2u);
  EXPECT_EQ(state.ep_dynamic_options_next_run_[1].first, "kvcache_rewind");
  EXPECT_EQ(state.ep_dynamic_options_next_run_[1].second, "7");
}

TEST(RewindTests, RejectsRewindThatSplitsThePrompt) {
  EXPECT_TRUE(Generators::RewindSplitsPrompt(3, 10));   // inside the prompt: reject
  EXPECT_FALSE(Generators::RewindSplitsPrompt(0, 10));  // full rewind: allowed
  EXPECT_FALSE(Generators::RewindSplitsPrompt(10, 10)); // at the boundary: allowed
  EXPECT_FALSE(Generators::RewindSplitsPrompt(15, 10)); // past the prompt: allowed
  EXPECT_FALSE(Generators::RewindSplitsPrompt(3, 0));   // no prompt to protect: allowed
}

TEST(RewindTests, RejectsRewindPastEvictedSlidingWindowPositions) {
  EXPECT_FALSE(Generators::CanRewindWindowedKvCache(64, 100, 10));  // evicted: reject
  EXPECT_TRUE(Generators::CanRewindWindowedKvCache(64, 100, 0));    // full rewind needs no history: allowed
  EXPECT_NO_THROW(Generators::CheckWindowedKvCacheRewind(64, 100, 0));
  EXPECT_TRUE(Generators::CanRewindWindowedKvCache(64, 100, 100));  // at current length: allowed
  EXPECT_TRUE(Generators::CanRewindWindowedKvCache(64, 50, 10));    // window never filled: allowed
  EXPECT_TRUE(Generators::CanRewindWindowedKvCache(0, 100, 10));    // no window configured: allowed
}

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
