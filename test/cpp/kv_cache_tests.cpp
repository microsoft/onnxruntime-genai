// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <string_view>

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

TEST(KvCacheTests, ComposeKeyValueNameAcceptsExactlyOneDecimalPlaceholder) {
  EXPECT_EQ(Generators::ComposeKeyValueName("past.%d.key", 3), "past.3.key");
  EXPECT_EQ(Generators::ComposeKeyValueName("%d_suffix", -7), "-7_suffix");
}

TEST(KvCacheTests, ComposeKeyValueNameRejectsUnsafeFormatSyntax) {
  for (const std::string_view value : {
           "past",
           "past_%",
           "past_%n",
           "past_%s",
           "past_%p",
           "past_%i",
           "past_%03d",
           "past_%d%s",
           "past_%d%n",
           "past_%d_%d",
       }) {
    EXPECT_THROW(
        Generators::ComposeKeyValueName(std::string{value}, 3),
        std::runtime_error)
        << value;
  }
}

TEST(KvCacheTests, ComposeKeyValueNamePreservesLengthLimits) {
  EXPECT_EQ(Generators::ComposeKeyValueName(std::string(62, 'a') + "%d", 3).size(), 63u);
  EXPECT_THROW(
      Generators::ComposeKeyValueName(std::string(63, 'a') + "%d", 3),
      std::runtime_error);
  EXPECT_EQ(Generators::ComposeKeyValueName(std::string(126, 'a') + "%d", 3, 128, "test name").size(), 127u);
}

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
