// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "models/io/shared_kv_cache.h"
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

TEST(KvCacheTests, LogicalOutputViewSharesBackingAllocation) {
  CacheTestModel model{false};
  constexpr std::array<int64_t, 4> backing_shape{1, 2, 96, 4};
  auto backing = OrtValue::CreateTensor(
      model.allocator_cpu_, backing_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  auto output_view = Generators::KeyValueCacheDetail::CreateLogicalKeyValueCacheOutputView(
      *backing, 34);

  EXPECT_EQ(output_view->GetTensorMutableRawData(), backing->GetTensorMutableRawData());
  EXPECT_EQ(output_view->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{1, 2, 34, 4}));
  EXPECT_EQ(backing->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{1, 2, 96, 4}));
}

TEST(KvCacheTests, LogicalOutputViewRejectsLengthBeyondCapacity) {
  CacheTestModel model{false};
  constexpr std::array<int64_t, 4> backing_shape{1, 2, 96, 4};
  auto backing = OrtValue::CreateTensor(
      model.allocator_cpu_, backing_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  EXPECT_THROW(Generators::KeyValueCacheDetail::CreateLogicalKeyValueCacheOutputView(
                   *backing, 97),
               std::runtime_error);
}

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
