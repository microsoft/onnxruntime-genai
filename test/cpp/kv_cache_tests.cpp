// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "models/io/dynamic_kv_cache.h"
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

struct CacheTestState : Generators::State {
  using State::State;

  Generators::DeviceSpan<float> Run(
      int, Generators::DeviceSpan<int32_t>&, Generators::DeviceSpan<int32_t>) override {
    return {};
  }
};

struct BFloat16Cache : Generators::DynamicKeyValueCache {
  BFloat16Cache(Generators::State& state, bool per_layer_shapes)
      : DynamicKeyValueCache{state} {
    type_ = Ort::TypeToTensorType<Ort::BFloat16_t>;
    if (per_layer_shapes) {
      shape_[2] = 8;
      layer_shapes_.push_back({shape_[0], 1, 8, shape_[1] * shape_[3]});
    }
    auto empty_shape = per_layer_shapes ? layer_shapes_[0] : shape_;
    empty_shape[2] = 0;
    empty_past_ = OrtValue::CreateTensor(Allocator(), empty_shape, type_);
  }
};

class BFloat16CacheTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    model_ = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
    params_ = Generators::CreateGeneratorParams(*model_);
    params_->search.num_beams = 2;
    params_->search.past_present_share_buffer = false;
    state_ = std::make_unique<CacheTestState>(*params_, *model_);
    cache_ = std::make_unique<BFloat16Cache>(*state_, GetParam());
    cache_->Add();
    cache_->Update({}, 3);
    for (auto& present : cache_->GetPresents()) {
      auto values = Generators::WrapTensor<Ort::BFloat16_t>(*model_->p_device_kvcache_, *present);
      auto cpu = values.CpuSpan();
      for (size_t i = 0; i < cpu.size(); ++i) {
        cpu[i].value = static_cast<uint16_t>(0x3f00 + i + expected_.size() * 128);
      }
      expected_.emplace_back(cpu.begin(), cpu.end());
      values.CopyCpuToDevice();
    }
  }

  std::shared_ptr<Generators::Model> model_;
  std::shared_ptr<Generators::GeneratorParams> params_;
  std::unique_ptr<CacheTestState> state_;
  std::unique_ptr<BFloat16Cache> cache_;
  std::vector<std::vector<Ort::BFloat16_t>> expected_;
};

TEST_P(BFloat16CacheTest, RewindPreservesBfloat16PrefixPerHead) {
  cache_->RewindTo(2);
  for (size_t tensor = 0; tensor < expected_.size(); ++tensor) {
    auto& input = *state_->inputs_[tensor];
    ASSERT_EQ(input.GetTensorTypeAndShapeInfo()->GetElementType(),
              Ort::TypeToTensorType<Ort::BFloat16_t>);
    const auto shape = input.GetTensorTypeAndShapeInfo()->GetShape();
    ASSERT_EQ(shape[2], 2);
    auto values = Generators::WrapTensor<Ort::BFloat16_t>(
                      *model_->p_device_kvcache_, input)
                      .CopyDeviceToCpu();
    const size_t width = static_cast<size_t>(shape[3]);
    for (size_t head = 0; head < static_cast<size_t>(shape[0] * shape[1]); ++head) {
      for (size_t i = 0; i < 2 * width; ++i) {
        EXPECT_EQ(values[head * 2 * width + i].value,
                  expected_[tensor][head * 3 * width + i].value);
      }
    }
  }
}

TEST_P(BFloat16CacheTest, BeamReorderPreservesBfloat16Bits) {
  auto indices = model_->p_device_kvcache_->Allocate<int32_t>(2);
  indices.CpuSpan()[0] = 1;
  indices.CpuSpan()[1] = 0;
  indices.CopyCpuToDevice();
  cache_->Update(indices, 4);
  for (size_t tensor = 0; tensor < expected_.size(); ++tensor) {
    auto& input = *state_->inputs_[tensor];
    ASSERT_EQ(input.GetTensorTypeAndShapeInfo()->GetElementType(),
              Ort::TypeToTensorType<Ort::BFloat16_t>);
    ASSERT_EQ(input.GetTensorTypeAndShapeInfo()->GetShape()[2], 3);
    auto values = Generators::WrapTensor<Ort::BFloat16_t>(
                      *model_->p_device_kvcache_, input)
                      .CopyDeviceToCpu();
    const size_t beam_size = values.size() / 2;
    for (size_t beam = 0; beam < 2; ++beam) {
      for (size_t i = 0; i < beam_size; ++i) {
        EXPECT_EQ(values[beam * beam_size + i].value,
                  expected_[tensor][(1 - beam) * beam_size + i].value);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(UniformAndPerLayer, BFloat16CacheTest, testing::Bool());

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
