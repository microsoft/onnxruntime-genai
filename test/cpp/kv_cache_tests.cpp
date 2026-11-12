// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include <gtest/gtest.h>

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

struct DenseCacheTestState : Generators::State {
  using State::State;

  Generators::DeviceSpan<float> Run(
      int, Generators::DeviceSpan<int32_t>&, Generators::DeviceSpan<int32_t>) override {
    return {};
  }
};

struct DenseCachePlacementTests : testing::TestWithParam<bool> {
  void SetUp() override {
    model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "longrope-context");
    params = Generators::CreateGeneratorParams(*model);
    params->search.max_length = 16;
    params->search.past_present_share_buffer = GetParam();
    state = std::make_unique<DenseCacheTestState>(*params, *model);
    cache = Generators::CreateStandardKeyValueCache(*state);
    ASSERT_NE(cache, nullptr);
    cache->Add();
  }

  void ExpectLocations() {
    const auto& session = model->session_info_.GetInputSession(state->input_names_.front());
    const auto input_names = session.GetInputNames();
    const auto output_names = session.GetOutputNames();
    const auto input_locations = session.GetMemoryInfoForInputs();
    const auto output_locations = session.GetMemoryInfoForOutputs();
    for (size_t i = 0; i < state->inputs_.size(); ++i) {
      SCOPED_TRACE(state->input_names_[i]);
      auto input = std::find(input_names.begin(), input_names.end(), state->input_names_[i]);
      auto output = std::find(output_names.begin(), output_names.end(), state->output_names_[i]);
      ASSERT_NE(input, input_names.end());
      ASSERT_NE(output, output_names.end());
      EXPECT_TRUE(state->inputs_[i]->GetTensorMemoryInfo() == *input_locations[input - input_names.begin()]);
      EXPECT_TRUE(state->outputs_[i]->GetTensorMemoryInfo() == *output_locations[output - output_names.begin()]);
      if (GetParam())
        EXPECT_EQ(state->inputs_[i], state->outputs_[i]);
      else
        EXPECT_NE(state->inputs_[i], state->outputs_[i]);
    }
  }

  std::shared_ptr<Generators::Model> model;
  std::shared_ptr<Generators::GeneratorParams> params;
  std::unique_ptr<DenseCacheTestState> state;
  std::unique_ptr<Generators::KeyValueCache> cache;
};

TEST_P(DenseCachePlacementTests, KeepsSessionLocationsAcrossDecodeAndRewind) {
  ExpectLocations();
  for (int length : {4, 5}) {
    cache->Update({}, length);
    ExpectLocations();
    for (size_t i = 0; i < state->outputs_.size(); ++i) {
      auto* data = state->outputs_[i]->GetTensorMutableData<float>();
      const auto count = state->outputs_[i]->GetTensorTypeAndShapeInfo()->GetElementCount();
      for (size_t j = 0; j < count; ++j)
        data[j] = static_cast<float>(100 * i + j + 1);
    }
  }

  cache->RewindTo(2);
  ExpectLocations();
  for (size_t i = 0; i < state->inputs_.size(); ++i) {
    const auto* data = state->inputs_[i]->GetTensorData<float>();
    EXPECT_EQ(data[0], static_cast<float>(100 * i + 1));
    EXPECT_EQ(data[1], static_cast<float>(100 * i + 2));
  }

  cache->RewindTo(0);
  ExpectLocations();
  if (!GetParam()) {
    for (auto* input : state->inputs_)
      EXPECT_EQ(input->GetTensorTypeAndShapeInfo()->GetElementCount(), 0);
  }
  cache->Update({}, 3);
  ExpectLocations();
}

INSTANTIATE_TEST_SUITE_P(SharedAndSeparateBuffers, DenseCachePlacementTests, testing::Bool());

struct CacheLocationValidator : Generators::DefaultKeyValueCacheBase {
  using DefaultKeyValueCacheBase::ValidatePastPresentLocations;
};

TEST(KvCacheTests, RejectsIncompatiblePastPresentMemoryLocations) {
  auto cpu = OrtMemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto cuda = OrtMemoryInfo::Create("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
  auto other_gpu = OrtMemoryInfo::Create("Cuda", OrtArenaAllocator, 1, OrtMemTypeDefault);
  auto pinned = OrtMemoryInfo::Create("CudaPinned", OrtDeviceAllocator, 0, OrtMemTypeCPUOutput);
  EXPECT_THROW(CacheLocationValidator::ValidatePastPresentLocations(
                   "past.key", "present.key", *cpu, *cuda),
               std::runtime_error);
  EXPECT_THROW(CacheLocationValidator::ValidatePastPresentLocations(
                   "past.key", "present.key", *cuda, *other_gpu),
               std::runtime_error);
  EXPECT_THROW(CacheLocationValidator::ValidatePastPresentLocations(
                   "past.key", "present.key", *cpu, *pinned),
               std::runtime_error);
}

TEST(KvCacheTests, AllowsCompatibleMemoryFromDifferentAllocatorTypes) {
  auto arena = OrtMemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto device = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  EXPECT_NO_THROW(CacheLocationValidator::ValidatePastPresentLocations(
      "past.key", "present.key", *arena, *device));
}

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
