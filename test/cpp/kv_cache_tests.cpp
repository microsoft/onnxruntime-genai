// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <thread>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "engine/engine.h"
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

std::unique_ptr<Generators::Config> MakeK1Config(bool share_buffer = false) {
  auto config = std::make_unique<Generators::Config>(fs::path{MODEL_PATH "dense-k1"}, std::string_view{});
  Generators::OverlayConfig(*config, R"({"model":{"decoder":{"declared_state":{"max_sequence_length":32}}}})");
  config->search.past_present_share_buffer = share_buffer;
  return config;
}

template <typename Api, typename = void>
struct HasK1Api : std::false_type {};

template <typename Api>
struct HasK1Api<Api, std::void_t<decltype(&Api::GetDeclaredStateSupport),
                                 decltype(&Api::SessionOptionsSetDeclaredStateContract)>> : std::true_type {};

template <typename Api>
bool SupportsK1(const Api& api) {
  if (Ort::api_version < 33)
    return false;
  if constexpr (HasK1Api<Api>::value) {
    uint32_t version = 0;
    auto* status = api.GetDeclaredStateSupport(&version);
    if (status && api.GetErrorCode(status) == ORT_NOT_IMPLEMENTED) {
      api.ReleaseStatus(status);
      return false;
    }
    Ort::ThrowOnError(status);
    if (version < 1)
      throw std::runtime_error("Successful K1 negotiation must support schema version 1.");
    return true;
  }
  return false;
}

TEST(K1ConfigTests, ProducesEscapedLayerTemplatesAndEffectiveAlias) {
  for (bool share : {false, true}) {
    auto config = MakeK1Config(share);
    config->model.decoder.num_hidden_layers = 2;
    config->model.decoder.inputs.past_key_names = "past\"\\.%d.key";
    config->model.decoder.declared_state->max_sequence_length = 24;
    const auto json = Generators::BuildDeclaredStateContract(*config);
    const auto expected_input = R"("input":"past\"\\.{layer}.key")";
    EXPECT_NE(json.find(expected_input), std::string::npos);
    EXPECT_NE(json.find(R"("output":"present.{layer}.value")"), std::string::npos);
    EXPECT_NE(json.find(R"("layers":[0,1],"capacity":24)"), std::string::npos);
    EXPECT_NE(json.find(share ? R"("alias":"past_present")" : R"("alias":"none")"), std::string::npos);
    EXPECT_NE(json.find(R"("max_sequence_length":24)"), std::string::npos);
  }
}

TEST(K1ConfigTests, RejectsInvalidOverlaysAtomically) {
  auto config = MakeK1Config();
  const auto original = Generators::BuildDeclaredStateContract(*config);
  for (const char* overlay : {
           R"({"model":{"decoder":{"declared_state":{"max_batch_size":2}}}})",
           R"({"model":{"decoder":{"declared_state":{"max_live_instances":2}}}})",
           R"({"model":{"decoder":{"declared_state":{"max_sequence_length":0}}}})",
           R"({"model":{"decoder":{"declared_state":{"max_sequence_length":33}}}})",
           R"({"model":{"decoder":{"declared_state":{"max_sequence_length":3.5}}}})",
           R"({"model":{"decoder":{"declared_state":{"unknown":1}}}})",
           R"({"model":{"decoder":{"inputs":{"past_key_names":"past.%s.key"}}}})",
           R"({"model":{"decoder":{"inputs":{"past_key_names":"past.%d.%d.key"}}}})",
           R"({"model":{"decoder":{"inputs":{"past_key_names":"past.{layer}.%d.key"}}}})",
           R"({"model":{"decoder":{"layer_types":["conv"]}}})",
           R"({"model":{"decoder":{"pipeline":[{"filename":"other.onnx"}]}}})",
           R"({"model":{"decoder":{"sliding_window":{"window_size":8}}}})",
           R"({"engine":{"dynamic_batching":{"max_batch_size":2}}})",
           R"({"search":{"max_length":33}})",
           R"({"search":{"max_length":0},"model":{"decoder":{"declared_state":{"max_sequence_length":16}}}})",
           R"({"search":{"num_beams":2}})"}) {
    SCOPED_TRACE(overlay);
    EXPECT_THROW(Generators::OverlayConfig(*config, overlay), std::exception);
    EXPECT_EQ(Generators::BuildDeclaredStateContract(*config), original);
  }
  EXPECT_NO_THROW(Generators::OverlayConfig(*config, R"({"model":{"decoder":{"declared_state":{"max_sequence_length":24}}}})"));
  EXPECT_EQ(config->model.decoder.declared_state->max_sequence_length, 24);
}

TEST(K1ConfigTests, RequiresCapabilityWithoutChangingLegacyModel) {
  auto& env = Generators::GetOrtEnv();
  EXPECT_NO_THROW(Generators::CreateModel(env, MODEL_PATH "longrope-context"));
  if (SupportsK1(*Ort::api)) {
    EXPECT_NO_THROW(Generators::CreateModel(env, MakeK1Config()));
  } else {
    EXPECT_THROW(Generators::CreateModel(env, MakeK1Config()), std::exception);
  }
}

struct K1EnvelopeTests : testing::TestWithParam<bool> {
  void SetUp() override {
    auto& env = Generators::GetOrtEnv();
    if (!SupportsK1(*Ort::api))
      GTEST_SKIP() << "Requires a matching ORT K1 SDK/runtime; capability explicitly unavailable.";
    model = Generators::CreateModel(env, MakeK1Config(GetParam()));
    params = Generators::CreateGeneratorParams(*model);
  }

  std::shared_ptr<Generators::Model> model;
  std::shared_ptr<Generators::GeneratorParams> params;
};

TEST_P(K1EnvelopeTests, KeepsResidentLeaseUntilDestructionAndRollsBackFailure) {
  EXPECT_THROW(Generators::Engine{model}, std::runtime_error);
  auto generator = Generators::CreateGenerator(*model, *params);
  EXPECT_THROW(Generators::CreateGenerator(*model, *params), std::runtime_error);
  const std::array<int32_t, 4> prompt{1, 2, 3, 4};
  generator->AppendTokens(prompt);
  for (int step = 0; step < params->search.max_length && !generator->IsDone(); ++step)
    generator->GenerateNextToken();
  EXPECT_TRUE(generator->IsDone());
  EXPECT_THROW(Generators::CreateGenerator(*model, *params), std::runtime_error);

  generator->RewindToLength(0);
  generator->AppendTokens(prompt);
  EXPECT_FALSE(generator->GetLogits().empty());
  std::thread release_on_another_thread{[owned = std::move(generator)]() mutable { owned.reset(); }};
  release_on_another_thread.join();

  params->search.do_sample = true;
  params->search.top_k = 2;
  params->search.top_p = 2.0f;
  EXPECT_THROW(Generators::CreateGenerator(*model, *params), std::runtime_error);
  params->search.do_sample = false;
  EXPECT_NO_THROW(Generators::CreateGenerator(*model, *params));
}

TEST_P(K1EnvelopeTests, FreezesGeometryAndBoundsBeforeAppendOrCacheGrowth) {
  auto generator = Generators::CreateGenerator(*model, *params);
  const std::array<int32_t, 4> prompt{1, 2, 3, 4};
  const auto original_max = params->search.max_length;
  params->search.max_length = original_max + 1;
  EXPECT_THROW(generator->AppendTokens(prompt), std::runtime_error);
  params->search.max_length = original_max;
  params->search.batch_size = 0;
  EXPECT_THROW(generator->AppendTokens(prompt), std::runtime_error);
  params->search.batch_size = 1;
  params->search.num_beams = 2;
  EXPECT_THROW(generator->GenerateNextToken(), std::runtime_error);
  params->search.num_beams = 1;
  params->search.past_present_share_buffer = !GetParam();
  EXPECT_THROW(generator->AppendTokens(prompt), std::runtime_error);
  params->search.past_present_share_buffer = GetParam();
  params->use_graph_capture = true;
  EXPECT_THROW(generator->GetLogits(), std::runtime_error);
  params->use_graph_capture = false;
  EXPECT_NO_THROW(generator->AppendTokens(prompt));
  EXPECT_EQ(generator->GetSequence(0).size(), prompt.size());
  EXPECT_THROW(generator->AppendTokens(std::vector<int32_t>(original_max, 1)), std::runtime_error);
  EXPECT_EQ(generator->GetSequence(0).size(), prompt.size());

  generator.reset();
  model->config_->model.decoder.declared_state->max_sequence_length = 64;
  params->search.max_length = 33;
  EXPECT_THROW(Generators::CreateGenerator(*model, *params), std::runtime_error);
}

TEST_P(K1EnvelopeTests, RejectsCacheGrowthBeforeChangingBindings) {
  DenseCacheTestState state{*params, *model};
  auto cache = Generators::CreateStandardKeyValueCache(state);
  ASSERT_NE(cache, nullptr);
  cache->Add();
  cache->Update({}, 4);
  const auto inputs = state.inputs_;
  const auto outputs = state.outputs_;
  EXPECT_THROW(cache->Update({}, params->search.max_length + 1), std::runtime_error);
  EXPECT_THROW(cache->Update({}, -1), std::runtime_error);
  EXPECT_EQ(state.inputs_, inputs);
  EXPECT_EQ(state.outputs_, outputs);
  EXPECT_NO_THROW(cache->Update({}, 5));
}

TEST_P(K1EnvelopeTests, RejectsUndeclaredLayerCaches) {
  auto config = MakeK1Config(GetParam());
  config->model.decoder.num_hidden_layers = 1;
  EXPECT_THROW(
      {
        auto incomplete = Generators::CreateModel(Generators::GetOrtEnv(), std::move(config));
        auto incomplete_params = Generators::CreateGeneratorParams(*incomplete);
        Generators::CreateGenerator(*incomplete, *incomplete_params);
      },
      std::exception);
}

INSTANTIATE_TEST_SUITE_P(SharedAndSeparateBuffers, K1EnvelopeTests, testing::Bool());

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
