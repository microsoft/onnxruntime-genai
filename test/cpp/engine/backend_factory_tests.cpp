// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "engine/backend_factory.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"

namespace Generators::test {
namespace {

struct IndependentSlotCache final : RecordingCacheManager {
  explicit IndependentSlotCache(std::shared_ptr<Model> model)
      : RecordingCacheManager(std::move(model), 2, nullptr, false) {}
  bool SupportsContinuousMultiSlotDecoding() const override { return true; }
  size_t MaxStaticBatchSize() const override { return 2; }
  size_t MaxStaticRequestLength() const override { return 64; }
};

struct SelectedTokenExecutor final : ModelExecutor {
  bool SelectsOwnTokens() const override { return true; }
  void Decode(ScheduledRequests&, ExecutionContext&) override {
    throw std::logic_error("This capability test must not execute device work.");
  }
};

std::shared_ptr<Model> LoadStaticDecoderOnlyModel() {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  config->model.type = "decoder";
  return CreateModel(GetOrtEnv(), std::move(config));
}

TEST(EngineBackendFactoryTest, DefaultDecoderBundleKeepsItsExecutionProtocol) {
  auto model = LoadStaticDecoderOnlyModel();
  auto bundle = CreateEngineBackend(model);
  EXPECT_FALSE(bundle.capabilities.UsesTransactions());
  EXPECT_FALSE(bundle.capabilities.SelectsOwnTokens());
  EXPECT_EQ(bundle.capabilities.kv_storage, BackendKvStorage::RebuiltContiguous);
  ASSERT_TRUE(bundle.dependencies.backend_capabilities);
  EXPECT_EQ(bundle.dependencies.backend_capabilities->max_slots,
            bundle.dependencies.cache_manager->MaxBatchSize());
  auto engine = std::make_shared<Engine>(model, std::move(bundle.dependencies));
  EXPECT_EQ(engine->GetCapabilities().configured_max_batch_size,
            bundle.capabilities.configured_max_batch_size);
}

TEST(EngineBackendFactoryTest, InjectedIndependentSlotsRemainNontransactional) {
  auto model = LoadDummyDecoderModel();
  auto cache = std::make_shared<IndependentSlotCache>(model);
  auto scheduler = Scheduler::Create(model, cache);
  auto executor = std::make_unique<SelectedTokenExecutor>();
  const auto capabilities = ResolveBackendCapabilities(*model, *cache, *scheduler, *executor);
  EXPECT_TRUE(capabilities.SupportsIndependentAdmission());
  EXPECT_TRUE(capabilities.SelectsOwnTokens());
  EXPECT_FALSE(capabilities.UsesTransactions());
  EXPECT_TRUE(capabilities.supports_stop_strings);
  EXPECT_FALSE(capabilities.supports_sampling);
  EXPECT_EQ(capabilities.max_request_length, 64u);
  auto engine = std::make_shared<Engine>(
      model, EngineDependencies{cache, std::move(scheduler), std::move(executor)});
  EXPECT_EQ(engine->GetCapabilities().configured_max_batch_size, 2u);
  EXPECT_EQ(engine->GetCapabilities().max_request_length, 64u);
  RequestOptions oversized;
  oversized.max_session_tokens = 65;
  EXPECT_THROW(engine->CreateRequest(oversized), std::runtime_error);
}

TEST(EngineBackendFactoryTest, ExplicitContradictionsFailBeforeAdmission) {
  auto model = LoadDummyDecoderModel();
  auto cache = std::make_shared<IndependentSlotCache>(model);
  auto scheduler = Scheduler::Create(model, cache);
  auto executor = std::make_unique<SelectedTokenExecutor>();
  auto capabilities = ResolveBackendCapabilities(*model, *cache, *scheduler, *executor);
  capabilities.execution_protocol = BackendExecutionProtocol::Transactional;
  EXPECT_THROW(ResolveBackendCapabilities(*model, *cache, *scheduler, *executor, capabilities),
               std::runtime_error);
  capabilities = ResolveBackendCapabilities(*model, *cache, *scheduler, *executor);
  capabilities.max_request_length = 65;
  EXPECT_THROW(ResolveBackendCapabilities(*model, *cache, *scheduler, *executor, capabilities),
               std::runtime_error);
  capabilities = ResolveBackendCapabilities(*model, *cache, *scheduler, *executor);
  capabilities.token_selection = BackendTokenSelection::Engine;
  EngineDependencies dependencies{cache, std::move(scheduler), std::move(executor)};
  dependencies.backend_capabilities = capabilities;
  EXPECT_THROW((void)std::make_shared<Engine>(model, std::move(dependencies)), std::runtime_error);
}

TEST(EngineBackendFactoryTest, SelectedTokensCannotClaimPagedTransactions) {
  auto model = LoadDummyDecoderModel();
  model->config_->engine.dynamic_batching = Config::Engine::DynamicBatching{};
  auto cache = std::make_shared<RecordingCacheManager>(model, 2);
  auto scheduler = Scheduler::Create(model, cache);
  SelectedTokenExecutor executor;
  EXPECT_THROW(ResolveBackendCapabilities(*model, *cache, *scheduler, executor), std::runtime_error);
}

TEST(EngineBackendFactoryTest, StaticDecoderCannotAdvertiseUnsupportedStopHandling) {
  auto model = LoadDummyDecoderModel();
  auto cache = std::make_shared<RecordingCacheManager>(model, 2, nullptr, false);
  auto scheduler = Scheduler::Create(model, cache);
  RecordingModelExecutor executor{model, cache, 5};
  auto capabilities = ResolveBackendCapabilities(*model, *cache, *scheduler, executor);
  capabilities.supports_stop_strings = true;
  EXPECT_THROW(ResolveBackendCapabilities(*model, *cache, *scheduler, executor, capabilities),
               std::runtime_error);
}

TEST(EngineBackendFactoryTest, ResolvedCapabilitiesDoNotChangeAfterAssembly) {
  auto model = LoadStaticDecoderOnlyModel();
  auto bundle = CreateEngineBackend(model);
  const auto configured_rows = bundle.capabilities.configured_max_batch_size;
  auto engine = std::make_shared<Engine>(model, std::move(bundle.dependencies));
  model->config_->engine.static_batching = Config::Engine::StaticBatching{};
  model->config_->engine.static_batching->max_batch_size = 17;
  EXPECT_EQ(engine->GetCapabilities().configured_max_batch_size, configured_rows);
}

TEST(EngineBackendFactoryTest, InjectedSelectedTokenPolicyIsValidatedBeforeMutation) {
  auto model = LoadDummyDecoderModel();
  auto cache = std::make_shared<IndependentSlotCache>(model);
  auto scheduler = Scheduler::Create(model, cache);
  auto engine = std::make_shared<Engine>(
      model, EngineDependencies{cache, std::move(scheduler), std::make_unique<SelectedTokenExecutor>()});
  auto request = engine->CreateRequest();
  const std::array<int32_t, 1> prompt{3};
  TurnOptions options;
  options.do_sample = true;
  options.top_k = 4;
  EXPECT_THROW(engine->BeginTurn(request, prompt, options), std::runtime_error);
  EXPECT_TRUE(request->IsAwaitingFirstTurn());
  EXPECT_EQ(request->CurrentSequenceLength(), 0);
}

}  // namespace
}  // namespace Generators::test
