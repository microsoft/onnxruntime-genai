// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include <gtest/gtest.h>

#include "engine/backend_factory.h"
#include "engine/native-device-memory-budget.h"
#include "engine/paged_key_value_cache.h"
#include "models/decoder_only.h"
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

TEST(EngineBackendFactoryTest, DeviceMemoryBudgetRejectsUnsupportedDeviceWithoutChangingSizing) {
  auto model = LoadSyntheticPagedModel();
  auto& batching = *model->config_->engine.dynamic_batching;
  const auto original_blocks = batching.num_blocks;
  batching.device_memory_budget_bytes = 1024;
  EXPECT_THROW(CreateEngineBackend(model), std::runtime_error);
  EXPECT_EQ(batching.num_blocks, original_blocks);
  EXPECT_FALSE(batching.gpu_utilization_factor);
}

TEST(EngineBackendFactoryTest, HeadroomDoesNotChangeSizingWithoutDeviceMemoryBudget) {
  auto model = LoadSyntheticPagedModel();
  auto& batching = *model->config_->engine.dynamic_batching;
  const auto original_blocks = batching.num_blocks;
  ASSERT_TRUE(original_blocks);
  batching.device_memory_headroom_bytes = std::numeric_limits<uint64_t>::max();
  const auto bundle = CreateEngineBackend(model);
  EXPECT_EQ(bundle.dependencies.cache_manager->TargetBlockCount(), *original_blocks);
  EXPECT_EQ(batching.num_blocks, original_blocks);
}

TEST(EngineBackendFactoryTest, ExplicitPagedPoolDoesNotDeductMediaAllowance) {
  auto model = LoadSyntheticPagedModel();
  auto decoder = std::dynamic_pointer_cast<DecoderOnly_Model>(model);
  ASSERT_NE(decoder, nullptr);
  decoder->vision_ = std::make_unique<PackedVision>();
  decoder->vision_->device = model->p_device_kvcache_;
  const auto blocks = model->config_->engine.dynamic_batching->num_blocks.value();

  const auto bundle = CreateEngineBackend(model);

  EXPECT_EQ(bundle.dependencies.cache_manager->TargetBlockCount(), blocks);
  EXPECT_EQ(model->config_->engine.dynamic_batching->num_blocks, blocks);
}

TEST(NativeDeviceMemoryBudgetTest, ChargesFixedStateButDoesNotDoubleChargeResidentSnapshots) {
  Config::Engine::DynamicBatching settings;
  settings.device_memory_budget_bytes = 600;
  settings.device_memory_headroom_bytes = 50;
  EXPECT_EQ(NativeSnapshotDeviceMemoryLimit(settings, 1000, 700, 0), 250u);
  EXPECT_EQ(NativeSnapshotDeviceMemoryLimit(settings, 1000, 600, 100), 250u);
  settings.gpu_utilization_factor = 0.1f;
  EXPECT_EQ(NativeSnapshotDeviceMemoryLimit(settings, 1000, 600, 100), 250u);
  EXPECT_THROW(NativeSnapshotDeviceMemoryLimit(settings, 1000, 300, 0), std::runtime_error);
  settings.device_memory_headroom_bytes = 300;
  EXPECT_EQ(NativeSnapshotDeviceMemoryLimit(settings, 1000, 700, 0), 0u);
  settings.device_memory_headroom_bytes = 301;
  EXPECT_THROW(NativeSnapshotDeviceMemoryLimit(settings, 1000, 700, 0), std::runtime_error);
  settings.device_memory_headroom_bytes = 0;
  settings.device_memory_budget_bytes = 300;
  EXPECT_EQ(NativeSnapshotDeviceMemoryLimit(settings, 1000, 700, 0), 0u);
}

TEST(NativeDeviceMemoryBudgetTest, UtilizationAndStandaloneHeadroomConstrainReclaimableMemory) {
  Config::Engine::DynamicBatching settings;
  EXPECT_EQ(NativeSnapshotDeviceMemoryLimit(settings, std::nullopt, 1000, 0), std::nullopt);
  settings.gpu_utilization_factor = 0.5f;
  EXPECT_EQ(NativeSnapshotDeviceMemoryLimit(settings, std::nullopt, 600, 100), 315u);
  settings.device_memory_headroom_bytes = 100;
  EXPECT_EQ(NativeSnapshotDeviceMemoryLimit(settings, std::nullopt, 600, 100), 270u);
  settings.gpu_utilization_factor.reset();
  EXPECT_EQ(NativeSnapshotDeviceMemoryLimit(settings, std::nullopt, 600, 100), 600u);
  settings.device_memory_headroom_bytes = 701;
  EXPECT_THROW(NativeSnapshotDeviceMemoryLimit(settings, std::nullopt, 600, 100), std::runtime_error);
}

TEST(NativeDeviceMemoryBudgetTest, RejectsMissingBaselineHeadroomInvalidFactorsAndOverflow) {
  Config::Engine::DynamicBatching settings;
  settings.device_memory_budget_bytes = 600;
  EXPECT_THROW(NativeSnapshotDeviceMemoryLimit(settings, 1000, 700, 0), std::runtime_error);
  settings.device_memory_headroom_bytes = 0;
  EXPECT_THROW(NativeSnapshotDeviceMemoryLimit(settings, std::nullopt, 700, 0), std::runtime_error);
  settings.device_memory_budget_bytes.reset();
  for (const float factor : {0.0f, -1.0f, 1.1f, std::numeric_limits<float>::quiet_NaN()}) {
    settings.gpu_utilization_factor = factor;
    EXPECT_THROW(NativeSnapshotDeviceMemoryLimit(settings, std::nullopt, 700, 0), std::runtime_error);
  }
  settings.gpu_utilization_factor = 1.0f;
  EXPECT_THROW(NativeSnapshotDeviceMemoryLimit(settings, std::nullopt, std::numeric_limits<size_t>::max(), 1),
               std::overflow_error);
}

#if USE_CUDA
void SetDeviceMemoryBudgetAfterStartup(const std::shared_ptr<Model>& model, uint64_t remaining_bytes) {
  size_t free_bytes, total_bytes;
  model->p_device_kvcache_->GetAvailableMemory(free_bytes, total_bytes);
  const size_t initial_free_bytes = model->device_memory_budget_initial_free_bytes_.value();
  const size_t startup_bytes = initial_free_bytes > free_bytes ? initial_free_bytes - free_bytes : 0;
  model->config_->engine.dynamic_batching->device_memory_budget_bytes = startup_bytes + remaining_bytes;
}

TEST(EngineBackendFactoryTest, AutomaticCudaMediaReservationUsesTheFeatureDevice) {
  if (!GetDeviceInterface(DeviceType::CUDA)) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  std::array<size_t, 2> blocks{};
  size_t bytes_per_block = 0;
  for (size_t index = 0; index < blocks.size(); ++index) {
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged");
    ClearProviders(*config);
    SetProviderOption(*config, "cuda", "enable_cuda_graph", "0");
    config->engine.dynamic_batching->device_memory_budget_bytes = 64 * 1024 * 1024;
    config->engine.dynamic_batching->device_memory_headroom_bytes = 16 * 1024 * 1024;
    config->engine.dynamic_batching->media_device_bytes = 16 * 1024 * 1024;
    auto model = CreateModel(GetOrtEnv(), std::move(config));
    auto decoder = std::dynamic_pointer_cast<DecoderOnly_Model>(model);
    ASSERT_NE(decoder, nullptr);
    decoder->vision_ = std::make_unique<PackedVision>();
    decoder->vision_->device = index == 0 ? GetDeviceInterface(DeviceType::CPU) : model->p_device_kvcache_;
    SetDeviceMemoryBudgetAfterStartup(model, 64 * 1024 * 1024);
    const auto bundle = CreateEngineBackend(model);
    ASSERT_FALSE(model->config_->engine.dynamic_batching->num_blocks);
    blocks[index] = bundle.dependencies.cache_manager->TargetBlockCount();
    bytes_per_block = PagedKeyValueCacheBytesPerBlock(model);
  }
  ASSERT_GT(blocks[0], blocks[1]);
  EXPECT_EQ((blocks[0] - blocks[1]) * bytes_per_block, 16u * 1024 * 1024);
}

TEST(EngineBackendFactoryTest, DeviceMemoryBudgetRequiresExplicitHeadroomBeforeChangingSizing) {
  if (!GetDeviceInterface(DeviceType::CUDA)) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", "enable_cuda_graph", "0");
  config->engine.dynamic_batching->device_memory_budget_bytes = 64 * 1024 * 1024;
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  const auto original_blocks = model->config_->engine.dynamic_batching->num_blocks;
  try {
    (void)CreateEngineBackend(model);
    FAIL() << "A budget requires explicit profiled headroom";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("device_memory_headroom_bytes"), std::string::npos);
  }
  EXPECT_EQ(model->config_->engine.dynamic_batching->num_blocks, original_blocks);
  EXPECT_FALSE(model->config_->engine.dynamic_batching->gpu_utilization_factor);
}

TEST(EngineBackendFactoryTest, DeviceMemoryBudgetOverridesProfileBlocksWithoutUtilization) {
  if (!GetDeviceInterface(DeviceType::CUDA)) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", "enable_cuda_graph", "0");
  config->engine.dynamic_batching->device_memory_budget_bytes = 64 * 1024 * 1024;
  Config::RuntimeProfile profile;
  profile.id = "budget-regression";
  profile.eligibility.minimum_total_device_memory_bytes = 0;
  profile.overlay.dynamic_batching.num_blocks = 1;
  profile.overlay.dynamic_batching.device_memory_headroom_bytes = 32 * 1024 * 1024;
  config->runtime_profiles.push_back(std::move(profile));
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  // Exercise startup usage larger than the cache allowance even on low-overhead CUDA runtimes.
  auto startup_allocation = model->p_device_kvcache_->Allocate<uint8_t>(80 * 1024 * 1024);
  SetDeviceMemoryBudgetAfterStartup(model, 64 * 1024 * 1024);
  ASSERT_EQ(model->config_->engine.dynamic_batching->num_blocks, 1u);
  ASSERT_FALSE(model->config_->engine.dynamic_batching->gpu_utilization_factor);
  auto bundle = CreateEngineBackend(model);
  EXPECT_FALSE(model->config_->engine.dynamic_batching->num_blocks);
  EXPECT_EQ(model->config_->engine.dynamic_batching->gpu_utilization_factor, 1.0f);
  EXPECT_GT(bundle.dependencies.cache_manager->TargetBlockCount(), 1u);
  EXPECT_LE(bundle.dependencies.cache_manager->TargetBlockCount() * PagedKeyValueCacheBytesPerBlock(model),
            32u * 1024 * 1024);
  EXPECT_GT(bundle.capabilities.max_request_length, 4u);
}

TEST(EngineBackendFactoryTest, DeviceMemoryBudgetRejectsMtpDsparkAndLateConfiguration) {
  if (!GetDeviceInterface(DeviceType::CUDA)) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  auto model = LoadSyntheticPagedCudaModel();
  model->config_->engine.dynamic_batching->device_memory_budget_bytes = 64 * 1024 * 1024;
  model->config_->engine.dynamic_batching->device_memory_headroom_bytes = 0;
  EXPECT_THROW(CreateEngineBackend(model), std::runtime_error);
  for (const bool dspark : {false, true}) {
    model->config_->model.mtp.filename = dspark ? "" : "not-loaded.onnx";
    model->config_->model.dflash2.filename = dspark ? "not-loaded.onnx" : "";
    model->config_->model.dflash2.is_dspark = dspark;
    try {
      (void)CreateEngineBackend(model);
      FAIL() << "Unsupported auxiliaries must be rejected before loading";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string{error.what()}.find("auxiliary models"), std::string::npos);
    }
  }
}

TEST(EngineBackendFactoryTest, DeviceMemoryBudgetSupportsWindowedDflash) {
  if (!GetDeviceInterface(DeviceType::CUDA)) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", "enable_cuda_graph", "0");
  auto& batching = *config->engine.dynamic_batching;
  batching.device_memory_budget_bytes = 64 * 1024 * 1024;
  batching.device_memory_headroom_bytes = 16 * 1024 * 1024;
  batching.max_batch_size = 1;
  auto& draft = config->model.dflash2;
  draft.filename = (fs::path{".."} / "synthetic-dspark" / "dflash2-fp16-aux.onnx").string();
  draft.num_hidden_layers = 1;
  draft.num_key_value_heads = 1;
  draft.head_size = 1;
  draft.block_size = 4;
  draft.num_draft_tokens = 3;
  draft.selector_top_k = 2;
  draft.mask_token_id = 31;
  draft.sliding_window = 8;
  draft.main_aux_hidden_states = "hidden_states";
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  SetDeviceMemoryBudgetAfterStartup(model, 64 * 1024 * 1024);
  auto bundle = CreateEngineBackend(model);
  ASSERT_NE(bundle.dependencies.dflash2_drafter, nullptr);
  EXPECT_FALSE(model->config_->engine.dynamic_batching->num_blocks);
  EXPECT_GT(bundle.dependencies.cache_manager->TargetBlockCount(), 0u);
  EXPECT_EQ(bundle.dependencies.cache_manager->DraftCheckpointCapacity(), 1u);
}

TEST(EngineBackendFactoryTest, DeviceMemoryBudgetBoundsOptionalDraftSnapshots) {
  if (!GetDeviceInterface(DeviceType::CUDA)) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", "enable_cuda_graph", "0");
  auto& batching = *config->engine.dynamic_batching;
  batching.device_memory_budget_bytes = 64 * 1024 * 1024;
  batching.device_memory_headroom_bytes = 32 * 1024 * 1024;
  batching.num_blocks.reset();
  batching.gpu_utilization_factor = 1.0f;
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  SetDeviceMemoryBudgetAfterStartup(model, 64 * 1024 * 1024);
  auto cache = CacheManager::Create(model, 0, 0, 40 * 1024 * 1024);
  EXPECT_EQ(cache->DraftCheckpointCapacity(), 0u);
  EXPECT_GT(cache->TargetBlockCount(), 0u);
}

TEST(EngineBackendFactoryTest, DeviceMemoryBudgetBoundsOptionalDraftSnapshotsWithCaptureTables) {
  if (!GetDeviceInterface(DeviceType::CUDA)) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  for (const size_t checkpoint_mib : {48u, 8u}) {
    SCOPED_TRACE(checkpoint_mib);
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite");
    ClearProviders(*config);
    SetProviderOption(*config, "cuda", "enable_cuda_graph", "1");
    config->model.context_length = 1 << 24;
    auto& batching = *config->engine.dynamic_batching;
    batching.max_batch_size = 1;
    batching.device_memory_budget_bytes = 64 * 1024 * 1024;
    batching.device_memory_headroom_bytes = 0;
    batching.num_blocks.reset();
    batching.gpu_utilization_factor = 1.0f;
    auto model = CreateModel(GetOrtEnv(), std::move(config));
    SetDeviceMemoryBudgetAfterStartup(model, 64 * 1024 * 1024);
    auto cache = CacheManager::Create(model, 0, 0, checkpoint_mib * 1024 * 1024);
    EXPECT_EQ(cache->DraftCheckpointCapacity(), checkpoint_mib == 48 ? 0u : 1u);
    EXPECT_GT(cache->TargetBlockCount(), 0u);
  }
}

TEST(EngineBackendFactoryTest, DeviceMemoryBudgetReservesCaptureBlockTablesBeforeAllocation) {
  if (!GetDeviceInterface(DeviceType::CUDA)) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-paged");
  ClearProviders(*config);
  SetProviderOption(*config, "cuda", "enable_cuda_graph", "1");
  config->model.context_length = 1 << 25;
  config->engine.dynamic_batching->device_memory_budget_bytes = 64 * 1024 * 1024;
  config->engine.dynamic_batching->device_memory_headroom_bytes = 0;
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  SetDeviceMemoryBudgetAfterStartup(model, 64 * 1024 * 1024);
  try {
    (void)CreateEngineBackend(model);
    FAIL() << "The context-bounded capture tables exceed this startup budget";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("reserved decoder state"), std::string::npos);
  }
}
#endif

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
  // Zero is no limit: a cache that holds 64 positions cannot declare it.
  capabilities.max_request_length = 0;
  EXPECT_THROW(ResolveBackendCapabilities(*model, *cache, *scheduler, *executor, capabilities),
               std::runtime_error);
  capabilities.max_request_length = 64;
  EXPECT_NO_THROW(ResolveBackendCapabilities(*model, *cache, *scheduler, *executor, capabilities));
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

TEST(EngineBackendFactoryTest, StaticBatchSchedulerCannotAdvertiseIndependentAdmission) {
  auto model = LoadDummyDecoderModel();
  auto cache = std::make_shared<IndependentSlotCache>(model);
  StaticBatchScheduler static_scheduler{model, cache};
  SelectedTokenExecutor executor;
  const auto resolved = ResolveBackendCapabilities(*model, *cache, static_scheduler, executor);
  EXPECT_FALSE(resolved.SupportsIndependentAdmission());
  auto declared = resolved;
  declared.admission = BackendAdmission::IndependentSlots;
  EXPECT_THROW(ResolveBackendCapabilities(*model, *cache, static_scheduler, executor, declared),
               std::runtime_error);
}

TEST(EngineBackendFactoryTest, StaticMaxRequestLengthIsClampedToModelContextLength) {
  auto model = LoadDummyDecoderModel();
  model->config_->model.context_length = 32;
  auto cache = std::make_shared<IndependentSlotCache>(model);  // MaxStaticRequestLength() == 64
  auto scheduler = Scheduler::Create(model, cache);
  SelectedTokenExecutor executor;
  const auto resolved = ResolveBackendCapabilities(*model, *cache, *scheduler, executor);
  EXPECT_EQ(resolved.max_request_length, 32u);
  auto declared = resolved;
  declared.max_request_length = 64;
  EXPECT_THROW(ResolveBackendCapabilities(*model, *cache, *scheduler, executor, declared),
               std::runtime_error);
}

}  // namespace
}  // namespace Generators::test
