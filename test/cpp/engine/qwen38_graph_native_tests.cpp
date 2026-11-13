// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_helpers.h"
#include "models/qwen38_graph.h"
#include "scoped-environment-variable.h"

#if USE_CUDA

namespace Generators::test {
namespace {

std::shared_ptr<Model> LoadCounterModel(bool warmup = true) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-qwen38");
  config->model.qwen38_graph.warmup = warmup;
  return CreateModel(GetOrtEnv(), std::move(config));
}

std::array<EngineEvent, 2> RunTwoLiveSteps(const std::shared_ptr<Model>& model) {
  auto engine = std::make_shared<Engine>(model);
  TurnOptions options;
  options.max_generated_tokens = 2;
  const std::array<int32_t, 2> prompt{0, 0};
  const auto request = CreateRequestWithPrompt(engine, prompt, options);
  return {RunOne(*engine), RunOne(*engine)};
}

void ExpectOneStateUpdatePerLiveStep(const std::array<EngineEvent, 2>& events) {
  for (size_t step = 0; step < events.size(); ++step) {
    SCOPED_TRACE(step);
    EXPECT_NE(events[step].flags & EngineEventFlagToken, 0u);
    EXPECT_EQ(events[step].error_code, EngineErrorCode::None);
    EXPECT_EQ(events[step].token, static_cast<int32_t>(step + 1));
  }
}

class Qwen38GraphNativeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Qwen38Plane plane{std::filesystem::path{}};
    const int error = plane.set_device(0);
    if (error != 0) GTEST_SKIP() << "CUDA device unavailable: " << plane.error_string(error);
  }
};

TEST_F(Qwen38GraphNativeTest, ModelLoadedOnAnotherThreadAdvancesOncePerLiveEngineStep) {
  const auto model = std::async(std::launch::async, [] { return LoadCounterModel(); }).get();
  ExpectOneStateUpdatePerLiveStep(RunTwoLiveSteps(model));
}

TEST_F(Qwen38GraphNativeTest, SameThreadWarmupAdvancesOncePerLiveEngineStep) {
  ExpectOneStateUpdatePerLiveStep(RunTwoLiveSteps(LoadCounterModel()));
}

TEST_F(Qwen38GraphNativeTest, SequentialEngineThreadsAndTheLoadingThreadAdvanceOncePerLiveStep) {
  const auto model = LoadCounterModel();
  for (int lease = 0; lease < 3; ++lease) {
    SCOPED_TRACE(lease);
    const auto events = std::async(std::launch::async, [&] { return RunTwoLiveSteps(model); }).get();
    ExpectOneStateUpdatePerLiveStep(events);
  }
  ExpectOneStateUpdatePerLiveStep(RunTwoLiveSteps(model));
  ExpectOneStateUpdatePerLiveStep(RunTwoLiveSteps(model));
}

TEST_F(Qwen38GraphNativeTest, DeviceMemoryBudgetValidatesFixedStateWithOrWithoutPrefixCaching) {
  for (bool prefix_caching : {false, true}) {
    for (bool fits : {false, true}) {
      SCOPED_TRACE(prefix_caching);
      SCOPED_TRACE(fits);
      auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-qwen38");
      config->engine.dynamic_batching.emplace();
      auto& batching = *config->engine.dynamic_batching;
      batching.max_batch_size = 1;
      batching.prefix_caching = prefix_caching;
      batching.device_memory_budget_bytes = fits ? uint64_t{1} << 40 : 1;
      batching.device_memory_headroom_bytes = 0;
      if (fits) {
        auto model = CreateModel(GetOrtEnv(), std::move(config));
        EXPECT_FALSE(model->config_->engine.dynamic_batching);
        ExpectOneStateUpdatePerLiveStep(RunTwoLiveSteps(model));
      } else {
        try {
          (void)CreateModel(GetOrtEnv(), std::move(config));
          FAIL() << "A one-byte budget cannot fit the native state";
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string{error.what()}.find("device_memory_budget_bytes is too small"), std::string::npos);
        }
      }
    }
  }
}

TEST_F(Qwen38GraphNativeTest, UtilizationLimitControlsDefaultOnSnapshotReuseAfterNormalization) {
  for (bool retain_snapshots : {false, true}) {
    SCOPED_TRACE(retain_snapshots);
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-qwen38");
    config->engine.dynamic_batching.emplace();
    auto& batching = *config->engine.dynamic_batching;
    batching.max_batch_size = 1;
    ASSERT_FALSE(batching.prefix_caching.has_value());
    batching.gpu_utilization_factor = retain_snapshots ? 1.0f : std::numeric_limits<float>::min();
    auto model = CreateModel(GetOrtEnv(), std::move(config));
    ASSERT_FALSE(model->config_->engine.dynamic_batching);
    auto engine = std::make_shared<Engine>(model);
    TurnOptions options;
    options.max_generated_tokens = 2;
    const std::array<int32_t, 2> prompt{0, 0};
    for (int repeat = 0; repeat < 2; ++repeat) {
      auto request = CreateRequestWithPrompt(engine, prompt, options);
      ExpectOneStateUpdatePerLiveStep({RunOne(*engine), RunOne(*engine)});
      EXPECT_EQ(request->TurnCachedPromptTokens(), retain_snapshots && repeat != 0 ? prompt.size() : 0u);
      request->Close();
    }
    ASSERT_TRUE(engine->PrefixCacheStats());
    EXPECT_EQ(engine->PrefixCacheStats()->hits, retain_snapshots ? 1u : 0u);
  }
}

TEST_F(Qwen38GraphNativeTest, SeparateDenseSnapshotsRestoreAndHandoffAcrossCompressionBoundaries) {
  for (bool disk : {false, true}) {
    SCOPED_TRACE(disk);
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-qwen38");
    config->model.decoder.filename = "decoder-handoff.onnx";
    auto& graph = config->model.qwen38_graph;
    graph.warmup = true;
    graph.fold.enabled = true;
    graph.fold.head_r2.filename = "head-r2.onnx";
    graph.mtp.enabled = true;
    graph.mtp.prefill_head = true;
    graph.prefill.enabled = true;
    graph.prefill.chunk = 4;
    graph.prefill.dense.filename = "dense-prefill.onnx";
    graph.prefill.head_kv.filename = "head-kv.onnx";
    graph.prefill.constants = "qsa.txt";
    graph.prefill.warmup_lengths = {3, 4, 5};
    config->engine.dynamic_batching.emplace();
    auto& batching = *config->engine.dynamic_batching;
    batching.max_batch_size = 1;
    batching.prefix_caching = true;
    if (disk)
      batching.snapshot_cache.disk_bytes = 65536;
    else
      batching.snapshot_cache.ram_bytes = 65536;
    auto model = std::dynamic_pointer_cast<Qwen38GraphModel>(CreateModel(GetOrtEnv(), std::move(config)));
    ASSERT_NE(model, nullptr);
    auto lease = model->AcquireDevice();
    Qwen38DecodeDevice& device = *lease;
    const auto decode = [&](size_t length) {
      // The folded graph checks the actual packed target/head state and publishes
      // validation failures through WaitCycle; parity alone could miss symmetric corruption.
      device.BeginDecode(true);
      const auto record = device.WaitCycle();
      EXPECT_EQ(record.pred0, static_cast<int64_t>(length + 20));
      EXPECT_EQ(record.pred1, static_cast<int64_t>(length + 21));
      EXPECT_EQ(record.pred2, static_cast<int64_t>(length + 22));
      EXPECT_EQ(record.accepted_count, 0);
      EXPECT_EQ(record.new_pending, static_cast<int64_t>(length + 20));
      EXPECT_EQ(record.new_length, static_cast<int64_t>(length + 1));
      device.Advance(record, false);
      return std::array<int64_t, 4>{record.pred0, record.pred1, record.new_pending, record.new_length};
    };
    for (size_t length : {size_t{3}, size_t{4}, size_t{5}}) {
      SCOPED_TRACE(length);
      const std::vector<int32_t> prompt(length, 0);
      device.Reset();
      device.PrefillChunk(prompt);
      const auto pending = device.PendingToken();
      EXPECT_EQ(pending, static_cast<int32_t>(length + 9));
      auto saved = device.CapturePrefix();
      ASSERT_NE(saved, nullptr);
      const auto* storage = dynamic_cast<const PrefixSnapshotCache::Entry*>(saved.get());
      ASSERT_NE(storage, nullptr);
      EXPECT_EQ(storage->tier, disk ? PrefixSnapshotCache::Tier::Disk : PrefixSnapshotCache::Tier::Ram);
      EXPECT_EQ(storage->bytes, 32u + length * (8u + 128u * 2u + 2u));
      EXPECT_EQ(storage->shapes.size(), 5u);
      const auto cold_output = decode(length);

      device.Reset();
      device.RestorePrefix(*saved, prompt);
      EXPECT_EQ(device.PendingToken(), pending);
      EXPECT_EQ(decode(length), cold_output);

      device.Reset();
      device.RestorePrefix(*saved, prompt);
      const std::vector<int32_t> suffix(5, 0);
      device.PrefillChunk(suffix);
      EXPECT_EQ(device.PendingToken(), static_cast<int32_t>(length + suffix.size() + 9));
      const auto continued = decode(length + suffix.size());
      device.Reset();
      device.PrefillChunk(std::vector<int32_t>(length + suffix.size(), 0));
      EXPECT_EQ(decode(length + suffix.size()), continued);
    }
  }
}

TEST_F(Qwen38GraphNativeTest, AdaptiveBudgetEnvironmentDefaultsAndOptOutReachTheNativeSession) {
  struct Case {
    std::optional<std::string> value;
    const char* reserve;
  };
  constexpr const char* kAdaptiveReserve = "5901778944";
  constexpr const char* kOptOutReserve = "6442450944";
  const Case cases[] = {
      {std::nullopt, kAdaptiveReserve},
      {"", kAdaptiveReserve},
      {"1", kAdaptiveReserve},
      {"true", kAdaptiveReserve},
      {"0", kOptOutReserve},
      {"false", kOptOutReserve},
  };
  for (const auto& [value, reserve] : cases) {
    SCOPED_TRACE(value ? *value : "<unset>");
    const ScopedEnvironmentVariable env{"ORT_QWEN38_ADAPTIVE_CTX_BUDGET", value};
    const auto model = LoadCounterModel(false);
    size_t size = 0;
    constexpr const char* key = "session.moe.offload.reserve_device_bytes";
    Ort::ThrowOnError(Ort::api->GetSessionConfigEntry(model->session_options_.get(), key, nullptr, &size));
    std::string actual(size, '\0');
    Ort::ThrowOnError(Ort::api->GetSessionConfigEntry(model->session_options_.get(), key, actual.data(), &size));
    if (!actual.empty() && actual.back() == '\0') actual.pop_back();
    EXPECT_EQ(actual, reserve);
  }
}

TEST_F(Qwen38GraphNativeTest, InvalidAdaptiveBudgetEnvironmentValuesFailModelCreation) {
  for (const char* value : {"2", "yes"}) {
    SCOPED_TRACE(value);
    const ScopedEnvironmentVariable env{"ORT_QWEN38_ADAPTIVE_CTX_BUDGET", std::string{value}};
    try {
      LoadCounterModel(false);
      ADD_FAILURE() << "model creation accepted an invalid ORT_QWEN38_ADAPTIVE_CTX_BUDGET";
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string{error.what()}.find("ORT_QWEN38_ADAPTIVE_CTX_BUDGET"), std::string::npos) << error.what();
    }
  }
}

}  // namespace
}  // namespace Generators::test

#endif
