// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
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

namespace Generators {

struct Qwen38GraphNativeTestAccess {
  struct PackedState {
    std::vector<int64_t> positions;
    std::vector<float> keys;
    std::vector<uint16_t> pending;
    std::vector<int32_t> lengths;
  };

  static void EnableDensePrefill(Qwen38GraphModel& model) {
    auto& prefill = model.config_->model.qwen38_graph.prefill;
    prefill.enabled = true;
    prefill.chunk = 4;
    model.session_dense_ = model.CreateSession(GetOrtEnv(), "dense-prefill.onnx", model.session_options_.get());
    model.session_head_kv_ = model.CreateSession(GetOrtEnv(), "decoder-handoff.onnx", model.session_options_.get());
    const std::filesystem::path root{MODEL_PATH "engine/synthetic-qwen38"};
    model.qsa_constants_ = std::make_unique<Qwen38QsaConstants>(
        *model.plane_, root / "qsa.txt", root, model.p_device_->GetCudaStream());
    model.engine_->BuildFastPrefill();
    model.engine_->Reset();
  }

  template <typename T>
  static std::vector<T> Read(Qwen38GraphEngine& engine, const DevTensor& tensor) {
    std::vector<T> values(tensor.Bytes() / sizeof(T));
    engine.SyncStream();
    engine.model_.p_device_->CopyDeviceToHost(values.data(), tensor.ptr, tensor.Bytes());
    return values;
  }

  static PackedState Handoff(Qwen38GraphEngine& engine) {
    engine.FastHandoff();
    return {
        Read<int64_t>(engine, *engine.store_.at("past_position_ids")),
        Read<float>(engine, *engine.shared_.at("past_key_values.0.index_key")),
        Read<uint16_t>(engine, *engine.banks_[0].tensors.at("past_key_values.0.index_buffer")),
        Read<int32_t>(engine, *engine.banks_[0].tensors.at("past_key_values.0.index_lengths")),
    };
  }
};

}  // namespace Generators

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

TEST_F(Qwen38GraphNativeTest, UtilizationLimitControlsActualSnapshotReuseAfterNormalization) {
  for (bool retain_snapshots : {false, true}) {
    SCOPED_TRACE(retain_snapshots);
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/synthetic-qwen38");
    config->engine.dynamic_batching.emplace();
    auto& batching = *config->engine.dynamic_batching;
    batching.max_batch_size = 1;
    batching.prefix_caching = true;
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
    config->model.qwen38_graph.warmup = true;
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
    // The tiny target is not folded; attach real dense/head sessions and invoke its handoff explicitly.
    Qwen38GraphNativeTestAccess::EnableDensePrefill(*model);
    auto& device = dynamic_cast<Qwen38GraphEngine&>(*lease);
    const auto check_packed = [](const Qwen38GraphNativeTestAccess::PackedState& packed, size_t length) {
      EXPECT_EQ(packed.lengths, (std::vector<int32_t>{static_cast<int32_t>(length / 4), static_cast<int32_t>(length % 4)}));
      for (size_t i = 0; i < packed.positions.size(); ++i)
        EXPECT_EQ(packed.positions[i], i < length ? static_cast<int64_t>(i) : 0);
      for (size_t i = 0; i < packed.keys.size(); ++i)
        EXPECT_NEAR(packed.keys[i], i / 128 < length / 4 && i % 128 == 0 ? std::sqrt(128.0f) : 0.0f, 0.02f);
      for (size_t i = 0; i < length % 4 * 128; ++i)
        EXPECT_EQ(packed.pending[i], i % 128 == 0 ? uint16_t{0x4300} : uint16_t{0});
    };
    const auto decode = [&] {
      device.BeginDecode(true);
      const auto record = device.WaitCycle();
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
      EXPECT_EQ(storage->bytes, 32u + length * (8u + 128u * 2u));
      EXPECT_EQ(storage->shapes.size(), 4u);
      const auto cold = Qwen38GraphNativeTestAccess::Handoff(device);
      check_packed(cold, length);
      const auto cold_output = decode();

      device.Reset();
      device.RestorePrefix(*saved, prompt);
      EXPECT_EQ(device.PendingToken(), pending);
      const auto warm = Qwen38GraphNativeTestAccess::Handoff(device);
      check_packed(warm, length);
      EXPECT_EQ(warm.positions, cold.positions);
      EXPECT_EQ(warm.keys, cold.keys);
      EXPECT_EQ(decode(), cold_output);

      device.Reset();
      device.RestorePrefix(*saved, prompt);
      const std::vector<int32_t> suffix(5, 0);
      device.PrefillChunk(suffix);
      EXPECT_EQ(device.PendingToken(), static_cast<int32_t>(length + suffix.size() + 9));
      check_packed(Qwen38GraphNativeTestAccess::Handoff(device), length + suffix.size());
      const auto continued = decode();
      device.Reset();
      device.PrefillChunk(std::vector<int32_t>(length + suffix.size(), 0));
      check_packed(Qwen38GraphNativeTestAccess::Handoff(device), length + suffix.size());
      EXPECT_EQ(decode(), continued);
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
