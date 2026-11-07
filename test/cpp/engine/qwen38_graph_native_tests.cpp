// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>

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

TEST_F(Qwen38GraphNativeTest, AdaptiveBudgetEnvironmentDefaultsAndOptOutReachTheNativeSession) {
  struct Case {
    std::optional<std::string> value;
    const char* reserve;
  };
  const Case cases[] = {{std::nullopt, "5901778944"}, {"", "5901778944"}, {"1", "5901778944"}, {"0", "6442450944"}};
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

}  // namespace
}  // namespace Generators::test

#endif
