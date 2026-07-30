// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "../src/telemetry/engine_telemetry.h"
#include "../src/telemetry/telemetry.h"

#if defined(ORTGENAI_ENABLE_TELEMETRY)

#include <optional>
#include <string>

namespace Generators::test {

namespace {

struct CapturedEngineRequest {
  uint32_t session_id{};
  uint32_t engine_id{};
  uint32_t request_id{};
  EngineRequestInfo info;
  std::string outcome;
  int64_t end_timestamp_ms{};
};

std::optional<CapturedEngineRequest> g_capture;
size_t g_capture_count{};

void CaptureEngineRequest(uint32_t session_id, uint32_t engine_id, uint32_t request_id,
                          const EngineRequestInfo& info, int64_t end_timestamp_ms) {
  g_capture = CapturedEngineRequest{
      session_id,
      engine_id,
      request_id,
      info,
      std::string{info.outcome},
      end_timestamp_ms,
  };
  g_capture->info.outcome = g_capture->outcome;
  ++g_capture_count;
}

class EngineTelemetryTests : public ::testing::Test {
 protected:
  void SetUp() override {
    g_capture.reset();
    g_capture_count = 0;
    EngineRequestTelemetry::SetFinalizeCallbackForTest(CaptureEngineRequest);
  }

  void TearDown() override {
    EngineRequestTelemetry::SetFinalizeCallbackForTest(nullptr);
  }
};

TEST_F(EngineTelemetryTests, CompletedRequestAggregatesMetricsOnce) {
  EngineRequestTelemetry telemetry;
  telemetry.Begin(7, 11, true, 10);
  telemetry.OnScheduled();
  telemetry.OnStep(11, 3, false);
  telemetry.OnStep(14, 2, true);
  telemetry.OnRemoved(true);

  ASSERT_TRUE(g_capture.has_value());
  EXPECT_EQ(g_capture_count, 1U);
  EXPECT_EQ(g_capture->session_id, 7U);
  EXPECT_EQ(g_capture->engine_id, 11U);
  EXPECT_NE(g_capture->request_id, 0U);
  EXPECT_TRUE(g_capture->info.dynamic_batching);
  EXPECT_EQ(g_capture->info.prompt_tokens, 10);
  EXPECT_EQ(g_capture->info.generated_tokens, 4);
  EXPECT_EQ(g_capture->info.step_count, 2);
  EXPECT_DOUBLE_EQ(g_capture->info.average_batch_size, 2.5);
  EXPECT_EQ(g_capture->info.max_batch_size, 3);
  EXPECT_EQ(g_capture->outcome, "completed");
  EXPECT_GE(g_capture->info.queue_time_ms, 0.0);
  EXPECT_GE(g_capture->info.time_to_first_token_ms, g_capture->info.queue_time_ms);
  EXPECT_GE(g_capture->info.total_time_ms, g_capture->info.time_to_first_token_ms);
  EXPECT_GT(g_capture->end_timestamp_ms, 0);
}

TEST_F(EngineTelemetryTests, ExplicitRemovalIsNotAbandonment) {
  EngineRequestTelemetry telemetry;
  telemetry.Begin(1, 2, false, 5);
  telemetry.OnScheduled();
  telemetry.OnStep(6, 1, false);
  telemetry.OnRemoved(false);

  ASSERT_TRUE(g_capture.has_value());
  EXPECT_EQ(g_capture_count, 1U);
  EXPECT_EQ(g_capture->outcome, "removed");
  EXPECT_FALSE(g_capture->info.dynamic_batching);
  EXPECT_EQ(g_capture->info.generated_tokens, 1);
}

TEST_F(EngineTelemetryTests, DestructionWithoutRemovalIsAbandoned) {
  {
    EngineRequestTelemetry telemetry;
    telemetry.Begin(1, 2, false, 5);
    telemetry.OnScheduled();
    telemetry.OnStep(6, 1, false);
  }

  ASSERT_TRUE(g_capture.has_value());
  EXPECT_EQ(g_capture_count, 1U);
  EXPECT_EQ(g_capture->outcome, "abandoned");
}

}  // namespace

}  // namespace Generators::test

#endif
