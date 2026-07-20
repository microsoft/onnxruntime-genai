// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telemetry/telemetry_sampling.h"

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

namespace Generators::test {

TEST(TelemetrySamplingTest, BoundaryRates) {
  EXPECT_FALSE(TelemetryInternal::ShouldSampleSession("process-a", 1, 0.0));
  EXPECT_FALSE(TelemetryInternal::ShouldSampleSession("process-a", 1, -1.0));
  EXPECT_FALSE(TelemetryInternal::ShouldSampleSession(
      "process-a", 1, std::numeric_limits<double>::quiet_NaN()));
  EXPECT_TRUE(TelemetryInternal::ShouldSampleSession("process-a", 1, 100.0));
  EXPECT_TRUE(TelemetryInternal::ShouldSampleSession("process-a", 1, 101.0));
}

TEST(TelemetrySamplingTest, DecisionIsStableForCorrelatedEvents) {
  const bool model_load = TelemetryInternal::ShouldSampleSession("process-a", 42, 10.0);
  const bool generator_create = TelemetryInternal::ShouldSampleSession("process-a", 42, 10.0);
  const bool generate_start = TelemetryInternal::ShouldSampleSession("process-a", 42, 10.0);
  const bool generate_end = TelemetryInternal::ShouldSampleSession("process-a", 42, 10.0);

  EXPECT_EQ(model_load, generator_create);
  EXPECT_EQ(model_load, generate_start);
  EXPECT_EQ(model_load, generate_end);
}

TEST(TelemetrySamplingTest, TenPercentRateHasExpectedDistribution) {
  int sampled = 0;
  constexpr int kSessionCount = 100000;
  for (uint32_t session_id = 0; session_id < kSessionCount; ++session_id) {
    if (TelemetryInternal::ShouldSampleSession("process-a", session_id, 10.0)) ++sampled;
  }

  EXPECT_GT(sampled, 9700);
  EXPECT_LT(sampled, 10300);
}

TEST(TelemetrySamplingTest, ProcessGuidAffectsDecision) {
  bool found_difference = false;
  for (uint32_t session_id = 0; session_id < 1000; ++session_id) {
    if (TelemetryInternal::ShouldSampleSession("process-a", session_id, 10.0) !=
        TelemetryInternal::ShouldSampleSession("process-b", session_id, 10.0)) {
      found_difference = true;
      break;
    }
  }
  EXPECT_TRUE(found_difference);
}

}  // namespace Generators::test
