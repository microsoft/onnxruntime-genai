// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telemetry/telemetry_environment.h"
#include "telemetry/telemetry_sampling.h"

#include <gtest/gtest.h>

namespace Generators::test {
namespace {

TEST(TelemetrySamplingTests, HonorsBoundaryRates) {
  EXPECT_FALSE(TelemetryInternal::ShouldSampleSession("process-guid", 16, 0.0));
  EXPECT_FALSE(TelemetryInternal::ShouldSampleSession("process-guid", 16, -1.0));
  EXPECT_TRUE(TelemetryInternal::ShouldSampleSession("process-guid", 16, 100.0));
  EXPECT_TRUE(TelemetryInternal::ShouldSampleSession("process-guid", 16, 101.0));
}

TEST(TelemetrySamplingTests, UsesStableOnePercentBuckets) {
  EXPECT_EQ(TelemetryInternal::HashSamplingKey("process-guid", 16),
            12086635946954002988ULL);
  EXPECT_TRUE(TelemetryInternal::ShouldSampleSession("process-guid", 16));

  EXPECT_EQ(TelemetryInternal::HashSamplingKey("process-guid", 42),
            16731315322573479350ULL);
  for (int repetition = 0; repetition < 10; ++repetition) {
    EXPECT_FALSE(TelemetryInternal::ShouldSampleSession("process-guid", 42));
  }
}

TEST(TelemetryEnvironmentClassificationTests, LeavesUnknownHostsUndetected) {
  const auto info =
      TelemetryInternal::ClassifyHostEnvironment(TelemetryInternal::HostEnvironmentEvidence{});

  EXPECT_FALSE(info.is_container);
  EXPECT_FALSE(info.is_virtual_machine);
  EXPECT_FALSE(info.is_emulator);
  EXPECT_STREQ(info.container_type, "none");
  EXPECT_STREQ(info.virtualization_type, "none");
  EXPECT_STREQ(info.environment_class, "undetected");
  EXPECT_STREQ(info.detection_confidence, "none");
  EXPECT_STREQ(info.device_id_scope, "installation");
}

TEST(TelemetryEnvironmentClassificationTests, PrioritizesSpecificContainerEvidence) {
  TelemetryInternal::HostEnvironmentEvidence evidence;
  evidence.docker_marker = true;
  evidence.kubernetes = true;

  const auto info = TelemetryInternal::ClassifyHostEnvironment(evidence);

  EXPECT_TRUE(info.is_container);
  EXPECT_STREQ(info.container_type, "kubernetes");
  EXPECT_STREQ(info.environment_class, "container");
  EXPECT_STREQ(info.detection_confidence, "high");
  EXPECT_STREQ(info.device_id_scope, "container");
}

TEST(TelemetryEnvironmentClassificationTests, ClassifiesContainersOnVirtualMachines) {
  TelemetryInternal::HostEnvironmentEvidence evidence;
  evidence.cgroup = "0::/docker/0123456789abcdef";
  evidence.kernel_release = "6.6.87.2-microsoft-standard-WSL2";

  const auto info = TelemetryInternal::ClassifyHostEnvironment(evidence);

  EXPECT_TRUE(info.is_container);
  EXPECT_TRUE(info.is_virtual_machine);
  EXPECT_STREQ(info.container_type, "docker");
  EXPECT_STREQ(info.virtualization_type, "wsl");
  EXPECT_STREQ(info.environment_class, "containerOnVirtualMachine");
  EXPECT_STREQ(info.device_id_scope, "container");
}

TEST(TelemetryEnvironmentClassificationTests, ClassifiesAndroidEmulators) {
  TelemetryInternal::HostEnvironmentEvidence evidence;
  evidence.android_emulator = true;

  const auto info = TelemetryInternal::ClassifyHostEnvironment(evidence);

  EXPECT_TRUE(info.is_virtual_machine);
  EXPECT_TRUE(info.is_emulator);
  EXPECT_STREQ(info.virtualization_type, "androidEmulator");
  EXPECT_STREQ(info.environment_class, "emulator");
  EXPECT_STREQ(info.detection_confidence, "high");
}

}  // namespace
}  // namespace Generators::test

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
