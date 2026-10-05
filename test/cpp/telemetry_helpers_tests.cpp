// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telemetry/telemetry_environment.h"
#include "telemetry/telemetry_sampling.h"
#include "telemetry/telemetry_io.h"
#include "telemetry/telemetry_redaction.h"
#include "models/env_utils.h"
#include "scoped-environment-variable.h"

#include <sstream>
#include <vector>

#include <gtest/gtest.h>

namespace Generators::test {
namespace {

TEST(EnvironmentTests, GetEnvPreservesExistingStringAndBooleanSemantics) {
  ScopedEnvironmentVariable variable{"ORTGENAI_TEST_ENV", std::nullopt};
  EXPECT_TRUE(GetEnv("ORTGENAI_TEST_ENV").empty());
  for (const bool initial : {false, true}) {
    bool value = initial;
    GetEnv("ORTGENAI_TEST_ENV", value);
    EXPECT_EQ(value, initial);
    variable.Set("");
    GetEnv("ORTGENAI_TEST_ENV", value);
    EXPECT_EQ(value, initial);
    variable.Set(std::nullopt);
  }
  for (const char* input : {"1", "true", "0", "false"}) {
    variable.Set(input);
    EXPECT_EQ(GetEnv("ORTGENAI_TEST_ENV"), input);
    bool value = input[0] == '0' || input[0] == 'f';
    GetEnv("ORTGENAI_TEST_ENV", value);
    EXPECT_EQ(value, input[0] == '1' || input[0] == 't');
  }
  for (const char* input : {"TRUE", " yes ", "random"}) {
    variable.Set(input);
    bool value = false;
    EXPECT_THROW(GetEnv("ORTGENAI_TEST_ENV", value), std::invalid_argument);
    EXPECT_FALSE(value);
  }
  const std::string large(kMaxTelemetryInputBytes + 1, 'x');
  variable.Set(large);
  EXPECT_EQ(GetEnv("ORTGENAI_TEST_ENV"), large);
}

TEST(TelemetryStringTests, BoundsAsciiAndAllUtf8Widths) {
  EXPECT_EQ(kMaxTelemetryStringLength, 1024u);
  for (const size_t size : {0u, 1023u, 1024u, 1025u, 1024u * 1024u}) {
    EXPECT_EQ(BoundTelemetryString(std::string(size, 'x')),
              std::string(std::min<size_t>(size, 1024), 'x'));
  }
  for (const std::string codepoint : {"\xC2\xA2", "\xE2\x82\xAC", "\xF0\x9F\x98\x80"}) {
    for (size_t remaining = 1; remaining < codepoint.size(); ++remaining) {
      const std::string prefix(1024 - remaining, 'x');
      EXPECT_EQ(BoundTelemetryString(prefix + codepoint), prefix);
    }
    const std::string exact = std::string(1024 - codepoint.size(), 'x') + codepoint;
    EXPECT_EQ(BoundTelemetryString(exact), exact);
    EXPECT_EQ(BoundTelemetryString(exact + "tail"), exact);
  }
  EXPECT_EQ(BoundTelemetryString(std::string(300, 'x'), 256), std::string(256, 'x'));
}

TEST(TelemetryStringTests, SanitizesMalformedUtf8) {
  EXPECT_EQ(BoundTelemetryString("\x80\xC0\xAF\xED\xA0\x80\xF4\x90\x80\x80"), "??????????");
  EXPECT_EQ(BoundTelemetryString("\xF0\x9F"), "??");
  EXPECT_EQ(BoundTelemetryString("valid \xC2\xA2"), "valid \xC2\xA2");
}

TEST(TelemetryStringTests, DoesNotScanUnterminatedCStringBeyondBudget) {
  const char buffer[] = {'a', 'b', 'c'};
  EXPECT_EQ(BoundedTelemetryCString(buffer, sizeof(buffer)), "abc");
  EXPECT_TRUE(BoundedTelemetryCString(nullptr).empty());
  const std::string huge(1024 * 1024, 'x');
  EXPECT_EQ(BoundedTelemetryCString(huge.c_str()).size(), kMaxTelemetryInputBytes + 1);
}

TEST(TelemetryStringTests, BoundsProviderListStorageAndIteration) {
  EXPECT_EQ(JoinTelemetryStrings(std::vector<std::string>{"CPU", "CUDA"}), "CPU,CUDA");
  EXPECT_EQ(JoinTelemetryStrings(std::vector<std::string>{"CPU", std::string(1024 * 1024, 'x'), "tail"}),
            "CPU," + std::string(1020, 'x'));
  EXPECT_EQ(JoinTelemetryStrings(std::vector<std::string>{std::string(1023, 'x'), "\xC2\xA2"}),
            std::string(1023, 'x') + ",");
  std::vector<std::string> empty_entries(1025);
  empty_entries.back() = "must not be processed";
  EXPECT_TRUE(JoinTelemetryStrings(empty_entries).empty());
}

TEST(TelemetryStringTests, BoundsEventPropertiesUsingTheProductionSetter) {
  struct Event {
    std::string name;
    std::string value;
    void SetProperty(const char* key, std::string property) {
      name = key;
      value = std::move(property);
    }
  } event;
  for (const char* name : {"modelType", "modelFamily", "executionProviders", "selectedDevice",
                           "modality", "inputModality", "cpuModel", "errorType", "context"}) {
    SetTelemetryStringProperty(event, name, std::string(1024 * 1024, 'x'));
    EXPECT_EQ(event.name, name);
    EXPECT_EQ(event.value, std::string(1024, 'x'));
  }
}

TEST(TelemetryInputTests, BoundsFileReadsAndCpuNameParsing) {
  std::istringstream input(std::string(1024 * 1024, 'x'));
  EXPECT_EQ(TelemetryInternal::ReadTelemetryInput(input).size(), kMaxTelemetryInputBytes);
  EXPECT_EQ(input.tellg(), static_cast<std::streamoff>(kMaxTelemetryInputBytes));
  EXPECT_EQ(TelemetryInternal::CpuModelFromTelemetryInput("processor : 0\nmodel name : Test CPU\n"),
            "Test CPU");
  EXPECT_EQ(TelemetryInternal::CpuModelFromTelemetryInput("model name : " + std::string(1024 * 1024, 'x')),
            std::string(1024, 'x'));
  EXPECT_EQ(TelemetryInternal::CpuModelFromTelemetryInput(
                std::string(kMaxTelemetryInputBytes, 'x') + "\nmodel name : hidden"),
            "unknown");
}

TEST(TelemetryInputTests, BoundsEnvironmentEvidenceProcessing) {
  EXPECT_EQ(TelemetryInternal::ToLowerAscii(std::string(1024 * 1024, 'A')).size(),
            TelemetryInternal::kMaxProcessingBytes);
  EXPECT_TRUE(TelemetryInternal::IsNonFalseValue(std::string(kMaxTelemetryInputBytes + 1, ' ')));
  TelemetryInternal::HostEnvironmentEvidence evidence;
  evidence.cgroup = std::string(1024 * 1024, 'x') + "docker";
  evidence.dmi = std::string(1024 * 1024, 'x') + "vmware";
  const auto unknown = TelemetryInternal::ClassifyHostEnvironment(evidence);
  EXPECT_FALSE(unknown.is_container);
  EXPECT_FALSE(unknown.is_virtual_machine);
  evidence.cgroup = "docker" + evidence.cgroup;
  evidence.dmi = "vmware" + evidence.dmi;
  const auto detected = TelemetryInternal::ClassifyHostEnvironment(evidence);
  EXPECT_STREQ(detected.container_type, "docker");
  EXPECT_STREQ(detected.virtualization_type, "vmware");
}

TEST(TelemetrySamplingTests, BoundsGuidHashingWithoutChangingGeneratedGuids) {
  const std::string guid = "11111111-2222-4333-8444-555555555555";
  EXPECT_EQ(TelemetryInternal::HashSamplingKey(guid + std::string(1024 * 1024, 'x'), 42),
            TelemetryInternal::HashSamplingKey(guid, 42));
}

TEST(TelemetrySamplingTests, HonorsBoundaryRates) {
  EXPECT_EQ(TelemetryInternal::kCriticalEventSampleRatePercent, 100.0);
  EXPECT_FALSE(TelemetryInternal::ShouldSampleSession("process-guid", 16, 0.0));
  EXPECT_FALSE(TelemetryInternal::ShouldSampleSession("process-guid", 16, -1.0));
  EXPECT_TRUE(TelemetryInternal::ShouldSampleSession(
      "process-guid", 16, TelemetryInternal::kCriticalEventSampleRatePercent));
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

TEST(TelemetryEnvironmentClassificationTests, MatchesNormalizedContainerEvidence) {
  struct Case {
    const char* value;
    const char* container_type;
  };
  for (const auto& entry : {Case{"KuBePoDs", "kubernetes"}, Case{"LiBpOd", "podman"},
                            Case{"PoDmAn", "podman"}, Case{"DoCkEr", "docker"},
                            Case{"CoNtAiNeRd", "containerd"}, Case{"LxC", "lxc"}}) {
    TelemetryInternal::HostEnvironmentEvidence evidence;
    evidence.cgroup = std::string{"0::/"} + entry.value + "/container";
    EXPECT_STREQ(TelemetryInternal::ClassifyHostEnvironment(evidence).container_type,
                 entry.container_type);
    evidence.cgroup.clear();
    evidence.systemd_container = std::string{" \t"} + entry.value + "\n";
    EXPECT_STREQ(TelemetryInternal::ClassifyHostEnvironment(evidence).container_type,
                 entry.container_type);
  }
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
