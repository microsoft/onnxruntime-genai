// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "telemetry/hardware_info.h"
#include "telemetry/telemetry_string.h"

namespace {

using namespace Generators;
using namespace Generators::TelemetryInternal;

TEST(HardwareInfoTests, StrictIntegerParsingRejectsMalformedAndNegativeValues) {
  EXPECT_EQ(ParseHardwareInteger(" 1234\n"), 1234);
  EXPECT_EQ(ParseHardwareInteger("10de", 16), 0x10de);
  EXPECT_EQ(ParseHardwareInteger(""), -1);
  EXPECT_EQ(ParseHardwareInteger("-1"), -1);
  EXPECT_EQ(ParseHardwareInteger("12 trailing"), -1);
  EXPECT_EQ(ParseHardwareInteger("xyz", 16), -1);
}

TEST(HardwareInfoTests, CpuInfoParsingIsBoundedAndRetainsNoUniqueIdentifiers) {
  const auto cpu = ParseCpuInfo(
      "processor: 0\nvendor_id: GenuineIntel\nmodel name: Example CPU\n"
      "serial: private\nHardware: private\n\nprocessor: 1\n");
  EXPECT_EQ(cpu.type, "CPU");
  EXPECT_EQ(cpu.vendor, "GenuineIntel");
  EXPECT_EQ(cpu.name, "Example CPU");
  EXPECT_LE(cpu.vendor.size(), kMaxTelemetryStringLength);
  EXPECT_LE(cpu.name.size(), kMaxTelemetryStringLength);
}

TEST(HardwareInfoTests, CpuInfoParsingRedactsPathsBeforeRetention) {
  const auto cpu = ParseCpuInfo(
      "vendor_id: Vendor /private/user/vendor\n"
      "model name: GPU /private/user/model\n\n");

  EXPECT_EQ(cpu.vendor, "Vendor [path]");
  EXPECT_EQ(cpu.name, "GPU [path]");
}

TEST(HardwareInfoTests, OrtFallbackResolvesPlatformUnknownWithoutMergingOrtDevices) {
  std::vector<HardwareDeviceInfo> platform_devices{
      {.type = "Unknown",
       .source = "dxcore",
       .vendor_id = 0x1234,
       .device_id = 0x5678}};
  const HardwareDeviceInfo ort_device{
      .type = "GPU",
      .source = "onnxruntime",
      .vendor_id = 0x1234,
      .device_id = 0x5678};

  std::vector<HardwareDeviceInfo> ort_devices{ort_device};
  EXPECT_EQ(FindOrtHardwareDeviceMatch(platform_devices, ort_device, ort_devices),
            &platform_devices.front());
  EXPECT_EQ(FindOrtHardwareDeviceMatch(ort_devices, ort_device, ort_devices), nullptr);
}

TEST(HardwareInfoTests, OrtProductMatchingRequiresUnambiguousUnknownDevice) {
  std::vector<HardwareDeviceInfo> platform_devices{
      {.type = "GPU", .source = "nvml", .vendor_id = 0x10de}};
  std::vector<HardwareDeviceInfo> ort_devices{
      {.type = "GPU", .source = "onnxruntime", .vendor_id = 0x10de, .device_id = 0x1234}};

  EXPECT_EQ(FindOrtHardwareDeviceMatch(platform_devices, ort_devices.front(), ort_devices),
            &platform_devices.front());

  ort_devices.push_back(
      {.type = "GPU", .source = "onnxruntime", .vendor_id = 0x10de, .device_id = 0x5678});
  EXPECT_EQ(FindOrtHardwareDeviceMatch(platform_devices, ort_devices.front(), ort_devices), nullptr);
  EXPECT_EQ(FindOrtHardwareDeviceMatch(platform_devices, ort_devices.back(), ort_devices), nullptr);

  ort_devices.pop_back();
  platform_devices.push_back(platform_devices.front());
  EXPECT_EQ(FindOrtHardwareDeviceMatch(platform_devices, ort_devices.front(), ort_devices), nullptr);

  platform_devices.back().device_id = ort_devices.front().device_id;
  EXPECT_EQ(FindOrtHardwareDeviceMatch(platform_devices, ort_devices.front(), ort_devices),
            &platform_devices.back());
}

TEST(HardwareInfoTests, NvmlProcessPolicyRejectsContainersSentinelsAndOverflow) {
  EXPECT_FALSE(ShouldQueryNvmlProcessMemory(true));
  EXPECT_TRUE(ShouldQueryNvmlProcessMemory(false));

  uint64_t total = 100;
  EXPECT_FALSE(AccumulateProcessMemoryBytes(
      total, (std::numeric_limits<uint64_t>::max)()));
  EXPECT_EQ(total, 100u);

  total = (std::numeric_limits<uint64_t>::max)() - 4;
  EXPECT_FALSE(AccumulateProcessMemoryBytes(total, 5));
  EXPECT_EQ(total, (std::numeric_limits<uint64_t>::max)() - 4);

  total = 100;
  EXPECT_TRUE(AccumulateProcessMemoryBytes(total, 25));
  EXPECT_EQ(total, 125u);
}

TEST(HardwareInfoTests, ProcessCpuTimeDeltaPreservesUnknownSemantics) {
  EXPECT_EQ(ProcessCpuTimeDelta(100, 145), 45);
  EXPECT_EQ(ProcessCpuTimeDelta(-1, 145), -1);
  EXPECT_EQ(ProcessCpuTimeDelta(145, -1), -1);
  EXPECT_EQ(ProcessCpuTimeDelta(145, 100), -1);
}

TEST(HardwareInfoTests, ResourceWindowKeepsActualCompletionSampleWhenEmissionIsDelayed) {
  ResourceSampleWindow window;
  window.Begin(ProcessResourceUsage{.resident_memory_mb = 10, .cpu_time_ms = 100});
  window.Complete(ProcessResourceUsage{.resident_memory_mb = 20, .cpu_time_ms = 140});
  window.Complete(ProcessResourceUsage{.resident_memory_mb = 999, .cpu_time_ms = 999});

  EXPECT_TRUE(window.IsComplete());
  EXPECT_EQ(window.End().resident_memory_mb, 20);
  EXPECT_EQ(ProcessCpuTimeDelta(window.Start().cpu_time_ms, window.End().cpu_time_ms), 40);

  window.Reset();
  EXPECT_FALSE(window.IsComplete());
  EXPECT_EQ(window.End().resident_memory_mb, -1);
}

TEST(HardwareInfoTests, ResourceSamplingStartsOnlyAtFirstLogicalAppend) {
  ResourceSampleWindow window;
  EXPECT_TRUE(ShouldBeginResourceSample(0, false, window.IsStarted(), true));
  window.Begin(ProcessResourceUsage{.resident_memory_mb = 10, .cpu_time_ms = 100});

  EXPECT_FALSE(ShouldBeginResourceSample(0, false, window.IsStarted(), true));
  EXPECT_FALSE(ShouldBeginResourceSample(4, false, window.IsStarted(), true));
  EXPECT_FALSE(ShouldBeginResourceSample(0, true, window.IsStarted(), true));
  EXPECT_EQ(window.Start().cpu_time_ms, 100);
}

TEST(HardwareInfoTests, UnsampledSessionsNeverBeginResourceSampling) {
  EXPECT_FALSE(ShouldBeginResourceSample(0, false, false, false));
  EXPECT_FALSE(ShouldBeginResourceSample(4, false, false, false));
  EXPECT_FALSE(ShouldBeginResourceSample(0, true, false, false));
}

TEST(HardwareInfoTests, CurrentProcessSamplesUseExplicitUnknowns) {
  const auto usage = CollectProcessResourceUsage();
  EXPECT_GE(usage.cpu_time_ms, 0);
  EXPECT_GE(usage.resident_memory_mb, 0);
  EXPECT_GE(usage.gpu_memory_mb, -1);
  EXPECT_GE(usage.npu_memory_mb, -1);
  EXPECT_FALSE(usage.accelerator_memory_source.empty());
}

TEST(HardwareInfoTests, HardwareInventoryIsBoundedAndContainsCpu) {
  const auto info = CollectHardwareInfo();
  EXPECT_LE(info.devices.size(), kMaxHardwareInventoryEntries);
  ASSERT_FALSE(info.devices.empty());
  EXPECT_EQ(info.devices.front().type, "CPU");
  for (const auto& device : info.devices) {
    EXPECT_LE(device.name.size(), kMaxTelemetryStringLength);
    EXPECT_LE(device.vendor.size(), kMaxTelemetryStringLength);
    EXPECT_LE(device.driver_version.size(), kMaxTelemetryStringLength);
  }
}

}  // namespace
