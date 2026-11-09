// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "generator/generators.h"
#include "telemetry_test_environment.h"

#if defined(_WIN32)
#define LEGACY_ADDON_IMPORT __declspec(dllimport)
#else
#define LEGACY_ADDON_IMPORT
#endif

extern "C" {
LEGACY_ADDON_IMPORT uint32_t GetInterfaceVersion();
LEGACY_ADDON_IMPORT void SetLegacyInterfaceVersion(uint32_t version);
LEGACY_ADDON_IMPORT uint32_t GetLegacyInterfaceCallCount();
}

namespace Generators::test {

class LegacyDeviceInterfaceVersionTest : public ::testing::TestWithParam<DeviceType> {
 protected:
  void RejectsVersionBeforeUsingInterface(uint32_t version) {
    SetLegacyInterfaceVersion(version);
    ASSERT_EQ(GetInterfaceVersion(), version);
    const auto calls_before = GetLegacyInterfaceCallCount();
    try {
      GetDeviceInterface(GetParam());
      ADD_FAILURE() << "An add-on without the combined preemption and media layouts must be rejected.";
    } catch (const std::runtime_error& error) {
      EXPECT_EQ(std::string{error.what()},
                "Cuda interface not available: CUDA add-on interface version mismatch: expected " +
                    std::to_string(kDeviceInterfaceVersion) +
                    ", got " + std::to_string(version) +
                    "; install a matching onnxruntime-genai-cuda package");
    }
    EXPECT_EQ(GetLegacyInterfaceCallCount(), calls_before);
  }
};

TEST_P(LegacyDeviceInterfaceVersionTest, RejectsVersion19BeforeUsingItsInterface) {
  RejectsVersionBeforeUsingInterface(19);
}

TEST_P(LegacyDeviceInterfaceVersionTest, RejectsVersion20BeforeUsingItsInterface) {
  RejectsVersionBeforeUsingInterface(20);
}

TEST_P(LegacyDeviceInterfaceVersionTest, RejectsVersion21BeforeUsingItsInterface) {
  RejectsVersionBeforeUsingInterface(21);
}

INSTANTIATE_TEST_SUITE_P(
    CudaAndTensorRtRtx, LegacyDeviceInterfaceVersionTest,
    ::testing::Values(DeviceType::CUDA, DeviceType::NvTensorRtRtx));

}  // namespace Generators::test

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}
