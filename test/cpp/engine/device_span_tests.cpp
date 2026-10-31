// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <gtest/gtest.h>

#include "smartptrs.h"

namespace Generators::test {
namespace {

struct UploadRecorder : DeviceBuffer {
  explicit UploadRecorder(size_t bytes) : storage{std::make_unique<uint8_t[]>(bytes)} {
    size_in_bytes_ = bytes;
    p_device_ = p_cpu_ = storage.get();
  }

  const char* GetType() const override { return "recorder"; }
  void AllocateCpu() override {}
  void CopyDeviceToCpu() override {}
  void CopyCpuToDevice() override { ++whole_uploads; }
  void CopyFrom(size_t, DeviceBuffer&, size_t, size_t) override {}
  void Zero() override {}

  std::unique_ptr<uint8_t[]> storage;
  size_t whole_uploads{};
};

struct RangeUploadRecorder final : UploadRecorder {
  using UploadRecorder::UploadRecorder;

  void CopyCpuRangeToDevice(size_t begin_bytes, size_t size_in_bytes) override {
    range = std::pair{begin_bytes, size_in_bytes};
  }

  std::optional<std::pair<size_t, size_t>> range;
};

}  // namespace

// DeviceBuffer objects are created by the dynamically loaded add-ons and called through by the host (and the other way
// around), so adding the virtual CopyCpuRangeToDevice moved the slots of CopyFrom, Zero and the methods after them. A host
// that accepted an add-on built before it would dispatch to the wrong methods; the version the host demands must therefore
// stay past the layout that introduced it.
TEST(DeviceInterfaceVersionTest, IsPastTheDeviceBufferRangedUploadLayout) {
  constexpr uint32_t kVersionOfTheRangedUpload = 15;
  EXPECT_GE(kDeviceInterfaceVersion, kVersionOfTheRangedUpload);
}

TEST(DeviceSpanTest, RangeUploadForwardsTheByteRangeOfTheSubspan) {
  auto buffer = std::make_shared<RangeUploadRecorder>(16 * sizeof(int32_t));
  auto* recorder = buffer.get();
  DeviceSpan<int32_t> whole{std::move(buffer)};

  whole.subspan(4, 3).CopyCpuRangeToDevice();

  ASSERT_TRUE(recorder->range.has_value());
  EXPECT_EQ(recorder->range->first, 4 * sizeof(int32_t));
  EXPECT_EQ(recorder->range->second, 3 * sizeof(int32_t));
  EXPECT_EQ(recorder->whole_uploads, 0u);
}

TEST(DeviceSpanTest, RangeUploadFallsBackToTheWholeBufferWithoutAnOverride) {
  auto buffer = std::make_shared<UploadRecorder>(16 * sizeof(int32_t));
  auto* recorder = buffer.get();
  DeviceSpan<int32_t> whole{std::move(buffer)};

  whole.subspan(4, 3).CopyCpuRangeToDevice();

  EXPECT_EQ(recorder->whole_uploads, 1u);
}

TEST(DeviceSpanTest, CopyCpuToDeviceStillUploadsTheWholeBufferForASubspan) {
  auto buffer = std::make_shared<RangeUploadRecorder>(16 * sizeof(int32_t));
  auto* recorder = buffer.get();
  DeviceSpan<int32_t> whole{std::move(buffer)};

  whole.subspan(4, 3).CopyCpuToDevice();

  EXPECT_EQ(recorder->whole_uploads, 1u);
  EXPECT_FALSE(recorder->range.has_value());
}

}  // namespace Generators::test
