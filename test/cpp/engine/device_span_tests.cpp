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
  void CopyDeviceToCpu() override { ++whole_downloads; }
  void CopyCpuToDevice() override { ++whole_uploads; }
  void CopyFrom(size_t, DeviceBuffer&, size_t, size_t) override {}
  void Zero() override {}

  std::unique_ptr<uint8_t[]> storage;
  size_t whole_uploads{};
  size_t whole_downloads{};
};

struct RangeUploadRecorder final : UploadRecorder {
  using UploadRecorder::UploadRecorder;

  void CopyCpuRangeToDevice(size_t begin_bytes, size_t size_in_bytes) override {
    range = std::pair{begin_bytes, size_in_bytes};
  }

  std::optional<std::pair<size_t, size_t>> range;
};

struct RangeDownloadRecorder final : UploadRecorder {
  using UploadRecorder::UploadRecorder;
  void CopyDeviceRangeToCpu(size_t begin_bytes, size_t bytes) override {
    range = std::pair{begin_bytes, bytes};
  }
  std::optional<std::pair<size_t, size_t>> range;
};

}  // namespace

// DeviceBuffer objects are created by the dynamically loaded add-ons and called through by the host (and the other way
// around), so adding the virtual CopyCpuRangeToDevice moved the slots of CopyFrom, Zero and the methods after them. A host
// that accepted an add-on built before it would dispatch to the wrong methods; the version the host demands must therefore
// stay past the layout that introduced it. An add-on built from a tree without the ranged upload reports at most 17, the
// version that added the qwen38_graph member to Config::Model.
TEST(DeviceInterfaceVersionTest, IsPastTheDeviceBufferRangedUploadLayout) {
  constexpr uint32_t kVersionOfTheRangedUpload = 18;
  EXPECT_GE(kDeviceInterfaceVersion, kVersionOfTheRangedUpload);
}

// Config is built by the host and read by the dynamically loaded add-ons, so the max_batch_size member that Qwen38Graph
// gained moved the offsets of the members of Config::Model and Config after it. An add-on built before it would read
// them from the wrong place; the version the host demands must therefore stay past the layout that introduced it. An
// add-on built from a tree without the member reports at most 18.
TEST(DeviceInterfaceVersionTest, IsPastTheQwen38GraphMaxBatchSizeConfigLayout) {
  constexpr uint32_t kVersionOfTheMaxBatchSize = 19;
  EXPECT_GE(kDeviceInterfaceVersion, kVersionOfTheMaxBatchSize);
}

TEST(DeviceInterfaceVersionTest, IsPastTheStatePromptInputLayout) {
  constexpr uint32_t kVersionOfThePromptInput = 21;
  EXPECT_GE(kDeviceInterfaceVersion, kVersionOfThePromptInput);
}

TEST(DeviceInterfaceVersionTest, IsPastTheStateInputSelectionLayout) {
  constexpr uint32_t kVersionOfTheInputSelection = 22;
  EXPECT_GE(kDeviceInterfaceVersion, kVersionOfTheInputSelection);
}

TEST(DeviceInterfaceVersionTest, IsPastTheRangedDownloadAndPromptChunkLayout) {
  constexpr uint32_t kVersionOfTheRangedDownloadAndPromptChunk = 23;
  EXPECT_GE(kDeviceInterfaceVersion, kVersionOfTheRangedDownloadAndPromptChunk);
}

TEST(DeviceInterfaceVersionTest, IsPastTheEngineMediaBudgetConfigLayout) {
  constexpr uint32_t kVersionOfTheEngineMediaBudget = 24;
  EXPECT_GE(kDeviceInterfaceVersion, kVersionOfTheEngineMediaBudget);
}

TEST(DeviceSpanTest, RangeDownloadForwardsTheSubspanAndReturnsTheSameHostRange) {
  auto buffer = std::make_shared<RangeDownloadRecorder>(16 * sizeof(int32_t));
  auto* recorder = buffer.get();
  DeviceSpan<int32_t> whole{std::move(buffer)};
  whole.CpuSpan()[4] = 7;
  auto values = whole.subspan(4, 3).CopyDeviceRangeToCpu();
  ASSERT_TRUE(recorder->range.has_value());
  EXPECT_EQ(*recorder->range, (std::pair<size_t, size_t>{4 * sizeof(int32_t), 3 * sizeof(int32_t)}));
  EXPECT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0], 7);
}

TEST(DeviceSpanTest, RangeDownloadFallsBackToTheAuthoritativeWholeBuffer) {
  auto buffer = std::make_shared<UploadRecorder>(16 * sizeof(int32_t));
  auto* recorder = buffer.get();
  DeviceSpan<int32_t> whole{std::move(buffer)};
  whole.CpuSpan()[4] = 7;
  auto values = whole.subspan(4, 3).CopyDeviceRangeToCpu();
  EXPECT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0], 7);
  EXPECT_EQ(recorder->whole_downloads, 1u);
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
