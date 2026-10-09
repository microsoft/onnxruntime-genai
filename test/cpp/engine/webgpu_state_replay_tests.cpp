// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "ep/webgpu/interface.h"
#include "models/graph_executor.h"
#include "models/model.h"
#include "models/session_options.h"

namespace Generators::test {
namespace {

TEST(DeviceSpanTest, BackingAndAbsoluteByteOffsetSurviveNestedViews) {
  auto buffer = GetDeviceInterface(DeviceType::CPU)->Allocate<float>(32);
  auto view = buffer.subspan(5, 20).subspan(3, 4);
  DeviceSpan<const float> const_view = view;
  EXPECT_EQ(buffer.BackingBuffer(), view.BackingBuffer());
  EXPECT_EQ(view.BackingBuffer(), const_view.BackingBuffer());
  EXPECT_EQ(buffer.ByteOffset(), 0u);
  EXPECT_EQ(view.ByteOffset(), 8 * sizeof(float));
  EXPECT_EQ(const_view.ByteOffset(), view.ByteOffset());
  view.Span()[0] = 3.5f;
  EXPECT_EQ(buffer.Span()[8], 3.5f);
  EXPECT_EQ(const_view.Span()[0], 3.5f);
  DeviceSpan<float> empty;
  EXPECT_EQ(empty.BackingBuffer(), nullptr);
  EXPECT_EQ(empty.ByteOffset(), 0u);
}

TEST(GraphBuilderTest, DefaultOnnxDomainStillExecutesCast) {
  std::array<float, 3> source{1.75f, -2.5f, 3.0f};
  std::array<int32_t, 3> destination{};
  auto memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  ExecuteCastOp(source.data(), destination.data(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                source.size(), DeviceType::CPU, "", memory_info.get());
  EXPECT_EQ(destination, (std::array<int32_t, 3>{1, -2, 3}));
}

std::string ConfigEntry(const OrtSessionOptions& options, const char* key) {
  size_t length = 0;
  Ort::ThrowOnError(Ort::api->GetSessionConfigEntry(&options, key, nullptr, &length));
  std::string value(length, '\0');
  Ort::ThrowOnError(Ort::api->GetSessionConfigEntry(&options, key, value.data(), &length));
  value.resize(length - 1);
  return value;
}

class WebGpuStateReplayHelperTest : public ::testing::TestWithParam<bool> {
 protected:
  static constexpr size_t kHv = 3, kDv = 2, kDk = 4, kHk = 2, kCapacity = 4;
  static constexpr size_t kStateElements = kHv * kDv * kDk;
  static constexpr size_t kDecayElements = kCapacity * kHv;
  static constexpr size_t kKeyElements = kCapacity * kHk * kDk;
  static constexpr size_t kDeltaElements = kCapacity * kHv * kDv;
  static constexpr size_t kCapsuleElements = kDecayElements + kKeyElements + kDeltaElements;
  static constexpr float kSentinel = -777.0f;

  void SetUp() override {
    GetOrtEnv();
    const auto providers = Ort::GetAvailableProviders();
    if (std::find(providers.begin(), providers.end(), "WebGpuExecutionProvider") == providers.end() &&
        FindRegisteredEpDevices("WebGpuExecutionProvider").empty()) {
      GTEST_SKIP() << "No native or registered WebGPU EP is available.";
    }
    device_ = GetDeviceInterface(DeviceType::WEBGPU);
    Config config;
    config.model.decoder.session_options.provider_options.emplace_back(
        "WebGPU", std::vector<std::pair<std::string, std::string>>{
                      {"deviceId", "0"}, {"validationMode", "full"}, {"maxNumPendingDispatches", "1"}, {"enableGraphCapture", "1"}});
    EnsureDeviceOrtInit(*device_, config);
    const auto& options =
        GetOrtGlobals()->device_allocators_[static_cast<int>(DeviceType::WEBGPU)].session_options_;
    ASSERT_NE(options, nullptr);
    EXPECT_EQ(ConfigEntry(*options, "ep.webgpuexecutionprovider.deviceId"), "0");
    EXPECT_EQ(ConfigEntry(*options, "ep.webgpuexecutionprovider.validationMode"), "full");
    EXPECT_EQ(ConfigEntry(*options, "ep.webgpuexecutionprovider.maxNumPendingDispatches"), "1");
    EXPECT_EQ(ConfigEntry(*options, "ep.webgpuexecutionprovider.enableGraphCapture"), "0");

    capsule_ = device_->Allocate<uint8_t>(3 * kCapsuleElements * sizeof(float));
    destination_ = device_->Allocate<uint8_t>(3 * kStateElements * sizeof(float));
    if (GetParam()) {
      source_ = capsule_;
    } else {
      // Exercise borrowed OrtValue-backed storage, as used by FixedStatePool.
      const std::array<int64_t, 1> shape{3 * kStateElements};
      source_tensor_ = OrtValue::CreateTensor(device_->GetAllocator(), shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      source_ = DeviceSpan<uint8_t>(device_->WrapMemoryBase(
          source_tensor_->GetTensorMutableRawData(), 3 * kStateElements * sizeof(float)));
    }
  }

  StateUpdateReplayDesc Descriptor(uint32_t kept_count) {
    StateUpdateReplayDesc desc{};
    desc.source_state = source_.subspan(kStateElements * sizeof(float), kStateElements * sizeof(float));
    desc.destination_state = destination_.subspan(kStateElements * sizeof(float), kStateElements * sizeof(float));
    auto row = capsule_.subspan(kCapsuleElements * sizeof(float), kCapsuleElements * sizeof(float));
    desc.decay = row.subspan(0, kDecayElements * sizeof(float));
    desc.key = row.subspan(kDecayElements * sizeof(float), kKeyElements * sizeof(float));
    desc.delta = row.subspan((kDecayElements + kKeyElements) * sizeof(float), kDeltaElements * sizeof(float));
    desc.channel_count = kHv;
    desc.state_width = kDv;
    desc.key_width = kDk;
    desc.key_head_count = kHk;
    desc.capacity = kCapacity;
    desc.kept_count = kept_count;
    desc.element_size = sizeof(float);
    desc.kind = StateUpdateReplayKind::GatedDeltaNet;
    return desc;
  }

  void Upload(uint32_t kept_count) {
    capsule_values_.assign(3 * kCapsuleElements, std::numeric_limits<float>::quiet_NaN());
    for (size_t t = 0; t < kept_count; ++t) {
      for (size_t h = 0; h < kHv; ++h) {
        capsule_values_[kCapsuleElements + t * kHv + h] = 0.75f + 0.01f * static_cast<float>(t + h);
        for (size_t v = 0; v < kDv; ++v) {
          capsule_values_[kCapsuleElements + kDecayElements + kKeyElements + (t * kHv + h) * kDv + v] =
              0.02f * static_cast<float>(1 + t + h + v);
        }
      }
      for (size_t i = 0; i < kHk * kDk; ++i) {
        capsule_values_[kCapsuleElements + kDecayElements + t * kHk * kDk + i] =
            -0.03f * static_cast<float>(1 + t + i);
      }
    }
    source_values_.assign(source_.size() / sizeof(float), 0.25f);
    for (size_t i = 0; i < kStateElements; ++i) {
      source_values_[kStateElements + i] = 0.1f * static_cast<float>(i) - 0.6f;
    }
    if (GetParam()) {
      std::copy_n(source_values_.begin() + kStateElements, kStateElements,
                  capsule_values_.begin() + kStateElements);
      source_values_ = capsule_values_;
    } else {
      source_.BackingBuffer()->CopyFromCpu(source_values_.data(), source_values_.size() * sizeof(float));
    }
    capsule_.BackingBuffer()->CopyFromCpu(capsule_values_.data(), capsule_values_.size() * sizeof(float));
    const std::vector<float> destination(3 * kStateElements, kSentinel);
    destination_.BackingBuffer()->CopyFromCpu(destination.data(), destination.size() * sizeof(float));
  }

  static std::vector<float> Read(DeviceSpan<uint8_t> buffer) {
    auto bytes = buffer.CopyDeviceToCpu();
    std::vector<float> values(bytes.size() / sizeof(float));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
  }

  DeviceInterface* device_{};
  std::unique_ptr<OrtValue> source_tensor_;
  DeviceSpan<uint8_t> source_, capsule_, destination_;
  std::vector<float> source_values_, capsule_values_;
};

TEST_P(WebGpuStateReplayHelperTest, ReplaysEveryPrefixOnExistingBackings) {
  // The second logical capsule row starts 16 bytes past a 256-byte binding boundary.
  ASSERT_EQ((kCapsuleElements * sizeof(float)) % 256, 16u);
  for (uint32_t kept = 1; kept <= kCapacity; ++kept) {
    SCOPED_TRACE(kept);
    Upload(kept);
    auto descriptor = Descriptor(kept);
    EXPECT_EQ(descriptor.decay.ByteOffset(), kCapsuleElements * sizeof(float));
    WebGPU::RunGatedDeltaNetStateReplay(descriptor);

    std::vector<float> expected(source_values_.begin() + kStateElements,
                                source_values_.begin() + 2 * kStateElements);
    // Independent transition-major oracle, not ReplayStateUpdatesOnCpu.
    for (size_t t = 0; t < kept; ++t) {
      for (size_t h = 0; h < kHv; ++h) {
        const auto hk = h * kHk / kHv;
        const float decay = capsule_values_[kCapsuleElements + t * kHv + h];
        for (size_t v = 0; v < kDv; ++v) {
          const float delta = capsule_values_[kCapsuleElements + kDecayElements + kKeyElements + (t * kHv + h) * kDv + v];
          for (size_t i = 0; i < kDk; ++i) {
            auto& state = expected[(h * kDv + v) * kDk + i];
            state = state * decay +
                    capsule_values_[kCapsuleElements + kDecayElements + (t * kHk + hk) * kDk + i] * delta;
          }
        }
      }
    }
    const auto actual = Read(destination_);
    for (size_t i = 0; i < actual.size(); ++i) {
      if (i >= kStateElements && i < 2 * kStateElements) {
        const float reference = expected[i - kStateElements];
        EXPECT_NEAR(actual[i], reference, 2e-6f + 5e-6f * std::abs(reference));
      } else {
        EXPECT_EQ(actual[i], kSentinel);
      }
    }
    const auto source_after = Read(source_);
    const auto capsule_after = Read(capsule_);
    EXPECT_EQ(std::memcmp(source_after.data(), source_values_.data(), source_values_.size() * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(capsule_after.data(), capsule_values_.data(), capsule_values_.size() * sizeof(float)), 0);
  }
}

TEST_P(WebGpuStateReplayHelperTest, RejectsInvalidViewsAndPropagatesOrtFailures) {
  Upload(1);
  ASSERT_NO_THROW(WebGPU::RunGatedDeltaNetStateReplay(Descriptor(1)));
  Upload(1);
  const auto expect_ort_error = [](const StateUpdateReplayDesc& desc, const char* message) {
    try {
      WebGPU::RunGatedDeltaNetStateReplay(desc);
      FAIL() << "Expected an ORT failure";
    } catch (const Ort::Exception& error) {
      EXPECT_NE(std::string(error.what()).find(message), std::string::npos) << error.what();
    }
  };
  auto desc = Descriptor(1);
  desc.key = desc.key.subspan(1, desc.key.size() - 1);
  EXPECT_THROW(WebGPU::RunGatedDeltaNetStateReplay(desc), std::invalid_argument);
  desc = Descriptor(kCapacity + 1);
  expect_ort_error(desc, "kept_count must be in [1, capacity]");
  desc = Descriptor(1);
  desc.destination_state = source_.subspan(kStateElements * sizeof(float), kStateElements * sizeof(float));
  expect_ort_error(desc, "destination backing must be distinct from source and capsule");
  for (const auto value : Read(destination_)) {
    EXPECT_EQ(value, kSentinel);
  }
  // A failed run must not poison subsequent use of the cached helper session.
  EXPECT_NO_THROW(WebGPU::RunGatedDeltaNetStateReplay(Descriptor(1)));
}

INSTANTIATE_TEST_SUITE_P(BackingOwners, WebGpuStateReplayHelperTest, ::testing::Bool());

}  // namespace
}  // namespace Generators::test
