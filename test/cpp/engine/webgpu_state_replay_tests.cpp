// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "models/graph_executor.h"
#include "models/model.h"
#include "models/session_options.h"
#include "state_update_replay.h"

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

class WebGpuStateReplayTest : public ::testing::TestWithParam<bool> {
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
    const bool already_initialized =
        GetOrtGlobals()->device_allocators_[static_cast<int>(DeviceType::WEBGPU)].allocator_ != nullptr;
    Config config;
    config.model.decoder.session_options.provider_options.emplace_back(
        "WebGPU", std::vector<std::pair<std::string, std::string>>{
                      {"deviceId", "0"}, {"validationMode", "full"}, {"maxNumPendingDispatches", "1"}, {"enableGraphCapture", "1"}});
    EnsureDeviceOrtInit(*device_, config);
    const auto& options =
        GetOrtGlobals()->device_allocators_[static_cast<int>(DeviceType::WEBGPU)].session_options_;
    ASSERT_NE(options, nullptr);
    if (!already_initialized) {
      EXPECT_EQ(ConfigEntry(*options, "ep.webgpuexecutionprovider.deviceId"), "0");
      EXPECT_EQ(ConfigEntry(*options, "ep.webgpuexecutionprovider.validationMode"), "full");
      EXPECT_EQ(ConfigEntry(*options, "ep.webgpuexecutionprovider.maxNumPendingDispatches"), "1");
    }
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

  StateUpdateReplayDesc CpuReference(const StateUpdateReplayDesc& descriptor) {
    auto* cpu = GetDeviceInterface(DeviceType::CPU);
    auto source = DeviceSpan<uint8_t>(cpu->WrapMemoryBase(
        source_values_.data(), source_values_.size() * sizeof(float)));
    auto capsule = DeviceSpan<uint8_t>(cpu->WrapMemoryBase(
        capsule_values_.data(), capsule_values_.size() * sizeof(float)));
    auto reference = descriptor;
    reference.source_state = source.subspan(descriptor.source_state.ByteOffset(), descriptor.source_state.size());
    reference.destination_state = cpu->Allocate<uint8_t>(descriptor.destination_state.size());
    reference.decay = capsule.subspan(descriptor.decay.ByteOffset(), descriptor.decay.size());
    reference.key = capsule.subspan(descriptor.key.ByteOffset(), descriptor.key.size());
    reference.delta = capsule.subspan(descriptor.delta.ByteOffset(), descriptor.delta.size());
    return reference;
  }

  static void PoisonCpuShadow(DeviceSpan<uint8_t> buffer) {
    auto backing = buffer.BackingBuffer();
    backing->AllocateCpu();
    std::memset(backing->p_cpu_, 0xa5, backing->size_in_bytes_);
  }

  static void ExpectPoisonedCpuShadow(DeviceSpan<uint8_t> buffer) {
    auto backing = buffer.BackingBuffer();
    const auto* begin = backing->p_cpu_;
    EXPECT_TRUE(std::all_of(begin, begin + backing->size_in_bytes_, [](uint8_t value) { return value == 0xa5; }))
        << "Replay unexpectedly downloaded or uploaded this backing through CPU staging.";
  }

  DeviceInterface* device_{};
  std::unique_ptr<OrtValue> source_tensor_;
  DeviceSpan<uint8_t> source_, capsule_, destination_;
  std::vector<float> source_values_, capsule_values_;
};

TEST_P(WebGpuStateReplayTest, ReplaysEveryPrefixOnGpuWithoutCpuStaging) {
  // The second logical capsule row starts 16 bytes past a 256-byte binding boundary.
  ASSERT_EQ((kCapsuleElements * sizeof(float)) % 256, 16u);
  for (uint32_t kept = 1; kept <= kCapacity; ++kept) {
    SCOPED_TRACE(kept);
    Upload(kept);
    auto descriptor = Descriptor(kept);
    EXPECT_EQ(descriptor.decay.ByteOffset(), kCapsuleElements * sizeof(float));
    auto reference = CpuReference(descriptor);
    ReplayStateUpdatesOnCpu(&reference, 1);
    PoisonCpuShadow(source_);
    PoisonCpuShadow(capsule_);
    PoisonCpuShadow(destination_);
    device_->ReplayStateUpdates(&descriptor, 1);
    // Check before readback: the old CPU path refreshes these poisoned host shadows.
    ExpectPoisonedCpuShadow(source_);
    ExpectPoisonedCpuShadow(capsule_);
    ExpectPoisonedCpuShadow(destination_);

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
        const float expected_value = expected[i - kStateElements];
        EXPECT_NEAR(actual[i], expected_value, 2e-6f + 5e-6f * std::abs(expected_value));
      } else {
        EXPECT_EQ(actual[i], kSentinel);
      }
    }
    const auto cpu_result = Read(reference.destination_state);
    for (size_t i = 0; i < kStateElements; ++i) {
      EXPECT_NEAR(actual[kStateElements + i], cpu_result[i], 2e-6f + 5e-6f * std::abs(cpu_result[i]));
    }
    const auto source_after = Read(source_);
    const auto capsule_after = Read(capsule_);
    EXPECT_EQ(std::memcmp(source_after.data(), source_values_.data(), source_values_.size() * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(capsule_after.data(), capsule_values_.data(), capsule_values_.size() * sizeof(float)), 0);
  }
}

TEST_P(WebGpuStateReplayTest, RejectsInvalidViewsAndPropagatesOrtFailures) {
  Upload(1);
  auto valid = Descriptor(1);
  ASSERT_NO_THROW(device_->ReplayStateUpdates(&valid, 1));
  Upload(1);
  const auto expect_ort_error = [this](const StateUpdateReplayDesc& desc, const char* message) {
    try {
      device_->ReplayStateUpdates(&desc, 1);
      FAIL() << "Expected an ORT failure";
    } catch (const Ort::Exception& error) {
      EXPECT_NE(std::string(error.what()).find(message), std::string::npos) << error.what();
    }
  };
  auto desc = Descriptor(1);
  desc.element_size = 2;
  EXPECT_THROW(device_->ReplayStateUpdates(&desc, 1), std::invalid_argument);
  desc = Descriptor(1);
  desc.kind = static_cast<StateUpdateReplayKind>(999);
  EXPECT_THROW(device_->ReplayStateUpdates(&desc, 1), std::invalid_argument);
  desc = Descriptor(1);
  desc.key = desc.key.subspan(1, desc.key.size() - 1);
  EXPECT_THROW(device_->ReplayStateUpdates(&desc, 1), std::invalid_argument);
  desc = Descriptor(kCapacity + 1);
  expect_ort_error(desc, "kept_count must be in [1, capacity]");
  desc = Descriptor(1);
  desc.destination_state = source_.subspan(kStateElements * sizeof(float), kStateElements * sizeof(float));
  expect_ort_error(desc, "destination backing must be distinct from source and capsule");
  for (const auto value : Read(destination_)) {
    EXPECT_EQ(value, kSentinel);
  }
  // A failed run must not poison subsequent use of the cached helper session.
  EXPECT_NO_THROW(device_->ReplayStateUpdates(&valid, 1));
}

TEST_P(WebGpuStateReplayTest, ConvolutionStillStagesThroughCpu) {
  Upload(2);
  std::array<float, 12> values{10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25};
  auto updates = device_->Allocate<uint8_t>(sizeof(values));
  updates.BackingBuffer()->CopyFromCpu(values.data(), sizeof(values));
  auto descriptor = Descriptor(2);
  auto reference = CpuReference(descriptor);
  descriptor.kind = reference.kind = StateUpdateReplayKind::CausalConv;
  descriptor.channel_count = reference.channel_count = kHv * kDv;
  descriptor.state_width = reference.state_width = kDk;
  descriptor.capacity = reference.capacity = 2;
  descriptor.value = updates;
  reference.value = DeviceSpan<uint8_t>(GetDeviceInterface(DeviceType::CPU)->WrapMemoryBase(values.data(), sizeof(values)));
  descriptor.decay = descriptor.key = descriptor.delta = {};
  reference.decay = reference.key = reference.delta = {};
  ReplayStateUpdatesOnCpu(&reference, 1);
  PoisonCpuShadow(source_);
  PoisonCpuShadow(capsule_);
  PoisonCpuShadow(destination_);
  PoisonCpuShadow(updates);
  device_->ReplayStateUpdates(&descriptor, 1);
  const auto expected = Read(reference.destination_state);
  // The fallback refreshes inputs and uploads the CPU-computed destination row.
  EXPECT_EQ(std::memcmp(source_.BackingBuffer()->p_cpu_, source_values_.data(),
                        source_values_.size() * sizeof(float)),
            0);
  EXPECT_EQ(std::memcmp(updates.BackingBuffer()->p_cpu_, values.data(), sizeof(values)), 0);
  EXPECT_EQ(std::memcmp(destination_.BackingBuffer()->p_cpu_ + descriptor.destination_state.ByteOffset(),
                        expected.data(), expected.size() * sizeof(float)),
            0);
  if (!GetParam()) ExpectPoisonedCpuShadow(capsule_);
  const auto actual = Read(destination_);
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i], i >= kStateElements && i < 2 * kStateElements
                             ? expected[i - kStateElements]
                             : kSentinel);
  }
}

TEST_P(WebGpuStateReplayTest, SeparateCapsuleBackingsFallBackBeforeGpuReplay) {
  Upload(2);
  auto descriptor = Descriptor(2);
  auto reference = CpuReference(descriptor);
  ReplayStateUpdatesOnCpu(&reference, 1);
  const auto separate = [&](DeviceSpan<const uint8_t> section) {
    auto buffer = device_->Allocate<uint8_t>(section.size());
    const auto* bytes = reinterpret_cast<const uint8_t*>(capsule_values_.data());
    buffer.BackingBuffer()->CopyFromCpu(bytes + section.ByteOffset(), section.size());
    return buffer;
  };
  descriptor.decay = separate(descriptor.decay);
  descriptor.key = separate(descriptor.key);
  descriptor.delta = separate(descriptor.delta);
  PoisonCpuShadow(source_);
  device_->ReplayStateUpdates(&descriptor, 1);
  EXPECT_EQ(std::memcmp(source_.BackingBuffer()->p_cpu_, source_values_.data(),
                        source_values_.size() * sizeof(float)),
            0);
  const auto actual = Read(descriptor.destination_state);
  const auto expected = Read(reference.destination_state);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_FLOAT_EQ(actual[i], expected[i]);
  }
}

TEST_P(WebGpuStateReplayTest, MixedDescriptorsPreserveDependentExecutionOrder) {
  std::array<float, 12> values{10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25};
  auto updates = device_->Allocate<uint8_t>(sizeof(values));
  updates.BackingBuffer()->CopyFromCpu(values.data(), sizeof(values));
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  auto cpu_updates = DeviceSpan<uint8_t>(cpu->WrapMemoryBase(
      values.data(), sizeof(values)));
  for (const bool recurrent_first : {false, true}) {
    SCOPED_TRACE(recurrent_first);
    Upload(2);
    std::array<StateUpdateReplayDesc, 3> descriptors;
    std::array<StateUpdateReplayDesc, 3> references;
    for (size_t i = 0; i < descriptors.size(); ++i) {
      auto& descriptor = descriptors[i];
      auto& reference = references[i];
      descriptor = Descriptor(2);
      reference = CpuReference(descriptor);
      if ((i % 2 == 0) != recurrent_first) {
        descriptor.kind = reference.kind = StateUpdateReplayKind::CausalConv;
        descriptor.channel_count = reference.channel_count = kHv * kDv;
        descriptor.state_width = reference.state_width = kDk;
        descriptor.value = updates;
        reference.value = cpu_updates;
        descriptor.decay = descriptor.key = descriptor.delta = {};
        reference.decay = reference.key = reference.delta = {};
      }
      descriptor.destination_state = device_->Allocate<uint8_t>(kStateElements * sizeof(float));
      reference.destination_state = cpu->Allocate<uint8_t>(kStateElements * sizeof(float));
      descriptor.destination_state.Zero();
      reference.destination_state.Zero();
      if (i > 0) {
        descriptor.source_state = descriptors[i - 1].destination_state;
        reference.source_state = references[i - 1].destination_state;
      }
    }
    ReplayStateUpdatesOnCpu(references.data(), references.size());
    device_->ReplayStateUpdates(descriptors.data(), descriptors.size());
    for (size_t i = 0; i < descriptors.size(); ++i) {
      const auto actual = Read(descriptors[i].destination_state);
      const auto expected = Read(references[i].destination_state);
      for (size_t j = 0; j < actual.size(); ++j) {
        EXPECT_NEAR(actual[j], expected[j], 2e-6f + 5e-6f * std::abs(expected[j]))
            << "descriptor=" << i << " element=" << j;
      }
    }
  }
}

TEST_P(WebGpuStateReplayTest, GpuRunFailureStopsMixedSequenceWithoutCpuRetry) {
  Upload(1);
  auto first = Descriptor(1);
  auto reference = CpuReference(first);
  ReplayStateUpdatesOnCpu(&reference, 1);
  auto failed = Descriptor(0);
  failed.source_state = first.destination_state;
  failed.destination_state = device_->Allocate<uint8_t>(kStateElements * sizeof(float));
  auto last = Descriptor(1);
  last.kind = StateUpdateReplayKind::CausalConv;
  last.channel_count = kHv * kDv;
  last.state_width = kDk;
  last.source_state = first.destination_state;
  last.destination_state = device_->Allocate<uint8_t>(kStateElements * sizeof(float));
  const std::vector<float> sentinel(kStateElements, kSentinel);
  last.destination_state.BackingBuffer()->CopyFromCpu(sentinel.data(), sentinel.size() * sizeof(float));
  const std::array<float, 6> values{10, 11, 12, 13, 14, 15};
  auto updates = device_->Allocate<uint8_t>(sizeof(values));
  updates.BackingBuffer()->CopyFromCpu(values.data(), sizeof(values));
  last.value = updates;
  last.decay = last.key = last.delta = {};
  PoisonCpuShadow(source_);
  PoisonCpuShadow(capsule_);
  PoisonCpuShadow(destination_);
  PoisonCpuShadow(failed.destination_state);
  PoisonCpuShadow(last.destination_state);
  const std::array<StateUpdateReplayDesc, 3> descriptors{first, failed, last};
  try {
    device_->ReplayStateUpdates(descriptors.data(), descriptors.size());
    FAIL() << "Expected the second ORT replay run to fail.";
  } catch (const Ort::Exception& error) {
    EXPECT_NE(std::string(error.what()).find("kept_count must be in [1, capacity]"), std::string::npos)
        << error.what();
  }
  ExpectPoisonedCpuShadow(source_);
  ExpectPoisonedCpuShadow(capsule_);
  ExpectPoisonedCpuShadow(destination_);
  ExpectPoisonedCpuShadow(failed.destination_state);
  ExpectPoisonedCpuShadow(last.destination_state);
  const auto actual = Read(first.destination_state);
  const auto expected = Read(reference.destination_state);
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], 2e-6f + 5e-6f * std::abs(expected[i]));
  }
  EXPECT_EQ(Read(last.destination_state), sentinel);
}

INSTANTIATE_TEST_SUITE_P(BackingOwners, WebGpuStateReplayTest, ::testing::Bool());

}  // namespace
}  // namespace Generators::test
