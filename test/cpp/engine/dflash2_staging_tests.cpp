// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "dflash2_staging.h"
#include "engine_test_doubles.h"

namespace Generators::test {
namespace {

constexpr size_t kCapacity = 64;

class Dflash2HostStagingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_.state->separate_host_mirror = true;
    device_.state->recycles_host_mirrors = true;
  }

  CountingCudaDevice device_;
  Dflash2HostStaging staging_;
  std::vector<uint8_t> first_tensor_ = std::vector<uint8_t>(kCapacity);
  std::vector<uint8_t> second_tensor_ = std::vector<uint8_t>(kCapacity);
};

TEST_F(Dflash2HostStagingTest, ATensorKeepsOneMirrorByDefaultAndAnotherTensorHasItsOwn) {
  EXPECT_FALSE(staging_.FreshMirrorPerUpload());
  auto first = staging_.Acquire(device_, &first_tensor_, first_tensor_.data(), kCapacity, 16);
  auto again = staging_.Acquire(device_, &first_tensor_, first_tensor_.data(), kCapacity, 32);
  auto other = staging_.Acquire(device_, &second_tensor_, second_tensor_.data(), kCapacity, 16);

  EXPECT_EQ(&first.GetBuffer(), &again.GetBuffer());
  EXPECT_NE(&first.GetBuffer(), &other.GetBuffer());
  EXPECT_EQ(first.size(), 16u);
  EXPECT_EQ(again.size(), 32u);
  EXPECT_EQ(first.CpuSpan().data(), again.CpuSpan().data());
  EXPECT_EQ(staging_.KeptMirrors(), 2u);
}

TEST_F(Dflash2HostStagingTest, ForgettingATensorDropsItsMirrorOnly) {
  auto first = staging_.Acquire(device_, &first_tensor_, first_tensor_.data(), kCapacity, 16);
  auto other = staging_.Acquire(device_, &second_tensor_, second_tensor_.data(), kCapacity, 16);
  const auto* kept_buffer = &first.GetBuffer();
  ASSERT_EQ(staging_.KeptMirrors(), 2u);

  staging_.Forget(&first_tensor_);
  staging_.Forget(&first_tensor_);  // forgetting twice, or a tensor never staged, is harmless
  staging_.Forget(this);
  EXPECT_EQ(staging_.KeptMirrors(), 1u);

  // The tensor gets a new mirror when it is staged again; the other tensor still has its own.
  auto restaged = staging_.Acquire(device_, &first_tensor_, first_tensor_.data(), kCapacity, 16);
  auto other_again = staging_.Acquire(device_, &second_tensor_, second_tensor_.data(), kCapacity, 16);
  EXPECT_NE(&restaged.GetBuffer(), kept_buffer);
  EXPECT_EQ(&other_again.GetBuffer(), &other.GetBuffer());
  EXPECT_EQ(staging_.KeptMirrors(), 2u);
}

TEST_F(Dflash2HostStagingTest, EveryUploadTakesItsOwnMirrorWhenFreshMirrorsAreRequested) {
  staging_.SetFreshMirrorPerUpload(true);
  ASSERT_TRUE(staging_.FreshMirrorPerUpload());

  // The proposals of two slots, back to back: the first one's mirror is still in use when the second one fills its own.
  auto first = staging_.Acquire(device_, &first_tensor_, first_tensor_.data(), kCapacity, 16);
  auto second = staging_.Acquire(device_, &first_tensor_, first_tensor_.data(), kCapacity, 24);
  EXPECT_NE(&first.GetBuffer(), &second.GetBuffer());
  EXPECT_EQ(first.size(), 16u);
  EXPECT_EQ(second.size(), 24u);

  auto first_host = first.CpuSpan();
  auto second_host = second.CpuSpan();
  ASSERT_NE(first_host.data(), second_host.data());
  std::fill(first_host.begin(), first_host.end(), uint8_t{0x11});
  std::fill(second_host.begin(), second_host.end(), uint8_t{0x22});
  EXPECT_TRUE(std::all_of(first_host.begin(), first_host.end(), [](uint8_t value) { return value == 0x11; }));
  EXPECT_EQ(staging_.KeptMirrors(), 0u);
}

TEST_F(Dflash2HostStagingTest, SwitchingToFreshMirrorsReleasesTheMirrorAKeptTensorHeld) {
  auto kept = staging_.Acquire(device_, &first_tensor_, first_tensor_.data(), kCapacity, 16);
  ASSERT_EQ(staging_.KeptMirrors(), 1u);

  staging_.SetFreshMirrorPerUpload(true);
  auto fresh = staging_.Acquire(device_, &first_tensor_, first_tensor_.data(), kCapacity, 16);
  EXPECT_NE(&kept.GetBuffer(), &fresh.GetBuffer());
  EXPECT_EQ(staging_.KeptMirrors(), 0u);
}

TEST_F(Dflash2HostStagingTest, ADeviceThatDoesNotRecycleMirrorsKeepsOneEvenWhenFreshMirrorsAreRequested) {
  device_.state->recycles_host_mirrors = false;
  staging_.SetFreshMirrorPerUpload(true);

  auto first = staging_.Acquire(device_, &first_tensor_, first_tensor_.data(), kCapacity, 16);
  auto second = staging_.Acquire(device_, &first_tensor_, first_tensor_.data(), kCapacity, 16);
  EXPECT_EQ(&first.GetBuffer(), &second.GetBuffer());
  EXPECT_EQ(staging_.KeptMirrors(), 1u);
}

}  // namespace
}  // namespace Generators::test
