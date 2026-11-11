// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "engine/prefix-checkpoint-index.h"
#include "engine/prefix_snapshot_cache.h"
#include "engine_test_doubles.h"

namespace Generators::test {
namespace {

struct Checkpoint {
  std::vector<int32_t> tokens;
  std::shared_ptr<PrefixSnapshotCache::Entry> state;
  std::span<const int32_t> Tokens() const { return tokens; }
  bool IsResident() const { return state->state == PrefixSnapshotCache::State::Resident; }
};

std::shared_ptr<PrefixSnapshotCache::Entry> ByteCheckpoint() {
  auto entry = std::make_shared<PrefixSnapshotCache::Entry>();
  entry->type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  entry->shapes = {{16}};
  return entry;
}

TEST(PrefixCheckpointIndexTest, FindsOnlyCompleteResidentTokenPrefixes) {
  PrefixSnapshotCache storage{{0, 128, 0}, 8};
  PrefixCheckpointIndex<Checkpoint> index;
  auto* device = GetDeviceInterface(DeviceType::CPU);
  for (const auto& tokens : std::vector<std::vector<int32_t>>{{}, {1, 2}, {1, 2, 3, 4}, {1, 2, 3, 9}}) {
    auto state = ByteCheckpoint();
    ASSERT_TRUE(storage.Allocate(state, device));
    index.Publish({tokens, std::move(state)});
  }
  const std::vector<int32_t> exact{1, 2, 3, 4};
  ASSERT_NE(index.Find(exact), nullptr);
  EXPECT_EQ(index.Find(exact)->tokens, exact);
  EXPECT_EQ(index.Find(std::vector<int32_t>{1, 2, 3, 4, 5})->tokens, exact);
  const std::vector<int32_t> shorter{1, 2};
  EXPECT_EQ(index.Find(std::vector<int32_t>{1, 2, 3, 5})->tokens, shorter);
  EXPECT_EQ(index.Find(std::vector<int32_t>{1}), nullptr);
  EXPECT_EQ(index.Find(std::vector<int32_t>{7, 2, 3, 4}), nullptr);
  EXPECT_EQ(index.Find(std::span<const int32_t>{}), nullptr);
  storage.Discard(*index.Find(exact)->state);
  EXPECT_EQ(index.Find(exact)->tokens, shorter);
  EXPECT_EQ(index.Prune(), 1u);
  EXPECT_EQ(index.Prune(), 0u);
}

TEST(PrefixCheckpointIndexTest, LookupDoesNotLeaseOrRefreshButAdoptionProtectsStorage) {
  PrefixSnapshotCache storage{{0, 32, 0}, 2};
  PrefixCheckpointIndex<Checkpoint> index;
  auto* device = GetDeviceInterface(DeviceType::CPU);
  const std::vector<int32_t> first{1, 2};
  const std::vector<int32_t> second{3, 4};
  for (const auto& tokens : {first, second}) {
    auto state = ByteCheckpoint();
    ASSERT_TRUE(storage.Allocate(state, device));
    index.Publish({tokens, std::move(state)});
  }
  ASSERT_NE(index.Find(first), nullptr);
  auto lease = index.Find(second)->state;
  auto replacement = ByteCheckpoint();
  index.ReserveForPublication();
  ASSERT_TRUE(storage.Allocate(replacement, device));
  EXPECT_EQ(index.Find(first), nullptr);
  ASSERT_NE(index.Find(second), nullptr);
  EXPECT_EQ(index.Prune(), 1u);
  index.Publish({{5, 6}, std::move(replacement)});

  auto second_lease = index.Find(std::vector<int32_t>{5, 6})->state;
  EXPECT_FALSE(storage.Allocate(ByteCheckpoint(), device));
  lease.reset();
  ASSERT_TRUE(storage.Allocate(ByteCheckpoint(), device));
  EXPECT_EQ(index.Find(second), nullptr);
  EXPECT_EQ(index.Prune(), 1u);
}

TEST(NativePrefixSnapshotPolicyTest, RequiresOptInAndHonorsExplicitDisables) {
  Config::Engine::DynamicBatching config;
  EXPECT_EQ(CreateNativePrefixSnapshotCache(config), nullptr);
  config.prefix_caching = false;
  EXPECT_EQ(CreateNativePrefixSnapshotCache(config), nullptr);
  config.prefix_caching = true;
  config.max_snapshots = 0;
  EXPECT_EQ(CreateNativePrefixSnapshotCache(config), nullptr);
  config.max_snapshots.reset();
  config.snapshot_cache.explicitly_set = true;
  EXPECT_EQ(CreateNativePrefixSnapshotCache(config), nullptr);
  config.snapshot_cache.explicitly_set = false;
  config.snapshot_cache.directory = "unused-snapshot-directory";
  EXPECT_EQ(CreateNativePrefixSnapshotCache(config), nullptr);
}

TEST(NativePrefixSnapshotPolicyTest, DefaultsToOneResidentCheckpointAndHonorsExplicitCount) {
  for (size_t count : {size_t{1}, size_t{2}}) {
    Config::Engine::DynamicBatching config;
    config.prefix_caching = true;
    if (count == 2) config.max_snapshots = count;
    auto storage = CreateNativePrefixSnapshotCache(config);
    ASSERT_NE(storage, nullptr);
    std::vector<std::shared_ptr<PrefixSnapshotCache::Entry>> retained;
    auto* device = GetDeviceInterface(DeviceType::CPU);
    for (size_t i = 0; i < count; ++i) {
      auto entry = ByteCheckpoint();
      ASSERT_TRUE(storage->Allocate(entry, device));
      EXPECT_EQ(entry->tier, PrefixSnapshotCache::Tier::Vram);
      retained.push_back(std::move(entry));
    }
    auto leases = retained;
    EXPECT_FALSE(storage->Allocate(ByteCheckpoint(), device));
    leases.clear();
    ASSERT_TRUE(storage->Allocate(ByteCheckpoint(), device));
    EXPECT_EQ(retained.front()->state, PrefixSnapshotCache::State::Evicted);
  }
}

TEST(NativePrefixSnapshotPolicyTest, ExplicitTierBudgetsDoNotImposeAnImplicitSingleEntryLimit) {
  for (size_t tier = 0; tier < 3; ++tier) {
    CountingCudaDevice cuda;
    Config::Engine::DynamicBatching config;
    config.prefix_caching = true;
    if (tier == 0) config.snapshot_cache.vram_bytes = 32;
    if (tier == 1) config.snapshot_cache.ram_bytes = 32;
    if (tier == 2) config.snapshot_cache.disk_bytes = 32;
    auto storage = CreateNativePrefixSnapshotCache(config);
    ASSERT_NE(storage, nullptr);
    auto* device = tier == 0 ? static_cast<DeviceInterface*>(&cuda) : GetDeviceInterface(DeviceType::CPU);
    auto first = ByteCheckpoint();
    auto second = ByteCheckpoint();
    ASSERT_TRUE(storage->Allocate(first, device));
    ASSERT_TRUE(storage->Allocate(second, device));
    EXPECT_EQ(static_cast<size_t>(first->tier), tier);
    EXPECT_EQ(first->state, PrefixSnapshotCache::State::Resident);
    auto lease = first;
    auto second_lease = second;
    EXPECT_FALSE(storage->Allocate(ByteCheckpoint(), device));
    second_lease.reset();
    ASSERT_TRUE(storage->Allocate(ByteCheckpoint(), device));
    EXPECT_EQ(first->state, PrefixSnapshotCache::State::Resident);
    EXPECT_EQ(second->state, PrefixSnapshotCache::State::Evicted);
  }
}

}  // namespace
}  // namespace Generators::test
