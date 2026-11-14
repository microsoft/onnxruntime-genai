// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <utility>

#include <gtest/gtest.h>

#include "engine/prefix-checkpoint-index.h"
#include "engine/prefix_snapshot_cache.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"

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
  PrefixCacheMetrics metrics;
  index.TryPublish(metrics, [&]() -> std::optional<Checkpoint> {
    auto replacement = ByteCheckpoint();
    if (!storage.Allocate(replacement, device)) return std::nullopt;
    return Checkpoint{{5, 6}, std::move(replacement)};
  });
  EXPECT_EQ(index.Find(first), nullptr);
  ASSERT_NE(index.Find(second), nullptr);
  EXPECT_EQ(metrics.evictions, 1u);
  auto second_lease = index.Find(std::vector<int32_t>{5, 6})->state;
  EXPECT_FALSE(storage.Allocate(ByteCheckpoint(), device));
  lease.reset();
  ASSERT_TRUE(storage.Allocate(ByteCheckpoint(), device));
  EXPECT_EQ(index.Find(second), nullptr);
}

TEST(PrefixCheckpointIndexTest, OptionalPublicationFailuresPruneButTransferErrorsEscape) {
  for (int failure = 0; failure < 4; ++failure) {
    SCOPED_TRACE(failure);
    PrefixSnapshotCache storage{{0, 16, 0}, 1};
    PrefixCheckpointIndex<Checkpoint> index;
    auto state = ByteCheckpoint();
    ASSERT_TRUE(storage.Allocate(state, GetDeviceInterface(DeviceType::CPU)));
    const std::vector<int32_t> tokens{1, 2};
    index.Publish({tokens, std::move(state)});
    PrefixCacheMetrics metrics;
    const auto publish = [&] {
      index.TryPublish(metrics, [&]() -> std::optional<Checkpoint> {
        storage.Discard(*index.Find(tokens)->state);
        if (failure == 1) throw std::bad_alloc{};
        if (failure == 2) throw std::system_error(std::make_error_code(std::errc::no_space_on_device));
        if (failure == 3) throw std::runtime_error("device transfer failure");
        ++metrics.publication_refusals;
        return std::nullopt;
      });
    };
    if (failure == 3) {
      EXPECT_THROW(publish(), std::runtime_error);
      EXPECT_EQ(metrics.publication_refusals, 0u);
    } else {
      publish();
      EXPECT_EQ(metrics.publication_refusals, 1u);
      EXPECT_EQ(metrics.evictions, 1u);
      EXPECT_EQ(index.Prune(), 0u);
    }
    EXPECT_EQ(index.Find(tokens), nullptr);
  }
}

TEST(PrefixCheckpointIndexTest, PublicationGrowsGeometricallyBeforeCapture) {
  struct MovingCheckpoint {
    MovingCheckpoint(int32_t token, size_t& moves) : tokens{token}, moves{&moves} {}
    MovingCheckpoint(MovingCheckpoint&& other) noexcept : tokens{std::move(other.tokens)}, moves{other.moves} {
      ++*moves;
    }
    MovingCheckpoint& operator=(MovingCheckpoint&&) noexcept = default;
    std::span<const int32_t> Tokens() const { return tokens; }
    bool IsResident() const { return true; }
    std::vector<int32_t> tokens;
    size_t* moves;
  };
  PrefixCheckpointIndex<MovingCheckpoint> index;
  PrefixCacheMetrics metrics;
  size_t moves = 0;
  for (int32_t token = 1; token <= 128; ++token) {
    index.TryPublish(metrics, [&]() -> std::optional<MovingCheckpoint> {
      return MovingCheckpoint{token, moves};
    });
    ASSERT_NE(index.Find(std::vector<int32_t>{token}), nullptr);
  }
  EXPECT_LT(moves, 128u * 8);
  EXPECT_EQ(metrics.publication_refusals, 0u);
}

TEST(PrefixCheckpointIndexTest, OperationalEventDeltasDoNotPublishUnacknowledgedAdoption) {
  PrefixCacheMetrics before;
  before.lookups = 8;
  before.matches = 4;
  before.hits = 3;
  before.queried_tokens = 64;
  before.matched_tokens = 24;
  auto after = before;
  ++after.lookups;
  ++after.matches;
  ++after.deferred_matches;
  after.queried_tokens += 16;
  after.registered_blocks += 2;
  ++after.duplicate_registrations;
  ++after.hash_collisions;
  ++after.evictions;
  ++after.retention_refusals;
  ++after.publication_refusals;
  after.RecordAdoption(8);
  PrefixCacheMetrics events;
  events.RecordAdoption(4);
  events.AddEventsSince(before, after);
  EXPECT_EQ(events.lookups, 1u);
  EXPECT_EQ(events.matches, 1u);
  EXPECT_EQ(events.deferred_matches, 1u);
  EXPECT_EQ(events.queried_tokens, 16u);
  EXPECT_EQ(events.registered_blocks, 2u);
  EXPECT_EQ(events.duplicate_registrations, 1u);
  EXPECT_EQ(events.hash_collisions, 1u);
  EXPECT_EQ(events.evictions, 1u);
  EXPECT_EQ(events.retention_refusals, 1u);
  EXPECT_EQ(events.publication_refusals, 1u);
  EXPECT_EQ(events.hits, 1u);
  EXPECT_EQ(events.matched_tokens, 4u);
  size_t pending = 8;
  for (int ack = 0; ack < 2; ++ack)
    events.RecordAdoption(std::exchange(pending, 0));
  EXPECT_EQ(events.hits, 2u);
  EXPECT_EQ(events.matched_tokens, 12u);
}

TEST(NativePrefixSnapshotPolicyTest, DefaultsDependOnTheTargetNotTheDrafterAndRespectExplicitPolicy) {
  Config config;
  EXPECT_EQ(CreateNativePrefixSnapshotCache(config), nullptr);
  config.engine.dynamic_batching.emplace();
  for (bool target_windowed : {false, true}) {
    for (bool drafter_windowed : {false, true}) {
      for (const auto enabled : {std::optional<bool>{}, std::optional<bool>{false}, std::optional<bool>{true}}) {
        SCOPED_TRACE(target_windowed);
        SCOPED_TRACE(drafter_windowed);
        SCOPED_TRACE(enabled.value_or(!target_windowed));
        config.model.decoder.sliding_window.reset();
        if (target_windowed) {
          config.model.decoder.sliding_window.emplace();
          config.model.decoder.sliding_window->window_size = 8;
          config.model.decoder.sliding_window->layers = {0};
        }
        config.model.dflash2.sliding_window = drafter_windowed ? 8 : -1;
        config.engine.dynamic_batching->prefix_caching = enabled;
        EXPECT_EQ(CreateNativePrefixSnapshotCache(config) != nullptr, enabled.value_or(!target_windowed));
      }
    }
  }
}

TEST(NativePrefixSnapshotPolicyTest, ExplicitDisablesOverrideTheDefaultAndTierBudgets) {
  for (const auto* settings : {
           R"({"prefix_caching":false,"snapshot_cache":{"ram_bytes":64}})",
           R"({"max_snapshots":0,"snapshot_cache":{"ram_bytes":64}})",
           R"({"snapshot_cache":{}})",
           R"({"snapshot_cache":{"directory":"unused-snapshot-directory"}})"}) {
    SCOPED_TRACE(settings);
    Config config;
    OverlayConfig(config, std::string{R"({"engine":{"dynamic_batching":)"} + settings + "}}");
    EXPECT_EQ(CreateNativePrefixSnapshotCache(config), nullptr);
  }
}

TEST(NativePrefixSnapshotPolicyTest, HonorsConfiguredCountsAndTierBudgets) {
  using Tier = PrefixSnapshotCache::Tier;
  struct Case {
    const char* settings;
    Tier tier;
    size_t retained;
  };
  const Case cases[] = {
      {R"({"max_batch_size":1})", Tier::Vram, 1},
      {R"({"max_snapshots":2,"snapshot_cache":{"ram_bytes":64}})", Tier::Ram, 2},
      {R"({"snapshot_cache":{"vram_bytes":32}})", Tier::Vram, 2},
      {R"({"snapshot_cache":{"ram_bytes":32}})", Tier::Ram, 2},
      {R"({"snapshot_cache":{"disk_bytes":32}})", Tier::Disk, 2},
      {R"({"max_snapshots":3,"snapshot_cache":{"ram_bytes":16}})", Tier::Ram, 1},
      {R"({"snapshot_cache":{"ram_bytes":15,"disk_bytes":15}})", Tier::Ram, 0},
  };
  for (const auto& [settings, tier, retained] : cases) {
    SCOPED_TRACE(settings);
    Config config;
    OverlayConfig(config, std::string{R"({"engine":{"dynamic_batching":)"} + settings + "}}");
    auto storage = CreateNativePrefixSnapshotCache(config);
    ASSERT_NE(storage, nullptr);
    CountingCudaDevice cuda;
    auto* device = tier == Tier::Vram ? static_cast<DeviceInterface*>(&cuda) : GetDeviceInterface(DeviceType::CPU);
    std::vector<std::shared_ptr<PrefixSnapshotCache::Entry>> entries;
    for (size_t index = 0; index <= retained; ++index) {
      auto entry = ByteCheckpoint();
      if (retained == 0) {
        EXPECT_FALSE(storage->Allocate(entry, device));
        break;
      }
      ASSERT_TRUE(storage->Allocate(entry, device));
      EXPECT_EQ(entry->tier, tier);
      entries.push_back(std::move(entry));
      if (entries.size() <= retained)
        for (const auto& previous : entries)
          EXPECT_EQ(previous->state, PrefixSnapshotCache::State::Resident);
    }
    if (retained != 0) {
      EXPECT_EQ(entries.front()->state, PrefixSnapshotCache::State::Evicted);
      for (size_t index = 1; index < entries.size(); ++index)
        EXPECT_EQ(entries[index]->state, PrefixSnapshotCache::State::Resident);
    }
    EXPECT_EQ(storage->Used()[static_cast<size_t>(tier)], retained * 16);
  }
}

TEST(NativePrefixSnapshotPolicyTest, PackedBufferAccountingRejectsOverflowBeforeAllocation) {
  using Range = PrefixSnapshotCache::Entry::Range;
  PrefixSnapshotCache::Entry entry;
  entry.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  entry.AddPackedBuffer(std::array<Range, 2>{{{0, 0, 8}, {8, 24, 4}}});
  EXPECT_EQ(entry.bytes, 12u);
  EXPECT_EQ(entry.shapes, (std::vector<std::vector<int64_t>>{{12}}));
  EXPECT_THROW(entry.AddPackedBuffer(std::array<Range, 2>{{{0, 0, std::numeric_limits<size_t>::max()}, {0, 0, 1}}}),
               std::overflow_error);
  entry.bytes = std::numeric_limits<size_t>::max() - 8;
  EXPECT_THROW(entry.AddPackedBuffer(std::array<Range, 1>{{{0, 0, 16}}}), std::overflow_error);
  EXPECT_EQ(entry.shapes.size(), 1u);
}

}  // namespace
}  // namespace Generators::test
