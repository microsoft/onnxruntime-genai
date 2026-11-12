// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_helpers.h"
#include "logging.h"
#include "models/qwen38_graph.h"

namespace Generators::test {
namespace {

std::shared_ptr<PrefixSnapshotCache::Entry> ByteEntry(size_t bytes) {
  auto entry = std::make_shared<PrefixSnapshotCache::Entry>();
  entry->type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  entry->shapes = {{static_cast<int64_t>(bytes)}};
  return entry;
}

TEST(Qwen38PrefixSnapshotTest, PolicyGatesOptInAndWarnsWhenInactive) {
  Config config;
  EXPECT_EQ(CreateQwen38PrefixSnapshotCache(ApplyQwen38DynamicBatchingConfig(config)), nullptr);
  OverlayConfig(config, R"({"engine":{"dynamic_batching":{"max_batch_size":1}}})");
  EXPECT_EQ(CreateQwen38PrefixSnapshotCache(ApplyQwen38DynamicBatchingConfig(config)), nullptr);
  for (const auto* settings : {
           R"({"prefix_caching":false,"max_batch_size":1})",
           R"({"prefix_caching":true,"max_batch_size":1,"max_snapshots":0})",
           R"({"prefix_caching":true,"max_batch_size":2})",
           R"({"prefix_caching":true,"max_batch_size":1,"snapshot_cache":{}})"}) {
    Config inactive;
    OverlayConfig(inactive, std::string{R"({"engine":{"dynamic_batching":)"} + settings + "}}");
    const auto previous_logging = g_log;
    SetLogBool("enabled", true);
    SetLogBool("warning", true);
    testing::internal::CaptureStderr();
    const auto resolved = ApplyQwen38DynamicBatchingConfig(inactive);
    const auto cache = CreateQwen38PrefixSnapshotCache(resolved);
    const auto log = testing::internal::GetCapturedStderr();
    g_log = previous_logging;
    EXPECT_EQ(cache, nullptr);
    EXPECT_EQ(log.find("prefix caching is inactive") != std::string::npos,
              resolved.batching->prefix_caching.value_or(false));
  }
}

TEST(Qwen38PrefixSnapshotTest, PolicyUsesRealCountEvictionAndTierBudgets) {
  for (const auto* settings : {
           R"({"prefix_caching":true,"max_batch_size":1})",
           R"({"prefix_caching":true,"max_batch_size":1,"max_snapshots":1,"snapshot_cache":{"ram_bytes":32}})",
           R"({"prefix_caching":true,"max_batch_size":1,"max_snapshots":1,"snapshot_cache":{"disk_bytes":32}})"}) {
    Config config;
    OverlayConfig(config, std::string{R"({"engine":{"dynamic_batching":)"} + settings + "}}");
    auto cache = CreateQwen38PrefixSnapshotCache(ApplyQwen38DynamicBatchingConfig(config));
    ASSERT_NE(cache, nullptr);
    auto* cpu = GetDeviceInterface(DeviceType::CPU);
    auto first = ByteEntry(16);
    ASSERT_TRUE(cache->Allocate(first, cpu, "qwen38"));
    auto lease = first;
    auto second = ByteEntry(16);
    EXPECT_FALSE(cache->Allocate(second, cpu, "qwen38"));
    EXPECT_EQ(first->state, PrefixSnapshotCache::State::Resident);
    lease.reset();
    ASSERT_TRUE(cache->Allocate(second, cpu, "qwen38"));
    EXPECT_EQ(first->state, PrefixSnapshotCache::State::Evicted);
    EXPECT_EQ(second->state, PrefixSnapshotCache::State::Resident);
    EXPECT_EQ(cache->Used()[static_cast<size_t>(second->tier)], 16u);
  }
  Config undersized;
  OverlayConfig(undersized, R"({"engine":{"dynamic_batching":{
    "prefix_caching":true,"max_batch_size":1,"snapshot_cache":{"ram_bytes":15,"disk_bytes":15}}}})");
  auto cache = CreateQwen38PrefixSnapshotCache(ApplyQwen38DynamicBatchingConfig(undersized));
  ASSERT_NE(cache, nullptr);
  EXPECT_FALSE(cache->Allocate(ByteEntry(16), GetDeviceInterface(DeviceType::CPU), "qwen38"));
}

TEST(Qwen38PrefixSnapshotTest, SingleSlotCreatesCacheFromResolvedSettingsAfterNormalization) {
  for (int explicit_chunk : {0, 8192}) {
    for (size_t scheduled_tokens : {size_t{2048}, size_t{8192}}) {
      Config config;
      config.model.qwen38_graph.prefill.enabled = true;
      config.model.qwen38_graph.prefill.dense.filename = "dense.onnx";
      EXPECT_EQ(EffectiveQwen38PrefillChunkTokens(config.model.qwen38_graph), 8192u);
      OverlayConfig(config, R"({"engine":{"dynamic_batching":{"prefix_caching":true,"max_batch_size":1}}})");
      config.model.qwen38_graph.prefill.chunk = explicit_chunk;
      config.engine.dynamic_batching->max_scheduled_tokens = scheduled_tokens;
      const auto resolved = ApplyQwen38DynamicBatchingConfig(config);
      EXPECT_FALSE(resolved.has_dynamic_batching_kv);
      EXPECT_FALSE(config.engine.dynamic_batching);
      auto cache = CreateQwen38PrefixSnapshotCache(resolved);
      ASSERT_NE(cache, nullptr);
      EXPECT_EQ(EffectiveQwen38PrefillChunkTokens(config.model.qwen38_graph),
                explicit_chunk ? static_cast<size_t>(explicit_chunk) : scheduled_tokens);
    }
  }
}

TEST(Qwen38PrefixSnapshotTest, ResolvedSnapshotPolicyDoesNotReadLaterConfigChanges) {
  Config config;
  OverlayConfig(config, R"({"engine":{"dynamic_batching":{
    "prefix_caching":true,"max_batch_size":1,"max_snapshots":1,"snapshot_cache":{"ram_bytes":16}}}})");
  const auto resolved = ApplyQwen38DynamicBatchingConfig(config);
  ASSERT_FALSE(config.engine.dynamic_batching);
  OverlayConfig(config, R"({"engine":{"dynamic_batching":{"prefix_caching":false,"max_batch_size":2}}})");
  auto cache = CreateQwen38PrefixSnapshotCache(resolved);
  ASSERT_NE(cache, nullptr);
  auto first = ByteEntry(16);
  auto second = ByteEntry(16);
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  ASSERT_TRUE(cache->Allocate(first, cpu, "qwen38"));
  EXPECT_EQ(first->tier, PrefixSnapshotCache::Tier::Ram);
  ASSERT_TRUE(cache->Allocate(second, cpu, "qwen38"));
  EXPECT_EQ(first->state, PrefixSnapshotCache::State::Evicted);
  EXPECT_EQ(second->tier, PrefixSnapshotCache::Tier::Ram);
}

TEST(Qwen38PrefixSnapshotTest, DflashEligibilityAndCapturePlanningKeepOnlyTheLastUsefulCheckpoint) {
  using Chunk = Qwen38DflashPromptFeatures::Chunk;
  EXPECT_TRUE(Qwen38PrefixCheckpointEligible(false, std::nullopt, 0));
  EXPECT_FALSE(Qwen38PrefixCheckpointEligible(true, std::nullopt, 3));
  EXPECT_TRUE(Qwen38PrefixCheckpointEligible(true, Chunk{0, 1}, 0));
  EXPECT_FALSE(Qwen38PrefixCheckpointEligible(true, Chunk{4, 4}, 0));
  EXPECT_FALSE(Qwen38PrefixCheckpointEligible(true, Chunk{4, 2}, 3));
  EXPECT_TRUE(Qwen38PrefixCheckpointEligible(true, Chunk{4, 3}, 3));
  for (size_t window : {size_t{0}, size_t{3}, size_t{5}}) {
    for (size_t total : {size_t{1}, size_t{8}, size_t{13}, size_t{16}}) {
      std::vector<size_t> eligible_captures;
      size_t refusals = 0;
      for (size_t position = 0; position < total; position += 4) {
        const Chunk saved{position, std::min(size_t{4}, total - position)};
        if (!ShouldCaptureQwen38Prefix(total - position - saved.rows, 4, true, saved, window)) continue;
        if (Qwen38PrefixCheckpointEligible(true, saved, window))
          eligible_captures.push_back(position + saved.rows);
        else
          ++refusals;
      }
      const bool final_eligible = total <= 4 || (window != 0 && (total - 1) % 4 + 1 >= window);
      EXPECT_EQ(eligible_captures, (std::vector<size_t>{
                                       final_eligible ? total : (window == 3 ? (total / 4) * 4 : 4)}));
      EXPECT_EQ(refusals, final_eligible ? 0u : 1u);
    }
  }
  EXPECT_FALSE(ShouldCaptureQwen38Prefix(8, 4, false, std::nullopt, 0));
  EXPECT_TRUE(ShouldCaptureQwen38Prefix(0, 4, false, std::nullopt, 0));
}

TEST(Qwen38PrefixSnapshotTest, RejectsInvalidAndOverflowingRanges) {
  EXPECT_THROW(PlanQwen38PrefixRanges("past_position_ids", {8}, 8, 64, 9), std::out_of_range);
  EXPECT_THROW(PlanQwen38PrefixRanges("d_prompt_hidden", {3}, 2, 48, 9), std::out_of_range);
  EXPECT_THROW(PlanQwen38PrefixRanges("past_key_values.0.key", {1, 0, 3}, 2, 0, 5), std::invalid_argument);
  EXPECT_THROW(PlanQwen38PrefixRanges("past_key_values.0.key", {1, 8, 3}, 2, 47, 5), std::invalid_argument);
  EXPECT_THROW(PlanQwen38PrefixRanges("d_dflash_prompt", {4}, 2, 64, std::numeric_limits<size_t>::max()),
               std::overflow_error);
}

class Qwen38PrefixRoundTripTest : public testing::TestWithParam<size_t> {};

TEST_P(Qwen38PrefixRoundTripTest, RestoresPlannedRowsAndAllBanksWithoutTouchingUnusedCapacity) {
  struct Buffer {
    const char* name;
    std::vector<int64_t> shape;
    size_t element_bytes, bytes, length, stride, kept_per_stride;
    bool index{};
  };
  const std::vector<Buffer> buffers{
      {"past_key_values.0.key", {1, 2, 8, 3}, 2, 96, 5, 48, 30},
      {"past_key_values.0.value", {1, 2, 8, 3}, 2, 96, 5, 48, 30},
      {"shared.index_key", {1, 2, 4, 3}, 4, 96, 5, 48, 24, true},
      {"past_position_ids", {1, 8}, 8, 64, 5, 64, 40},
      {"bank0", {16}, 1, 16, 5, 16, 16},
      {"bank1", {16}, 1, 16, 5, 16, 16},
      {"bank2", {16}, 1, 16, 5, 16, 16},
      {"ctx_commit", {16}, 1, 16, 5, 16, 16},
      {"last_hidden", {1, 3}, 2, 6, 5, 6, 6},
      {"d_prompt_hidden", {8, 3}, 2, 48, 5, 48, 30},
      {"d_dflash_prompt", {8, 4}, 2, 64, 3, 64, 24},
      {"dense.state", {2, 3}, 2, 12, 5, 12, 12},
  };
  std::array<size_t, 3> limits{};
  limits[GetParam()] = 4096;
  PrefixSnapshotCache cache{limits, 1, {}, GetParam() == 0};
  auto snapshot = std::make_shared<PrefixSnapshotCache::Entry>();
  snapshot->type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  for (const auto& buffer : buffers)
    snapshot->shapes.push_back({static_cast<int64_t>(buffer.bytes / buffer.stride * buffer.kept_per_stride)});
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  ASSERT_TRUE(cache.Allocate(snapshot, cpu, "qwen38"));
  for (size_t i = 0; i < buffers.size(); ++i) {
    const auto& buffer = buffers[i];
    auto ranges = PlanQwen38PrefixRanges(buffer.name, buffer.shape, buffer.element_bytes,
                                         buffer.bytes, buffer.length, buffer.index);
    Tensor source{cpu, snapshot->type};
    source.CreateTensor(std::array<int64_t, 1>{static_cast<int64_t>(buffer.bytes)});
    std::iota(source.GetMutableData<uint8_t>(), source.GetMutableData<uint8_t>() + buffer.bytes,
              static_cast<uint8_t>(i + 1));
    snapshot->Write(i, source, ranges);
  }
  snapshot->Flush();
  for (size_t i = 0; i < buffers.size(); ++i) {
    const auto& buffer = buffers[i];
    auto ranges = PlanQwen38PrefixRanges(buffer.name, buffer.shape, buffer.element_bytes,
                                         buffer.bytes, buffer.length, buffer.index);
    Tensor destination{cpu, snapshot->type};
    destination.CreateTensor(std::array<int64_t, 1>{static_cast<int64_t>(buffer.bytes)});
    auto* data = destination.GetMutableData<uint8_t>();
    std::fill_n(data, buffer.bytes, uint8_t{231});
    snapshot->Read(i, destination, ranges);
    std::vector<uint8_t> expected(buffer.bytes, 231);
    for (size_t j = 0; j < buffer.bytes; ++j)
      if (j % buffer.stride < buffer.kept_per_stride) expected[j] = static_cast<uint8_t>(i + j + 1);
    EXPECT_EQ(std::vector<uint8_t>(data, data + buffer.bytes), expected) << buffer.name;
  }
  // The incomplete index group needs its extra row; KV rows clamp at physical capacity.
  EXPECT_EQ(PlanQwen38PrefixRanges("index", {1, 4, 3}, 4, 48, 8, true).front().bytes, 36u);
  EXPECT_EQ(PlanQwen38PrefixRanges("cache.key", {1, 8, 3}, 2, 48, 20).front().bytes, 48u);
}

INSTANTIATE_TEST_SUITE_P(StorageTiers, Qwen38PrefixRoundTripTest, testing::Values(0u, 1u, 2u));

}  // namespace
}  // namespace Generators::test
