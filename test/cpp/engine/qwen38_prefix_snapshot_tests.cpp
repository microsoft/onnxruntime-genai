// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_helpers.h"
#include "logging.h"
#include "models/qwen38_graph.h"

namespace Generators::test {
namespace {

TEST(Qwen38PrefixSnapshotTest, DefaultsOnForSingleSlotAndWarnsOnlyForInactiveExplicitEnable) {
  Config config;
  EXPECT_EQ(CreateQwen38PrefixSnapshotCache(config), nullptr);
  OverlayConfig(config, R"({"engine":{"dynamic_batching":{"max_batch_size":1}}})");
  EXPECT_NE(CreateQwen38PrefixSnapshotCache(config), nullptr);
  for (const auto* settings : {
           R"({"prefix_caching":false,"max_batch_size":1})",
           R"({"prefix_caching":true,"max_batch_size":1,"max_snapshots":0})",
           R"({"prefix_caching":true,"max_batch_size":2})",
           R"({"prefix_caching":true,"max_batch_size":1,"snapshot_cache":{}})",
           R"({"max_batch_size":2})",
           R"({"max_batch_size":1,"max_snapshots":0})",
           R"({"max_batch_size":1,"snapshot_cache":{}})"}) {
    SCOPED_TRACE(settings);
    Config inactive;
    OverlayConfig(inactive, std::string{R"({"engine":{"dynamic_batching":)"} + settings + "}}");
    const auto previous_logging = g_log;
    SetLogBool("enabled", true);
    SetLogBool("warning", true);
    testing::internal::CaptureStderr();
    const auto cache = CreateQwen38PrefixSnapshotCache(inactive);
    const auto log = testing::internal::GetCapturedStderr();
    g_log = previous_logging;
    EXPECT_EQ(cache, nullptr);
    EXPECT_EQ(log.find("prefix caching is inactive") != std::string::npos,
              inactive.engine.dynamic_batching->prefix_caching.value_or(false));
  }
}

TEST(Qwen38PrefixSnapshotTest, PreservesDefaultOnStorageAndCountLimitsAcrossSingleSlotNormalization) {
  Config config;
  OverlayConfig(config, R"({"engine":{"dynamic_batching":{
    "max_batch_size":1,"max_snapshots":2,"snapshot_cache":{"ram_bytes":32}}}})");
  auto cache = CreateQwen38PrefixSnapshotCache(config);
  ApplyQwen38DynamicBatchingConfig(config);
  ASSERT_FALSE(config.engine.dynamic_batching);
  ASSERT_NE(cache, nullptr);
  std::vector<std::shared_ptr<PrefixSnapshotCache::Entry>> entries;
  for (int i = 0; i < 2; ++i) {
    auto entry = std::make_shared<PrefixSnapshotCache::Entry>();
    entry->type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    entry->shapes = {{16}};
    ASSERT_TRUE(cache->Allocate(entry, GetDeviceInterface(DeviceType::CPU)));
    EXPECT_EQ(entry->tier, PrefixSnapshotCache::Tier::Ram);
    entries.push_back(std::move(entry));
  }
  for (const auto& entry : entries) EXPECT_EQ(entry->state, PrefixSnapshotCache::State::Resident);
  EXPECT_EQ(cache->Used()[1], 32u);
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

}  // namespace
}  // namespace Generators::test
