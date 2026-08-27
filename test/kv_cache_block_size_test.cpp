// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the kv_cache_block_size sizing helpers (models/kv_cache_block_size.h):
// when block rounding applies, how sliding-window (local) layers are identified,
// and how the rounding is applied to the shared and per-layer cache shapes.
//
// These mirror the decisions DefaultKeyValueCache makes in its constructor, but on
// the pure helpers so the rounding behavior -- and the uniform vs. alternating
// sliding-window cases -- can be checked without a model or ONNX session.

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "models/kv_cache_block_size.h"

namespace Generators::test {
namespace {

using Shape = std::array<int64_t, 4>;

// [batch, kv_heads, seq_len, head_size]; only index 2 (seq_len) is rounded.
Shape MakeShape(int64_t seq_len) { return Shape{1, 2, seq_len, 16}; }

}  // namespace

// ---------------------------------------------------------------------------
// RoundUpToKvCacheBlock
// ---------------------------------------------------------------------------

TEST(KvCacheBlockSizeTest, RoundUpLeavesValueWhenNotConfigured) {
  // block_size <= 0 means "not configured": the length is returned unchanged.
  EXPECT_EQ(RoundUpToKvCacheBlock(1000, 0), 1000);
  EXPECT_EQ(RoundUpToKvCacheBlock(1000, -8), 1000);
}

TEST(KvCacheBlockSizeTest, RoundUpAlreadyAlignedIsUnchanged) {
  EXPECT_EQ(RoundUpToKvCacheBlock(512, 256), 512);
  EXPECT_EQ(RoundUpToKvCacheBlock(256, 256), 256);
}

TEST(KvCacheBlockSizeTest, RoundUpRoundsToNextMultiple) {
  EXPECT_EQ(RoundUpToKvCacheBlock(1, 256), 256);
  EXPECT_EQ(RoundUpToKvCacheBlock(257, 256), 512);
  EXPECT_EQ(RoundUpToKvCacheBlock(8192, 384), 8448);  // gpt-oss-20b style window rounding
}

// ---------------------------------------------------------------------------
// ShouldRoundKvCacheToBlock: the default (no option) case vs. the enabled cases.
// ---------------------------------------------------------------------------

TEST(KvCacheBlockSizeTest, DefaultCaseDoesNotRound) {
  // No search option set -> the default sizing path, no rounding.
  EXPECT_FALSE(ShouldRoundKvCacheToBlock(/*share_buffer=*/true, /*fixed_kv_seq_len=*/0,
                                         /*windowed=*/false, /*layers_empty=*/true,
                                         /*block_size=*/std::nullopt));
  // A zero block size is parsed as "not configured".
  EXPECT_FALSE(ShouldRoundKvCacheToBlock(true, 0, false, true, std::optional<size_t>{0}));
}

TEST(KvCacheBlockSizeTest, EnabledOnlyWithSharedBuffer) {
  EXPECT_TRUE(ShouldRoundKvCacheToBlock(true, 0, false, true, std::optional<size_t>{256}));
  // Without a shared past/present buffer the sequence dim is not fixed at
  // allocation time, so the option does not apply.
  EXPECT_FALSE(ShouldRoundKvCacheToBlock(false, 0, false, true, std::optional<size_t>{256}));
}

TEST(KvCacheBlockSizeTest, FixedShapeModelDoesNotRound) {
  // A fixed KV shape comes from the model graph and must not be resized.
  EXPECT_FALSE(ShouldRoundKvCacheToBlock(true, /*fixed_kv_seq_len=*/2048, false, true,
                                         std::optional<size_t>{256}));
}

TEST(KvCacheBlockSizeTest, UniformSlidingWindowCacheDoesNotRound) {
  // Windowed cache with an empty layers list => every layer is a sliding-window
  // (local) layer, so the whole cache is local and must not be grown.
  EXPECT_FALSE(ShouldRoundKvCacheToBlock(true, 0, /*windowed=*/true, /*layers_empty=*/true,
                                         std::optional<size_t>{256}));
}

TEST(KvCacheBlockSizeTest, AlternatingSlidingWindowCacheRounds) {
  // Windowed cache with an explicit layers list has global layers to round.
  EXPECT_TRUE(ShouldRoundKvCacheToBlock(true, 0, /*windowed=*/true, /*layers_empty=*/false,
                                        std::optional<size_t>{256}));
}

// ---------------------------------------------------------------------------
// ComputeLocalAttentionSlots
// ---------------------------------------------------------------------------

TEST(KvCacheBlockSizeTest, NoSlidingWindowLayersMeansNoLocalSlots) {
  const auto is_local = ComputeLocalAttentionSlots(/*layer_count=*/4, /*kv_layer_indices=*/{}, /*layers=*/{});
  EXPECT_EQ(is_local, (std::vector<std::uint8_t>{0, 0, 0, 0}));
}

TEST(KvCacheBlockSizeTest, AlternatingLayersMarkListedSlotsLocal) {
  // Layers 0 and 2 are sliding-window; slot index == model layer index here.
  const auto is_local = ComputeLocalAttentionSlots(4, {}, {0, 2});
  EXPECT_EQ(is_local, (std::vector<std::uint8_t>{1, 0, 1, 0}));
}

TEST(KvCacheBlockSizeTest, SparseKvLayoutMapsModelLayerToCacheSlot) {
  // Sparse KV layout: cache slot i holds model layer kv_layer_indices[i].
  // Model layer 5 is sliding-window and lives in cache slot 1.
  const std::vector<int> kv_layer_indices{3, 5, 7};
  const auto is_local = ComputeLocalAttentionSlots(3, kv_layer_indices, {5});
  EXPECT_EQ(is_local, (std::vector<std::uint8_t>{0, 1, 0}));
}

// ---------------------------------------------------------------------------
// ApplyKvCacheBlockSize: uniform vs. alternating rounding on real shapes.
// ---------------------------------------------------------------------------

TEST(KvCacheBlockSizeTest, ApplyRoundsSharedShapeWhenNoPerLayerShapes) {
  Shape shape = MakeShape(1000);
  std::vector<Shape> layer_shapes;  // uniform cache: no per-layer shapes
  ApplyKvCacheBlockSize(256, /*is_local=*/{}, shape, layer_shapes);
  EXPECT_EQ(shape[2], 1024);
}

TEST(KvCacheBlockSizeTest, ApplyRoundsEveryLayerWhenNoneLocal) {
  Shape shape = MakeShape(1000);
  std::vector<Shape> layer_shapes{MakeShape(1000), MakeShape(1000)};
  const std::vector<std::uint8_t> is_local{0, 0};
  ApplyKvCacheBlockSize(256, is_local, shape, layer_shapes);
  EXPECT_EQ(shape[2], 1024);
  EXPECT_EQ(layer_shapes[0][2], 1024);
  EXPECT_EQ(layer_shapes[1][2], 1024);
}

TEST(KvCacheBlockSizeTest, ApplySkipsLocalLayersOnAlternatingPath) {
  // Global layers sit at max_length (1000); local layers at a smaller window (300).
  Shape shape = MakeShape(1000);
  std::vector<Shape> layer_shapes{MakeShape(300), MakeShape(1000), MakeShape(300), MakeShape(1000)};
  const std::vector<std::uint8_t> is_local{1, 0, 1, 0};
  ApplyKvCacheBlockSize(256, is_local, shape, layer_shapes);

  EXPECT_EQ(shape[2], 1024);            // shared/global bound rounded
  EXPECT_EQ(layer_shapes[0][2], 300);   // local: untouched
  EXPECT_EQ(layer_shapes[1][2], 1024);  // global: rounded
  EXPECT_EQ(layer_shapes[2][2], 300);   // local: untouched
  EXPECT_EQ(layer_shapes[3][2], 1024);  // global: rounded
}

TEST(KvCacheBlockSizeTest, ApplyIsNoOpWhenBlockSizeNotConfigured) {
  Shape shape = MakeShape(1000);
  std::vector<Shape> layer_shapes{MakeShape(1000)};
  ApplyKvCacheBlockSize(0, {0}, shape, layer_shapes);
  EXPECT_EQ(shape[2], 1000);
  EXPECT_EQ(layer_shapes[0][2], 1000);
}

}  // namespace Generators::test
