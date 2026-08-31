// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <set>
#include <vector>

#include "engine/window_ring.h"

using Generators::WindowLiveSpan;
using Generators::WindowRingBlocks;
using Generators::WindowRingColumn;
using Generators::WindowRingSlot;

// A sliding-window paged layer is served from a ring of blocks rather than one block per position.
// These tests pin the two things this has to get right: the ring is big enough that a step never
// overwrites a position it still has to read, and the addressing the runtime builds into the block
// table agrees with the addressing the operator performs.

namespace {

// Positions a step is allowed to read: the queries at [past, past + scheduled) each reach back
// window_size positions, so the oldest one still needed is past - (window_size - 1).
std::vector<size_t> LivePositions(size_t past, size_t scheduled, size_t window_size) {
  const size_t newest = past + scheduled;  // exclusive
  const size_t oldest = past + 1 >= window_size ? past + 1 - window_size : 0;
  std::vector<size_t> positions;
  for (size_t p = oldest; p < newest; ++p) {
    positions.push_back(p);
  }
  return positions;
}

}  // namespace

// ===========================================================================
// Sizing
// ===========================================================================

TEST(WindowRingTest, SpanCoversTheChunkAndTheWindowBehindIt) {
  // 256 new positions, each looking back over 128, touches 383 positions in total.
  EXPECT_EQ(WindowLiveSpan(256, 128), 383u);
  // A pure decode step still needs the whole window.
  EXPECT_EQ(WindowLiveSpan(1, 128), 128u);
}

TEST(WindowRingTest, RingRoundsUpToWholeBlocks) {
  // gpt-oss-20b served at block_size 256: 383 live positions fit in two blocks.
  EXPECT_EQ(WindowRingBlocks(256, 128, 256), 2u);
  // An exact fit must not round up to a spare block.
  EXPECT_EQ(WindowRingBlocks(129, 128, 256), 1u);
  // One position past the block boundary does.
  EXPECT_EQ(WindowRingBlocks(130, 128, 256), 2u);
}

TEST(WindowRingTest, SmallerChunksBuyASmallerRing) {
  // The ring is what the sliding-window layers cost, so trading prefill chunk size for capacity
  // is the knob this layout offers.
  EXPECT_EQ(WindowRingBlocks(256, 128, 64), 6u);
  EXPECT_EQ(WindowRingBlocks(64, 128, 64), 3u);
  EXPECT_EQ(WindowRingBlocks(1, 128, 64), 2u);
}

// ===========================================================================
// Addressing
// ===========================================================================

TEST(WindowRingTest, SlotFollowsTheRepeatedBlockTable) {
  // The runtime writes ring[j % ring_blocks] into column j and the operator reads column
  // position / block_size, so the two have to compose into position % (ring_blocks * block_size).
  constexpr size_t ring_blocks = 2;
  constexpr size_t block_size = 256;
  for (size_t position = 0; position < 4000; ++position) {
    const size_t column = position / block_size;
    const size_t expected = WindowRingColumn(column, ring_blocks) * block_size + position % block_size;
    EXPECT_EQ(WindowRingSlot(position, ring_blocks, block_size), expected) << "position " << position;
    EXPECT_EQ(WindowRingSlot(position, ring_blocks, block_size), position % (ring_blocks * block_size));
  }
}

TEST(WindowRingTest, PositionsRepeatOncePerRing) {
  constexpr size_t ring_blocks = 3;
  constexpr size_t block_size = 8;
  constexpr size_t ring_slots = ring_blocks * block_size;

  EXPECT_EQ(WindowRingSlot(0, ring_blocks, block_size), WindowRingSlot(ring_slots, ring_blocks, block_size));
  EXPECT_NE(WindowRingSlot(0, ring_blocks, block_size), WindowRingSlot(ring_slots - 1, ring_blocks, block_size));
}

// ===========================================================================
// The invariant the sizing exists to guarantee
// ===========================================================================

TEST(WindowRingTest, NoLivePositionIsOverwrittenDuringPrefill) {
  constexpr size_t window_size = 128;
  constexpr size_t block_size = 256;
  constexpr size_t chunk_size = 256;
  const size_t ring_blocks = WindowRingBlocks(chunk_size, window_size, block_size);

  for (size_t past = 0; past < 3000; past += chunk_size) {
    const auto positions = LivePositions(past, chunk_size, window_size);
    std::set<size_t> slots;
    for (size_t position : positions) {
      slots.insert(WindowRingSlot(position, ring_blocks, block_size));
    }
    // Every position the step reads must sit in a slot of its own, or one of them was clobbered.
    EXPECT_EQ(slots.size(), positions.size()) << "prefill step at past " << past;
  }
}

TEST(WindowRingTest, NoLivePositionIsOverwrittenDuringDecode) {
  constexpr size_t window_size = 128;
  constexpr size_t block_size = 64;
  constexpr size_t chunk_size = 64;
  const size_t ring_blocks = WindowRingBlocks(chunk_size, window_size, block_size);

  for (size_t past = 0; past < 1000; ++past) {
    const auto positions = LivePositions(past, /*scheduled=*/1, window_size);
    std::set<size_t> slots;
    for (size_t position : positions) {
      slots.insert(WindowRingSlot(position, ring_blocks, block_size));
    }
    EXPECT_EQ(slots.size(), positions.size()) << "decode step at past " << past;
  }
}

TEST(WindowRingTest, ARingSizedForASmallerChunkFailsOnABiggerStep) {
  // Documents why the cache rejects a step that carries more tokens than it was sized for: with a
  // ring built for chunk_size 64, a 256-token step overwrites positions it still has to attend to.
  constexpr size_t window_size = 128;
  constexpr size_t block_size = 64;
  const size_t ring_blocks = WindowRingBlocks(/*chunk_size=*/64, window_size, block_size);

  const auto positions = LivePositions(/*past=*/512, /*scheduled=*/256, window_size);
  std::set<size_t> slots;
  for (size_t position : positions) {
    slots.insert(WindowRingSlot(position, ring_blocks, block_size));
  }
  EXPECT_LT(slots.size(), positions.size());
}
