// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Pure unit tests for the paged-cache block arithmetic and ownership primitives
// (Generators::Block and Generators::BlockPool). These require no model, no ONNX session, and no
// GPU; they validate slot arithmetic, allocation/reservation semantics, capacity accounting, and
// free-time ownership guards deterministically.

#include <array>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "engine/block.h"

namespace Generators {

namespace {

constexpr size_t kBlockSize = 4;

// ---------------------------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------------------------

TEST(BlockTest, ConstructEmpty) {
  Block block(/*id=*/0, /*slots=*/0, kBlockSize);
  EXPECT_EQ(block.Id(), 0u);
  EXPECT_EQ(block.Size(), 0u);
  EXPECT_EQ(block.Capacity(), kBlockSize);
  EXPECT_EQ(block.EmptySlots(), kBlockSize);
  EXPECT_FALSE(block.IsFull());
  EXPECT_TRUE(block.SlotIds().empty());
}

TEST(BlockTest, ConstructPartial) {
  Block block(/*id=*/2, /*slots=*/1, kBlockSize);
  EXPECT_EQ(block.Size(), 1u);
  EXPECT_EQ(block.EmptySlots(), kBlockSize - 1);
  EXPECT_FALSE(block.IsFull());
  // Slot ids are contiguous and offset by id * capacity.
  EXPECT_EQ(block.SlotIds(), (std::vector<size_t>{2 * kBlockSize}));
}

TEST(BlockTest, ConstructFull) {
  Block block(/*id=*/1, /*slots=*/kBlockSize, kBlockSize);
  EXPECT_EQ(block.Size(), kBlockSize);
  EXPECT_EQ(block.EmptySlots(), 0u);
  EXPECT_TRUE(block.IsFull());
  EXPECT_EQ(block.SlotIds(),
            (std::vector<size_t>{kBlockSize, kBlockSize + 1, kBlockSize + 2, kBlockSize + 3}));
}

TEST(BlockTest, IdentityAndSlotIdsAreStable) {
  static_assert(std::is_copy_constructible_v<Block>);
  static_assert(!std::is_copy_assignable_v<Block>);
  static_assert(!std::is_move_assignable_v<Block>);
  Block block(/*id=*/3, /*slots=*/2, kBlockSize);
  EXPECT_EQ(block.Id(), 3u);
  EXPECT_EQ(block.SlotIds(), (std::vector<size_t>{3 * kBlockSize, 3 * kBlockSize + 1}));
}

// ---------------------------------------------------------------------------------------------
// BlockPool: BlocksNeeded
// ---------------------------------------------------------------------------------------------

TEST(BlockPoolTest, BlocksNeededBoundaries) {
  constexpr size_t kNumBlocks = 8;
  BlockPool pool(kBlockSize, kNumBlocks);
  EXPECT_EQ(pool.BlocksNeeded(0), 0u);
  EXPECT_EQ(pool.BlocksNeeded(1), 1u);
  EXPECT_EQ(pool.BlocksNeeded(kBlockSize - 1), 1u);
  EXPECT_EQ(pool.BlocksNeeded(kBlockSize), 1u);
  EXPECT_EQ(pool.BlocksNeeded(kBlockSize + 1), 2u);
  EXPECT_EQ(pool.BlocksNeeded(kNumBlocks * kBlockSize), kNumBlocks);
  EXPECT_EQ(
      pool.BlocksNeeded(std::numeric_limits<size_t>::max()),
      std::numeric_limits<size_t>::max() / kBlockSize + 1);
  EXPECT_THROW(
      pool.AllocateBlocks(std::numeric_limits<size_t>::max()),
      std::runtime_error);
}

// ---------------------------------------------------------------------------------------------
// BlockPool: allocation vs reservation
// ---------------------------------------------------------------------------------------------

TEST(BlockPoolTest, AllocateMarksSlotsUsed) {
  BlockPool pool(kBlockSize, /*num_blocks=*/4);
  auto blocks = pool.AllocateBlocks(kBlockSize + 1);  // 2 blocks: one full, one with a single slot.
  ASSERT_EQ(blocks.size(), 2u);
  EXPECT_TRUE(blocks[0]->IsFull());
  EXPECT_EQ(blocks[0]->Size(), kBlockSize);
  EXPECT_EQ(blocks[1]->Size(), 1u);
  EXPECT_EQ(pool.AvailableBlocks(), 2u);
  EXPECT_EQ(pool.Size(), 2u);
}

TEST(BlockPoolTest, ReserveLeavesSlotsEmpty) {
  BlockPool pool(kBlockSize, /*num_blocks=*/4);
  auto blocks = pool.ReserveBlocks(kBlockSize + 1);  // 2 blocks reserved, no slots used yet.
  ASSERT_EQ(blocks.size(), 2u);
  EXPECT_EQ(blocks[0]->Size(), 0u);
  EXPECT_EQ(blocks[1]->Size(), 0u);
  // Reservation still consumes pool capacity even though the slots are empty.
  EXPECT_EQ(pool.AvailableBlocks(), 2u);
}

TEST(BlockPoolTest, ExactFitConsumesFinalBlock) {
  constexpr size_t kNumBlocks = 3;
  BlockPool pool(kBlockSize, kNumBlocks);
  auto blocks = pool.AllocateBlocks(kNumBlocks * kBlockSize);
  EXPECT_EQ(blocks.size(), kNumBlocks);
  EXPECT_EQ(pool.AvailableBlocks(), 0u);
  EXPECT_EQ(pool.Size(), kNumBlocks);
}

TEST(BlockPoolTest, OneOverCapacityFailsWithoutPartialAllocation) {
  constexpr size_t kNumBlocks = 3;
  BlockPool pool(kBlockSize, kNumBlocks);
  EXPECT_THROW(pool.AllocateBlocks(kNumBlocks * kBlockSize + 1), std::runtime_error);
  // A rejected allocation must not consume any capacity.
  EXPECT_EQ(pool.AvailableBlocks(), kNumBlocks);
  EXPECT_EQ(pool.Size(), 0u);
}

TEST(BlockPoolTest, FailedMultiBlockAllocationLeavesPoolUnchanged) {
  constexpr size_t kNumBlocks = 4;
  BlockPool pool(kBlockSize, kNumBlocks);
  auto held = pool.AllocateBlocks(2 * kBlockSize);  // Occupy 2 of 4 blocks.
  ASSERT_EQ(pool.AvailableBlocks(), 2u);

  // Asking for 3 blocks when only 2 remain must fail atomically, leaving the earlier allocation and
  // the free count untouched.
  EXPECT_THROW(pool.AllocateBlocks(3 * kBlockSize), std::runtime_error);
  EXPECT_EQ(pool.AvailableBlocks(), 2u);
  EXPECT_EQ(pool.Size(), 2u);
}

// ---------------------------------------------------------------------------------------------
// BlockPool: free
// ---------------------------------------------------------------------------------------------

TEST(BlockPoolTest, FreeRestoresCapacity) {
  BlockPool pool(kBlockSize, /*num_blocks=*/4);
  auto blocks = pool.AllocateBlocks(2 * kBlockSize);
  ASSERT_EQ(pool.AvailableBlocks(), 2u);
  pool.Free(blocks);
  EXPECT_EQ(pool.AvailableBlocks(), 4u);
  EXPECT_EQ(pool.Size(), 0u);
}

TEST(BlockPoolTest, FreeNullBlockRejected) {
  BlockPool pool(kBlockSize, /*num_blocks=*/4);
  std::vector<std::shared_ptr<Block>> blocks{nullptr};
  EXPECT_THROW(pool.Free(blocks), std::runtime_error);
}

TEST(BlockPoolTest, FreeOutOfRangeIdRejected) {
  BlockPool pool(kBlockSize, /*num_blocks=*/2);
  std::vector<std::shared_ptr<Block>> blocks{std::make_shared<Block>(/*id=*/5, 0, kBlockSize)};
  EXPECT_THROW(pool.Free(blocks), std::runtime_error);
  EXPECT_EQ(pool.AvailableBlocks(), 2u);
}

TEST(BlockPoolTest, FreeForeignBlockRejected) {
  BlockPool pool(kBlockSize, /*num_blocks=*/4);
  // A block with a valid id that this pool never handed out (the slot is currently free).
  std::vector<std::shared_ptr<Block>> foreign{std::make_shared<Block>(/*id=*/0, 0, kBlockSize)};
  EXPECT_THROW(pool.Free(foreign), std::runtime_error);
  EXPECT_EQ(pool.AvailableBlocks(), 4u);
}

TEST(BlockPoolTest, DuplicateFreeInSameCallRejected) {
  BlockPool pool(kBlockSize, /*num_blocks=*/4);
  auto blocks = pool.AllocateBlocks(kBlockSize);
  ASSERT_EQ(blocks.size(), 1u);
  std::vector<std::shared_ptr<Block>> duplicated{blocks[0], blocks[0]};
  EXPECT_THROW(pool.Free(duplicated), std::runtime_error);
  // Rejected before any mutation: the block is still owned by the pool.
  EXPECT_EQ(pool.AvailableBlocks(), 3u);
}

TEST(BlockPoolTest, DoubleFreeAcrossCallsRejected) {
  BlockPool pool(kBlockSize, /*num_blocks=*/4);
  auto blocks = pool.AllocateBlocks(kBlockSize);
  pool.Free(blocks);
  ASSERT_EQ(pool.AvailableBlocks(), 4u);
  // Freeing the same (now stale) block again must be rejected, not silently double-freed.
  EXPECT_THROW(pool.Free(blocks), std::runtime_error);
  EXPECT_EQ(pool.AvailableBlocks(), 4u);
}

TEST(BlockPoolTest, RepeatedAllocateFreeCyclesPreserveTotals) {
  constexpr size_t kNumBlocks = 4;
  BlockPool pool(kBlockSize, kNumBlocks);
  for (int cycle = 0; cycle < 5; ++cycle) {
    auto blocks = pool.AllocateBlocks(kNumBlocks * kBlockSize);
    EXPECT_EQ(pool.AvailableBlocks(), 0u);
    pool.Free(blocks);
    EXPECT_EQ(pool.AvailableBlocks(), kNumBlocks);
    EXPECT_EQ(pool.Capacity(), kNumBlocks);
  }
}

TEST(BlockPoolTest, ValidateFreeIsPureAndPublicationIsNoexcept) {
  BlockPool pool(kBlockSize, 2);
  auto blocks = pool.AllocateBlocks(2 * kBlockSize);
  const auto generation = pool.MutationGeneration();

  pool.ValidateFree(blocks);

  EXPECT_EQ(pool.AvailableBlocks(), 0u);
  EXPECT_EQ(pool.MutationGeneration(), generation);
  const std::span<const std::shared_ptr<Block>> block_span{blocks};
  static_assert(noexcept(pool.FreeValidated(block_span)));
  pool.FreeValidated(block_span);
  EXPECT_EQ(pool.AvailableBlocks(), 2u);
  EXPECT_EQ(pool.MutationGeneration(), generation + 1);
}

TEST(BlockPoolTest, FreeValidatedMisuseIsANoop) {
  BlockPool pool(kBlockSize, 1);
  const auto blocks = pool.AllocateBlocks(kBlockSize);
  const auto generation = pool.MutationGeneration();
  const std::array<std::shared_ptr<Block>, 1> null_block{nullptr};
  const std::array<std::shared_ptr<Block>, 1> out_of_range{
      std::make_shared<Block>(pool.Capacity(), 0, kBlockSize)};
  const std::array<std::shared_ptr<Block>, 2> duplicate{
      blocks[0], blocks[0]};

  pool.FreeValidated(null_block);
  pool.FreeValidated(out_of_range);
  pool.FreeValidated(duplicate);

  EXPECT_TRUE(pool.Owns(blocks[0]));
  EXPECT_EQ(pool.AvailableBlocks(), 0u);
  EXPECT_EQ(pool.MutationGeneration(), generation);
}

}  // namespace
}  // namespace Generators
