// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "engine/paged_cache_reservation.h"

namespace Generators {
namespace {

constexpr size_t kBlockSize = 4;
const char kRequestStorageA{};
const char kRequestStorageB{};
const void* const kRequestA = &kRequestStorageA;
const void* const kRequestB = &kRequestStorageB;

TEST(PagedCacheReservationTest, SumsPerRequestBlockCeilings) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables;
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 1, true},
      PagedCacheReservationRequest{kRequestB, 1, true},
  };

  PagedCacheReservation reservation{pool, tables, requests};

  EXPECT_EQ(reservation.ReservedBlockCount(), 2u);
  EXPECT_EQ(pool.AvailableBlocks(), 2u);
}

TEST(PagedCacheReservationTest, AggregateRejectionDoesNotConsumeBlocks) {
  BlockPool pool{kBlockSize, 1};
  std::vector<PagedCacheBlockTable> tables;
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 1, true},
      PagedCacheReservationRequest{kRequestB, 1, true},
  };

  EXPECT_THROW((PagedCacheReservation{pool, tables, requests}), std::runtime_error);
  EXPECT_TRUE(tables.empty());
  EXPECT_EQ(pool.AvailableBlocks(), 1u);
}

TEST(PagedCacheReservationTest, CommitConsumesExistingTailWithoutNewBlock) {
  BlockPool pool{kBlockSize, 2};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 3, pool.AllocateBlocks(3)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 4, false},
  };

  PagedCacheReservation reservation{pool, tables, requests};
  ASSERT_EQ(reservation.Deltas().size(), 1u);
  EXPECT_EQ(reservation.Deltas()[0].tail_slots_to_consume, 1u);
  EXPECT_EQ(reservation.ReservedBlockCount(), 0u);

  reservation.Commit();

  EXPECT_EQ(tables[0].committed_slots, 4u);
  EXPECT_EQ(tables[0].blocks.size(), 1u);
  EXPECT_TRUE(tables[0].blocks[0]->IsFull());
}

TEST(PagedCacheReservationTest, CommitPrefixCommitsOnlyTheAcceptedSlots) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 8, false},
  };

  PagedCacheReservation reservation{pool, tables, requests};
  ASSERT_EQ(reservation.ReservedBlockCount(), 1u);
  // A four-token verify step of which two were accepted.
  reservation.CommitPrefix(kRequestA, /*step_slots=*/4, /*kept_slots=*/2);
  ASSERT_EQ(reservation.Deltas().size(), 1u);
  EXPECT_EQ(reservation.Deltas()[0].target_slots, 6u);

  reservation.Commit();

  // The rejected slots stay inside the block the request now owns, ready for the next step.
  EXPECT_EQ(tables[0].committed_slots, 6u);
  EXPECT_EQ(tables[0].blocks.size(), 2u);
  EXPECT_EQ(tables[0].blocks[1]->EmptySlots(), 2u);
}

TEST(PagedCacheReservationTest, CommitPrefixRejectsArgumentsOutsideThePlannedStep) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 8, false},
  };

  PagedCacheReservation reservation{pool, tables, requests};
  EXPECT_THROW(reservation.CommitPrefix(kRequestB, 4, 2), std::runtime_error);
  EXPECT_THROW(reservation.CommitPrefix(kRequestA, 4, 0), std::runtime_error);
  EXPECT_THROW(reservation.CommitPrefix(kRequestA, 4, 5), std::runtime_error);
  // The step length has to be the growth this reservation actually planned.
  EXPECT_THROW(reservation.CommitPrefix(kRequestA, 3, 2), std::runtime_error);
  reservation.Commit();
  EXPECT_THROW(reservation.CommitPrefix(kRequestA, 4, 2), std::logic_error);
}

TEST(PagedCacheReservationTest, ProposedBlockTableIncludesReservationsAndPadding) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
      PagedCacheReservationRequest{kRequestB, 1, true},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  const std::array request_ids{kRequestB, kRequestA};
  std::array<int32_t, 6> block_table;

  reservation.FillBlockTable(request_ids, 3, block_table);

  EXPECT_EQ(block_table[0], 2);
  EXPECT_EQ(block_table[1], -1);
  EXPECT_EQ(block_table[2], -1);
  EXPECT_EQ(block_table[3], 0);
  EXPECT_EQ(block_table[4], 1);
  EXPECT_EQ(block_table[5], -1);
}

TEST(PagedCacheReservationTest, ReleaseIsIdempotentAndBlocksCanBeReused) {
  BlockPool pool{kBlockSize, 1};
  std::vector<PagedCacheBlockTable> tables;
  const std::array request_a{
      PagedCacheReservationRequest{kRequestA, 1, true},
  };
  PagedCacheReservation first{pool, tables, request_a};
  std::array<int32_t, 1> first_table;
  const std::array request_a_id{kRequestA};
  first.FillBlockTable(request_a_id, 1, first_table);

  first.Release();
  first.Release();

  EXPECT_EQ(pool.AvailableBlocks(), 1u);
  const std::array request_b{
      PagedCacheReservationRequest{kRequestB, 1, true},
  };
  PagedCacheReservation second{pool, tables, request_b};
  std::array<int32_t, 1> second_table;
  const std::array request_b_id{kRequestB};
  second.FillBlockTable(request_b_id, 1, second_table);
  EXPECT_EQ(second_table[0], first_table[0]);
}

TEST(PagedCacheReservationTest, CommitPublishesNewOwnershipExactlyOnce) {
  BlockPool pool{kBlockSize, 2};
  std::vector<PagedCacheBlockTable> tables;
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, true},
  };
  PagedCacheReservation reservation{pool, tables, requests};

  reservation.Commit();

  ASSERT_EQ(tables.size(), 1u);
  EXPECT_EQ(tables[0].request_id, kRequestA);
  EXPECT_EQ(tables[0].committed_slots, 5u);
  ASSERT_EQ(tables[0].blocks.size(), 2u);
  EXPECT_EQ(tables[0].blocks[0]->Size(), kBlockSize);
  EXPECT_EQ(tables[0].blocks[1]->Size(), 1u);
  EXPECT_THROW(reservation.Commit(), std::logic_error);
  EXPECT_THROW(reservation.Release(), std::logic_error);
}

// A newly admitted chunked prefill takes the blocks for its whole prompt up front, so the rest of
// the pool cannot be handed to another request between chunks, but only the chunk is committed.
TEST(PagedCacheReservationTest, ChunkedPrefillHoldsWholePromptButCommitsOnlyTheChunk) {
  BlockPool pool{kBlockSize, 3};
  std::vector<PagedCacheBlockTable> tables;
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, /*target_slots=*/2, /*newly_admitted=*/true,
                                   /*reserved_slots=*/9},
  };

  PagedCacheReservation reservation{pool, tables, requests};
  EXPECT_EQ(reservation.ReservedBlockCount(), 3u);
  EXPECT_EQ(pool.AvailableBlocks(), 0u);

  reservation.Commit();

  ASSERT_EQ(tables.size(), 1u);
  EXPECT_EQ(tables[0].committed_slots, 2u);
  EXPECT_EQ(tables[0].blocks.size(), 3u);

  // The next chunk finds its capacity already owned and needs no new block.
  const std::array next_requests{
      PagedCacheReservationRequest{kRequestA, /*target_slots=*/6, /*newly_admitted=*/false,
                                   /*reserved_slots=*/9},
  };
  PagedCacheReservation next{pool, tables, next_requests};
  EXPECT_EQ(next.ReservedBlockCount(), 0u);

  next.Commit();
  EXPECT_EQ(tables[0].committed_slots, 6u);
  EXPECT_EQ(tables[0].blocks.size(), 3u);
}

TEST(PagedCacheReservationTest, ReleaseReturnsWindowRingBlocks) {
  BlockPool pool{kBlockSize, 1};
  BlockPool window_pool{kBlockSize, 2};
  std::vector<PagedCacheBlockTable> tables;
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 1, true},
  };

  PagedCacheReservation reservation{pool, tables, requests, &window_pool, 2};
  EXPECT_EQ(window_pool.AvailableBlocks(), 0u);

  reservation.Release();

  EXPECT_EQ(pool.AvailableBlocks(), 1u);
  EXPECT_EQ(window_pool.AvailableBlocks(), 2u);
  EXPECT_TRUE(tables.empty());
}

TEST(PagedCacheReservationTest, RollbackReturnsBothPoolsForMixedExistingAndNewRequests) {
  BlockPool pool{kBlockSize, 3};
  BlockPool window_pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4),
                           window_pool.AllocateBlocks(2 * kBlockSize)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
      PagedCacheReservationRequest{kRequestB, 1, true},
  };

  PagedCacheReservation reservation{pool, tables, requests, &window_pool, 2};
  EXPECT_EQ(pool.AvailableBlocks(), 0u);
  EXPECT_EQ(window_pool.AvailableBlocks(), 0u);

  reservation.Release();

  ASSERT_EQ(tables.size(), 1u);
  EXPECT_EQ(tables[0].request_id, kRequestA);
  EXPECT_EQ(tables[0].committed_slots, 4u);
  EXPECT_EQ(tables[0].blocks.size(), 1u);
  EXPECT_EQ(tables[0].window_blocks.size(), 2u);
  EXPECT_EQ(pool.AvailableBlocks(), 2u);
  EXPECT_EQ(window_pool.AvailableBlocks(), 2u);
}

TEST(PagedCacheReservationTest, CommittedWindowBlocksCanBeRemovedAndReused) {
  BlockPool pool{kBlockSize, 1};
  BlockPool window_pool{kBlockSize, 2};
  std::vector<PagedCacheBlockTable> tables;
  const std::array request_a{
      PagedCacheReservationRequest{kRequestA, 1, true},
  };
  PagedCacheReservation first{pool, tables, request_a, &window_pool, 2};
  first.Commit();
  ASSERT_EQ(tables.size(), 1u);
  const auto first_window_blocks = tables[0].window_blocks;

  RemovePagedCacheBlockTable(pool, &window_pool, tables, kRequestA);

  const std::array request_b{
      PagedCacheReservationRequest{kRequestB, 1, true},
  };
  PagedCacheReservation second{pool, tables, request_b, &window_pool, 2};
  second.Commit();
  ASSERT_EQ(tables.size(), 1u);
  EXPECT_EQ(tables[0].window_blocks[0]->Id(), first_window_blocks[0]->Id());
  EXPECT_EQ(tables[0].window_blocks[1]->Id(), first_window_blocks[1]->Id());
}

TEST(PagedCacheReservationTest, WindowTableUsesReservedRingBeforeCommit) {
  BlockPool pool{kBlockSize, 1};
  BlockPool window_pool{kBlockSize, 2};
  std::vector<PagedCacheBlockTable> tables;
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 1, true},
  };
  PagedCacheReservation reservation{pool, tables, requests, &window_pool, 2};
  const std::array request_ids{kRequestA};
  std::array<int32_t, 5> window_table;

  reservation.FillWindowBlockTable(request_ids, 5, window_table);

  EXPECT_EQ(window_table, (std::array<int32_t, 5>{0, 1, 0, 1, 0}));
  reservation.Commit();
  ASSERT_EQ(tables.size(), 1u);
  ASSERT_EQ(tables[0].window_blocks.size(), 2u);
  EXPECT_EQ(tables[0].window_blocks[0]->Id(), 0u);
  EXPECT_EQ(tables[0].window_blocks[1]->Id(), 1u);
}

// ValidateCommit must be a pure precondition check: it may not touch the committed cache, the
// block pool, the reserved blocks, or the reservation's own state, so a composite transaction can
// validate every reservation up front before any of them publish.
TEST(PagedCacheReservationTest, ValidateCommitDoesNotMutateAndIsRepeatable) {
  BlockPool pool{kBlockSize, 2};
  std::vector<PagedCacheBlockTable> tables;
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, true},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  const size_t available_before = pool.AvailableBlocks();

  reservation.ValidateCommit();
  reservation.ValidateCommit();

  // Nothing was published or consumed and the reservation is still committable.
  EXPECT_TRUE(tables.empty());
  EXPECT_EQ(pool.AvailableBlocks(), available_before);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
  EXPECT_EQ(reservation.ReservedBlockCount(), 2u);

  // The still-Reserved reservation commits normally afterwards.
  reservation.CommitValidated();
  ASSERT_EQ(tables.size(), 1u);
  EXPECT_EQ(tables[0].committed_slots, 5u);
  EXPECT_EQ(tables[0].blocks.size(), 2u);
}

// CommitValidated publishes a reservation that ValidateCommit already accepted, producing exactly
// the same committed state as the single-call Commit wrapper.
TEST(PagedCacheReservationTest, CommitValidatedPublishesPreviouslyValidatedReservation) {
  BlockPool pool{kBlockSize, 2};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};

  reservation.ValidateCommit();
  reservation.CommitValidated();

  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Committed);
  ASSERT_EQ(tables.size(), 1u);
  EXPECT_EQ(tables[0].committed_slots, 5u);
  ASSERT_EQ(tables[0].blocks.size(), 2u);
  // Committing again is rejected exactly as the legacy single-call path is.
  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_THROW(reservation.CommitValidated(), std::logic_error);
  EXPECT_THROW(reservation.Commit(), std::logic_error);
}

// If ownership changes between reserve and commit -- here a committed table for the request
// appears after a newly-admitted reservation was taken -- ValidateCommit rejects it before any
// block is published.
TEST(PagedCacheReservationTest, ValidateCommitRejectsOwnershipChangeBeforePublication) {
  BlockPool pool{kBlockSize, 3};
  std::vector<PagedCacheBlockTable> tables;
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 1, true},
  };
  PagedCacheReservation reservation{pool, tables, requests};

  // Simulate another transaction admitting the same request id first.
  tables.push_back(PagedCacheBlockTable{kRequestA, 1, pool.AllocateBlocks(1)});
  const size_t blocks_before = tables[0].blocks.size();

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_THROW(reservation.Commit(), std::logic_error);

  // The pre-existing table was not extended and the reservation never published.
  EXPECT_EQ(tables[0].blocks.size(), blocks_before);
  EXPECT_EQ(tables[0].committed_slots, 1u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

// If the committed token boundary moves out from under an existing-request reservation,
// ValidateCommit rejects it before publication rather than advancing to a stale target.
TEST(PagedCacheReservationTest, ValidateCommitRejectsTokenBoundaryChangeBeforePublication) {
  BlockPool pool{kBlockSize, 2};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 3, pool.AllocateBlocks(3)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 4, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};

  // Another transaction advanced this request's committed boundary after the reservation planned
  // against committed_slots == 3.
  tables[0].committed_slots = 2;
  const size_t blocks_before = tables[0].blocks.size();

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);

  // No slots were advanced and no reserved block was appended.
  EXPECT_EQ(tables[0].committed_slots, 2u);
  EXPECT_EQ(tables[0].blocks.size(), blocks_before);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

// If the committed blocks backing an existing request shrink so the reservation can no longer
// reach its target token boundary, ValidateCommit rejects it before publication.
TEST(PagedCacheReservationTest, ValidateCommitRejectsUnreachableTargetBeforePublication) {
  BlockPool pool{kBlockSize, 3};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};

  // Simulate the committed capacity vanishing between reserve and commit.
  tables[0].blocks.clear();

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_EQ(tables[0].committed_slots, 4u);
  EXPECT_TRUE(tables[0].blocks.empty());
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

// The blocks.capacity() guard is the sole guarantee that CommitValidated's insert into an existing
// table cannot reallocate. If the committed table grows into its preallocated headroom between
// reserve and commit -- the target stays reachable, but there is no longer spare capacity for the
// appended reserved block -- ValidateCommit must reject it before any block is published.
TEST(PagedCacheReservationTest, ValidateCommitRejectsExhaustedPreallocatedCapacityBeforePublication) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  ASSERT_EQ(reservation.ReservedBlockCount(), 1u);

  // The constructor preallocated headroom for exactly the committed block plus the reserved block;
  // the guard under test only fires once that headroom is consumed, so pin the precondition here.
  ASSERT_EQ(tables[0].blocks.capacity(), 2u);

  // Another actor appended a committed block, consuming the reservation's preallocated headroom so
  // that blocks.capacity() (2) is now less than the post-commit block count (2 committed + 1
  // reserved). The target boundary of 5 slots is still reachable, so only the capacity guard trips.
  tables[0].blocks.push_back(pool.AllocateBlocks(kBlockSize)[0]);
  ASSERT_EQ(tables[0].blocks.size(), 2u);

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_THROW(reservation.Commit(), std::logic_error);

  // Nothing was published: the reserved block was not appended and no slots advanced.
  EXPECT_EQ(tables[0].blocks.size(), 2u);
  EXPECT_EQ(tables[0].committed_slots, 4u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

}  // namespace
}  // namespace Generators
