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

  pool.Free(tables[0].blocks);
  window_pool.Free(tables[0].window_blocks);
  tables.erase(tables.begin());

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

}  // namespace
}  // namespace Generators
