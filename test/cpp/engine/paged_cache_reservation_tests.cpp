// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <limits>
#include <type_traits>
#include <type_traits>
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

void ReplaceTable(
    PagedCacheBlockTable& table,
    size_t committed_slots,
    std::vector<std::shared_ptr<Block>> blocks,
    std::vector<std::shared_ptr<Block>> window_blocks = {}) {
  table = PagedCacheBlockTable{
      table.RequestId(), committed_slots, std::move(blocks),
      std::move(window_blocks)};
}

TEST(PagedCacheReservationTest, TableReplacementAdvancesGeneration) {
  PagedCacheBlockTable table{kRequestA, 0};

  EXPECT_EQ(table.MutationGeneration(), 0u);
  ReplaceTable(table, 1, {});
  EXPECT_EQ(table.MutationGeneration(), 1u);
  PagedCacheBlockTable replacement{kRequestA, 2};
  table = replacement;
  EXPECT_EQ(table.MutationGeneration(), 2u);
}

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

TEST(PagedCacheReservationTest, ReservationRejectsBlockCapacityOverflow) {
  constexpr size_t block_size =
      std::numeric_limits<size_t>::max() / 2 + 1;
  BlockPool pool{block_size, 2};
  std::vector<PagedCacheBlockTable> tables;
  const std::array requests{
      PagedCacheReservationRequest{
          kRequestA, std::numeric_limits<size_t>::max(), true},
  };

  EXPECT_THROW(
      (PagedCacheReservation{pool, tables, requests}),
      std::overflow_error);
  EXPECT_TRUE(tables.empty());
  EXPECT_EQ(pool.AvailableBlocks(), 2u);
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
  EXPECT_EQ(reservation.ReservedBlockCount(), 0u);

  reservation.Commit();

  EXPECT_EQ(tables[0].CommittedSlots(), 4u);
  EXPECT_EQ(tables[0].Blocks().size(), 1u);
  EXPECT_TRUE(tables[0].Blocks()[0]->IsFull());
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

TEST(PagedCacheReservationTest, DestructorDoesNotReleaseReissuedBlocks) {
  static_assert(std::is_nothrow_destructible_v<PagedCacheReservation>);
  BlockPool pool{kBlockSize, 1};
  std::vector<PagedCacheBlockTable> tables;
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 1, true},
  };
  std::shared_ptr<Block> current;
  {
    PagedCacheReservation reservation{pool, tables, requests};
    const std::vector<std::shared_ptr<Block>> stale{
        reservation.ReservedBlocks().begin(),
        reservation.ReservedBlocks().end()};
    pool.Free(stale);
    current = pool.ReserveBlocks(1)[0];
  }

  EXPECT_TRUE(pool.Owns(current));
  EXPECT_EQ(pool.AvailableBlocks(), 0u);
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
  EXPECT_EQ(tables[0].RequestId(), kRequestA);
  EXPECT_EQ(tables[0].CommittedSlots(), 5u);
  ASSERT_EQ(tables[0].Blocks().size(), 2u);
  EXPECT_EQ(tables[0].Blocks()[0]->Size(), kBlockSize);
  EXPECT_EQ(tables[0].Blocks()[1]->Size(), 1u);
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
  EXPECT_EQ(tables[0].CommittedSlots(), 2u);
  EXPECT_EQ(tables[0].Blocks().size(), 3u);

  // The next chunk finds its capacity already owned and needs no new block.
  const std::array next_requests{
      PagedCacheReservationRequest{kRequestA, /*target_slots=*/6, /*newly_admitted=*/false,
                                   /*reserved_slots=*/9},
  };
  PagedCacheReservation next{pool, tables, next_requests};
  EXPECT_EQ(next.ReservedBlockCount(), 0u);

  next.Commit();
  EXPECT_EQ(tables[0].CommittedSlots(), 6u);
  EXPECT_EQ(tables[0].Blocks().size(), 3u);
}

TEST(PagedCacheReservationTest, FutureReservedCapacityRemainsImmutableAfterValidation) {
  BlockPool pool{kBlockSize, 2};
  std::vector<PagedCacheBlockTable> tables;
  const std::array requests{
      PagedCacheReservationRequest{
          kRequestA, /*target_slots=*/1, /*newly_admitted=*/true,
          /*reserved_slots=*/2 * kBlockSize},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  ASSERT_EQ(reservation.ReservedBlocks().size(), 2u);
  static_assert(!std::is_assignable_v<Block&, const Block&>);
  const size_t future_block_id = reservation.ReservedBlocks()[1]->Id();

  reservation.ValidateCommit();
  reservation.CommitValidated();

  ASSERT_EQ(tables.size(), 1u);
  ASSERT_EQ(tables[0].Blocks().size(), 2u);
  EXPECT_EQ(tables[0].Blocks()[1]->Id(), future_block_id);
  EXPECT_EQ(tables[0].Blocks()[1]->Size(), 0u);
  EXPECT_TRUE(pool.Owns(tables[0].Blocks()[1]));
}

TEST(PagedCacheReservationTest, AliasReservationInvalidatesOriginalPublication) {
  BlockPool pool{kBlockSize, 2};
  std::vector<PagedCacheBlockTable> tables;
  const std::array requests{
      PagedCacheReservationRequest{
          kRequestA, /*target_slots=*/1, /*newly_admitted=*/true,
          /*reserved_slots=*/2 * kBlockSize},
  };
  PagedCacheReservation original{pool, tables, requests};
  original.ValidateCommit();

  std::vector<std::shared_ptr<Block>> alias_blocks{
      original.ReservedBlocks().begin(), original.ReservedBlocks().end()};
  std::vector<PagedCacheBlockTable> alias_tables{
      PagedCacheBlockTable{kRequestB, kBlockSize, std::move(alias_blocks)},
  };
  const std::array alias_requests{
      PagedCacheReservationRequest{
          kRequestB, /*target_slots=*/kBlockSize + 1,
          /*newly_admitted=*/false,
          /*reserved_slots=*/2 * kBlockSize},
  };
  PagedCacheReservation alias{pool, alias_tables, alias_requests};
  alias.Commit();

  EXPECT_THROW(original.CommitValidated(), std::logic_error);
  EXPECT_TRUE(tables.empty());
  EXPECT_EQ(original.State(), PagedCacheReservationState::Reserved);
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
  EXPECT_EQ(tables[0].RequestId(), kRequestA);
  EXPECT_EQ(tables[0].CommittedSlots(), 4u);
  EXPECT_EQ(tables[0].Blocks().size(), 1u);
  EXPECT_EQ(tables[0].WindowBlocks().size(), 2u);
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
  const auto first_window_blocks = tables[0].WindowBlocks();

  RemovePagedCacheBlockTable(pool, &window_pool, tables, kRequestA);

  const std::array request_b{
      PagedCacheReservationRequest{kRequestB, 1, true},
  };
  PagedCacheReservation second{pool, tables, request_b, &window_pool, 2};
  second.Commit();
  ASSERT_EQ(tables.size(), 1u);
  EXPECT_EQ(tables[0].WindowBlocks()[0]->Id(), first_window_blocks[0]->Id());
  EXPECT_EQ(tables[0].WindowBlocks()[1]->Id(), first_window_blocks[1]->Id());
}

TEST(PagedCacheReservationTest, RemovalPreflightValidatesBothPoolsBeforeMutation) {
  BlockPool pool{kBlockSize, 1};
  BlockPool window_pool{kBlockSize, 1};
  auto blocks = pool.AllocateBlocks(kBlockSize);
  const auto block = blocks[0];
  const auto foreign_window =
      std::make_shared<Block>(0, kBlockSize, kBlockSize);
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{
          kRequestA, kBlockSize, std::move(blocks), {foreign_window}},
  };

  EXPECT_THROW(
      ValidateRemovePagedCacheBlockTable(
          pool, &window_pool, tables, kRequestA),
      std::runtime_error);
  EXPECT_TRUE(pool.Owns(block));
  EXPECT_EQ(pool.AvailableBlocks(), 0u);
  EXPECT_EQ(window_pool.AvailableBlocks(), 1u);
  EXPECT_EQ(tables.size(), 1u);

  EXPECT_DEATH_IF_SUPPORTED(
      RemoveValidatedPagedCacheBlockTable(
          pool, &window_pool, tables, kRequestA),
      "");
  EXPECT_TRUE(pool.Owns(block));
  EXPECT_EQ(tables.size(), 1u);
}

TEST(PagedCacheReservationTest, ValidatedRemovalPublishesBothPoolsNoexcept) {
  BlockPool pool{kBlockSize, 1};
  BlockPool window_pool{kBlockSize, 1};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{
          kRequestA, kBlockSize, pool.AllocateBlocks(kBlockSize),
          window_pool.AllocateBlocks(kBlockSize)},
  };

  ValidateRemovePagedCacheBlockTable(
      pool, &window_pool, tables, kRequestA);
  RemoveValidatedPagedCacheBlockTable(
      pool, &window_pool, tables, kRequestA);

  EXPECT_TRUE(tables.empty());
  EXPECT_EQ(pool.AvailableBlocks(), 1u);
  EXPECT_EQ(window_pool.AvailableBlocks(), 1u);
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
  ASSERT_EQ(tables[0].WindowBlocks().size(), 2u);
  EXPECT_EQ(tables[0].WindowBlocks()[0]->Id(), 0u);
  EXPECT_EQ(tables[0].WindowBlocks()[1]->Id(), 1u);
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
  EXPECT_EQ(tables[0].CommittedSlots(), 5u);
  EXPECT_EQ(tables[0].Blocks().size(), 2u);
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
  EXPECT_EQ(tables[0].CommittedSlots(), 5u);
  ASSERT_EQ(tables[0].Blocks().size(), 2u);
  // Committing again is rejected exactly as the legacy single-call path is.
  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_THROW(reservation.CommitValidated(), std::logic_error);
  EXPECT_THROW(reservation.Commit(), std::logic_error);
}

TEST(PagedCacheReservationTest, CommitValidatedPreflightsAllRequestsBeforePublication) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
      PagedCacheBlockTable{kRequestB, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
      PagedCacheReservationRequest{kRequestB, 5, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  reservation.ValidateCommit();
  const size_t available_before = pool.AvailableBlocks();

  // Invalidate only the later request after validation. Publication must reject this before the
  // first request consumes or appends any reserved state.
  ReplaceTable(
      tables[1], 3, tables[1].Blocks(), tables[1].WindowBlocks());

  EXPECT_THROW(reservation.CommitValidated(), std::logic_error);
  EXPECT_EQ(tables[0].CommittedSlots(), 4u);
  EXPECT_EQ(tables[0].Blocks().size(), 1u);
  EXPECT_EQ(tables[0].Blocks()[0]->Size(), 4u);
  EXPECT_EQ(tables[1].CommittedSlots(), 3u);
  EXPECT_EQ(tables[1].Blocks().size(), 1u);
  EXPECT_EQ(pool.AvailableBlocks(), available_before);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, CommitValidatedRejectsTailAliasingBeforePublication) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 3, pool.AllocateBlocks(3)},
      PagedCacheBlockTable{kRequestB, 3, pool.AllocateBlocks(3)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 4, false},
      PagedCacheReservationRequest{kRequestB, 4, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  reservation.ValidateCommit();

  // Mapping mutation cannot alias a later request's touched tail after validation and allow the
  // first request to publish.
  auto aliased_blocks = tables[1].Blocks();
  aliased_blocks[0] = tables[0].Blocks()[0];
  ReplaceTable(tables[1], 3, std::move(aliased_blocks));

  EXPECT_THROW(reservation.CommitValidated(), std::logic_error);
  EXPECT_EQ(tables[0].CommittedSlots(), 3u);
  EXPECT_EQ(tables[0].Blocks()[0]->Size(), 3u);
  EXPECT_EQ(tables[1].CommittedSlots(), 3u);
  EXPECT_EQ(tables[1].Blocks()[0]->Size(), 3u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, CommitValidatedRejectsOmittedResidentMutationBeforePublication) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
      PagedCacheBlockTable{kRequestB, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  ASSERT_EQ(reservation.ReservedBlocks().size(), 1u);
  reservation.ValidateCommit();

  // B is omitted from this step, but it must not be allowed to claim A's reserved block without
  // invalidating the publication preflight after the earlier validation succeeded.
  ReplaceTable(
      tables[1], 4, {reservation.ReservedBlocks()[0]});

  EXPECT_THROW(reservation.CommitValidated(), std::logic_error);
  EXPECT_EQ(tables[0].CommittedSlots(), 4u);
  EXPECT_EQ(tables[0].Blocks().size(), 1u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, ValidateCommitRejectsOmittedResidentMutation) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
      PagedCacheBlockTable{kRequestB, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};

  ReplaceTable(
      tables[1], 4, {reservation.ReservedBlocks()[0]});

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_EQ(tables[0].CommittedSlots(), 4u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, OmittedResidentUnchangedAllowsMovedReservationCommit) {
  BlockPool pool{kBlockSize, 3};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
      PagedCacheBlockTable{kRequestB, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  PagedCacheReservation moved{std::move(reservation)};

  moved.Commit();

  EXPECT_EQ(tables[0].CommittedSlots(), 5u);
  EXPECT_EQ(tables[1].CommittedSlots(), 4u);
  EXPECT_EQ(moved.State(), PagedCacheReservationState::Committed);
}

TEST(PagedCacheReservationTest, RejectsSameScalarResidentTableReplacement) {
  BlockPool pool{kBlockSize, 3};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
      PagedCacheBlockTable{kRequestB, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  const size_t block_id = tables[1].Blocks()[0]->Id();

  // Preserve request ID, vector sizes, block ID, capacity, and occupancy. Only the mapping storage
  // and physical block identity change; move assignment also advances the destination generation.
  tables[1] = PagedCacheBlockTable{
      kRequestB, 4, {std::make_shared<Block>(block_id, kBlockSize, kBlockSize)}};

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_EQ(tables[0].CommittedSlots(), 4u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, RejectsCopyAssignedOmittedResidentMapping) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
      PagedCacheBlockTable{kRequestB, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  PagedCacheBlockTable replacement{
      kRequestB, 4, {reservation.ReservedBlocks()[0]}};

  // Copy assignment may reuse the destination vector's storage. Its explicit generation advance
  // must still invalidate the omitted resident snapshot.
  tables[1] = replacement;

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_EQ(tables[0].CommittedSlots(), 4u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, RejectsOmittedResidentBoundaryReplacement) {
  BlockPool pool{kBlockSize, 2};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 0, {}},
      PagedCacheBlockTable{kRequestB, 0, {}},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 1, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  reservation.ValidateCommit();

  tables[1] = PagedCacheBlockTable{kRequestB, 1, {}};

  EXPECT_THROW(reservation.CommitValidated(), std::logic_error);
  EXPECT_EQ(tables[0].CommittedSlots(), 0u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, ReservationRejectsStableMalformedLaterTail) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 3, pool.AllocateBlocks(3)},
      PagedCacheBlockTable{kRequestB, 3, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 4, false},
      PagedCacheReservationRequest{kRequestB, 4, false},
  };
  // Request B's unchanged generation cannot make its full tail consistent with committed_slots=3.
  EXPECT_THROW((PagedCacheReservation{pool, tables, requests}), std::logic_error);
  EXPECT_EQ(tables[0].CommittedSlots(), 3u);
  EXPECT_EQ(tables[0].Blocks()[0]->Size(), 3u);
  EXPECT_EQ(tables[1].CommittedSlots(), 3u);
  EXPECT_TRUE(tables[1].Blocks()[0]->IsFull());
  EXPECT_EQ(pool.AvailableBlocks(), 2u);
}

TEST(PagedCacheReservationTest, ReservationRejectsSharedTouchedBlockBeforePublication) {
  BlockPool pool{kBlockSize, 3};
  auto shared_tail = pool.AllocateBlocks(3);
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 3, shared_tail},
      PagedCacheBlockTable{kRequestB, 3, shared_tail},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 4, false},
      PagedCacheReservationRequest{kRequestB, 4, false},
  };

  EXPECT_THROW((PagedCacheReservation{pool, tables, requests}), std::logic_error);
  EXPECT_EQ(tables[0].CommittedSlots(), 3u);
  EXPECT_EQ(tables[1].CommittedSlots(), 3u);
  EXPECT_EQ(tables[0].Blocks()[0]->Size(), 3u);
  EXPECT_EQ(tables[1].Blocks()[0]->Size(), 3u);
  EXPECT_EQ(pool.AvailableBlocks(), 2u);
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
  const size_t blocks_before = tables[0].Blocks().size();

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_THROW(reservation.Commit(), std::logic_error);

  // The pre-existing table was not extended and the reservation never published.
  EXPECT_EQ(tables[0].Blocks().size(), blocks_before);
  EXPECT_EQ(tables[0].CommittedSlots(), 1u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

// If the committed token boundary changes under an existing-request reservation,
// ValidateCommit rejects it before publication rather than publishing against stale state.
TEST(PagedCacheReservationTest, ValidateCommitRejectsTokenBoundaryChangeBeforePublication) {
  BlockPool pool{kBlockSize, 2};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 3, pool.AllocateBlocks(3)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 4, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};

  // Another transaction moved this request's committed boundary backward after the reservation
  // planned against committed_slots == 3.
  ReplaceTable(
      tables[0], 2, tables[0].Blocks(), tables[0].WindowBlocks());
  const size_t blocks_before = tables[0].Blocks().size();

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);

  // Validation did not further change the boundary or append the reserved block.
  EXPECT_EQ(tables[0].CommittedSlots(), 2u);
  EXPECT_EQ(tables[0].Blocks().size(), blocks_before);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, ValidateCommitRejectsReorderedCommittedBlocks) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 8, pool.AllocateBlocks(8)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 9, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  ASSERT_EQ(tables[0].Blocks().size(), 2u);
  const auto first_id = tables[0].Blocks()[0]->Id();
  const auto second_id = tables[0].Blocks()[1]->Id();

  auto reordered_blocks = tables[0].Blocks();
  std::swap(reordered_blocks[0], reordered_blocks[1]);
  ReplaceTable(tables[0], 8, std::move(reordered_blocks));

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_EQ(tables[0].Blocks()[0]->Id(), second_id);
  EXPECT_EQ(tables[0].Blocks()[1]->Id(), first_id);
  EXPECT_EQ(tables[0].CommittedSlots(), 8u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, ValidateCommitRejectsDuplicateCommittedBlocks) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 8, pool.AllocateBlocks(8)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 9, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  ASSERT_EQ(tables[0].Blocks().size(), 2u);

  auto duplicate_blocks = tables[0].Blocks();
  duplicate_blocks[1] = duplicate_blocks[0];
  ReplaceTable(tables[0], 8, std::move(duplicate_blocks));

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_EQ(tables[0].Blocks()[0], tables[0].Blocks()[1]);
  EXPECT_EQ(tables[0].CommittedSlots(), 8u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, ValidateCommitRejectsForeignSameIdCommittedBlock) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 8, pool.AllocateBlocks(8)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 9, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  ASSERT_EQ(tables[0].Blocks().size(), 2u);
  const size_t second_id = tables[0].Blocks()[1]->Id();

  auto foreign_blocks = tables[0].Blocks();
  foreign_blocks[1] =
      std::make_shared<Block>(second_id, kBlockSize, kBlockSize);
  ReplaceTable(tables[0], 8, std::move(foreign_blocks));

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_EQ(tables[0].Blocks()[1]->Id(), second_id);
  EXPECT_EQ(tables[0].Blocks()[1]->Size(), kBlockSize);
  EXPECT_EQ(tables[0].CommittedSlots(), 8u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, ValidateCommitRejectsReallocatedSameIdCommittedBlock) {
  BlockPool pool{kBlockSize, 3};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  const size_t original_id = tables[0].Blocks()[0]->Id();

  pool.Free(tables[0].Blocks());
  auto replacement = pool.AllocateBlocks(4);
  ASSERT_EQ(replacement.size(), 1u);
  ASSERT_EQ(replacement[0]->Id(), original_id);
  ReplaceTable(tables[0], 4, std::move(replacement));

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_EQ(tables[0].Blocks()[0]->Id(), original_id);
  EXPECT_EQ(tables[0].CommittedSlots(), 4u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, ValidateCommitRejectsDuplicateCommittedRequestTable) {
  BlockPool pool{kBlockSize, 4};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
  };
  PagedCacheReservation reservation{pool, tables, requests};
  tables.push_back(
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4)});

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  ASSERT_EQ(tables.size(), 2u);
  EXPECT_EQ(tables[0].RequestId(), kRequestA);
  EXPECT_EQ(tables[1].RequestId(), kRequestA);
  EXPECT_EQ(tables[0].CommittedSlots(), 4u);
  EXPECT_EQ(tables[1].CommittedSlots(), 4u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, ValidateCommitRejectsReorderedResidentWindowRing) {
  BlockPool pool{kBlockSize, 2};
  BlockPool window_pool{kBlockSize, 3};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4),
                           window_pool.AllocateBlocks(2 * kBlockSize)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
  };
  PagedCacheReservation reservation{pool, tables, requests, &window_pool, 2};
  ASSERT_EQ(tables[0].WindowBlocks().size(), 2u);
  const auto first_id = tables[0].WindowBlocks()[0]->Id();
  const auto second_id = tables[0].WindowBlocks()[1]->Id();

  auto reordered_window_blocks = tables[0].WindowBlocks();
  std::swap(reordered_window_blocks[0], reordered_window_blocks[1]);
  ReplaceTable(
      tables[0], 4, tables[0].Blocks(),
      std::move(reordered_window_blocks));

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_EQ(tables[0].WindowBlocks()[0]->Id(), second_id);
  EXPECT_EQ(tables[0].WindowBlocks()[1]->Id(), first_id);
  EXPECT_EQ(tables[0].CommittedSlots(), 4u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, ValidateCommitRejectsReallocatedResidentWindowRing) {
  BlockPool pool{kBlockSize, 2};
  BlockPool window_pool{kBlockSize, 2};
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 4, pool.AllocateBlocks(4),
                           window_pool.AllocateBlocks(2 * kBlockSize)},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 5, false},
  };
  PagedCacheReservation reservation{pool, tables, requests, &window_pool, 2};
  const auto original_ids = std::array{
      tables[0].WindowBlocks()[0]->Id(),
      tables[0].WindowBlocks()[1]->Id(),
  };

  window_pool.Free(tables[0].WindowBlocks());
  auto replacements = window_pool.AllocateBlocks(2 * kBlockSize);
  ASSERT_EQ(replacements.size(), 2u);
  ASSERT_EQ(replacements[0]->Id(), original_ids[0]);
  ASSERT_EQ(replacements[1]->Id(), original_ids[1]);
  ReplaceTable(
      tables[0], 4, tables[0].Blocks(), std::move(replacements));

  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_EQ(tables[0].WindowBlocks()[0]->Id(), original_ids[0]);
  EXPECT_EQ(tables[0].WindowBlocks()[1]->Id(), original_ids[1]);
  EXPECT_EQ(tables[0].CommittedSlots(), 4u);
  EXPECT_EQ(reservation.State(), PagedCacheReservationState::Reserved);
}

TEST(PagedCacheReservationTest, ReservationRejectsSharedResidentWindowBlock) {
  BlockPool pool{kBlockSize, 2};
  BlockPool window_pool{kBlockSize, 3};
  auto shared_window = window_pool.AllocateBlocks(kBlockSize);
  auto other_window = window_pool.AllocateBlocks(kBlockSize);
  std::vector<PagedCacheBlockTable> tables{
      PagedCacheBlockTable{kRequestA, 3, pool.AllocateBlocks(3), {shared_window[0], other_window[0]}},
      PagedCacheBlockTable{kRequestB, 3, pool.AllocateBlocks(3), {shared_window[0], other_window[0]}},
  };
  const std::array requests{
      PagedCacheReservationRequest{kRequestA, 4, false},
      PagedCacheReservationRequest{kRequestB, 4, false},
  };

  EXPECT_THROW(
      (PagedCacheReservation{pool, tables, requests, &window_pool, 2}),
      std::logic_error);
  EXPECT_EQ(tables[0].Blocks()[0]->Size(), 3u);
  EXPECT_EQ(tables[1].Blocks()[0]->Size(), 3u);
  EXPECT_EQ(window_pool.AvailableBlocks(), 1u);
}

}  // namespace
}  // namespace Generators
