// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Pure unit tests for the Engine invariant validator (Generators::ValidateInvariants and
// friends). The validator operates only on immutable snapshot value types, so these tests need no
// model, cache, scheduler, or GPU: they construct snapshots directly and assert which invariant
// violations are (and are not) reported.

#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engine/engine_invariants.h"

namespace Generators {
namespace {

// Distinct fake request identities. The validator only compares/prints these, never dereferences,
// so any two distinct, stable addresses work. Using addresses of static storage avoids
// implementation-defined integer-to-pointer casts (which can trip UB tooling).
const char kRequestStorageA{};
const char kRequestStorageB{};
const void* const kRequestA = &kRequestStorageA;
const void* const kRequestB = &kRequestStorageB;
const char kRequestStorageC{};
const void* const kRequestC = &kRequestStorageC;

constexpr size_t kBlockSize = 4;

// A consistent two-request cache: A owns blocks {0,1} (full + one used slot), B owns block {2}
// (fully used); one block free. Helpers below mutate copies to violate a single invariant at a time.
PagedCacheSnapshot MakeValidCache() {
  PagedCacheSnapshot cache;
  cache.block_size = kBlockSize;
  cache.total_blocks = 4;
  cache.free_blocks = 1;
  cache.block_table_columns = 2;
  cache.requests = {
      RequestBlockSnapshot{kRequestA, {0, 1}, /*used=*/kBlockSize + 1, /*empty=*/kBlockSize - 1},
      RequestBlockSnapshot{kRequestB, {2}, /*used=*/kBlockSize, /*empty=*/0},
  };
  return cache;
}

RequestStateSnapshot MakeValidRequest(const void* id, RequestStatus status, int64_t current,
                                      int64_t processed) {
  RequestStateSnapshot request;
  request.request_id = id;
  request.status = status;
  request.current_sequence_length = current;
  request.processed_sequence_length = processed;
  request.is_prefill = false;
  request.has_current_turn = status != RequestStatus::Unassigned;
  request.current_turn_id =
      status == RequestStatus::Unassigned ? 0 : 1;
  request.finish_reason =
      status == RequestStatus::TurnComplete
          ? GenerationFinishReason::ContextLimit
          : GenerationFinishReason::None;
  return request;
}

// A consistent fixed-state pool that matches MakeValidCache(): A and B each own one committed slot
// whose committed_tokens equals that request's paged used_slots (A: kBlockSize+1, B: kBlockSize);
// one slot is free. Helpers below mutate copies to violate a single invariant at a time.
FixedStatePoolSnapshot MakeValidFixedState() {
  FixedStatePoolSnapshot fixed;
  fixed.capacity = 3;
  fixed.free_slots = 1;
  fixed.reserved_slots = 0;
  fixed.committed_slots = 2;
  fixed.healthy = true;
  fixed.slots = {
      FixedStateSlotSnapshot{kRequestA, 0, /*generation=*/1, /*state_generation=*/2,
                             /*committed_tokens=*/kBlockSize + 1,
                             FixedStateSlotOwnership::Committed},
      FixedStateSlotSnapshot{kRequestB, 1, 1, 1, kBlockSize, FixedStateSlotOwnership::Committed},
      FixedStateSlotSnapshot{nullptr, 2, 0, 0, 0, FixedStateSlotOwnership::Free},
  };
  return fixed;
}

std::vector<RequestStateSnapshot> MakeValidCompositeRequests() {
  return {
      MakeValidRequest(kRequestA, RequestStatus::Active, 6, kBlockSize + 1),
      MakeValidRequest(kRequestB, RequestStatus::Active, 5, kBlockSize),
  };
}

// ---------------------------------------------------------------------------------------------
// Cache invariants
// ---------------------------------------------------------------------------------------------

TEST(InvariantValidatorTest, ValidCacheHasNoViolations) {
  EXPECT_TRUE(ValidateCacheInvariants(MakeValidCache()).empty());
}

TEST(InvariantValidatorTest, ValidWindowBlockPoolHasNoViolations) {
  auto cache = MakeValidCache();
  cache.window_blocks.total_blocks = 4;
  cache.window_blocks.free_blocks = 0;
  cache.window_blocks.blocks_per_request = 2;
  cache.window_blocks.requests = {
      RequestBlockSnapshot{kRequestA, {0, 1}},
      RequestBlockSnapshot{kRequestB, {2, 3}},
  };

  EXPECT_TRUE(ValidateCacheInvariants(cache).empty());
}

TEST(InvariantValidatorTest, WindowBlockPoolValidatesRingSizeAndOwnershipIndependently) {
  auto cache = MakeValidCache();
  cache.window_blocks.total_blocks = 4;
  cache.window_blocks.free_blocks = 1;
  cache.window_blocks.blocks_per_request = 2;
  cache.window_blocks.requests = {
      RequestBlockSnapshot{kRequestA, {0}},
      RequestBlockSnapshot{kRequestB, {1, 1}},
  };

  const auto violations = ValidateCacheInvariants(cache);

  EXPECT_NE(std::find_if(violations.begin(), violations.end(),
                         [](const InvariantViolation& violation) {
                           return violation.message.find("window blocks instead of 2") !=
                                  std::string::npos;
                         }),
            violations.end());
  EXPECT_NE(std::find_if(violations.begin(), violations.end(),
                         [](const InvariantViolation& violation) {
                           return violation.message.find("Window block id 1 is listed more than once") !=
                                  std::string::npos;
                         }),
            violations.end());
}

TEST(InvariantValidatorTest, WindowBlockPoolIncludesTransactionReservationsInAccounting) {
  auto cache = MakeValidCache();
  cache.window_blocks.total_blocks = 2;
  cache.window_blocks.free_blocks = 0;
  cache.window_blocks.blocks_per_request = 2;
  cache.window_blocks.transaction_reserved_block_ids = {0, 1};

  EXPECT_TRUE(ValidateCacheInvariants(cache).empty());

  cache.window_blocks.transaction_reserved_block_ids.pop_back();
  EXPECT_FALSE(ValidateCacheInvariants(cache).empty());
}

TEST(InvariantValidatorTest, AllocatedBlocksCountsOwnedBlocks) {
  const auto cache = MakeValidCache();
  EXPECT_EQ(cache.AllocatedBlocks(), 3u);
  EXPECT_EQ(cache.free_blocks + cache.AllocatedBlocks(), cache.total_blocks);
  EXPECT_EQ(cache.TransactionReservedBlocks(), 0u);
}

TEST(InvariantValidatorTest, ValidTransactionReservationBalancesAccounting) {
  auto cache = MakeValidCache();
  cache.free_blocks = 0;
  cache.transaction_reserved_block_ids = {3};
  cache.reservations = {
      RequestReservationSnapshot{
          kRequestA,
          /*committed_slots=*/kBlockSize + 1,
          /*target_slots=*/2 * kBlockSize + 1,
          /*reserved_block_ids=*/{3},
      },
  };

  EXPECT_TRUE(ValidateCacheInvariants(cache).empty());
  EXPECT_EQ(cache.free_blocks + cache.TransactionReservedBlocks() +
                cache.AllocatedBlocks(),
            cache.total_blocks);
}

TEST(InvariantValidatorTest, InitialAdmissionReservationValidatesWithoutCommittedRows) {
  PagedCacheSnapshot cache;
  cache.block_size = kBlockSize;
  cache.total_blocks = 1;
  cache.free_blocks = 0;
  cache.transaction_reserved_block_ids = {0};
  cache.reservations = {
      RequestReservationSnapshot{kRequestA, 0, 1, {0}},
  };

  EXPECT_TRUE(ValidateCacheInvariants(cache).empty());
}

TEST(InvariantValidatorTest, InitialAdmissionRequiresEveryReservedBlockInADelta) {
  PagedCacheSnapshot cache;
  cache.block_size = kBlockSize;
  cache.total_blocks = 1;
  cache.free_blocks = 0;
  cache.transaction_reserved_block_ids = {0};

  const auto violations = ValidateCacheInvariants(cache);

  ASSERT_EQ(violations.size(), 1u);
  EXPECT_NE(violations[0].message.find(
                "Not every transaction-reserved block belongs to exactly one Request delta"),
            std::string::npos);
}

TEST(InvariantValidatorTest, InitialAdmissionRejectsUnreservedDeltaBlock) {
  PagedCacheSnapshot cache;
  cache.block_size = kBlockSize;
  cache.total_blocks = 2;
  cache.free_blocks = 1;
  cache.transaction_reserved_block_ids = {0};
  cache.reservations = {
      RequestReservationSnapshot{kRequestA, 0, 1, {1}},
  };

  const auto violations = ValidateCacheInvariants(cache);

  EXPECT_NE(std::find_if(violations.begin(), violations.end(),
                         [](const InvariantViolation& violation) {
                           return violation.message.find("that is not transaction-reserved") !=
                                  std::string::npos;
                         }),
            violations.end());
  EXPECT_NE(std::find_if(violations.begin(), violations.end(),
                         [](const InvariantViolation& violation) {
                           return violation.message.find(
                                      "Not every transaction-reserved block belongs to exactly one Request delta") !=
                                  std::string::npos;
                         }),
            violations.end());
}

TEST(InvariantValidatorTest, ReservedBlockCannotAlsoBeCommitted) {
  auto cache = MakeValidCache();
  cache.free_blocks = 0;
  cache.transaction_reserved_block_ids = {2};
  cache.reservations = {
      RequestReservationSnapshot{kRequestB, 4, 5, {2}},
  };

  EXPECT_FALSE(ValidateCacheInvariants(cache).empty());
}

TEST(InvariantValidatorTest, EveryReservedBlockNeedsOneRequestDelta) {
  auto cache = MakeValidCache();
  cache.free_blocks = 0;
  cache.transaction_reserved_block_ids = {3};

  const auto violations = ValidateCacheInvariants(cache);
  ASSERT_EQ(violations.size(), 1u);
  EXPECT_NE(violations[0].message.find(
                "Not every transaction-reserved block belongs to exactly one Request delta"),
            std::string::npos);
}

TEST(InvariantValidatorTest, BlockAccountingMismatchReported) {
  auto cache = MakeValidCache();
  cache.free_blocks = 2;  // free(2) + allocated(3) != total(4)
  EXPECT_FALSE(ValidateCacheInvariants(cache).empty());
}

TEST(InvariantValidatorTest, FreeExceedingTotalReported) {
  auto cache = MakeValidCache();
  cache.total_blocks = 2;  // free(1)+allocated(3) mismatch and allocated ids now out of range too
  EXPECT_FALSE(ValidateCacheInvariants(cache).empty());
}

TEST(InvariantValidatorTest, OutOfRangeBlockIdReported) {
  auto cache = MakeValidCache();
  cache.requests[1].block_ids = {9};  // id 9 >= total_blocks (4)
  // Keep accounting otherwise sound so the out-of-range id is the reported problem.
  cache.free_blocks = 1;
  const auto violations = ValidateCacheInvariants(cache);
  ASSERT_FALSE(violations.empty());
}

TEST(InvariantValidatorTest, DuplicateBlockWithinRequestReported) {
  auto cache = MakeValidCache();
  cache.requests[0].block_ids = {0, 0};  // same block listed twice within request A
  EXPECT_FALSE(ValidateCacheInvariants(cache).empty());
}

TEST(InvariantValidatorTest, BlockOwnedByTwoRequestsReported) {
  auto cache = MakeValidCache();
  cache.requests[1].block_ids = {1};  // block 1 already owned by request A
  cache.free_blocks = 2;              // keep total accounting consistent (A:2 + B:1 - shared)
  const auto violations = ValidateCacheInvariants(cache);
  bool found_shared = false;
  for (const auto& v : violations) {
    if (v.message.find("more than one Request") != std::string::npos) found_shared = true;
  }
  EXPECT_TRUE(found_shared);
}

TEST(InvariantValidatorTest, DuplicateRequestBlockTableReported) {
  auto cache = MakeValidCache();
  // Request A appears twice in the block-table listing (a malformed/duplicated row).
  cache.requests.push_back(cache.requests[0]);
  const auto violations = ValidateCacheInvariants(cache);
  bool found_duplicate_row = false;
  for (const auto& v : violations) {
    if (v.message.find("more than one block table") != std::string::npos) found_duplicate_row = true;
  }
  EXPECT_TRUE(found_duplicate_row);
}

TEST(InvariantValidatorTest, SlotCapacityMismatchReported) {
  auto cache = MakeValidCache();
  cache.requests[1].used_slots = kBlockSize + 1;  // one block can hold at most block_size slots
  cache.requests[1].empty_slots = 0;
  EXPECT_FALSE(ValidateCacheInvariants(cache).empty());
}

TEST(InvariantValidatorTest, NarrowBlockTableColumnsReported) {
  auto cache = MakeValidCache();
  cache.block_table_columns = 1;  // request A owns 2 blocks, so 1 column cannot hold it
  EXPECT_FALSE(ValidateCacheInvariants(cache).empty());
}

TEST(InvariantValidatorTest, ZeroBlockTableColumnsIsAllowed) {
  auto cache = MakeValidCache();
  cache.block_table_columns = 0;  // 0 means "cache does not use a block table"
  EXPECT_TRUE(ValidateCacheInvariants(cache).empty());
}

// ---------------------------------------------------------------------------------------------
// Fixed and composite state invariants
// ---------------------------------------------------------------------------------------------

TEST(InvariantValidatorTest, ValidFixedStateHasNoViolations) {
  EXPECT_TRUE(ValidateFixedStateInvariants(MakeValidFixedState()).empty());
}

TEST(InvariantValidatorTest, FixedStateRejectsDuplicateOwnership) {
  auto fixed = MakeValidFixedState();
  fixed.slots[1].request_id = kRequestA;  // A now owns two committed slots
  EXPECT_FALSE(ValidateFixedStateInvariants(fixed).empty());
}

TEST(InvariantValidatorTest, FixedStateRejectsUnhealthyPool) {
  auto fixed = MakeValidFixedState();
  fixed.healthy = false;
  EXPECT_FALSE(ValidateFixedStateInvariants(fixed).empty());
}

TEST(InvariantValidatorTest, FixedStateRejectsTotalAccountingMismatch) {
  auto fixed = MakeValidFixedState();
  fixed.committed_slots = 3;  // free + reserved + committed no longer equals capacity
  EXPECT_FALSE(ValidateFixedStateInvariants(fixed).empty());
}

TEST(InvariantValidatorTest, FixedStateRejectsResidualFreeSlot) {
  auto fixed = MakeValidFixedState();
  fixed.slots[2].committed_tokens = 1;  // a free slot must retain no committed progress
  EXPECT_FALSE(ValidateFixedStateInvariants(fixed).empty());
}

TEST(InvariantValidatorTest, ValidCompositeOwnershipHasNoViolations) {
  EXPECT_TRUE(ValidateCompositeStateInvariants(
                  MakeValidCache(), MakeValidFixedState(), MakeValidCompositeRequests())
                  .empty());
}

TEST(InvariantValidatorTest, CompositeRejectsPagedOwnerWithoutFixedSlot) {
  auto fixed = MakeValidFixedState();
  // Drop B's fixed slot: the paged cache still owns a table for B, so the two disagree.
  fixed.slots[1] = FixedStateSlotSnapshot{nullptr, 1, 0, 0, 0, FixedStateSlotOwnership::Free};
  fixed.committed_slots = 1;
  fixed.free_slots = 2;
  EXPECT_FALSE(ValidateCompositeStateInvariants(
                   MakeValidCache(), fixed, MakeValidCompositeRequests())
                   .empty());
}

TEST(InvariantValidatorTest, CompositeRejectsFixedOwnerWithoutPagedTable) {
  auto fixed = MakeValidFixedState();
  // Commit the third fixed slot to a request the paged cache does not own.
  fixed.slots[2] = FixedStateSlotSnapshot{kRequestC, 2, 1, 1, kBlockSize,
                                          FixedStateSlotOwnership::Committed};
  fixed.committed_slots = 3;
  fixed.free_slots = 0;
  auto requests = MakeValidCompositeRequests();
  requests.push_back(MakeValidRequest(kRequestC, RequestStatus::Active, 5, kBlockSize));
  EXPECT_FALSE(
      ValidateCompositeStateInvariants(MakeValidCache(), fixed, requests).empty());
}

TEST(InvariantValidatorTest, CompositeRejectsMismatchedTokenBoundary) {
  auto fixed = MakeValidFixedState();
  fixed.slots[0].committed_tokens = kBlockSize + 2;  // A's fixed boundary no longer matches paged used
  EXPECT_FALSE(ValidateCompositeStateInvariants(
                   MakeValidCache(), fixed, MakeValidCompositeRequests())
                   .empty());
}

TEST(InvariantValidatorTest, CompositeViolationsFollowSnapshotOrder) {
  auto fixed = MakeValidFixedState();
  fixed.slots[0] = FixedStateSlotSnapshot{
      nullptr, 0, 0, 0, 0, FixedStateSlotOwnership::Free};
  ++fixed.free_slots;
  --fixed.committed_slots;
  ++fixed.slots[1].committed_tokens;

  const auto violations = ValidateCompositeStateInvariants(
      MakeValidCache(), fixed, MakeValidCompositeRequests());

  ASSERT_EQ(violations.size(), 2u);
  EXPECT_NE(violations[0].message.find("has no committed fixed state slot"),
            std::string::npos);
  EXPECT_NE(violations[1].message.find(
                "different paged and fixed committed token boundaries"),
            std::string::npos);
}

TEST(InvariantValidatorTest, CompositeRejectsRequestStateBoundaryMismatch) {
  auto requests = MakeValidCompositeRequests();
  --requests[0].processed_sequence_length;

  const auto violations = ValidateCompositeStateInvariants(
      MakeValidCache(), MakeValidFixedState(), requests);

  EXPECT_NE(std::find_if(
                violations.begin(), violations.end(),
                [](const InvariantViolation& violation) {
                  return violation.message.find(
                             "request and decoder-state committed token boundaries") !=
                         std::string::npos;
                }),
            violations.end());
}

TEST(InvariantValidatorTest, ThrowIfCompositeInvariantsViolatedThrowsOnMismatch) {
  auto fixed = MakeValidFixedState();
  fixed.slots[0].committed_tokens = kBlockSize + 2;
  EXPECT_THROW(ThrowIfCompositeStateInvariantsViolated(
                   MakeValidCache(), fixed, MakeValidCompositeRequests()),
               std::runtime_error);
  EXPECT_NO_THROW(ThrowIfCompositeStateInvariantsViolated(
      MakeValidCache(), MakeValidFixedState(), MakeValidCompositeRequests()));
}

// ---------------------------------------------------------------------------------------------
// Request invariants
// ---------------------------------------------------------------------------------------------

TEST(InvariantValidatorTest, ValidRequestHasNoViolations) {
  EXPECT_TRUE(ValidateRequestInvariants(
                  MakeValidRequest(kRequestA, RequestStatus::Active, 10, 4))
                  .empty());
}

TEST(InvariantValidatorTest, ProcessedBeyondCurrentReported) {
  auto request = MakeValidRequest(kRequestA, RequestStatus::Active, 10, 12);
  EXPECT_FALSE(ValidateRequestInvariants(request).empty());
}

TEST(InvariantValidatorTest, ZeroTurnIdIsRejectedWhenAssigned) {
  auto request = MakeValidRequest(kRequestA, RequestStatus::Active, 10, 4);
  request.current_turn_id = 0;
  request.has_current_turn = true;
  EXPECT_FALSE(ValidateRequestInvariants(request).empty());
}

TEST(InvariantValidatorTest, TurnCompleteRequestWithFinalUnprocessedTokenIsValid) {
  // At completion the just-generated final token is appended but never fed back to the model, so a
  // TurnComplete Request may legitimately report unprocessed tokens. This must not fire.
  auto request = MakeValidRequest(kRequestA, RequestStatus::TurnComplete, 10, 9);
  EXPECT_TRUE(ValidateRequestInvariants(request).empty());
}

TEST(InvariantValidatorTest, TurnCompleteRequestFullyProcessedIsValid) {
  EXPECT_TRUE(ValidateRequestInvariants(
                  MakeValidRequest(kRequestA, RequestStatus::TurnComplete, 10, 10))
                  .empty());
}

TEST(InvariantValidatorTest, ExecutableRequestRequiresTurnMetadata) {
  auto request =
      MakeValidRequest(kRequestA, RequestStatus::Active, 10, 10);
  request.has_current_turn = false;
  request.finish_reason = GenerationFinishReason::Canceled;
  EXPECT_FALSE(ValidateRequestInvariants(request).empty());
}

TEST(InvariantValidatorTest, TurnCompleteRequestRequiresFinishReason) {
  auto request =
      MakeValidRequest(kRequestA, RequestStatus::TurnComplete, 10, 10);
  request.finish_reason = GenerationFinishReason::None;
  EXPECT_FALSE(ValidateRequestInvariants(request).empty());
}

TEST(InvariantValidatorTest, CanceledTurnMayRetainUnprocessedInput) {
  auto request =
      MakeValidRequest(kRequestA, RequestStatus::TurnComplete, 10, 4);
  request.is_prefill = true;
  request.finish_reason = GenerationFinishReason::Canceled;
  EXPECT_TRUE(ValidateRequestInvariants(request).empty());
}

// ---------------------------------------------------------------------------------------------
// Combined / cross-cutting invariants
// ---------------------------------------------------------------------------------------------

TEST(InvariantValidatorTest, ConsistentSnapshotsValidateClean) {
  const auto cache = MakeValidCache();
  const std::vector<RequestStateSnapshot> requests{
      MakeValidRequest(kRequestA, RequestStatus::Active, 9, 9),
      MakeValidRequest(kRequestB, RequestStatus::Active, 4, 4),
  };
  EXPECT_TRUE(ValidateInvariants(cache, requests).empty());
  EXPECT_NO_THROW(ThrowIfInvariantsViolated(cache, requests));
}

TEST(InvariantValidatorTest, BlockTableForUnknownRequestReported) {
  const auto cache = MakeValidCache();  // owns tables for A and B
  const std::vector<RequestStateSnapshot> requests{
      MakeValidRequest(kRequestA, RequestStatus::Active, 9, 9),
      // B is missing from the request set, yet the cache holds a block table for it.
  };
  EXPECT_FALSE(ValidateInvariants(cache, requests).empty());
}

TEST(InvariantValidatorTest, ThrowWrapperListsViolations) {
  auto cache = MakeValidCache();
  cache.free_blocks = 0;  // break block accounting
  const std::vector<RequestStateSnapshot> requests{
      MakeValidRequest(kRequestA, RequestStatus::Active, 9, 9),
      MakeValidRequest(kRequestB, RequestStatus::Active, 4, 4),
  };
  EXPECT_THROW(ThrowIfInvariantsViolated(cache, requests), std::runtime_error);
}

}  // namespace
}  // namespace Generators
