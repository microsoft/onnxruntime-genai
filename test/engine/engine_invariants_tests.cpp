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
                                      int64_t processed, int64_t seen) {
  RequestStateSnapshot request;
  request.request_id = id;
  request.status = status;
  request.current_sequence_length = current;
  request.processed_sequence_length = processed;
  request.seen_sequence_length = seen;
  request.is_prefill = false;
  return request;
}

// ---------------------------------------------------------------------------------------------
// Cache invariants
// ---------------------------------------------------------------------------------------------

TEST(InvariantValidatorTest, ValidCacheHasNoViolations) {
  EXPECT_TRUE(ValidateCacheInvariants(MakeValidCache()).empty());
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
          /*tail_slots_to_consume=*/kBlockSize - 1,
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
      RequestReservationSnapshot{kRequestA, 0, 1, 0, {0}},
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
      RequestReservationSnapshot{kRequestA, 0, 1, 0, {1}},
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
      RequestReservationSnapshot{kRequestB, 4, 5, 0, {2}},
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
// Request invariants
// ---------------------------------------------------------------------------------------------

TEST(InvariantValidatorTest, ValidRequestHasNoViolations) {
  EXPECT_TRUE(ValidateRequestInvariants(
                  MakeValidRequest(kRequestA, RequestStatus::InProgress, 10, 4, 6))
                  .empty());
}

TEST(InvariantValidatorTest, ProcessedBeyondCurrentReported) {
  auto request = MakeValidRequest(kRequestA, RequestStatus::InProgress, 10, 12, 6);
  EXPECT_FALSE(ValidateRequestInvariants(request).empty());
}

TEST(InvariantValidatorTest, SeenBeyondCurrentReported) {
  auto request = MakeValidRequest(kRequestA, RequestStatus::InProgress, 10, 4, 11);
  EXPECT_FALSE(ValidateRequestInvariants(request).empty());
}

TEST(InvariantValidatorTest, CompletedRequestWithFinalUnprocessedTokenIsValid) {
  // At completion the just-generated final token is appended but never fed back to the model, so a
  // Completed Request legitimately reports one (or more) unprocessed token(s). This must not fire.
  auto request = MakeValidRequest(kRequestA, RequestStatus::Completed, 10, 9, 10);
  EXPECT_TRUE(ValidateRequestInvariants(request).empty());
}

TEST(InvariantValidatorTest, CompletedRequestFullyProcessedIsValid) {
  EXPECT_TRUE(ValidateRequestInvariants(
                  MakeValidRequest(kRequestA, RequestStatus::Completed, 10, 10, 10))
                  .empty());
}

// ---------------------------------------------------------------------------------------------
// Combined / cross-cutting invariants
// ---------------------------------------------------------------------------------------------

TEST(InvariantValidatorTest, ConsistentSnapshotsValidateClean) {
  const auto cache = MakeValidCache();
  const std::vector<RequestStateSnapshot> requests{
      MakeValidRequest(kRequestA, RequestStatus::InProgress, 9, 9, 9),
      MakeValidRequest(kRequestB, RequestStatus::InProgress, 4, 4, 4),
  };
  EXPECT_TRUE(ValidateInvariants(cache, requests).empty());
  EXPECT_NO_THROW(ThrowIfInvariantsViolated(cache, requests));
}

TEST(InvariantValidatorTest, BlockTableForUnknownRequestReported) {
  const auto cache = MakeValidCache();  // owns tables for A and B
  const std::vector<RequestStateSnapshot> requests{
      MakeValidRequest(kRequestA, RequestStatus::InProgress, 9, 9, 9),
      // B is missing from the request set, yet the cache holds a block table for it.
  };
  EXPECT_FALSE(ValidateInvariants(cache, requests).empty());
}

TEST(InvariantValidatorTest, ThrowWrapperListsViolations) {
  auto cache = MakeValidCache();
  cache.free_blocks = 0;  // break block accounting
  const std::vector<RequestStateSnapshot> requests{
      MakeValidRequest(kRequestA, RequestStatus::InProgress, 9, 9, 9),
      MakeValidRequest(kRequestB, RequestStatus::InProgress, 4, 4, 4),
  };
  EXPECT_THROW(ThrowIfInvariantsViolated(cache, requests), std::runtime_error);
}

}  // namespace
}  // namespace Generators
