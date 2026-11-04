// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>

#include <gtest/gtest.h>

#include "engine/decode_first_scheduler_policy.h"

namespace Generators::test {
namespace {

TEST(DecodeFirstSchedulerPolicyTest, OrdersDecodesFirstAndKeepsEachPhaseStable) {
  const std::array candidates{
      DecodeFirstBudgetCandidate{true, 8, std::nullopt},
      DecodeFirstBudgetCandidate{false, 1, std::nullopt},
      DecodeFirstBudgetCandidate{true, 4, std::nullopt},
      DecodeFirstBudgetCandidate{false, 1, std::nullopt},
  };

  EXPECT_EQ(DecodeFirstCandidateOrder(candidates),
            (std::vector<size_t>{1, 3, 0, 2}));
}

TEST(DecodeFirstSchedulerPolicyTest, AccountsForEveryTokenExactly) {
  const std::array selected{
      DecodeFirstBudgetCandidate{false, 1, std::nullopt},
      DecodeFirstBudgetCandidate{false, 1, std::nullopt},
      DecodeFirstBudgetCandidate{true, 8, std::nullopt},
      DecodeFirstBudgetCandidate{true, 8, std::nullopt},
  };

  const auto counts = AllocateDecodeFirstTokenBudget(selected, 7);

  EXPECT_EQ(counts, (std::vector<size_t>{1, 1, 4, 1}));
}

TEST(DecodeFirstSchedulerPolicyTest, DecodeDemandExhaustsTheBudget) {
  const std::array selected{
      DecodeFirstBudgetCandidate{false, 1, std::nullopt},
      DecodeFirstBudgetCandidate{false, 1, std::nullopt},
      DecodeFirstBudgetCandidate{false, 1, std::nullopt},
  };

  EXPECT_EQ(AllocateDecodeFirstTokenBudget(selected, 3),
            (std::vector<size_t>{1, 1, 1}));
  EXPECT_THROW(AllocateDecodeFirstTokenBudget(selected, 2),
               std::invalid_argument);
}

TEST(DecodeFirstSchedulerPolicyTest, SpeculativeRowsConsumeBudgetAfterEveryRequestGetsOneToken) {
  const std::array selected{
      DecodeFirstBudgetCandidate{false, 1, std::nullopt, 3},
      DecodeFirstBudgetCandidate{false, 1, std::nullopt, 2},
      DecodeFirstBudgetCandidate{true, 8, std::nullopt},
  };

  EXPECT_EQ(AllocateDecodeFirstTokenBudget(selected, 5),
            (std::vector<size_t>{3, 1, 1}));
  EXPECT_EQ(AllocateDecodeFirstTokenBudget(selected, 8),
            (std::vector<size_t>{4, 3, 1}));
}

TEST(DecodeFirstSchedulerPolicyTest, UnreadCommittedTokensAreChunkWorkWithinTheirCap) {
  const std::array selected{
      DecodeFirstBudgetCandidate{false, 4, std::nullopt},
      DecodeFirstBudgetCandidate{false, 6, 2},
      DecodeFirstBudgetCandidate{true, 8, std::nullopt},
  };

  EXPECT_EQ(AllocateDecodeFirstTokenBudget(selected, 4),
            (std::vector<size_t>{2, 1, 1}));
  EXPECT_EQ(AllocateDecodeFirstTokenBudget(selected, 9),
            (std::vector<size_t>{4, 2, 3}));
}

// Reading several unread tokens beside a verify row would cost the peer its drafts, so the row
// reads one and the prefill still gets the rest.
TEST(DecodeFirstSchedulerPolicyTest, UnreadCommittedTokensReadOneWhileAPeerVerifiesDrafts) {
  const std::array selected{
      DecodeFirstBudgetCandidate{false, 1, std::nullopt, 3},
      DecodeFirstBudgetCandidate{false, 8, std::nullopt},
      DecodeFirstBudgetCandidate{true, 10, std::nullopt},
  };

  EXPECT_EQ(AllocateDecodeFirstTokenBudget(selected, 16),
            (std::vector<size_t>{4, 1, 10}));
}

// Spare tokens go to the chunk-work row that waited longest, whatever order the rows arrive in.
TEST(DecodeFirstSchedulerPolicyTest, ChunkWorkIsServedLongestWaitingFirstInAnyOrder) {
  const std::array rows{
      DecodeFirstBudgetCandidate{true, 10, 4, 0, /*last_chunk_service=*/7},
      DecodeFirstBudgetCandidate{false, 8, 3, 0, /*last_chunk_service=*/3},
      DecodeFirstBudgetCandidate{true, 10, 4, 0, /*last_chunk_service=*/5},
  };
  // Three rows leave five spare tokens: the stamp-3 row takes its cap of 2, the stamp-5 row 3.
  const std::array<size_t, 3> expected{1, 3, 4};

  std::array<size_t, 3> order{0, 1, 2};
  do {
    std::array<DecodeFirstBudgetCandidate, 3> selected;
    for (size_t i = 0; i < order.size(); ++i)
      selected[i] = rows[order[i]];
    const auto counts = AllocateDecodeFirstTokenBudget(selected, 8);
    for (size_t i = 0; i < order.size(); ++i)
      EXPECT_EQ(counts[i], expected[order[i]]) << "row " << order[i];
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST(DecodeFirstSchedulerPolicyTest, DecodeRowCannotCarryDraftsAndUnreadCommittedTokens) {
  const std::array selected{DecodeFirstBudgetCandidate{false, 2, std::nullopt, 1}};
  EXPECT_THROW(AllocateDecodeFirstTokenBudget(selected, 8), std::invalid_argument);
}

TEST(DecodeFirstSchedulerPolicyTest, PrefillsRespectPendingCapAndGlobalBudget) {
  const std::array selected{
      DecodeFirstBudgetCandidate{true, 2, std::nullopt},
      DecodeFirstBudgetCandidate{true, 10, 3},
      DecodeFirstBudgetCandidate{true, 10, std::nullopt},
  };

  EXPECT_EQ(AllocateDecodeFirstTokenBudget(selected, 7),
            (std::vector<size_t>{2, 3, 2}));
}

TEST(DecodeFirstSchedulerPolicyTest, RequestLimitUsesBothIndependentCaps) {
  EXPECT_EQ(DecodeFirstProvisionalRequestLimit(3, 8), 3u);
  EXPECT_EQ(DecodeFirstProvisionalRequestLimit(8, 3), 3u);
  EXPECT_THROW(DecodeFirstProvisionalRequestLimit(0, 3),
               std::invalid_argument);
}

}  // namespace
}  // namespace Generators::test
