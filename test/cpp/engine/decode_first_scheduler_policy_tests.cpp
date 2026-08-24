// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
