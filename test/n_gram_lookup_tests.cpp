// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "n_gram_lookup.h"

#include <array>
#include <vector>

#include <gtest/gtest.h>

namespace Generators::test {

TEST(NGramLookupTest, ReturnsEmptyWithoutPreviousMatch) {
  NGramLookup lookup{3};
  const std::array<int32_t, 3> history{1, 2, 3};
  lookup.Sync(history);
  EXPECT_TRUE(lookup.Propose(4).empty());
}

TEST(NGramLookupTest, ProposesContinuationFromPreviousMatch) {
  NGramLookup lookup{3};
  const std::array<int32_t, 5> history{1, 2, 3, 1, 2};
  lookup.Sync(history);
  EXPECT_EQ(lookup.Propose(3), (std::vector<int32_t>{3, 1, 2}));
}

TEST(NGramLookupTest, UsesMostRecentPreviousMatch) {
  NGramLookup lookup{3};
  const std::array<int32_t, 8> history{1, 2, 7, 1, 2, 8, 1, 2};
  lookup.Sync(history);
  EXPECT_EQ(lookup.Propose(1), (std::vector<int32_t>{8}));
}

TEST(NGramLookupTest, SupportsOverlappingMatches) {
  NGramLookup lookup{2};
  const std::array<int32_t, 3> history{4, 4, 4};
  lookup.Sync(history);
  EXPECT_EQ(lookup.Propose(2), (std::vector<int32_t>{4}));
}

TEST(NGramLookupTest, ClampsToAvailableContinuationAndRequestedLimit) {
  NGramLookup lookup{3};
  const std::array<int32_t, 6> history{1, 2, 3, 4, 1, 2};
  lookup.Sync(history);
  EXPECT_EQ(lookup.Propose(2), (std::vector<int32_t>{3, 4}));
  EXPECT_EQ(lookup.Propose(10), (std::vector<int32_t>{3, 4, 1, 2}));
}

TEST(NGramLookupTest, ExtendsIncrementally) {
  NGramLookup lookup{3};
  const std::array<int32_t, 3> initial{1, 2, 3};
  const std::array<int32_t, 5> extended{1, 2, 3, 1, 2};
  lookup.Sync(initial);
  lookup.Sync(extended);
  EXPECT_EQ(lookup.HistorySize(), extended.size());
  EXPECT_EQ(lookup.Propose(1), (std::vector<int32_t>{3}));
}

TEST(NGramLookupTest, RebuildsAfterRewindOrReplacement) {
  NGramLookup lookup{3};
  const std::array<int32_t, 5> initial{1, 2, 3, 1, 2};
  const std::array<int32_t, 4> rewound{1, 2, 3, 1};
  lookup.Sync(initial);
  lookup.Sync(rewound);
  EXPECT_EQ(lookup.HistorySize(), rewound.size());
  EXPECT_TRUE(lookup.Propose(4).empty());

  const std::array<int32_t, 5> replacement{9, 8, 7, 9, 8};
  lookup.Sync(replacement);
  EXPECT_EQ(lookup.Propose(1), (std::vector<int32_t>{7}));
}

}  // namespace Generators::test
