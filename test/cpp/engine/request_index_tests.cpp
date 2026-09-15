// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <unordered_map>

#include <gtest/gtest.h>

#include "engine/request_index.h"

namespace Generators {
namespace {

TEST(RequestIndexTest, SupportsInsertFindEraseAndReuseWithoutAllocation) {
  RequestIndex index{3};
  const std::array<int, 4> requests{};

  EXPECT_TRUE(index.Insert(&requests[0], 10));
  EXPECT_TRUE(index.Insert(&requests[1], 20));
  EXPECT_EQ(index.Find(&requests[0]), 10u);
  EXPECT_EQ(index.Find(&requests[1]), 20u);
  EXPECT_FALSE(index.Insert(&requests[0], 30));

  EXPECT_TRUE(index.Erase(&requests[0]));
  EXPECT_FALSE(index.Find(&requests[0]));
  EXPECT_TRUE(index.Insert(&requests[2], 30));
  EXPECT_EQ(index.Find(&requests[2]), 30u);
  EXPECT_EQ(index.Size(), 2u);
}

TEST(RequestIndexTest, EnforcesConfiguredCapacityAndClearsEntries) {
  RequestIndex index{2};
  const std::array<int, 3> requests{};

  EXPECT_TRUE(index.Insert(&requests[0], 0));
  EXPECT_TRUE(index.Insert(&requests[1], 1));
  EXPECT_FALSE(index.Insert(&requests[2], 2));

  index.Clear();
  EXPECT_EQ(index.Size(), 0u);
  EXPECT_FALSE(index.Find(&requests[0]));
  EXPECT_TRUE(index.Insert(&requests[2], 2));
}

TEST(RequestIndexTest, SupportsSustainedIdentityChurn) {
  RequestIndex index{1};

  for (uintptr_t request = 1; request <= 64; ++request) {
    const auto* request_id = reinterpret_cast<const void*>(request);
    ASSERT_TRUE(index.Insert(request_id, request));
    EXPECT_EQ(index.Find(request_id), request);
    ASSERT_TRUE(index.Erase(request_id));
  }

  EXPECT_TRUE(index.Insert(reinterpret_cast<const void*>(65), 65));
  EXPECT_EQ(index.Find(reinterpret_cast<const void*>(65)), 65u);
}

TEST(RequestIndexTest, PreservesCollidingEntriesAcrossBackshiftDeletion) {
  RequestIndex index{8};
  std::unordered_map<const void*, size_t> expected;

  uint32_t random = 1;
  for (size_t step = 0; step < 5000; ++step) {
    random = random * 1664525u + 1013904223u;
    const auto* request_id =
        reinterpret_cast<const void*>(static_cast<uintptr_t>(random % 32 + 1));
    const auto existing = expected.find(request_id);
    if (existing != expected.end()) {
      ASSERT_TRUE(index.Erase(request_id));
      expected.erase(existing);
    } else if (expected.size() < index.Capacity()) {
      ASSERT_TRUE(index.Insert(request_id, step));
      expected.emplace(request_id, step);
    }

    EXPECT_EQ(index.Size(), expected.size());
    for (uintptr_t request = 1; request <= 32; ++request) {
      const auto* candidate = reinterpret_cast<const void*>(request);
      const auto found = index.Find(candidate);
      const auto expected_entry = expected.find(candidate);
      if (expected_entry == expected.end()) {
        EXPECT_FALSE(found);
      } else {
        ASSERT_TRUE(found);
        EXPECT_EQ(*found, expected_entry->second);
      }
    }
  }
}

}  // namespace
}  // namespace Generators
