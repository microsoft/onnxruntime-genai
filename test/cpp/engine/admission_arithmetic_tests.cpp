// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Pure unit tests for the admission arithmetic in src/engine/admission.h.

#include <cstddef>

#include <gtest/gtest.h>

#include "engine/admission.h"

namespace Generators {
namespace test {
namespace {

TEST(AdmissionArithmeticTest, PrefillRequiresOneSlotPerInputToken) {
  EXPECT_EQ(RequiredSlots(/*processed_sequence_length=*/0, /*unprocessed_token_count=*/0), 0u);
  EXPECT_EQ(RequiredSlots(/*processed_sequence_length=*/0, /*unprocessed_token_count=*/1), 1u);
  EXPECT_EQ(RequiredSlots(/*processed_sequence_length=*/0, /*unprocessed_token_count=*/7), 7u);
}

TEST(AdmissionArithmeticTest, DecodeAdvancesFromProcessedCursor) {
  EXPECT_EQ(RequiredSlots(/*processed_sequence_length=*/4, /*unprocessed_token_count=*/1), 5u);
  EXPECT_EQ(RequiredSlots(/*processed_sequence_length=*/10, /*unprocessed_token_count=*/2), 12u);
}

TEST(AdmissionArithmeticTest, NoPendingTokensPreservesProcessedCursor) {
  EXPECT_EQ(RequiredSlots(/*processed_sequence_length=*/9, /*unprocessed_token_count=*/0), 9u);
}

}  // namespace
}  // namespace test
}  // namespace Generators
