// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Pure unit tests for the admission arithmetic in src/engine/admission.h. RequiredSlots is a
// plain calculation over token counters, so these tests need no model, device, or Search -- they
// pin the prefill/decode branch behavior that the paged cache manager relies on to size block
// reservations.

#include <cstddef>

#include <gtest/gtest.h>

#include "engine/admission.h"

namespace Generators {
namespace test {
namespace {

// While prefilling, every token in the sequence is unprocessed, so the required slot count is just
// the sequence length and any unprocessed_count argument is ignored.
TEST(AdmissionArithmeticTest, PrefillRequiresOneSlotPerSequenceToken) {
  EXPECT_EQ(RequiredSlots(/*sequence_length=*/0, /*unprocessed_count=*/0, /*is_prefill=*/true), 0u);
  EXPECT_EQ(RequiredSlots(/*sequence_length=*/1, /*unprocessed_count=*/0, /*is_prefill=*/true), 1u);
  EXPECT_EQ(RequiredSlots(/*sequence_length=*/7, /*unprocessed_count=*/0, /*is_prefill=*/true), 7u);
}

// The unprocessed_count is not part of the prefill slot count: a prefill request's sequence length
// already accounts for its unprocessed prompt tokens.
TEST(AdmissionArithmeticTest, PrefillIgnoresUnprocessedCount) {
  EXPECT_EQ(RequiredSlots(/*sequence_length=*/5, /*unprocessed_count=*/3, /*is_prefill=*/true), 5u);
}

// While decoding, the sequence length counts only already-processed tokens, so the freshly
// generated unprocessed tokens must be added on top to reserve room for the pending step.
TEST(AdmissionArithmeticTest, DecodeAddsUnprocessedTokensOnTopOfSequence) {
  EXPECT_EQ(RequiredSlots(/*sequence_length=*/4, /*unprocessed_count=*/1, /*is_prefill=*/false), 5u);
  EXPECT_EQ(RequiredSlots(/*sequence_length=*/10, /*unprocessed_count=*/2, /*is_prefill=*/false), 12u);
}

// A decode step with nothing unprocessed reserves exactly the current sequence length.
TEST(AdmissionArithmeticTest, DecodeWithNoUnprocessedTokensEqualsSequenceLength) {
  EXPECT_EQ(RequiredSlots(/*sequence_length=*/9, /*unprocessed_count=*/0, /*is_prefill=*/false), 9u);
}

}  // namespace
}  // namespace test
}  // namespace Generators
