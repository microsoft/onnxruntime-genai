// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the absolute-position arithmetic in `src/engine/sequence_positions.h`, which the
// varlen decoder and the paged KV cache both use to place a step's tokens in the cache.
//
// The invariant under test is that the base position is the number of tokens already written to
// the cache, not the length of the sequence. The two differ on every decode step, because token
// selection appends the next token before the step that processes it runs. Deriving the base from
// the sequence length left every generated token one slot too high, skipped the slot below it and
// shifted the rotary positions of every decode token by one.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "engine/sequence_positions.h"

namespace Generators::test {
namespace {

// Mirrors the two cursors a Request keeps over its own sequence: the search's sequence length,
// which grows as soon as a token is selected, and the processed length, which only grows once the
// tokens have been through the model.
struct SequenceCursor {
  int64_t sequence_length{};
  int64_t processed_length{};

  void AppendPrompt(int64_t prompt_length) { sequence_length += prompt_length; }

  size_t Unprocessed() const { return static_cast<size_t>(sequence_length - processed_length); }

  // One engine step: the tokens between the two cursors go through the model, then the token the
  // step predicted is appended to the sequence.
  void Step() {
    processed_length = sequence_length;
    sequence_length += 1;
  }
};

// The absolute positions a step writes to, as the operator resolves them.
std::vector<int64_t> WrittenPositions(const SequenceCursor& cursor) {
  std::vector<int64_t> positions;
  for (size_t j = 0; j < cursor.Unprocessed(); ++j) {
    positions.push_back(cursor.processed_length + static_cast<int64_t>(j));
  }
  return positions;
}

}  // namespace

TEST(SequencePositionsTest, PrefillStartsAtZero) {
  SequenceCursor cursor;
  cursor.AppendPrompt(5);

  EXPECT_EQ(cursor.processed_length, 0);
  EXPECT_EQ(cursor.Unprocessed(), 5u);
  EXPECT_EQ(SlotsAfterStep(cursor.processed_length, cursor.Unprocessed()), 5u);
}

TEST(SequencePositionsTest, DecodeStepsWriteConsecutivePositions) {
  // The regression: on a five token prompt the operator used to be told 0, 6, 7, 8 where it should
  // have been told 0, 5, 6, 7.
  SequenceCursor cursor;
  cursor.AppendPrompt(5);

  std::vector<int64_t> reported_past;
  std::vector<int64_t> all_written;
  for (int step = 0; step < 4; ++step) {
    reported_past.push_back(cursor.processed_length);
    for (int64_t position : WrittenPositions(cursor)) {
      all_written.push_back(position);
    }
    cursor.Step();
  }

  EXPECT_EQ(reported_past, (std::vector<int64_t>{0, 5, 6, 7}));
  // Every position from 0 up to the last one written is covered exactly once: no gap, no overlap.
  EXPECT_EQ(all_written, (std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(SequencePositionsTest, SlotsAfterStepCoversEveryWrittenPosition) {
  SequenceCursor cursor;
  cursor.AppendPrompt(3);

  for (int step = 0; step < 6; ++step) {
    const size_t slots = SlotsAfterStep(cursor.processed_length, cursor.Unprocessed());
    for (int64_t position : WrittenPositions(cursor)) {
      // A slot count that does not cover the highest position written is an out-of-bounds block
      // table entry as soon as that position lands on a block boundary.
      EXPECT_LT(static_cast<size_t>(position), slots);
    }
    cursor.Step();
  }
}

TEST(SequencePositionsTest, AdmissionCoversTheWholePrompt) {
  SequenceCursor cursor;
  cursor.AppendPrompt(7);

  // Admission reserves for the whole sequence, which the pending step must never exceed.
  for (int step = 0; step < 5; ++step) {
    EXPECT_LE(SlotsAfterStep(cursor.processed_length, cursor.Unprocessed()),
              SlotsForWholeSequence(cursor.sequence_length));
    cursor.Step();
  }
}

}  // namespace Generators::test
