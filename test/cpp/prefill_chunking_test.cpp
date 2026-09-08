// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the chunked prefill schedule: how many tokens a request contributes to one engine
// step, and where in the KV cache the model writes them.
//
// The schedule and the cache accounting come from the same two helpers the engine uses
// (`ScheduledTokenCount` and `SlotsAfterStep`), so a step can never be sized differently from the
// tokens it carries. The simulation below mirrors what Request does with them: schedule a chunk,
// run it, advance the cursor, and -- once the prompt is exhausted -- append the predicted token.

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engine/sequence_positions.h"

namespace Generators::test {
namespace {

// (past_sequence_length, query_length) as the model sees them on one step.
using Step = std::pair<int64_t, size_t>;

// Mirrors the cursors Request keeps: the search's sequence length, which grows as soon as a token
// is selected, and the processed length, which only grows once tokens have been through the model.
struct RequestSimulator {
  int64_t sequence_length{};
  int64_t processed_length{};
  size_t scheduled{};

  void AddTokens(int64_t count) { sequence_length += count; }

  size_t Unprocessed() const { return static_cast<size_t>(sequence_length - processed_length); }

  void ScheduleTokens(std::optional<size_t> chunk_size) {
    scheduled = ScheduledTokenCount(Unprocessed(), chunk_size);
  }

  bool IsChunkComplete() const {
    return processed_length + static_cast<int64_t>(scheduled) >= sequence_length;
  }

  void AdvanceChunk() { processed_length += static_cast<int64_t>(scheduled); }
};

// Runs `num_steps` engine steps over a prompt and returns what the model was told on each one.
std::vector<Step> RunSteps(int64_t prompt_length, std::optional<size_t> chunk_size, int num_steps) {
  RequestSimulator request;
  request.AddTokens(prompt_length);

  std::vector<Step> steps;
  for (int i = 0; i < num_steps; ++i) {
    request.ScheduleTokens(chunk_size);
    steps.emplace_back(request.processed_length, request.scheduled);

    const bool selects_token = request.IsChunkComplete();
    request.AdvanceChunk();
    if (selects_token) {
      request.AddTokens(1);  // The step's last logits row predicted a token.
    }
  }
  return steps;
}

}  // namespace

TEST(PrefillChunkingTest, WholePromptWhenNoChunkSizeIsConfigured) {
  EXPECT_EQ(RunSteps(512, std::nullopt, 3), (std::vector<Step>{{0, 512}, {512, 1}, {513, 1}}));
  // A chunk size of zero means "not configured", matching how Config parses it.
  EXPECT_EQ(RunSteps(512, 0, 3), (std::vector<Step>{{0, 512}, {512, 1}, {513, 1}}));
}

TEST(PrefillChunkingTest, PromptIsSpreadOverSteps) {
  // A 512 token prompt at chunk 128: four full chunks, then decode one token at a time.
  EXPECT_EQ(RunSteps(512, 128, 6),
            (std::vector<Step>{{0, 128}, {128, 128}, {256, 128}, {384, 128}, {512, 1}, {513, 1}}));
}

TEST(PrefillChunkingTest, LastChunkIsShort) {
  EXPECT_EQ(RunSteps(300, 128, 4), (std::vector<Step>{{0, 128}, {128, 128}, {256, 44}, {300, 1}}));
}

TEST(PrefillChunkingTest, ChunkLargerThanPromptRunsInOneStep) {
  EXPECT_EQ(RunSteps(10, 128, 3), (std::vector<Step>{{0, 10}, {10, 1}, {11, 1}}));
}

TEST(PrefillChunkingTest, ChunkOfOneDegradesToTokenAtATime) {
  EXPECT_EQ(RunSteps(4, 1, 6),
            (std::vector<Step>{{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}}));
}

TEST(PrefillChunkingTest, EveryPositionIsWrittenExactlyOnce) {
  // Whatever the chunk size, the steps tile the sequence: no position is skipped and none is
  // written twice. This is what makes a chunked prefill produce the same KV cache as an unchunked
  // one.
  for (size_t chunk_size : {size_t{1}, size_t{7}, size_t{37}, size_t{64}, size_t{128}, size_t{512}}) {
    int64_t next_expected_position = 0;
    for (const auto& [past, query_length] : RunSteps(551, chunk_size, 24)) {
      EXPECT_EQ(past, next_expected_position) << "chunk_size=" << chunk_size;
      EXPECT_GT(query_length, 0u) << "chunk_size=" << chunk_size;
      EXPECT_LE(query_length, chunk_size) << "chunk_size=" << chunk_size;
      next_expected_position = past + static_cast<int64_t>(query_length);
    }
  }
}

TEST(PrefillChunkingTest, CacheIsSizedForExactlyTheScheduledTokens) {
  // The slots the cache must own after a step cover the tokens the step actually writes, never the
  // whole remaining prompt. A chunked prefill therefore grows the cache one chunk at a time.
  const auto chunked = RunSteps(551, 128, 6);
  const auto unchunked = RunSteps(551, std::nullopt, 6);

  EXPECT_EQ(SlotsAfterStep(chunked[0].first, chunked[0].second), 128u);
  EXPECT_EQ(SlotsAfterStep(unchunked[0].first, unchunked[0].second), 551u);

  // Both reach the same total once the prompt is in: five chunks, then one decoded token.
  EXPECT_EQ(SlotsAfterStep(chunked.back().first, chunked.back().second), 552u);
}

}  // namespace Generators::test
