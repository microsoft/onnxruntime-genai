// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Engine-level tests of the static single-slot Qwen3.8-Flash CUDA-graph backend. A scripted device replaces the CUDA
// loop (see qwen38_graph_test_doubles.h); everything else is the real Engine: StaticBatchScheduler, the single-slot
// cache manager, the executor, Request::CommitExternalTokens and the Engine event path. The fixture model is the
// synthetic-paged one because it carries a tiny tokenizer for the stop-string cases (token 5 -> "ST", 6 -> "OP",
// 0/1 special, EOS is 1).

#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engine/qwen38_graph_executor.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"
#include "qwen38_graph_test_doubles.h"
#include "models/qwen38_suffix_drafter.h"

namespace Generators::test {
namespace {

constexpr int32_t kEos = 1;

TEST(Qwen38SuffixDrafterTest, ProposesOnlyHistoricalContinuation) {
  Qwen38SuffixDrafter drafter;
  const std::array<int32_t, 9> history{10, 11, 12, 13, 14, 15, 99, 10, 11};
  drafter.AppendSpan(history);
  std::array<int32_t, 3> drafts{-1, -1, -1};
  ASSERT_EQ(drafter.ProposeDrafts(12, drafts), 3u);
  EXPECT_EQ(drafts, (std::array<int32_t, 3>{13, 14, 15}));
}

TEST(Qwen38SuffixConfigTest, ParsesOnlyTheThreeSupportedModes) {
  Config config;
  EXPECT_EQ(config.model.qwen38_graph.suffix_lookup, 0);
  for (int mode : {0, 1, 2}) {
    OverlayConfig(config, "{\"model\":{\"qwen38_graph\":{\"suffix_lookup\":" + std::to_string(mode) + "}}}");
    EXPECT_EQ(config.model.qwen38_graph.suffix_lookup, mode);
  }
  for (const char* mode : {"-1", "3", "1.5", "\"1\"", "true"}) {
    EXPECT_THROW(OverlayConfig(config, std::string{"{\"model\":{\"qwen38_graph\":{\"suffix_lookup\":"} + mode + "}}}"),
                 std::exception);
  }
}

TEST(Qwen38SuffixDrafterTest, TwoGramFallbackIsExplicit) {
  Qwen38SuffixDrafter drafter;
  const std::array<int32_t, 5> history{10, 11, 12, 99, 10};
  drafter.AppendSpan(history);
  std::array<int32_t, 1> drafts{-1};
  EXPECT_EQ(drafter.ProposeDrafts(11, drafts), 0u);
  ASSERT_EQ(drafter.ProposeDrafts(11, drafts, 2), 1u);
  EXPECT_EQ(drafts[0], 12);
}

TEST(Qwen38SuffixDrafterTest, DoesNotAliasHighTokenBits) {
  Qwen38SuffixDrafter drafter;
  const std::array<int32_t, 6> history{5 + (1 << 21), 6, 7, 42, 5, 6};
  drafter.AppendSpan(history);
  std::array<int32_t, 3> drafts{-1, -1, -1};
  EXPECT_EQ(drafter.ProposeDrafts(7, drafts), 0u);
  EXPECT_EQ(drafts, (std::array<int32_t, 3>{-1, -1, -1}));
}

TEST(Qwen38SuffixDrafterTest, ResetsHistoryAndClipsAtItsEnd) {
  Qwen38SuffixDrafter drafter;
  const std::array<int32_t, 6> history{10, 11, 12, 42, 10, 11};
  drafter.AppendSpan(history);
  std::array<int32_t, 8> drafts{};
  ASSERT_EQ(drafter.ProposeDrafts(12, drafts), 3u);
  EXPECT_EQ(drafts[0], 42);
  EXPECT_EQ(drafts[1], 10);
  EXPECT_EQ(drafts[2], 11);
  EXPECT_EQ(drafts[3], 0);
  drafter.Reset();
  EXPECT_EQ(drafter.ProposeDrafts(12, drafts), 0u);
}

TEST(Qwen38SuffixDrafterTest, RejectsInvalidInputsWithoutAppendingPartialSpan) {
  Qwen38SuffixDrafter drafter;
  const std::array<int32_t, 4> invalid{1, 2, 3, -1};
  EXPECT_THROW(drafter.AppendSpan(invalid), std::invalid_argument);
  EXPECT_THROW(drafter.AppendToken(-1), std::invalid_argument);
  std::array<int32_t, 1> drafts{};
  EXPECT_EQ(drafter.ProposeDrafts(3, drafts), 0u);
  EXPECT_THROW(drafter.ProposeDrafts(-1, drafts), std::invalid_argument);
  EXPECT_THROW(drafter.ProposeDrafts(3, drafts, 1), std::invalid_argument);
  EXPECT_THROW(drafter.ProposeDrafts(3, drafts, 4), std::invalid_argument);
}

TEST(Qwen38SuffixDrafterTest, HashCollisionsNeverProposeAnUnmatchedContinuation) {
  Qwen38SuffixDrafter drafter;
  std::vector<int32_t> history;
  std::mt19937 random{1839};
  for (size_t step = 0; step < 20000; ++step) {
    history.push_back(static_cast<int32_t>(random() % 8));
    drafter.AppendToken(history.back());
    if (history.size() < 3) continue;
    const int32_t pending = static_cast<int32_t>(random() % 8);
    std::array<int32_t, 3> drafts{};
    const size_t count = drafter.ProposeDrafts(pending, drafts);
    if (count == 0) continue;
    bool matched = false;
    for (size_t position = 3; position + count <= history.size(); ++position) {
      if (history[position - 3] == history[history.size() - 2] &&
          history[position - 2] == history.back() && history[position - 1] == pending &&
          std::equal(drafts.begin(), drafts.begin() + count, history.begin() + position)) {
        matched = true;
        break;
      }
    }
    ASSERT_TRUE(matched);
  }
}

struct Outcome {
  std::vector<int32_t> tokens;
  GenerationFinishReason finish_reason{GenerationFinishReason::None};
  int32_t matched_stop_string_index{-1};
  TurnUsage usage{};
  size_t runs{};
  size_t finished_events{};
  uint32_t failure_flags{};
};

class Qwen38GraphEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadSyntheticPagedModel();
    model_->config_->engine.dynamic_batching.reset();
    model_->config_->model.qwen38_graph.prefill_chunk_tokens = 4;
    device_ = std::make_shared<ScriptedQwen38Device>();
  }

  std::shared_ptr<Engine> MakeEngine() {
    return std::make_shared<Engine>(model_, CreateQwen38GraphDependencies(model_, device_));
  }

  static std::vector<int32_t> Prompt(size_t count, int32_t first = 12) {
    std::vector<int32_t> prompt(count);
    for (size_t i = 0; i < count; ++i) prompt[i] = first + static_cast<int32_t>(i % 20);
    return prompt;
  }

  // Runs the Engine until it has no work and returns what `request` produced.
  static Outcome RunToCompletion(Engine& engine, const std::shared_ptr<Request>& request, size_t max_runs = 10000) {
    Outcome outcome;
    std::array<EngineEvent, 8> buffer;
    while (engine.HasPendingRequests()) {
      if (++outcome.runs > max_runs) {
        ADD_FAILURE() << "the Engine did not finish";
        break;
      }
      const size_t count = engine.Run(buffer);
      for (size_t i = 0; i < count; ++i) {
        const EngineEvent& event = buffer[i];
        if (event.request != request) continue;
        if (event.flags & EngineEventFlagToken) outcome.tokens.push_back(event.token);
        if (event.flags & (EngineEventFlagFailed | EngineEventFlagRetryable | EngineEventFlagCapacityBlocked)) {
          outcome.failure_flags |= event.flags;
        }
        if (event.flags & EngineEventFlagTurnFinished) {
          ++outcome.finished_events;
          outcome.finish_reason = event.finish_reason;
          outcome.matched_stop_string_index = event.matched_stop_string_index;
          outcome.usage = event.usage;
        }
      }
    }
    return outcome;
  }

  std::vector<int32_t> Expected(size_t first_position, size_t count) const {
    std::vector<int32_t> tokens;
    for (size_t i = 0; i < count; ++i) tokens.push_back(device_->TokenAt(first_position + i));
    return tokens;
  }

  static TurnOptions Limit(size_t max_generated_tokens) {
    TurnOptions options;
    options.max_generated_tokens = max_generated_tokens;
    return options;
  }

  std::shared_ptr<Request> Begin(Engine& engine, std::span<const int32_t> prompt, size_t max_generated_tokens,
                                 std::optional<size_t> max_session_tokens = std::nullopt) {
    RequestOptions request_options;
    request_options.max_session_tokens = max_session_tokens;
    auto request = engine.CreateRequest(request_options);
    auto options = Limit(max_generated_tokens);
    options.request = request;
    request->BeginTurn(prompt, options);
    return request;
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<ScriptedQwen38Device> device_;
};

// ---- token path ------------------------------------------------------------------------------

TEST_F(Qwen38GraphEngineTest, GeneratesTheScriptedStreamAndEndsAtTheTurnLimit) {
  auto engine = MakeEngine();
  const auto prompt = Prompt(5);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/7);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 7));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::TurnLimit);
  EXPECT_EQ(outcome.finished_events, 1u);
  EXPECT_EQ(outcome.usage.prompt_tokens, 5u);
  EXPECT_EQ(outcome.usage.generated_tokens, 7u);
  EXPECT_EQ(request->TokensCpu().size(), 12u);

  // Accept pattern {accept, reject}: tokens 1 | 2 | 1 | 2 | 1 = 7 over 4 cycles; the last cycle was waited for and
  // no further cycle was enqueued.
  EXPECT_EQ(device_->CallCount("wait"), 4u);
  EXPECT_EQ(device_->cycles_enqueued, 4u);
  EXPECT_FALSE(device_->inflight());
  EXPECT_EQ(device_->calls.back(), "advance:0");
  EXPECT_EQ(device_->ingested, prompt);

  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.rounds, 4u);
  EXPECT_EQ(stats.draft_tokens_proposed, 4u);
  EXPECT_EQ(stats.draft_tokens_accepted, 2u);
  EXPECT_EQ(stats.bonus_tokens, 2u);
  EXPECT_EQ(stats.correction_tokens, 2u);
  EXPECT_EQ(stats.tokens_emitted, 7u);
  EXPECT_EQ(stats.acceptance_length_histogram[0], 2u);
  EXPECT_EQ(stats.acceptance_length_histogram[1], 2u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 0.5f);
  EXPECT_FLOAT_EQ(stats.avg_draft_tokens_per_round, 1.0f);
}

TEST_F(Qwen38GraphEngineTest, TheTurnLimitCutsAnAcceptedCycleAndEndsTheLoop) {
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/2);

  const auto outcome = RunToCompletion(*engine, request);

  // First token + the accepted first cycle would commit 3 tokens; only 2 are allowed.
  EXPECT_EQ(outcome.tokens, Expected(5, 2));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::TurnLimit);
  EXPECT_EQ(device_->calls.back(), "advance:0");
  EXPECT_FALSE(device_->inflight());
  EXPECT_EQ(engine->GetSpeculativeStats().tokens_discarded, 1u);
}

TEST_F(Qwen38GraphEngineTest, SuffixLookupStatisticsReportAttemptsIncludingMisses) {
  device_->suffix_lookup_pattern = {true, false};
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/7);
  const auto outcome = RunToCompletion(*engine, request);
  EXPECT_EQ(outcome.tokens, Expected(5, 7));
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.ngram_lookup_hits, 2u);
  EXPECT_EQ(stats.ngram_lookup_misses, 2u);
  EXPECT_EQ(stats.ngram_lookup_tokens_proposed, 2u);
}

TEST_F(Qwen38GraphEngineTest, MultiDraftCyclesCommitOnlyTheMatchingPrefix) {
  for (int drafts : {2, 3}) {
    device_->SetDraftCount(drafts);
    device_->suffix_lookup_pattern = {true};
    device_->accepted_prefix_pattern = {drafts, 1, 0};
    auto engine = MakeEngine();
    auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/static_cast<size_t>(drafts + 6));
    const auto outcome = RunToCompletion(*engine, request);
    EXPECT_EQ(outcome.tokens, Expected(5, static_cast<size_t>(drafts + 6)));
    EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::TurnLimit);
    EXPECT_FALSE(device_->inflight());
    const auto stats = engine->GetSpeculativeStats();
    EXPECT_EQ(stats.rounds, 4u);
    EXPECT_EQ(stats.draft_tokens_proposed, static_cast<size_t>(4 * drafts));
    EXPECT_EQ(stats.draft_tokens_accepted, static_cast<size_t>(drafts + 2));
    EXPECT_EQ(stats.full_accept_rounds, 1u);
    EXPECT_EQ(stats.partial_accept_rounds, 2u);
    EXPECT_EQ(stats.bonus_tokens, 1u);
    EXPECT_EQ(stats.correction_tokens, 2u);
    EXPECT_EQ(stats.acceptance_length_histogram[drafts], 1u);
    EXPECT_EQ(stats.acceptance_length_histogram[1], 2u);
    EXPECT_EQ(stats.ngram_lookup_hits, 4u);
    EXPECT_EQ(stats.ngram_lookup_tokens_proposed, static_cast<size_t>(4 * drafts));
    device_->Reset();
  }
}

TEST_F(Qwen38GraphEngineTest, MultiDraftTurnLimitClipsTokensAndAcceptanceStats) {
  device_->SetDraftCount(3);
  device_->accepted_prefix_pattern = {3};
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/3);
  const auto outcome = RunToCompletion(*engine, request);
  EXPECT_EQ(outcome.tokens, Expected(5, 3));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::TurnLimit);
  EXPECT_FALSE(device_->inflight());
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_proposed, 3u);
  EXPECT_EQ(stats.draft_tokens_accepted, 2u);
  EXPECT_EQ(stats.acceptance_length_histogram[2], 1u);
  EXPECT_EQ(stats.bonus_tokens, 0u);
  EXPECT_EQ(stats.full_accept_rounds, 0u);
  EXPECT_EQ(stats.partial_accept_rounds, 1u);
  EXPECT_EQ(stats.tokens_discarded, 2u);
}

TEST_F(Qwen38GraphEngineTest, MultiDraftEosAtEveryRowStopsWithoutExposingLaterPredictions) {
  for (size_t row = 0; row < 4; ++row) {
    device_->SetDraftCount(3);
    device_->accepted_prefix_pattern = {3};
    device_->overrides.clear();
    device_->overrides[6 + row] = kEos;
    auto engine = MakeEngine();
    auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/20);
    const auto outcome = RunToCompletion(*engine, request);
    EXPECT_EQ(outcome.tokens, Expected(5, 1 + row));
    EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::EosToken);
    EXPECT_FALSE(device_->inflight());
    const auto stats = engine->GetSpeculativeStats();
    EXPECT_EQ(stats.draft_tokens_accepted, std::min<size_t>(3, row));
    EXPECT_EQ(stats.tokens_discarded, 4 - row);
    EXPECT_EQ(stats.bonus_tokens, 0u);
    device_->Reset();
  }
}

TEST(Qwen38CycleRecordTest, RejectsInvalidAcceptanceCounts) {
  Qwen38CycleRecord record;
  record.proposed_count = 3;
  for (int accepted = 0; accepted <= 3; ++accepted) {
    record.accepted_count = accepted;
    EXPECT_EQ(record.TokenCount(), static_cast<size_t>(accepted + 1));
  }
  record.accepted_count = 4;
  EXPECT_THROW(record.TokenCount(), std::logic_error);
  record.accepted_count = -1;
  EXPECT_THROW(record.TokenCount(), std::logic_error);
  record.accepted_count = 0;
  record.proposed_count = 4;
  EXPECT_THROW(record.TokenCount(), std::logic_error);
  record.proposed_count = 0;
  EXPECT_THROW(record.TokenCount(), std::logic_error);
}

TEST_F(Qwen38GraphEngineTest, AnEndOfSequenceTokenFirstInACycleEndsTheTurnWithoutBeingDelivered) {
  device_->overrides[6] = kEos;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 1));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(outcome.usage.generated_tokens, 1u);
  EXPECT_EQ(device_->calls.back(), "advance:0");
  EXPECT_FALSE(device_->inflight());
  EXPECT_EQ(device_->CallCount("wait"), 1u);
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.tokens_emitted, 1u);
  EXPECT_EQ(stats.tokens_discarded, 2u);
  EXPECT_EQ(stats.draft_tokens_accepted, 0u);
  EXPECT_EQ(stats.bonus_tokens, 0u);
}

TEST_F(Qwen38GraphEngineTest, AnEndOfSequenceTokenSecondInAnAcceptedCycleKeepsTheFirstToken) {
  device_->overrides[7] = kEos;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 2));  // positions 5 and 6; the EOS at 7 is not delivered
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(outcome.usage.generated_tokens, 2u);
  EXPECT_EQ(device_->calls.back(), "advance:0");
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.tokens_emitted, 2u);
  EXPECT_EQ(stats.tokens_discarded, 1u);
  EXPECT_EQ(stats.draft_tokens_accepted, 1u);
  EXPECT_EQ(stats.bonus_tokens, 0u);
}

TEST_F(Qwen38GraphEngineTest, AnEndOfSequenceFirstTokenNeverStartsACycle) {
  device_->overrides[5] = kEos;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_TRUE(outcome.tokens.empty());
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(outcome.finished_events, 1u);
  EXPECT_EQ(device_->CallCount("begin:0"), 1u);
  EXPECT_EQ(device_->CallCount("wait"), 0u);
  EXPECT_EQ(device_->cycles_enqueued, 0u);
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.tokens_emitted, 0u);
  EXPECT_EQ(stats.tokens_discarded, 0u);
  EXPECT_EQ(stats.rounds, 0u);
}

TEST_F(Qwen38GraphEngineTest, TheSessionLimitEndsTheTurnWithTheContextLimitReason) {
  auto engine = MakeEngine();
  // 5 prompt tokens, room for 3 more.
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/50, /*max_session_tokens=*/8);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 3));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::ContextLimit);
  EXPECT_EQ(request->CurrentSequenceLength(), 8);
  EXPECT_EQ(device_->calls.back(), "advance:0");
}

TEST_F(Qwen38GraphEngineTest, AStopStringEndsTheTurnAndTheWastedLookAheadCycleIsHarmless) {
  // Tokens 5 and 6 decode to "ST" and "OP": the stop string "STOP" completes on the first token of the first cycle.
  device_->overrides[5] = 5;
  device_->overrides[6] = 6;
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  options.max_generated_tokens = 20;
  options.stop_strings = {"STOP"};
  request->BeginTurn(Prompt(5), options);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(outcome.matched_stop_string_index, 0);
  EXPECT_EQ(outcome.tokens, (std::vector<int32_t>{5, 6}));
  const auto stop_stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stop_stats.tokens_emitted, 2u);
  EXPECT_EQ(stop_stats.tokens_discarded, 1u);
  EXPECT_EQ(stop_stats.draft_tokens_accepted, 1u);
  EXPECT_EQ(stop_stats.bonus_tokens, 0u);
  // The device cannot know about stop strings: the cycle after the one that matched is still in flight...
  EXPECT_TRUE(device_->inflight());

  // ...and a new request waits for it (Reset) before reusing the device.
  const size_t resets = device_->CallCount("reset");
  device_->overrides.clear();
  auto next = Begin(*engine, Prompt(5), /*max_generated_tokens=*/3);
  const auto next_outcome = RunToCompletion(*engine, next);
  EXPECT_EQ(device_->CallCount("reset"), resets + 1);
  EXPECT_EQ(device_->status_checks, 2u);
  EXPECT_EQ(next_outcome.tokens, Expected(5, 3));
}

TEST_F(Qwen38GraphEngineTest, StopStringsSpanningTheTwoTokensOfACycleAreMatched) {
  device_->overrides[6] = 5;
  device_->overrides[7] = 6;
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  options.stop_strings = {"STOP"};
  request->BeginTurn(Prompt(5), options);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(outcome.tokens, (std::vector<int32_t>{device_->TokenAt(5), 5, 6}));
}

TEST_F(Qwen38GraphEngineTest, EveryDraftRejectedCommitsTheGreedyTokenOfEachCycleOnly) {
  device_->accept_pattern = {false};  // the fold.force_reject hook: no draft ever survives its verify
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/6);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 6));  // the same stream as with any other accept pattern
  EXPECT_EQ(device_->CallCount("wait"), 5u);  // the first token comes from the prompt, then one token per cycle
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.rounds, 5u);
  EXPECT_EQ(stats.draft_tokens_accepted, 0u);
  EXPECT_EQ(stats.zero_accept_rounds, 5u);
  EXPECT_EQ(stats.acceptance_length_histogram[0], 5u);
  EXPECT_EQ(device_->ledger_length(), 5u + 5u);
}

TEST_F(Qwen38GraphEngineTest, EveryDraftAcceptedCommitsTwoTokensPerCycle) {
  device_->accept_pattern = {true};
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/9);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 9));
  EXPECT_EQ(device_->CallCount("wait"), 4u);  // 1 + 4 * 2 tokens
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_accepted, 4u);
  EXPECT_EQ(stats.full_accept_rounds, 4u);
  EXPECT_EQ(device_->ledger_length(), 5u + 8u);
  EXPECT_EQ(device_->ledger_bank(), static_cast<int>(device_->ledger_length() % 3));
}

TEST_F(Qwen38GraphEngineTest, WithoutMtpEveryCycleCommitsOneToken) {
  device_ = std::make_shared<ScriptedQwen38Device>(/*capacity=*/68, /*mtp=*/false);
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(4), /*max_generated_tokens=*/5);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(4, 5));
  EXPECT_EQ(device_->CallCount("wait"), 4u);
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_proposed, 0u);
  EXPECT_EQ(stats.target_forward_passes, 4u);
  EXPECT_EQ(stats.tokens_emitted, 5u);
}

// ---- prefill, cancellation, queueing ---------------------------------------------------------

TEST_F(Qwen38GraphEngineTest, ThePromptIsIngestedInEvenChunksOverSeveralSteps) {
  model_->config_->model.qwen38_graph.prefill_chunk_tokens = 5;  // rounded down to 4: S=2 pairs stay aligned
  auto engine = MakeEngine();
  const auto prompt = Prompt(11);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/2);

  std::array<EngineEvent, 8> buffer;
  EXPECT_EQ(engine->Run(buffer), 0u);  // chunk 1: no event
  EXPECT_EQ(engine->Run(buffer), 0u);  // chunk 2
  EXPECT_TRUE(engine->HasPendingRequests());
  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(device_->CallCount("prefill:4"), 2u);
  EXPECT_EQ(device_->CallCount("prefill:3"), 1u);
  EXPECT_EQ(device_->ingested, prompt);
  EXPECT_EQ(outcome.tokens, Expected(11, 2));
}

TEST_F(Qwen38GraphEngineTest, CancelDuringPrefillStopsBeforeDecodingAndTheNextRequestStartsClean) {
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  const uint64_t turn = request->BeginTurn(Prompt(11), options);
  std::array<EngineEvent, 8> buffer;
  ASSERT_EQ(engine->Run(buffer), 0u);  // first chunk only

  EXPECT_TRUE(engine->CancelRequest(request, turn));
  const size_t count = engine->Run(buffer);
  ASSERT_EQ(count, 1u);
  EXPECT_EQ(buffer[0].finish_reason, GenerationFinishReason::Canceled);
  EXPECT_FALSE(engine->HasPendingRequests());
  EXPECT_EQ(device_->CallCount("begin:1") + device_->CallCount("begin:0"), 0u);

  auto next = Begin(*engine, Prompt(6, 20), /*max_generated_tokens=*/3);
  const auto outcome = RunToCompletion(*engine, next);
  EXPECT_EQ(outcome.tokens, Expected(6, 3));
  EXPECT_EQ(device_->ingested, Prompt(6, 20));  // the canceled prompt is gone from the device
}

TEST_F(Qwen38GraphEngineTest, CancelWithACycleInFlightIsSafeAndTheNextRequestWaitsForIt) {
  model_->config_->model.qwen38_graph.prefill_chunk_tokens = 64;  // the first step ingests the whole prompt
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  const uint64_t turn = request->BeginTurn(Prompt(5), options);
  std::array<EngineEvent, 8> buffer;
  ASSERT_GE(engine->Run(buffer), 1u);  // prefill + first token (the first cycle is now in flight)
  ASSERT_TRUE(device_->inflight());

  EXPECT_TRUE(engine->CancelRequest(request, turn));
  ASSERT_EQ(engine->Run(buffer), 1u);
  EXPECT_EQ(buffer[0].finish_reason, GenerationFinishReason::Canceled);
  EXPECT_TRUE(device_->inflight());  // still running on the device

  auto next = Begin(*engine, Prompt(5, 25), /*max_generated_tokens=*/4);
  const auto outcome = RunToCompletion(*engine, next);
  EXPECT_EQ(outcome.tokens, Expected(5, 4));
  EXPECT_GE(device_->CallCount("reset"), 1u);
}

TEST_F(Qwen38GraphEngineTest, AFurtherRequestQueuesBehindTheResidentOneAndRunsAfterIt) {
  auto engine = MakeEngine();
  auto first = Begin(*engine, Prompt(5), /*max_generated_tokens=*/4);
  auto second = Begin(*engine, Prompt(6, 25), /*max_generated_tokens=*/3);

  std::array<EngineEvent, 8> buffer;
  std::vector<int32_t> first_tokens, second_tokens;
  std::vector<const Request*> order;
  while (engine->HasPendingRequests()) {
    const size_t count = engine->Run(buffer);
    for (size_t i = 0; i < count; ++i) {
      if (!(buffer[i].flags & EngineEventFlagToken)) continue;
      (buffer[i].request == first ? first_tokens : second_tokens).push_back(buffer[i].token);
      order.push_back(buffer[i].request.get());
    }
  }

  EXPECT_EQ(first_tokens, Expected(5, 4));
  EXPECT_EQ(second_tokens, Expected(6, 3));
  // Every event of the first request precedes every event of the second.
  const auto last_first = std::find(order.rbegin(), order.rend(), first.get());
  const auto first_second = std::find(order.begin(), order.end(), second.get());
  EXPECT_LT(std::distance(order.begin(), last_first.base()), std::distance(order.begin(), first_second) + 1);
  EXPECT_GE(device_->CallCount("reset"), 1u);
  EXPECT_EQ(device_->ingested, Prompt(6, 25));
  EXPECT_EQ(engine->GetCapabilities().configured_max_batch_size, 1u);
}

// ---- continuation, rewind --------------------------------------------------------------------

TEST_F(Qwen38GraphEngineTest, AContinuedTurnReIngestsTheWholeHistory) {
  auto engine = MakeEngine();
  const auto prompt = Prompt(5);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/3);
  ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(5, 3));
  const size_t resets = device_->CallCount("reset");

  const std::vector<int32_t> follow_up{30, 31};
  TurnOptions options = Limit(2);
  options.request = request;
  request->BeginTurn(follow_up, options);
  const auto outcome = RunToCompletion(*engine, request);

  std::vector<int32_t> history = prompt;
  for (const int32_t token : Expected(5, 3)) history.push_back(token);
  history.insert(history.end(), follow_up.begin(), follow_up.end());
  EXPECT_EQ(device_->CallCount("reset"), resets + 1);
  EXPECT_EQ(device_->ingested, history);
  EXPECT_EQ(outcome.tokens, Expected(history.size(), 2));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::TurnLimit);
}

TEST_F(Qwen38GraphEngineTest, RewindToTheStartOfATurnReleasesTheSlotAndReplaysTheRetainedPrefix) {
  auto engine = MakeEngine();
  const auto prompt = Prompt(5);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/3);
  ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(5, 3));
  TurnOptions options = Limit(2);
  options.request = request;
  const uint64_t discarded_turn = request->BeginTurn(std::array<int32_t, 2>{30, 31}, options);
  ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(10, 2));

  request->RewindToStartOfTurn(discarded_turn);
  EXPECT_FALSE(engine->HasPendingRequests());
  EXPECT_EQ(request->CurrentSequenceLength(), 8);

  const std::vector<int32_t> replacement{40};
  options = Limit(2);
  options.request = request;
  request->BeginTurn(replacement, options);
  const auto outcome = RunToCompletion(*engine, request);

  std::vector<int32_t> history = prompt;
  for (const int32_t token : Expected(5, 3)) history.push_back(token);
  history.push_back(40);
  EXPECT_EQ(device_->ingested, history);
  EXPECT_EQ(outcome.tokens, Expected(history.size(), 2));
}

// ---- dense (fast) prefill: chunk streaming and hand-off ---------------------------------------

TEST_F(Qwen38GraphEngineTest, ADeviceWithAPreferredChunkReceivesThePromptInChunksOfThatSizeAndHandsOffOnce) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 8;  // replaces prefill_chunk_tokens (4) of the configuration
  auto engine = MakeEngine();
  const auto prompt = Prompt(20);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/5);

  std::array<EngineEvent, 8> buffer;
  EXPECT_EQ(engine->Run(buffer), 0u);  // chunk 1 of 3: no event, no hand-off yet
  EXPECT_EQ(device_->handoffs, 0u);
  EXPECT_EQ(engine->Run(buffer), 0u);  // chunk 2
  EXPECT_EQ(device_->handoffs, 0u);
  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(device_->CallCount("prefill:8"), 2u);
  EXPECT_EQ(device_->CallCount("prefill:4"), 1u);
  EXPECT_EQ(device_->ingested, prompt);
  EXPECT_EQ(device_->handoffs, 1u);
  EXPECT_EQ(device_->handoff_length, 20u);
  EXPECT_EQ(outcome.tokens, Expected(20, 5));
}

TEST_F(Qwen38GraphEngineTest, DenseChunksMayHaveAnOddSizeAndTheLastChunkEndsTheStreamExactly) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 7;
  auto engine = MakeEngine();
  const auto prompt = Prompt(15);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/3);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(device_->CallCount("prefill:7"), 2u);
  EXPECT_EQ(device_->CallCount("prefill:1"), 1u);
  EXPECT_EQ(device_->ingested, prompt);
  EXPECT_EQ(outcome.tokens, Expected(15, 3));
}

TEST_F(Qwen38GraphEngineTest, ADensePromptShorterThanTheChunkIsOneChunk) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 64;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/2);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(device_->CallCount("prefill:5"), 1u);
  EXPECT_EQ(outcome.tokens, Expected(5, 2));
}

TEST_F(Qwen38GraphEngineTest, ADensePromptOfExactlyTheChunkSizeHasNoEmptyTailChunk) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 8;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(16), /*max_generated_tokens=*/2);

  RunToCompletion(*engine, request);

  EXPECT_EQ(device_->CallCount("prefill:8"), 2u);
  EXPECT_EQ(device_->CallCount("prefill:0"), 0u);
}

TEST_F(Qwen38GraphEngineTest, Dense8192TokenChunksPreserveBoundaryAndTailIngestion) {
  for (const size_t length : {8191u, 8192u, 8193u, 16385u}) {
    SCOPED_TRACE(length);
    device_ = std::make_shared<ScriptedQwen38Device>(33028);
    device_->dense_prefill = true;
    device_->preferred_chunk = 8192;
    model_->config_->model.context_length = 33024;
    model_->config_->search.max_length = 33024;
    auto engine = MakeEngine();
    const auto prompt = Prompt(length);
    auto request = Begin(*engine, prompt, /*max_generated_tokens=*/3);
    const auto outcome = RunToCompletion(*engine, request);
    EXPECT_EQ(device_->CallCount("prefill:8192"), length / 8192);
    if (length % 8192 != 0) {
      EXPECT_EQ(device_->CallCount("prefill:" + std::to_string(length % 8192)), 1u);
    }
    EXPECT_EQ(device_->CallCount("prefill:0"), 0u);
    EXPECT_EQ(device_->ingested, prompt);
    EXPECT_EQ(device_->handoffs, 1u);
    EXPECT_EQ(device_->handoff_length, length);
    EXPECT_EQ(outcome.tokens, Expected(length, 3));
    EXPECT_EQ(outcome.failure_flags, 0u);
  }
}

TEST_F(Qwen38GraphEngineTest, ContinuationAndRewindReIngestTheHistoryThroughTheDenseChunksAndHandOffAgain) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 4;
  auto engine = MakeEngine();
  const auto prompt = Prompt(5);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/3);
  ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(5, 3));
  ASSERT_EQ(device_->handoffs, 1u);

  TurnOptions options = Limit(2);
  options.request = request;
  const uint64_t turn = request->BeginTurn(std::array<int32_t, 2>{30, 31}, options);
  ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(10, 2));
  EXPECT_EQ(device_->handoffs, 2u);
  EXPECT_EQ(device_->handoff_length, 10u);  // 5 prompt + 3 generated + 2 follow-up

  request->RewindToStartOfTurn(turn);
  options = Limit(2);
  options.request = request;
  request->BeginTurn(std::array<int32_t, 1>{40}, options);
  const auto outcome = RunToCompletion(*engine, request);
  std::vector<int32_t> history = prompt;
  for (const int32_t token : Expected(5, 3)) history.push_back(token);
  history.push_back(40);
  EXPECT_EQ(device_->ingested, history);
  EXPECT_EQ(device_->handoffs, 3u);
  EXPECT_EQ(device_->handoff_length, history.size());
  EXPECT_EQ(outcome.tokens, Expected(history.size(), 2));
}

TEST_F(Qwen38GraphEngineTest, AFirstTokenThatEndsTheTurnSkipsTheHandOff) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 4;
  device_->overrides[5] = kEos;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_TRUE(outcome.tokens.empty());
  EXPECT_EQ(device_->handoffs, 0u);
  EXPECT_EQ(device_->CallCount("begin:0"), 1u);
  EXPECT_EQ(device_->status_checks, 0u);
}

TEST_F(Qwen38GraphEngineTest, CancelBetweenDenseChunksNeverHandsOffAndTheNextRequestStartsClean) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 4;
  auto engine = MakeEngine();
  auto first = engine->CreateRequest();
  TurnOptions first_options;
  first_options.request = first;
  const uint64_t turn = first->BeginTurn(Prompt(12), first_options);
  std::array<EngineEvent, 8> buffer;
  ASSERT_EQ(engine->Run(buffer), 0u);  // chunk 1 of 3
  EXPECT_TRUE(engine->CancelRequest(first, turn));
  while (engine->HasPendingRequests()) engine->Run(buffer);
  EXPECT_EQ(device_->handoffs, 0u);

  const auto prompt = Prompt(9, 20);
  auto second = Begin(*engine, prompt, /*max_generated_tokens=*/3);
  const auto outcome = RunToCompletion(*engine, second);
  EXPECT_EQ(device_->ingested, prompt);  // the chunk of the cancelled request is gone
  EXPECT_EQ(device_->handoffs, 1u);
  EXPECT_EQ(outcome.tokens, Expected(9, 3));
}

// ---- end-of-decode device status ---------------------------------------------------------------

TEST_F(Qwen38GraphEngineTest, TheDeviceStatusIsCheckedOnceWhenATurnEndsWithNothingInFlight) {
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/7);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 7));
  EXPECT_EQ(device_->status_checks, 1u);
}

TEST_F(Qwen38GraphEngineTest, DeviceErrorBitsAtTheEndOfATurnAreReportedAsAnExecutionFailure) {
  device_->fail_status_check = true;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/3);

  std::array<EngineEvent, 8> buffer;
  bool saw_failure = false;
  for (int i = 0; i < 100 && engine->HasPendingRequests() && !saw_failure; ++i) {
    const size_t count = engine->Run(buffer);
    for (size_t j = 0; j < count; ++j) {
      if (buffer[j].flags & EngineEventFlagFailed) {
        saw_failure = true;
        EXPECT_EQ(buffer[j].error_code, EngineErrorCode::EngineExecutionFailure);
      }
    }
  }
  EXPECT_TRUE(saw_failure);
}

TEST_F(Qwen38GraphEngineTest, DeviceErrorBitsAfterAStopStringTurnAreSurfacedBeforeReset) {
  device_->overrides[5] = 5;
  device_->overrides[6] = 6;
  device_->fail_status_check = true;
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  options.max_generated_tokens = 20;
  options.stop_strings = {"STOP"};
  const uint64_t turn = request->BeginTurn(Prompt(5), options);

  const auto outcome = RunToCompletion(*engine, request);
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::StopString);

  // Releasing the slot for rewind (or starting a new turn/request) drains the look-ahead cycle and checks decode status.
  EXPECT_THROW(request->RewindToStartOfTurn(turn), std::runtime_error);
  EXPECT_EQ(device_->status_checks, 1u);
}

TEST_F(Qwen38GraphEngineTest, DeviceErrorBitsAfterACanceledTurnAreSurfacedOnTheNextRun) {
  model_->config_->model.qwen38_graph.prefill_chunk_tokens = 64;
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  const uint64_t turn = request->BeginTurn(Prompt(5), options);
  std::array<EngineEvent, 8> buffer;
  ASSERT_GE(engine->Run(buffer), 1u);
  ASSERT_TRUE(device_->inflight());

  device_->fail_status_check = true;
  EXPECT_TRUE(engine->CancelRequest(request, turn));
  ASSERT_EQ(engine->Run(buffer), 1u);
  EXPECT_EQ(buffer[0].finish_reason, GenerationFinishReason::Canceled);

  // When the next request claims the slot, Release() drains the canceled cycle and surfaces the device status error.
  auto next = Begin(*engine, Prompt(5, 25), /*max_generated_tokens=*/4);
  bool saw_failure = false;
  for (int i = 0; i < 10 && engine->HasPendingRequests() && !saw_failure; ++i) {
    const size_t count = engine->Run(buffer);
    for (size_t j = 0; j < count; ++j) {
      if (buffer[j].flags & EngineEventFlagFailed) {
        saw_failure = true;
        EXPECT_EQ(buffer[j].error_code, EngineErrorCode::EngineExecutionFailure);
      }
    }
  }
  EXPECT_TRUE(saw_failure);
  EXPECT_EQ(device_->status_checks, 1u);
}

// ---- capabilities and errors -----------------------------------------------------------------

TEST_F(Qwen38GraphEngineTest, CapabilitiesAreHonest) {
  auto engine = MakeEngine();
  const auto capabilities = engine->GetCapabilities();
  EXPECT_EQ(capabilities.configured_max_batch_size, 1u);
  EXPECT_EQ(capabilities.max_request_length, static_cast<uint64_t>(device_->Capacity() - Qwen38DecodeDevice::kReservedTail));
  EXPECT_EQ(engine->MaxDraftTokensPerStep(), 0u);
  EXPECT_FALSE(engine->PrefixCacheStats().has_value());

  RequestOptions too_long;
  too_long.max_session_tokens = capabilities.max_request_length + 1;
  try {
    engine->CreateRequest(too_long);
    FAIL() << "a session longer than the KV capacity must be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("max_request_length"), std::string::npos) << error.what();
  }
  RequestOptions at_limit;
  at_limit.max_session_tokens = capabilities.max_request_length;
  EXPECT_NO_THROW(engine->CreateRequest(at_limit));
}

TEST_F(Qwen38GraphEngineTest, UnsupportedTurnOptionsAreRejectedBeforeTheRequestChanges) {
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  const auto prompt = Prompt(4);
  const auto expect_rejected = [&](const char* label, TurnOptions options, const char* expected_fragment) {
    options.request = request;
    try {
      request->BeginTurn(prompt, options);
      ADD_FAILURE() << label << " must be rejected";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string{error.what()}.find(expected_fragment), std::string::npos) << label << ": " << error.what();
    }
    EXPECT_TRUE(request->IsAwaitingFirstTurn()) << label << " mutated the request";
  };

  TurnOptions sampled;
  sampled.do_sample = true;
  sampled.top_k = 5;
  sampled.temperature = 0.7f;
  expect_rejected("sampling", sampled, "greedily");

  TurnOptions penalty;
  penalty.repetition_penalty = 1.2f;
  expect_rejected("repetition_penalty", penalty, "repetition_penalty");

  TurnOptions ngram;
  ngram.no_repeat_ngram_size = 3;
  expect_rejected("no_repeat_ngram_size", ngram, "no_repeat_ngram_size");

  TurnOptions minimum;
  minimum.min_generated_tokens = 4;
  expect_rejected("min_generated_tokens", minimum, "min_generated_tokens");

  TurnOptions seeded;
  seeded.seed = 7;
  expect_rejected("seed", seeded, "seed");

  TurnOptions guided;
  guided.guidance_type = "lark_grammar";
  guided.guidance_data = "start: \"a\"";
  expect_rejected("guidance", guided, "guidance");

  // The same request still starts a plain greedy turn.
  TurnOptions greedy = Limit(2);
  greedy.request = request;
  EXPECT_NO_THROW(request->BeginTurn(prompt, greedy));
  EXPECT_EQ(RunToCompletion(*engine, request).tokens, Expected(4, 2));
}

TEST_F(Qwen38GraphEngineTest, ADeviceFailureIsFatalAndReportedThroughTheEngineErrorTaxonomy) {
  model_->config_->model.qwen38_graph.prefill_chunk_tokens = 64;  // the first step ingests the whole prompt
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/10);
  std::array<EngineEvent, 8> buffer;
  ASSERT_GE(engine->Run(buffer), 1u);  // prefill and the first token succeed
  device_->fail_wait = true;

  const size_t count = engine->Run(buffer);
  ASSERT_GE(count, 1u);
  bool saw_failure = false;
  for (size_t i = 0; i < count; ++i) {
    if (buffer[i].flags & EngineEventFlagFailed) {
      saw_failure = true;
      EXPECT_EQ(buffer[i].error_code, EngineErrorCode::EngineExecutionFailure);
    }
  }
  EXPECT_TRUE(saw_failure);
  EXPECT_THROW(engine->Run(buffer), std::exception);  // the Engine is no longer healthy
}

TEST_F(Qwen38GraphEngineTest, DestroyingTheEngineDrainsTheDevice) {
  model_->config_->model.qwen38_graph.prefill_chunk_tokens = 64;  // the first step ingests the whole prompt
  {
    auto engine = MakeEngine();
    auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/10);
    std::array<EngineEvent, 8> buffer;
    ASSERT_GE(engine->Run(buffer), 1u);
    ASSERT_TRUE(device_->inflight());
  }
  EXPECT_FALSE(device_->inflight());
  EXPECT_GE(device_->CallCount("drain"), 1u);
}

TEST_F(Qwen38GraphEngineTest, ANewEngineOverAUsedDeviceStartsFromACleanDevice) {
  model_->config_->model.qwen38_graph.prefill_chunk_tokens = 64;
  {
    auto engine = MakeEngine();
    auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/3);
    ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(5, 3));
  }
  const size_t resets = device_->CallCount("reset");

  // The device kept the first Engine's decode state; the next Engine must not prefill on top of it.
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(6, 20), /*max_generated_tokens=*/3);
  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(device_->CallCount("reset"), resets + 1);
  EXPECT_EQ(device_->ingested, Prompt(6, 20));
  EXPECT_EQ(outcome.tokens, Expected(6, 3));
}

// ---- cache manager contract ------------------------------------------------------------------

TEST_F(Qwen38GraphEngineTest, TheCacheManagerHoldsExactlyOneRow) {
  auto engine = MakeEngine();
  auto session = std::make_shared<Qwen38GraphSession>(device_, Qwen38GraphSession::Options{});
  Qwen38GraphCacheManager cache{model_, session};
  auto a = engine->CreateRequest();
  auto b = engine->CreateRequest();
  a->BeginTurn(Prompt(3));
  b->BeginTurn(Prompt(3));

  EXPECT_FALSE(cache.CanAllocate({}));
  EXPECT_FALSE(cache.CanAllocate({a, b}));
  ASSERT_TRUE(cache.CanAllocate({a}));
  cache.Allocate({a});
  EXPECT_EQ(cache.ResidentRequestCount(), 1u);
  EXPECT_TRUE(cache.IsResident(a));
  EXPECT_FALSE(cache.IsResident(b));
  EXPECT_FALSE(cache.CanAllocate({b})) << "an active resident row cannot be replaced";
  EXPECT_THROW(cache.Allocate({b}), std::runtime_error);

  std::vector<std::shared_ptr<Request>> wrong{b};
  EXPECT_THROW(cache.Deallocate(wrong), std::runtime_error);
  std::vector<std::shared_ptr<Request>> right{a};
  cache.Deallocate(right);
  EXPECT_EQ(cache.ResidentRequestCount(), 0u);
  EXPECT_TRUE(cache.CanAllocate({b}));
  EXPECT_EQ(cache.MaxBatchSize(), 1u);
  EXPECT_EQ(cache.MaxStaticRequestLength(), static_cast<size_t>(device_->Capacity() - Qwen38DecodeDevice::kReservedTail));
  EXPECT_FALSE(cache.SupportsDynamicBatching());
}

TEST_F(Qwen38GraphEngineTest, DynamicBatchingConfigurationIsRejected) {
  model_->config_->engine.dynamic_batching = Config::Engine::DynamicBatching{};
  EXPECT_THROW(CreateQwen38GraphDependencies(model_, device_), std::runtime_error);
  model_->config_->engine.dynamic_batching.reset();
  model_->config_->model.qwen38_graph.prefill_chunk_tokens = 1;
  EXPECT_THROW(CreateQwen38GraphDependencies(model_, device_), std::runtime_error);
}

TEST_F(Qwen38GraphEngineTest, GenericPrefillChunkingIsRejected) {
  model_->config_->search.chunk_size = 4;
  EXPECT_THROW(CreateQwen38GraphDependencies(model_, device_), std::runtime_error);
}

TEST(Qwen38GraphConfigTest, ParsesEnginePrefillChunkSize) {
  const Config config{
      fs::path{std::string{MODEL_PATH "engine/dummy-decoder"}},
      R"({"model":{"qwen38_graph":{"prefill_chunk_tokens":512}}})"};
  EXPECT_EQ(config.model.qwen38_graph.prefill_chunk_tokens, 512);
}

// Host cost of one Engine step (Run) with a device that answers instantly: everything the Engine adds around a cycle. Run
// manually: engine_unit_tests --gtest_also_run_disabled_tests --gtest_filter='*RunOverhead*'.
TEST_F(Qwen38GraphEngineTest, DISABLED_RunOverheadPerCycle) {
  device_ = std::make_shared<ScriptedQwen38Device>(/*capacity=*/33028);
  model_->config_->model.qwen38_graph.prefill_chunk_tokens = 2048;
  auto engine = MakeEngine();
  const size_t generated = 20000;
  auto request = Begin(*engine, Prompt(16), generated, /*max_session_tokens=*/33024);
  std::array<EngineEvent, 16> buffer;
  size_t runs = 0, tokens = 0;
  const auto start = std::chrono::steady_clock::now();
  while (engine->HasPendingRequests()) {
    const size_t count = engine->Run(buffer);
    ++runs;
    tokens += count;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << "Engine::Run: " << runs << " runs, " << tokens << " events in " << seconds * 1e3 << " ms = "
            << seconds / static_cast<double>(runs) * 1e6 << " us per run" << std::endl;
  EXPECT_EQ(tokens, generated + 0);
}

}  // namespace
}  // namespace Generators::test
