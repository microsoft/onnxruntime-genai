// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "models/qwen38_ort_env.h"

namespace Generators::test {

namespace {

constexpr const char* kVariable = "GENAI_QWEN38_ORT_ENV_TEST_VARIABLE";

// Reads through the C runtime the scope writes with.
std::optional<std::string> Read(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) return std::nullopt;
  return std::string{value};
}

// ORT resets the one-shot notices it consumed with setenv/_putenv_s from inside the Run.
void WriteLikeOrt(const char* name, const char* value) {
#if defined(_WIN32)
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

std::vector<Qwen38OrtEnvVariable> Publish(const std::string& value, bool only_if_unset = false) {
  return {Qwen38OrtEnvVariable{kVariable, value, only_if_unset}};
}

const Qwen38OrtEnvVariable* Find(const std::vector<Qwen38OrtEnvVariable>& variables, const char* name) {
  for (const auto& variable : variables)
    if (variable.name == name) return &variable;
  return nullptr;
}

}  // namespace

class Qwen38OrtEnvScopeTest : public ::testing::Test {
 protected:
  void SetUp() override { ASSERT_FALSE(Read(kVariable)) << kVariable << " must not be set by the test environment"; }
};

TEST_F(Qwen38OrtEnvScopeTest, PublishesForTheScopeAndLeavesAnUnsetVariableUnset) {
  {
    Qwen38OrtEnvScope scope{Publish("1")};
    EXPECT_EQ(Read(kVariable), "1");
  }
  EXPECT_FALSE(Read(kVariable));
}

TEST_F(Qwen38OrtEnvScopeTest, RestoresThePreviousValueOfNestedScopes) {
  {
    Qwen38OrtEnvScope outer{Publish("outer")};
    {
      Qwen38OrtEnvScope inner{Publish("inner")};
      EXPECT_EQ(Read(kVariable), "inner");
    }
    EXPECT_EQ(Read(kVariable), "outer");
  }
  EXPECT_FALSE(Read(kVariable));
}

TEST_F(Qwen38OrtEnvScopeTest, ARepeatedVariableUnwindsInReverseOrder) {
  const std::vector<Qwen38OrtEnvVariable> twice{{kVariable, "first", false}, {kVariable, "second", false}};
  {
    Qwen38OrtEnvScope scope{twice};
    EXPECT_EQ(Read(kVariable), "second");
  }
  EXPECT_FALSE(Read(kVariable));
}

TEST_F(Qwen38OrtEnvScopeTest, OnlyIfUnsetNeverOverridesAValueSetByTheUser) {
  {
    Qwen38OrtEnvScope user{Publish("user")};
    {
      Qwen38OrtEnvScope scope{Publish("1", /*only_if_unset=*/true)};
      EXPECT_EQ(Read(kVariable), "user");
    }
    EXPECT_EQ(Read(kVariable), "user");
  }
  {
    Qwen38OrtEnvScope scope{Publish("1", /*only_if_unset=*/true)};
    EXPECT_EQ(Read(kVariable), "1");
  }
  EXPECT_FALSE(Read(kVariable));
}

TEST_F(Qwen38OrtEnvScopeTest, RestoresWhenTheRunThrows) {
  EXPECT_THROW(
      {
        Qwen38OrtEnvScope scope{Publish("1")};
        throw std::runtime_error("Run failed");
      },
      std::runtime_error);
  EXPECT_FALSE(Read(kVariable));
}

TEST_F(Qwen38OrtEnvScopeTest, ReportsWhetherThePublishedValueSurvivedTheRun) {
  Qwen38OrtEnvScope scope{Publish("1")};
  EXPECT_TRUE(scope.StillPublished(kVariable));
  WriteLikeOrt(kVariable, "0");
  EXPECT_FALSE(scope.StillPublished(kVariable));
  EXPECT_FALSE(scope.StillPublished("GENAI_QWEN38_ORT_ENV_TEST_NEVER_PUBLISHED"));
}

TEST_F(Qwen38OrtEnvScopeTest, ScopesOfDifferentThreadsNeverOverlap) {
  std::atomic<int> inside{0};
  std::atomic<int> violations{0};
  auto worker = [&](const std::string& value) {
    for (int i = 0; i < 300; ++i) {
      Qwen38OrtEnvScope scope{Publish(value)};
      if (inside.fetch_add(1) != 0) ++violations;
      if (Read(kVariable) != value) ++violations;
      std::this_thread::yield();
      if (Read(kVariable) != value) ++violations;
      inside.fetch_sub(1);
    }
  };
  std::thread first{worker, "first"};
  std::thread second{worker, "second"};
  first.join();
  second.join();
  EXPECT_EQ(violations.load(), 0);
  EXPECT_FALSE(Read(kVariable));
}

TEST(Qwen38OrtRunHintsTest, TheNewTurnNoticeStaysPendingUntilOrtResetsIt) {
  const auto before = Read(Qwen38OrtEnv::kNewTurn);
  Qwen38OrtRunHints hints;
  EXPECT_TRUE(hints.Collect(nullptr).empty());
  hints.BeginTurn();
  const auto pending = hints.Collect(nullptr);
  ASSERT_EQ(pending.size(), 1u);
  EXPECT_EQ(pending[0].name, Qwen38OrtEnv::kNewTurn);
  EXPECT_EQ(pending[0].value, "1");

  // A Run that does not consume the notice (a session without a QMoE layer that reads it) leaves it pending.
  std::optional<std::string> seen;
  RunWithQwen38OrtHints(hints, nullptr, [&] { seen = Read(Qwen38OrtEnv::kNewTurn); });
  EXPECT_EQ(seen, "1");
  EXPECT_NE(Find(hints.Collect(nullptr), Qwen38OrtEnv::kNewTurn), nullptr);

  // The Run in which ORT consumes it resets the variable; later Runs carry no notice.
  RunWithQwen38OrtHints(hints, nullptr, [] { WriteLikeOrt(Qwen38OrtEnv::kNewTurn, "0"); });
  EXPECT_TRUE(hints.Collect(nullptr).empty());
  EXPECT_EQ(Read(Qwen38OrtEnv::kNewTurn), before) << "the environment is restored after the Run";
}

TEST(Qwen38OrtRunHintsTest, TheHandoffNoticeStaysPendingUntilOrtResetsIt) {
  const auto before = Read(Qwen38OrtEnv::kHandoffDone);
  Qwen38OrtRunHints hints;
  hints.MarkHandoff();
  const auto pending = hints.Collect(nullptr);
  ASSERT_EQ(pending.size(), 1u);
  EXPECT_EQ(pending[0].name, Qwen38OrtEnv::kHandoffDone);
  EXPECT_EQ(pending[0].value, "1");

  RunWithQwen38OrtHints(hints, nullptr, [] {});
  EXPECT_NE(Find(hints.Collect(nullptr), Qwen38OrtEnv::kHandoffDone), nullptr);

  RunWithQwen38OrtHints(hints, nullptr, [] { WriteLikeOrt(Qwen38OrtEnv::kHandoffDone, "0"); });
  EXPECT_TRUE(hints.Collect(nullptr).empty());
  EXPECT_EQ(Read(Qwen38OrtEnv::kHandoffDone), before);
}

TEST(Qwen38OrtRunHintsTest, ANewTurnDropsEverythingOfThePreviousOne) {
  Qwen38OrtRunHints hints;
  hints.ExpectPrefillTokens(4096);
  hints.MarkHandoff();
  hints.BeginTurn();
  const auto variables = hints.Collect(nullptr);
  EXPECT_NE(Find(variables, Qwen38OrtEnv::kNewTurn), nullptr);
  EXPECT_EQ(Find(variables, Qwen38OrtEnv::kHandoffDone), nullptr);
  EXPECT_EQ(hints.ExpectedPrefillTokens(), 0u);
}

TEST(Qwen38OrtRunHintsTest, HandingOffSpendsTheAnnouncedPromptLength) {
  Qwen38OrtRunHints hints;
  hints.BeginTurn();
  hints.ExpectPrefillTokens(4096);
  hints.MarkHandoff();
  EXPECT_EQ(hints.ExpectedPrefillTokens(), 0u);
}

TEST(Qwen38OrtRunHintsTest, APromptChunkReportsTheAnnouncedTotalNeverLessThanWhatWasIngested) {
  Qwen38OrtRunHints hints;
  hints.BeginTurn();
  EXPECT_EQ(hints.BeginPromptChunk(0, 100), 100u);
  EXPECT_EQ(hints.BeginPromptChunk(100, 50), 150u);

  hints.BeginTurn();
  hints.ExpectPrefillTokens(1000);
  EXPECT_EQ(hints.BeginPromptChunk(0, 100), 1000u);
  EXPECT_EQ(hints.BeginPromptChunk(900, 200), 1100u);
}

TEST(Qwen38OrtRunHintsTest, OnlyTheFirstPromptChunkRearmsTheNewTurnNotice) {
  Qwen38OrtRunHints hints;
  hints.BeginTurn();
  RunWithQwen38OrtHints(hints, nullptr, [] { WriteLikeOrt(Qwen38OrtEnv::kNewTurn, "0"); });  // consumed early
  EXPECT_EQ(Find(hints.Collect(nullptr), Qwen38OrtEnv::kNewTurn), nullptr);
  hints.BeginPromptChunk(0, 10);
  EXPECT_NE(Find(hints.Collect(nullptr), Qwen38OrtEnv::kNewTurn), nullptr);
  RunWithQwen38OrtHints(hints, nullptr, [] { WriteLikeOrt(Qwen38OrtEnv::kNewTurn, "0"); });
  hints.BeginPromptChunk(10, 10);
  EXPECT_EQ(Find(hints.Collect(nullptr), Qwen38OrtEnv::kNewTurn), nullptr);
}

TEST(Qwen38OrtRunHintsTest, APromptChunkWithdrawsAHandoffNoticeOfTheEarlierPrompt) {
  Qwen38OrtRunHints hints;
  hints.MarkHandoff();
  hints.BeginPromptChunk(0, 10);
  EXPECT_EQ(Find(hints.Collect(nullptr), Qwen38OrtEnv::kHandoffDone), nullptr);
}

TEST(Qwen38OrtRunHintsTest, PromptRunsPublishTheTotalAndTheCompactExpertsOfTheirChunk) {
  Qwen38OrtRunHints hints;
  const Qwen38PromptRunHints chunk{4096, 128};
  const auto with_chunk = hints.Collect(&chunk);
  ASSERT_NE(Find(with_chunk, Qwen38OrtEnv::kPrefillTotalTokens), nullptr);
  EXPECT_EQ(Find(with_chunk, Qwen38OrtEnv::kPrefillTotalTokens)->value, "4096");
  ASSERT_NE(Find(with_chunk, Qwen38OrtEnv::kCurrentMaxCompactExperts), nullptr);
  EXPECT_EQ(Find(with_chunk, Qwen38OrtEnv::kCurrentMaxCompactExperts)->value, "128");

  const Qwen38PromptRunHints unannounced{};
  EXPECT_TRUE(hints.Collect(&unannounced).empty());
  EXPECT_TRUE(hints.Collect(nullptr).empty());
}

TEST(Qwen38OrtRunHintsTest, WarmupIsAnnouncedOnlyWhileItRunsAndDeterminismStaysFromThenOn) {
  Qwen38OrtRunHints hints;
  EXPECT_TRUE(hints.Collect(nullptr).empty());
  EXPECT_FALSE(hints.InWarmup());

  hints.SetWarmup(true);
  EXPECT_TRUE(hints.InWarmup());
  const auto warmup = hints.Collect(nullptr);
  ASSERT_NE(Find(warmup, Qwen38OrtEnv::kInWarmup), nullptr);
  EXPECT_EQ(Find(warmup, Qwen38OrtEnv::kInWarmup)->value, "1");
  ASSERT_NE(Find(warmup, Qwen38OrtEnv::kForceDeterministicMoe), nullptr);
  EXPECT_EQ(Find(warmup, Qwen38OrtEnv::kForceDeterministicMoe)->value, "1");
  EXPECT_TRUE(Find(warmup, Qwen38OrtEnv::kForceDeterministicMoe)->only_if_unset) << "a value of the user wins";

  hints.SetWarmup(false);
  EXPECT_FALSE(hints.InWarmup());
  const auto after = hints.Collect(nullptr);
  EXPECT_EQ(Find(after, Qwen38OrtEnv::kInWarmup), nullptr);
  EXPECT_NE(Find(after, Qwen38OrtEnv::kForceDeterministicMoe), nullptr);
}

TEST(Qwen38OrtRunHintsTest, DeterministicMoeCanBeRequiredWithoutAWarmup) {
  Qwen38OrtRunHints hints;
  hints.RequireDeterministicMoe();
  EXPECT_FALSE(hints.InWarmup());
  const auto variables = hints.Collect(nullptr);
  EXPECT_EQ(Find(variables, Qwen38OrtEnv::kInWarmup), nullptr);
  ASSERT_NE(Find(variables, Qwen38OrtEnv::kForceDeterministicMoe), nullptr);
  EXPECT_EQ(Find(variables, Qwen38OrtEnv::kForceDeterministicMoe)->value, "1");
  EXPECT_TRUE(Find(variables, Qwen38OrtEnv::kForceDeterministicMoe)->only_if_unset) << "a value of the user wins";

  hints.BeginTurn();  // a new turn does not drop it
  EXPECT_NE(Find(hints.Collect(nullptr), Qwen38OrtEnv::kForceDeterministicMoe), nullptr);
}

TEST(Qwen38OrtRunHintsTest, TheWarmupVariablesLeaveNoTraceAfterTheRunAndNeverOverrideTheUser) {
  const auto in_warmup = Read(Qwen38OrtEnv::kInWarmup);
  const auto deterministic = Read(Qwen38OrtEnv::kForceDeterministicMoe);
  Qwen38OrtRunHints hints;
  hints.SetWarmup(true);
  std::optional<std::string> seen_in_warmup, seen_deterministic;
  RunWithQwen38OrtHints(hints, nullptr, [&] {
    seen_in_warmup = Read(Qwen38OrtEnv::kInWarmup);
    seen_deterministic = Read(Qwen38OrtEnv::kForceDeterministicMoe);
  });
  EXPECT_EQ(seen_in_warmup, "1");
  EXPECT_EQ(seen_deterministic, deterministic ? deterministic : std::optional<std::string>{"1"});
  EXPECT_EQ(Read(Qwen38OrtEnv::kInWarmup), in_warmup);
  EXPECT_EQ(Read(Qwen38OrtEnv::kForceDeterministicMoe), deterministic);

  const std::vector<Qwen38OrtEnvVariable> user{{Qwen38OrtEnv::kForceDeterministicMoe, "0", false}};
  Qwen38OrtEnvScope user_scope{user};
  RunWithQwen38OrtHints(hints, nullptr, [&] { seen_deterministic = Read(Qwen38OrtEnv::kForceDeterministicMoe); });
  EXPECT_EQ(seen_deterministic, "0");
}

TEST(Qwen38OrtRunHintsTest, TheHintsAreVisibleOnlyDuringTheirRun) {
  const auto before_new_turn = Read(Qwen38OrtEnv::kNewTurn);
  const auto before_total = Read(Qwen38OrtEnv::kPrefillTotalTokens);
  const auto before_compact = Read(Qwen38OrtEnv::kCurrentMaxCompactExperts);
  Qwen38OrtRunHints hints;
  hints.BeginTurn();
  const Qwen38PromptRunHints chunk{300, 64};
  std::optional<std::string> new_turn, total, compact;
  RunWithQwen38OrtHints(hints, &chunk, [&] {
    new_turn = Read(Qwen38OrtEnv::kNewTurn);
    total = Read(Qwen38OrtEnv::kPrefillTotalTokens);
    compact = Read(Qwen38OrtEnv::kCurrentMaxCompactExperts);
  });
  EXPECT_EQ(new_turn, "1");
  EXPECT_EQ(total, "300");
  EXPECT_EQ(compact, "64");
  EXPECT_EQ(Read(Qwen38OrtEnv::kNewTurn), before_new_turn);
  EXPECT_EQ(Read(Qwen38OrtEnv::kPrefillTotalTokens), before_total);
  EXPECT_EQ(Read(Qwen38OrtEnv::kCurrentMaxCompactExperts), before_compact);
}

TEST(Qwen38OrtRunHintsTest, AFailedRunRestoresTheEnvironmentAndKeepsTheNoticesPending) {
  const auto before = Read(Qwen38OrtEnv::kNewTurn);
  Qwen38OrtRunHints hints;
  hints.BeginTurn();
  EXPECT_THROW(RunWithQwen38OrtHints(hints, nullptr,
                                     [] {
                                       WriteLikeOrt(Qwen38OrtEnv::kNewTurn, "0");
                                       // Not unconditional on purpose: the bookkeeping after a Run that provably never
                                       // returns would be unreachable code, which MSVC reports as C4702.
                                       if (Read(Qwen38OrtEnv::kNewTurn) == "0") throw std::runtime_error("Run failed");
                                     }),
               std::runtime_error);
  EXPECT_EQ(Read(Qwen38OrtEnv::kNewTurn), before);
  EXPECT_NE(Find(hints.Collect(nullptr), Qwen38OrtEnv::kNewTurn), nullptr);
}

TEST(Qwen38OrtRunHintsTest, ThePromptLengthIsNeverReadBackFromTheEnvironment) {
  Qwen38OrtRunHints hints;
  hints.BeginTurn();
  hints.ExpectPrefillTokens(512);
  // Another engine or the caller leaving a stale (or hostile) value there changes nothing the engine publishes.
  const std::vector<Qwen38OrtEnvVariable> stale{{Qwen38OrtEnv::kPrefillTotalTokens, "999999", false}};
  Qwen38OrtEnvScope poisoned{stale};
  const size_t total = hints.BeginPromptChunk(0, 100);
  EXPECT_EQ(total, 512u);

  const Qwen38PromptRunHints chunk{total, 64};
  std::optional<std::string> seen;
  RunWithQwen38OrtHints(hints, &chunk, [&] { seen = Read(Qwen38OrtEnv::kPrefillTotalTokens); });
  EXPECT_EQ(seen, "512");
  EXPECT_EQ(Read(Qwen38OrtEnv::kPrefillTotalTokens), "999999") << "the previous value is restored, not clobbered";

  Qwen38OrtRunHints unannounced;
  EXPECT_EQ(unannounced.BeginPromptChunk(0, 100), 100u);
}

// ---- per slot ------------------------------------------------------------------------------------------------------------

namespace {

// ORT consumes the new-turn notice of the Run it reads it in.
void ConsumeNoticeLikeOrt(const char* name) {
  if (Read(name) == "1") WriteLikeOrt(name, "0");
}

struct SeenByRun {
  std::optional<std::string> new_turn, handoff, total, compact;
};

// One ORT Run that serves `participants`: records what it sees in the environment and consumes the notices like ORT.
SeenByRun RunLikeOrt(Qwen38OrtRunHints& hints, const Qwen38PromptRunHints* prompt, Qwen38SlotMask participants) {
  SeenByRun seen;
  RunWithQwen38OrtHints(
      hints, prompt,
      [&] {
        seen.new_turn = Read(Qwen38OrtEnv::kNewTurn);
        seen.handoff = Read(Qwen38OrtEnv::kHandoffDone);
        seen.total = Read(Qwen38OrtEnv::kPrefillTotalTokens);
        seen.compact = Read(Qwen38OrtEnv::kCurrentMaxCompactExperts);
        ConsumeNoticeLikeOrt(Qwen38OrtEnv::kNewTurn);
        ConsumeNoticeLikeOrt(Qwen38OrtEnv::kHandoffDone);
      },
      participants);
  return seen;
}

}  // namespace

TEST(Qwen38OrtRunHintsSlotTest, SlotMasksNameTheSlotsOfARun) {
  EXPECT_EQ(Qwen38SlotBit(0), 1u);
  EXPECT_EQ(Qwen38SlotBit(3), 8u);
  EXPECT_EQ(Qwen38FirstSlots(1), 1u);
  EXPECT_EQ(Qwen38FirstSlots(3), 7u);
  EXPECT_EQ(Qwen38FirstSlots(8), 0xFFu);
  EXPECT_EQ(Qwen38SlotMaskOf(std::vector<size_t>{0, 2}), 5u);
  EXPECT_EQ(Qwen38SlotMaskOf(std::vector<size_t>{}), 0u);
}

TEST(Qwen38OrtRunHintsSlotTest, TheSlotCountIsBoundedLikeTheDevice) {
  EXPECT_EQ(Qwen38OrtRunHints{}.SlotCount(), 1u);
  EXPECT_EQ(Qwen38OrtRunHints{8}.SlotCount(), 8u);
  EXPECT_THROW(Qwen38OrtRunHints{0}, std::invalid_argument);
  EXPECT_THROW(Qwen38OrtRunHints{9}, std::invalid_argument);
  Qwen38OrtRunHints hints{2};
  hints.ExpectPrefillTokens(1, 70);
  hints.SetSlotCount(4);  // the slots that exist keep their state
  EXPECT_EQ(hints.SlotCount(), 4u);
  EXPECT_EQ(hints.ExpectedPrefillTokens(1), 70u);
  EXPECT_EQ(hints.ExpectedPrefillTokens(3), 0u);
}

TEST(Qwen38OrtRunHintsSlotTest, AnUnknownSlotIsRejectedBeforeAnythingIsPublished) {
  Qwen38OrtRunHints hints{2};
  EXPECT_THROW(hints.BeginTurn(2), std::out_of_range);
  EXPECT_THROW(hints.ExpectPrefillTokens(2, 10), std::out_of_range);
  EXPECT_THROW(hints.ExpectedPrefillTokens(2), std::out_of_range);
  EXPECT_THROW(hints.BeginPromptChunk(2, 0, 10), std::out_of_range);
  EXPECT_THROW(hints.MarkHandoff(2), std::out_of_range);
  EXPECT_THROW(hints.Collect(nullptr, Qwen38SlotBit(2)), std::out_of_range);
  EXPECT_THROW(hints.Collect(nullptr, Qwen38SlotBit(1) | Qwen38SlotBit(5)), std::out_of_range);
  EXPECT_THROW(hints.Collect(nullptr, 0), std::out_of_range) << "a Run serves at least one slot";
  const std::vector<Qwen38OrtEnvVariable> none;
  const Qwen38OrtEnvScope scope{none};
  EXPECT_THROW(hints.RunReturned(scope, Qwen38SlotBit(2)), std::out_of_range);
}

TEST(Qwen38OrtRunHintsSlotTest, TheTurnStateOfASlotIsIndependentOfTheOthers) {
  Qwen38OrtRunHints hints{3};
  hints.BeginTurn(1);
  EXPECT_TRUE(hints.Collect(nullptr, Qwen38SlotBit(0)).empty());
  EXPECT_NE(Find(hints.Collect(nullptr, Qwen38SlotBit(1)), Qwen38OrtEnv::kNewTurn), nullptr);
  EXPECT_TRUE(hints.Collect(nullptr, Qwen38SlotBit(2)).empty());

  hints.MarkHandoff(2);
  EXPECT_EQ(Find(hints.Collect(nullptr, Qwen38SlotBit(0)), Qwen38OrtEnv::kHandoffDone), nullptr);
  EXPECT_EQ(Find(hints.Collect(nullptr, Qwen38SlotBit(1)), Qwen38OrtEnv::kHandoffDone), nullptr);
  EXPECT_NE(Find(hints.Collect(nullptr, Qwen38SlotBit(2)), Qwen38OrtEnv::kHandoffDone), nullptr);
}

TEST(Qwen38OrtRunHintsSlotTest, ANewTurnOfOneSlotLeavesTheOthersAlone) {
  Qwen38OrtRunHints hints{2};
  hints.ExpectPrefillTokens(0, 20);
  hints.MarkHandoff(0);
  hints.ExpectPrefillTokens(1, 50);
  hints.BeginTurn(1);
  EXPECT_EQ(hints.ExpectedPrefillTokens(1), 0u) << "nothing of the previous turn of the slot survives";
  EXPECT_EQ(hints.ExpectedPrefillTokens(0), 0u) << "slot 0 handed off: its length was spent";
  EXPECT_NE(Find(hints.Collect(nullptr, Qwen38SlotBit(0)), Qwen38OrtEnv::kHandoffDone), nullptr)
      << "the hand-off notice of slot 0 is still pending";

  hints.ExpectPrefillTokens(0, 20);
  hints.BeginTurn(1);
  EXPECT_EQ(hints.ExpectedPrefillTokens(0), 20u) << "the length of another slot is not touched";
  hints.BeginAllTurns();
  EXPECT_EQ(hints.ExpectedPrefillTokens(0), 0u);
  EXPECT_EQ(Find(hints.Collect(nullptr, Qwen38SlotBit(0)), Qwen38OrtEnv::kHandoffDone), nullptr);
  EXPECT_NE(Find(hints.Collect(nullptr, Qwen38SlotBit(0)), Qwen38OrtEnv::kNewTurn), nullptr);
  EXPECT_NE(Find(hints.Collect(nullptr, Qwen38SlotBit(1)), Qwen38OrtEnv::kNewTurn), nullptr);
}

TEST(Qwen38OrtRunHintsSlotTest, ThePromptLengthIsKeptPerSlotAndASlotStartedLaterNeverReplacesIt) {
  Qwen38OrtRunHints hints{2};
  hints.BeginAllTurns();
  hints.ExpectPrefillTokens(0, 20);
  hints.ExpectPrefillTokens(1, 5);  // announced after slot 0's, before slot 0's second chunk
  EXPECT_EQ(hints.BeginPromptChunk(0, 0, 8), 20u);
  EXPECT_EQ(hints.BeginPromptChunk(1, 0, 5), 5u);
  EXPECT_EQ(hints.BeginPromptChunk(0, 8, 8), 20u);
  EXPECT_EQ(hints.BeginPromptChunk(0, 16, 4), 20u);
  EXPECT_EQ(hints.ExpectedPrefillTokens(0), 20u);
  EXPECT_EQ(hints.ExpectedPrefillTokens(1), 5u);

  // Never less than what the slot has ingested, whatever another slot was announced.
  hints.ExpectPrefillTokens(0, 0);
  hints.ExpectPrefillTokens(1, 4096);
  EXPECT_EQ(hints.BeginPromptChunk(0, 24, 8), 32u);
  EXPECT_EQ(hints.BeginPromptChunk(1, 0, 8), 4096u);
}

TEST(Qwen38OrtRunHintsSlotTest, HandingOffASlotSpendsOnlyItsOwnPromptLength) {
  Qwen38OrtRunHints hints{2};
  hints.ExpectPrefillTokens(0, 20);
  hints.ExpectPrefillTokens(1, 5);
  hints.MarkHandoff(1);
  EXPECT_EQ(hints.ExpectedPrefillTokens(1), 0u);
  EXPECT_EQ(hints.ExpectedPrefillTokens(0), 20u);
}

TEST(Qwen38OrtRunHintsSlotTest, OnlyTheFirstChunkOfASlotRearmsItsNewTurnNotice) {
  Qwen38OrtRunHints hints{2};
  hints.BeginPromptChunk(1, 0, 8);
  RunWithQwen38OrtHints(hints, nullptr, [] { WriteLikeOrt(Qwen38OrtEnv::kNewTurn, "0"); }, Qwen38SlotBit(1));
  hints.BeginPromptChunk(1, 8, 8);
  EXPECT_EQ(Find(hints.Collect(nullptr, Qwen38SlotBit(1)), Qwen38OrtEnv::kNewTurn), nullptr);
  hints.BeginPromptChunk(0, 0, 8);  // the first chunk of slot 0 arms slot 0's notice, not slot 1's
  EXPECT_NE(Find(hints.Collect(nullptr, Qwen38SlotBit(0)), Qwen38OrtEnv::kNewTurn), nullptr);
  EXPECT_EQ(Find(hints.Collect(nullptr, Qwen38SlotBit(1)), Qwen38OrtEnv::kNewTurn), nullptr);
}

TEST(Qwen38OrtRunHintsSlotTest, AChunkOfOneSlotWithdrawsOnlyItsOwnHandoffNotice) {
  Qwen38OrtRunHints hints{2};
  hints.MarkHandoff(0);
  hints.MarkHandoff(1);
  hints.BeginPromptChunk(1, 0, 8);
  EXPECT_NE(Find(hints.Collect(nullptr, Qwen38SlotBit(0)), Qwen38OrtEnv::kHandoffDone), nullptr);
  EXPECT_EQ(Find(hints.Collect(nullptr, Qwen38SlotBit(1)), Qwen38OrtEnv::kHandoffDone), nullptr);
}

TEST(Qwen38OrtRunHintsSlotTest, ARunPublishesAndConsumesTheNoticesOfItsParticipantsOnly) {
  Qwen38OrtRunHints hints{3};
  hints.BeginAllTurns();
  hints.MarkHandoff(0);
  hints.MarkHandoff(2);

  // A Run that serves slot 1 sees only slot 1's new-turn notice, and consuming it leaves the others pending.
  const auto only_one = RunLikeOrt(hints, nullptr, Qwen38SlotBit(1));
  EXPECT_EQ(only_one.new_turn, "1");
  EXPECT_NE(only_one.handoff, "1");
  EXPECT_EQ(Find(hints.Collect(nullptr, Qwen38SlotBit(1)), Qwen38OrtEnv::kNewTurn), nullptr);
  EXPECT_NE(Find(hints.Collect(nullptr, Qwen38SlotBit(0)), Qwen38OrtEnv::kNewTurn), nullptr);
  EXPECT_NE(Find(hints.Collect(nullptr, Qwen38SlotBit(2)), Qwen38OrtEnv::kHandoffDone), nullptr);

  // A batched Run of slots 0 and 2 publishes each notice once and clears both slots.
  const auto several = RunLikeOrt(hints, nullptr, Qwen38SlotBit(0) | Qwen38SlotBit(2));
  EXPECT_EQ(several.new_turn, "1");
  EXPECT_EQ(several.handoff, "1");
  EXPECT_TRUE(hints.Collect(nullptr, Qwen38SlotMask{7}).empty());
}

TEST(Qwen38OrtRunHintsSlotTest, ANoticeThatOrtDidNotConsumeStaysPendingForItsSlotOnly) {
  Qwen38OrtRunHints hints{2};
  hints.BeginAllTurns();
  RunWithQwen38OrtHints(hints, nullptr, [] {}, Qwen38SlotBit(0));  // ORT left the notice alone
  EXPECT_NE(Find(hints.Collect(nullptr, Qwen38SlotBit(0)), Qwen38OrtEnv::kNewTurn), nullptr);
  RunWithQwen38OrtHints(hints, nullptr, [] { WriteLikeOrt(Qwen38OrtEnv::kNewTurn, "0"); }, Qwen38SlotBit(1));
  EXPECT_NE(Find(hints.Collect(nullptr, Qwen38SlotBit(0)), Qwen38OrtEnv::kNewTurn), nullptr)
      << "the Run of slot 1 consumed the notice of slot 1 only";
  EXPECT_EQ(Find(hints.Collect(nullptr, Qwen38SlotBit(1)), Qwen38OrtEnv::kNewTurn), nullptr);
}

// The schedule that a process-wide value gets wrong: A ingests a chunk, B starts its prompt, A ingests the next one. Every
// Run must be told the values of its own slot, and nothing may stay behind in the process environment.
TEST(Qwen38OrtRunHintsSlotTest, InterleavedPromptChunksOfTwoSlotsEachSeeTheirOwnValues) {
  const char* const names[] = {Qwen38OrtEnv::kNewTurn, Qwen38OrtEnv::kHandoffDone, Qwen38OrtEnv::kPrefillTotalTokens,
                               Qwen38OrtEnv::kCurrentMaxCompactExperts};
  std::vector<std::optional<std::string>> before;
  for (const char* name : names) before.push_back(Read(name));

  Qwen38OrtRunHints hints{2};
  hints.BeginAllTurns();
  hints.ExpectPrefillTokens(0, 4096);  // slot A: a long prompt, 3 chunks follow
  hints.ExpectPrefillTokens(1, 100);   // slot B: announced after A started, before A's second chunk

  const auto chunk = [&](size_t slot, size_t ingested, size_t tokens, int compact) {
    const Qwen38PromptRunHints prompt{hints.BeginPromptChunk(slot, ingested, tokens), compact};
    return RunLikeOrt(hints, &prompt, Qwen38SlotBit(slot));
  };
  const auto a0 = chunk(0, 0, 1024, 128);
  const auto b0 = chunk(1, 0, 100, 64);
  const auto a1 = chunk(0, 1024, 1024, 128);
  const auto a2 = chunk(0, 2048, 2048, 64);

  EXPECT_EQ(a0.total, "4096");
  EXPECT_EQ(a0.compact, "128");
  EXPECT_EQ(a0.new_turn, "1");
  EXPECT_EQ(b0.total, "100") << "the total of the slot that ran last is never merged into the other's";
  EXPECT_EQ(b0.compact, "64");
  EXPECT_EQ(b0.new_turn, "1") << "slot B's first chunk carries its own notice although slot A consumed one before";
  EXPECT_EQ(a1.total, "4096") << "slot B started in between and changed nothing for slot A";
  EXPECT_EQ(a1.compact, "128");
  EXPECT_EQ(a1.new_turn, before[0]) << "the notice of slot A was consumed by its first chunk";
  EXPECT_EQ(a2.total, "4096");
  EXPECT_EQ(a2.compact, "64") << "recomputed for every Run, the larger wave of the earlier chunks is not kept";

  for (size_t i = 0; i < std::size(names); ++i) EXPECT_EQ(Read(names[i]), before[i]) << names[i];
}

TEST(Qwen38OrtRunHintsSlotTest, ADecodeRunOfSomeSlotsCarriesNoPromptValuesAndNoNoticeOfTheOthers) {
  const auto before_total = Read(Qwen38OrtEnv::kPrefillTotalTokens);
  const auto before_compact = Read(Qwen38OrtEnv::kCurrentMaxCompactExperts);
  Qwen38OrtRunHints hints{3};
  hints.BeginAllTurns();
  hints.ExpectPrefillTokens(2, 64);  // slot 2 is still prefilling
  hints.BeginPromptChunk(2, 0, 8);
  hints.MarkHandoff(0);  // slots 0 and 1 decode
  hints.MarkHandoff(1);
  const auto decode = RunLikeOrt(hints, nullptr, Qwen38SlotBit(0) | Qwen38SlotBit(1));
  EXPECT_EQ(decode.handoff, "1");
  EXPECT_EQ(decode.total, before_total) << "a decode Run publishes no prompt length";
  EXPECT_EQ(decode.compact, before_compact);
  EXPECT_NE(Find(hints.Collect(nullptr, Qwen38SlotBit(2)), Qwen38OrtEnv::kNewTurn), nullptr)
      << "the notice of the prefilling slot is still pending";
  EXPECT_EQ(hints.ExpectedPrefillTokens(2), 64u);
}

}  // namespace Generators::test
