// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <cstdlib>
#include <initializer_list>
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
                                       throw std::runtime_error("Run failed");
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

}  // namespace Generators::test
