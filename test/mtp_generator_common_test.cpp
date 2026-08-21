// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "mtp_generator_common.h"

namespace Generators::test {
namespace {

// Drives MtpGeneratorBase with a scripted round so the delivery/stats plumbing can be tested
// without a model.
struct ScriptedMtpGenerator : MtpGeneratorBase {
  void AppendTokens(cpu_span<const int32_t> input_ids) override {
    sequence_.assign(input_ids.begin(), input_ids.end());
    emitted_sequence_ = sequence_;
    primed_ = true;
  }

  using MtpGeneratorBase::eos_token_ids_;
  using MtpGeneratorBase::max_length_;

  std::vector<int32_t> round_tokens_;

 private:
  void RunRound() override { sequence_.insert(sequence_.end(), round_tokens_.begin(), round_tokens_.end()); }
};

ScriptedMtpGenerator MakeGenerator(std::vector<int32_t> round_tokens, int max_length,
                                   std::vector<int32_t> eos = {}) {
  ScriptedMtpGenerator generator;
  generator.round_tokens_ = std::move(round_tokens);
  generator.max_length_ = max_length;
  generator.eos_token_ids_ = std::move(eos);
  return generator;
}

constexpr std::array<int32_t, 2> kPrompt{1, 2};

}  // namespace

TEST(MtpGeneratorCommonTest, ArgmaxSelectsLargestValue) {
  constexpr std::array<float, 5> values{-2.0f, 4.0f, 3.5f, 8.0f, 1.0f};
  EXPECT_EQ(ArgmaxRow(values.data(), static_cast<int>(values.size())), 3);
}

TEST(MtpGeneratorCommonTest, FinalizesDerivedStatistics) {
  SpeculativeStats stats{};
  stats.rounds = 2;
  stats.draft_tokens_proposed = 6;
  stats.draft_tokens_evaluated = 4;
  stats.draft_tokens_accepted = 3;
  stats.tokens_emitted = 8;

  const auto finalized = FinalizeSpeculativeStats(stats, 2);

  EXPECT_EQ(finalized.tokens_buffered, 2);
  EXPECT_FLOAT_EQ(finalized.acceptance_rate, 0.75f);
  EXPECT_FLOAT_EQ(finalized.avg_draft_tokens_per_round, 3.0f);
  EXPECT_FLOAT_EQ(finalized.mean_emitted_tokens_per_round, 4.0f);
}

TEST(CountAcceptedDraftsTest, RejectsWhenFirstDraftDiffers) {
  constexpr std::array<int32_t, 3> drafts{7, 8, 9};
  // Deliberately empty: a rejected first draft must not read the verify rows at all.
  EXPECT_EQ(CountAcceptedDrafts(drafts.data(), nullptr, static_cast<int>(drafts.size()), 5), 0);
}

TEST(CountAcceptedDraftsTest, AcceptsMatchingPrefix) {
  constexpr std::array<int32_t, 4> drafts{5, 8, 9, 10};
  constexpr std::array<int32_t, 4> verify{8, 99, 99, 99};
  EXPECT_EQ(CountAcceptedDrafts(drafts.data(), verify.data(), static_cast<int>(drafts.size()), 5), 2);
}

TEST(CountAcceptedDraftsTest, AcceptsEveryDraft) {
  constexpr std::array<int32_t, 3> drafts{5, 8, 9};
  constexpr std::array<int32_t, 3> verify{8, 9, 11};
  EXPECT_EQ(CountAcceptedDrafts(drafts.data(), verify.data(), static_cast<int>(drafts.size()), 5), 3);
}

TEST(CountAcceptedDraftsTest, HandlesEmptyRound) {
  EXPECT_EQ(CountAcceptedDrafts(nullptr, nullptr, 0, 5), 0);
}

TEST(MtpGeneratorBaseTest, RequiresAppendTokensFirst) {
  auto generator = MakeGenerator({3}, 100);
  EXPECT_THROW(generator.GenerateNextToken(), std::runtime_error);
}

TEST(MtpGeneratorBaseTest, DeliversOneBufferedTokenPerCall) {
  auto generator = MakeGenerator({3, 4}, 100);
  generator.AppendTokens(cpu_span<const int32_t>(kPrompt.data(), kPrompt.size()));

  generator.GenerateNextToken();
  EXPECT_EQ(generator.GetSequence(), (std::vector<int32_t>{1, 2, 3}));
  EXPECT_EQ(generator.GetSpeculativeStats().tokens_buffered, 1);

  generator.GenerateNextToken();
  EXPECT_EQ(generator.GetSequence(), (std::vector<int32_t>{1, 2, 3, 4}));

  const auto stats = generator.GetSpeculativeStats();
  EXPECT_EQ(stats.tokens_queued, 2);
  EXPECT_EQ(stats.tokens_emitted, 2);
  EXPECT_EQ(stats.tokens_buffered, 0);
  EXPECT_FALSE(generator.IsDone());
}

TEST(MtpGeneratorBaseTest, StopsQueueingAtEos) {
  auto generator = MakeGenerator({3, 4, 5}, 100, {4});
  generator.AppendTokens(cpu_span<const int32_t>(kPrompt.data(), kPrompt.size()));

  generator.GenerateNextToken();
  generator.GenerateNextToken();

  EXPECT_EQ(generator.GetSequence(), (std::vector<int32_t>{1, 2, 3, 4}));
  EXPECT_EQ(generator.GetSpeculativeStats().tokens_queued, 2);
  EXPECT_TRUE(generator.IsDone());
}

TEST(MtpGeneratorBaseTest, StopsQueueingAtMaxLength) {
  auto generator = MakeGenerator({3, 4, 5}, 4);
  generator.AppendTokens(cpu_span<const int32_t>(kPrompt.data(), kPrompt.size()));

  generator.GenerateNextToken();
  generator.GenerateNextToken();

  EXPECT_EQ(generator.GetSequence(), (std::vector<int32_t>{1, 2, 3, 4}));
  EXPECT_TRUE(generator.IsDone());
}

}  // namespace Generators::test
