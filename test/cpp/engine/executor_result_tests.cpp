// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "engine/scheduled_requests.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"

namespace Generators::test {
namespace {

class ExecutorResultTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadDummyDecoderModel();
    engine_ = MakeStaticDoublesEngine(model_, 4, EosToken(*model_)).engine;
  }

  std::shared_ptr<Request> ActiveRequest() {
    auto request = CreateRequestWithPrompt(engine_, std::array<int32_t, 2>{2, 3});
    request->Schedule();
    return request;
  }

  BackendPlannedRow Plan(const std::shared_ptr<Request>& request, size_t row = 0) {
    return {row, {17, 2, {static_cast<uint32_t>(row), 1}, request->CurrentTurnId(), 1}, 2, true};
  }

  BackendRowResult Tokens(const BackendPlannedRow& plan, std::initializer_list<int32_t> values) {
    BackendRowResult row;
    row.key = plan.key;
    row.selected_tokens.emplace();
    std::copy(values.begin(), values.end(), row.selected_tokens->candidates.begin());
    row.selected_tokens->candidate_count = values.size();
    return row;
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<Engine> engine_;
};

TEST_F(ExecutorResultTest, ExplicitPrefillProgressNeverSamplesOrAppends) {
  auto request = ActiveRequest();
  auto plan = Plan(request);
  ScheduledRequests batch{{request}, model_, nullptr, nullptr};
  batch.BeginBackendStep({plan});
  batch.StageBackendRow(0, {plan.key, PrefillProgress{2, false}, std::nullopt});
  std::vector<RequestStepResult> results;
  batch.GenerateNextTokens(results);
  EXPECT_EQ(request->CurrentSequenceLength(), 2);
  ASSERT_EQ(batch.CommitAcks().size(), 1);
  EXPECT_EQ(batch.CommitAcks()[0].candidates_consumed, 0);
  EXPECT_EQ(batch.CommitAcks()[0].request_sequence_length_after, 2);
  EXPECT_FALSE(batch.CommitAcks()[0].turn_finished);
  EXPECT_THROW(batch.GenerateNextTokens(results), std::logic_error);
}

TEST_F(ExecutorResultTest, ProgressAndTokensCanShareOneRowInAMixedBatch) {
  auto first = ActiveRequest();
  auto second = ActiveRequest();
  const auto a = Plan(first);
  const auto b = Plan(second, 1);
  ScheduledRequests batch{{first, second}, model_, nullptr, nullptr};
  batch.BeginBackendStep({a, b});
  auto completed = Tokens(b, {4});
  completed.prefill = PrefillProgress{2, true};
  batch.StageBackendRow(1, completed);
  batch.StageBackendRow(0, {a.key, PrefillProgress{1, false}, std::nullopt});
  // The result owns its small token block through commit and settlement.
  completed.selected_tokens->candidates[0] = 7;
  std::vector<RequestStepResult> results;
  batch.GenerateNextTokens(results);
  EXPECT_EQ(first->CurrentSequenceLength(), 2);
  EXPECT_EQ(second->TokensCpu().back(), 4);
  ASSERT_EQ(batch.CommitAcks().size(), 2);
  EXPECT_EQ(batch.CommitAcks()[0].tokens_appended, 1);
  EXPECT_EQ(batch.CommitAcks()[0].tokens_staged_for_events, 1);
}

TEST_F(ExecutorResultTest, MalformedSecondRowCannotMutateFirstRequest) {
  auto first = ActiveRequest();
  auto second = ActiveRequest();
  auto a = Plan(first);
  auto b = Plan(second, 1);
  for (int violation = 0; violation < 7; ++violation) {
    ScheduledRequests batch{{first, second}, model_, nullptr, nullptr};
    batch.BeginBackendStep({a, b});
    batch.StageBackendRow(0, Tokens(a, {4}));
    auto bad = Tokens(b, {5});
    switch (violation) {
      case 0:
        ++bad.key.lease_epoch;
        break;
      case 1:
        bad.selected_tokens->candidates[0] = -1;
        break;
      case 2:
        bad.selected_tokens->candidates[0] = model_->config_->model.vocab_size;
        break;
      case 3:
        bad.selected_tokens->candidate_count = kMaxGeneratedTokensPerStep + 1;
        break;
      case 4:
        bad.selected_tokens.reset();
        break;
      case 5:
        bad.prefill = PrefillProgress{3, true};
        break;
      case 6:
        bad.prefill = PrefillProgress{0, false};
        break;
    }
    batch.StageBackendRow(1, std::move(bad));
    std::vector<RequestStepResult> results;
    EXPECT_THROW(batch.GenerateNextTokens(results), std::logic_error) << violation;
    EXPECT_EQ(first->CurrentSequenceLength(), 2) << violation;
    EXPECT_EQ(second->CurrentSequenceLength(), 2) << violation;
    EXPECT_TRUE(batch.CommitAcks().empty());
  }
}

TEST_F(ExecutorResultTest, CoverageAndDuplicateRowsAreValidated) {
  auto first = ActiveRequest();
  auto second = ActiveRequest();
  auto a = Plan(first);
  auto b = Plan(second, 1);
  ScheduledRequests missing{{first, second}, model_, nullptr, nullptr};
  missing.BeginBackendStep({a, b});
  missing.StageBackendRow(0, Tokens(a, {4}));
  EXPECT_THROW(missing.StageBackendRow(0, Tokens(a, {4})), std::logic_error);
  std::vector<RequestStepResult> results;
  EXPECT_THROW(missing.GenerateNextTokens(results), std::logic_error);
  EXPECT_EQ(first->CurrentSequenceLength(), 2);
  ScheduledRequests unexpected{{first, second}, model_, nullptr, nullptr};
  unexpected.BeginBackendStep({a});
  unexpected.StageBackendRow(1, Tokens(b, {4}));
  EXPECT_THROW(unexpected.GenerateNextTokens(results), std::logic_error);
  EXPECT_EQ(second->CurrentSequenceLength(), 2);
}

TEST_F(ExecutorResultTest, NegotiatedResultAndSlotLimitsAreEnforced) {
  auto request = ActiveRequest();
  auto plan = Plan(request);
  ScheduledRequests batch{{request}, model_, nullptr, nullptr};
  batch.SetBackendLimits(1, 1);
  batch.BeginBackendStep({plan});
  batch.StageBackendRow(0, Tokens(plan, {4, 5}));
  std::vector<RequestStepResult> results;
  EXPECT_THROW(batch.GenerateNextTokens(results), std::logic_error);
  ScheduledRequests bad_slot{{request}, model_, nullptr, nullptr};
  bad_slot.SetBackendLimits(1, 1);
  plan.key.slot.index = 1;
  EXPECT_THROW(bad_slot.BeginBackendStep({plan}), std::logic_error);
  EXPECT_EQ(request->CurrentSequenceLength(), 2);
}

TEST_F(ExecutorResultTest, EosConsumptionDiffersFromAppendedTokensAtEveryPosition) {
  for (size_t eos_position = 0; eos_position < kMaxGeneratedTokensPerStep; ++eos_position) {
    auto request = ActiveRequest();
    auto plan = Plan(request);
    auto row = Tokens(plan, {});
    row.selected_tokens->candidates.fill(4);
    row.selected_tokens->candidates[eos_position] = EosToken(*model_);
    row.selected_tokens->candidate_count = kMaxGeneratedTokensPerStep;
    ScheduledRequests batch{{request}, model_, nullptr, nullptr};
    batch.BeginBackendStep({plan});
    batch.StageBackendRow(0, std::move(row));
    std::vector<RequestStepResult> results;
    batch.GenerateNextTokens(results);
    const auto& ack = batch.CommitAcks().at(0);
    EXPECT_EQ(ack.candidates_consumed, eos_position + 1);
    EXPECT_EQ(ack.tokens_appended, eos_position);
    EXPECT_EQ(ack.tokens_staged_for_events, eos_position);
    EXPECT_EQ(ack.request_sequence_length_after, eos_position + 2);
    EXPECT_TRUE(ack.turn_finished);
    EXPECT_EQ(ack.disposition, ContinuationDisposition::Invalidate);
    EXPECT_EQ(ack.finish_reason, GenerationFinishReason::EosToken);
  }
}

}  // namespace
}  // namespace Generators::test
