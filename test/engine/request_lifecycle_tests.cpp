// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Lifecycle tests for the engine Request state machine. Because a tiny real
// CPU fixture model is available, these tests drive genuine Request objects (rather than a mock
// Search) and pin the transition policy: which mutations each status permits, and how
// create/assign/schedule/remove move a request between Unassigned, Assigned, InProgress, and
// Completed.

#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_helpers.h"
#include "engine_test_doubles.h"
#include "engine/request_status.h"

namespace Generators {
namespace test {
namespace {

std::vector<int32_t> Prompt() { return {2, 3, 4}; }

DeviceSpan<float> LogitsForToken(Model& model, int32_t token) {
  auto logits = model.p_device_inputs_->Allocate<float>(
      static_cast<size_t>(model.config_->model.vocab_size));
  auto cpu_logits = logits.CpuSpan();
  std::fill(cpu_logits.begin(), cpu_logits.end(), 0.0f);
  cpu_logits[token] = 100.0f;
  logits.CopyCpuToDevice();
  return logits;
}

class RequestLifecycleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadDummyDecoderModel();
    engine_ = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  }

  std::shared_ptr<Request> NewRequest() {
    return std::make_shared<Request>(MakeGreedyParams(*model_));
  }

  std::shared_ptr<Model> model_;
  DoublesEngine engine_;
};

// A request must carry at least one token per append; an empty batch is rejected.
TEST_F(RequestLifecycleTest, EmptyAppendIsRejected) {
  auto request = NewRequest();
  EXPECT_THROW(request->AddTokens({}), std::runtime_error);
}

TEST_F(RequestLifecycleTest, InvalidEosTokenIsRejected) {
  for (const int invalid_eos_token_id : {-1, 5}) {
    const std::string overlay =
        R"({ "model": { "vocab_size": 5, "eos_token_id": )" +
        std::to_string(invalid_eos_token_id) + " } }";
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder", nullptr, overlay);
    auto model = CreateModel(GetOrtEnv(), std::move(config));
    auto params = MakeGreedyParams(*model);

    try {
      std::make_shared<Request>(params);
      FAIL() << "Expected invalid eos_token_id to be rejected";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find("eos_token_id"), std::string::npos);
    }
  }
}

TEST_F(RequestLifecycleTest, EmptyRequestIsRejectedBeforeAssignment) {
  auto request = NewRequest();

  EXPECT_THROW(engine_.engine->AddRequest(request), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Unassigned);
}

// An append that would exceed the model's max length is rejected before any tokens are buffered, so
// a subsequent valid append and assign reflect only the accepted tokens.
TEST_F(RequestLifecycleTest, AppendBeyondContextIsRejectedBeforeMutation) {
  auto request = NewRequest();
  const size_t over_capacity = static_cast<size_t>(model_->config_->model.context_length) + 1;
  std::vector<int32_t> too_many(over_capacity, 2);
  EXPECT_THROW(request->AddTokens(too_many), std::runtime_error);

  auto prompt = Prompt();
  request->AddTokens(prompt);
  request->Assign(engine_.engine);
  EXPECT_EQ(request->CurrentSequenceLength(), static_cast<int64_t>(prompt.size()));
}

// A request can only be assigned while Unassigned; assigning an already-assigned request throws and
// leaves the request untouched.
TEST_F(RequestLifecycleTest, AssignIsRejectedWhenAlreadyAssigned) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
  ASSERT_EQ(request->status_, RequestStatus::Assigned);
  const int64_t length_before = request->CurrentSequenceLength();

  EXPECT_THROW(request->Assign(engine_.engine), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_EQ(request->CurrentSequenceLength(), length_before);
}

// Scheduling requires a prior assign; an unassigned request cannot be scheduled and stays
// Unassigned.
TEST_F(RequestLifecycleTest, ScheduleIsRejectedBeforeAssign) {
  auto prompt = Prompt();
  auto request = MintRequest(*model_, prompt);
  ASSERT_EQ(request->status_, RequestStatus::Unassigned);

  EXPECT_THROW(request->Schedule(), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Unassigned);
}

// An assigned, non-empty request schedules cleanly and moves to InProgress.
TEST_F(RequestLifecycleTest, ScheduleFromAssignedMovesToInProgress) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
  request->Schedule();
  EXPECT_EQ(request->status_, RequestStatus::InProgress);
}

// While a request is in progress its token stream is owned by the engine, so external appends are
// rejected without mutating the request.
TEST_F(RequestLifecycleTest, AppendIsRejectedWhileInProgress) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
  request->Schedule();
  ASSERT_EQ(request->status_, RequestStatus::InProgress);
  const int64_t length_before = request->CurrentSequenceLength();

  std::vector<int32_t> more{5, 6};
  EXPECT_THROW(request->AddTokens(more), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::InProgress);
  EXPECT_EQ(request->CurrentSequenceLength(), length_before);
}

// After a request completes, appending tokens resumes its sequence (the continuation path) rather
// than being rejected.
TEST_F(RequestLifecycleTest, AppendAfterCompletedExtendsSequence) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
  const int64_t assigned_length = request->CurrentSequenceLength();

  request->status_ = RequestStatus::Completed;
  std::vector<int32_t> more{5, 6};
  request->AddTokens(more);

  EXPECT_EQ(request->CurrentSequenceLength(), assigned_length + static_cast<int64_t>(more.size()));
}

// Removing a request releases it from the engine (deallocating its cache resources) and returns it
// to the Unassigned state.
TEST_F(RequestLifecycleTest, RemoveReturnsRequestToUnassigned) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
  ASSERT_EQ(request->status_, RequestStatus::Assigned);

  request->Remove();
  EXPECT_EQ(request->status_, RequestStatus::Unassigned);
  EXPECT_EQ(engine_.cache->deallocate_calls, 1);
  EXPECT_EQ(engine_.cache->AllocatedCount(), 0u);
}

TEST_F(RequestLifecycleTest, TransactionalLogitsStageUntilCommit) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
  const auto before = request->Snapshot();
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.target_cache_slots = static_cast<size_t>(before.current_sequence_length);
  const int32_t next_token = 5;
  auto logits = LogitsForToken(*model_, next_token);

  request->SaveStateForTransaction();
  const auto result = request->ApplyLogitsForTransaction(logits);

  const auto staged = request->Snapshot();
  EXPECT_TRUE(result.token_appended);
  EXPECT_FALSE(result.done);
  EXPECT_EQ(staged.status, before.status);
  EXPECT_EQ(staged.processed_sequence_length, before.processed_sequence_length);
  EXPECT_EQ(staged.is_prefill, before.is_prefill);
  EXPECT_EQ(staged.current_sequence_length, before.current_sequence_length + 1);

  request->CommitStateForTransaction();
  request->CommitStep(plan, result);

  const auto committed = request->Snapshot();
  EXPECT_EQ(committed.status, RequestStatus::InProgress);
  EXPECT_EQ(committed.processed_sequence_length, before.current_sequence_length);
  EXPECT_FALSE(committed.is_prefill);
  ASSERT_TRUE(request->HasUnseenTokens());
  EXPECT_EQ(request->UnseenToken(), next_token);
}

TEST_F(RequestLifecycleTest, TransactionalLogitsRollbackRestoresSearchState) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
  const auto before = request->Snapshot();
  auto logits = LogitsForToken(*model_, 5);

  request->SaveStateForTransaction();
  request->ApplyLogitsForTransaction(logits);
  request->RestoreStateForTransaction();

  const auto restored = request->Snapshot();
  EXPECT_EQ(restored.status, before.status);
  EXPECT_EQ(restored.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(restored.processed_sequence_length, before.processed_sequence_length);
  EXPECT_EQ(restored.is_prefill, before.is_prefill);
  EXPECT_FALSE(request->HasUnseenTokens());
}

TEST_F(RequestLifecycleTest, PartialPrefillAdvancesOnlyAtCommit) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
  const auto before = request->Snapshot();
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.unprocessed_token_count = 2;
  plan.target_cache_slots = 2;

  request->SaveStateForTransaction();
  request->CommitStateForTransaction();
  request->CommitStep(plan, RequestStepResult{});

  const auto committed = request->Snapshot();
  EXPECT_EQ(committed.status, RequestStatus::InProgress);
  EXPECT_EQ(committed.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(committed.processed_sequence_length, 2);
  // Two of the three prompt tokens are in the cache, so the request is still prefilling.
  EXPECT_TRUE(committed.is_prefill);
  EXPECT_FALSE(request->HasUnseenTokens());
}

TEST_F(RequestLifecycleTest, FirstTransactionalStepCanCommitDirectlyToCompleted) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
  const auto before = request->Snapshot();
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.target_cache_slots = static_cast<size_t>(before.current_sequence_length);
  auto logits = LogitsForToken(*model_, EosToken(*model_));

  request->SaveStateForTransaction();
  const auto result = request->ApplyLogitsForTransaction(logits);
  request->CommitStateForTransaction();
  request->CommitStep(plan, result);

  EXPECT_TRUE(result.done);
  EXPECT_EQ(request->status_, RequestStatus::Completed);
}

TEST_F(RequestLifecycleTest, RequestRejectsMultiSequenceSearch) {
  auto params = MakeGreedyParams(*model_);
  params->search.batch_size = 2;
  EXPECT_THROW(
      {
        auto request = std::make_shared<Request>(params);
        static_cast<void>(request);
      },
      std::runtime_error);
}

}  // namespace
}  // namespace test
}  // namespace Generators
