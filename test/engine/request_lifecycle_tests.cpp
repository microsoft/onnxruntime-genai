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

}  // namespace
}  // namespace test
}  // namespace Generators
