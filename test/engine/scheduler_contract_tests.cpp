// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Component-contract tests for the continuous-batching DynamicBatchScheduler. They drive a real
// scheduler against a recording CacheManager double and a tiny CPU fixture model that is used only
// to mint real Request objects: the model never runs, so these tests need no GPU and no real
// inference. They assert the scheduler's admission contract -- which requests it admits, in what
// order, how it honors capacity backpressure, and how it releases completed and removed requests --
// by observing the calls it makes to the cache manager.

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "engine/scheduler.h"
#include "engine_test_helpers.h"
#include "engine_test_doubles.h"

namespace Generators {
namespace test {
namespace {

// A distinct in-vocabulary, non-EOS prompt per request; contents are irrelevant because the model
// never runs. Kept well within the fixture model's small vocabulary.
std::vector<int32_t> Prompt(int32_t seed) {
  const int32_t base = 2 + (seed % 8);
  return {base, base + 1, base + 2};
}

class SchedulerContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadDummyDecoderModel();
    // A permissive engine used purely as the assignment target when minting requests.
    assign_target_ = MakeDoublesEngine(model_, /*capacity=*/1024, EosToken(*model_)).engine;
  }

  std::shared_ptr<Request> Assigned(int32_t seed) {
    auto prompt = Prompt(seed);
    return MintAssignedRequest(assign_target_, *model_, prompt);
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<Engine> assign_target_;
};

// A single assigned request is admitted: the scheduler asks the cache once, allocates it, moves the
// request to InProgress, and returns it in the scheduled batch.
TEST_F(SchedulerContractTest, DynamicAdmitsSingleAssignedRequest) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto request = Assigned(10);
  scheduler.AddRequest(request);

  auto scheduled = scheduler.Schedule();

  EXPECT_EQ(scheduled.size(), 1u);
  EXPECT_EQ(request->status_, RequestStatus::InProgress);
  EXPECT_EQ(cache->allocate_calls, 1);
  EXPECT_EQ(cache->AllocatedCount(), 1u);
}

// The dynamic scheduler admits each assignable request independently and preserves insertion order
// in the scheduled batch.
TEST_F(SchedulerContractTest, DynamicPreservesRequestOrder) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto first = Assigned(10);
  auto second = Assigned(20);
  auto third = Assigned(30);
  scheduler.AddRequest(first);
  scheduler.AddRequest(second);
  scheduler.AddRequest(third);

  auto scheduled = scheduler.Schedule();

  ASSERT_EQ(scheduled.size(), 3u);
  EXPECT_EQ(scheduled[0], first);
  EXPECT_EQ(scheduled[1], second);
  EXPECT_EQ(scheduled[2], third);
}

// When the cache cannot fit every request, the dynamic scheduler admits only those that fit and
// leaves the rest Assigned for a later step rather than forcing an over-capacity batch.
TEST_F(SchedulerContractTest, DynamicHonorsCapacityBackpressure) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/2);
  DynamicBatchScheduler scheduler(model_, cache);

  auto first = Assigned(10);
  auto second = Assigned(20);
  auto third = Assigned(30);
  scheduler.AddRequest(first);
  scheduler.AddRequest(second);
  scheduler.AddRequest(third);

  auto scheduled = scheduler.Schedule();

  EXPECT_EQ(scheduled.size(), 2u);
  EXPECT_EQ(first->status_, RequestStatus::InProgress);
  EXPECT_EQ(second->status_, RequestStatus::InProgress);
  EXPECT_EQ(third->status_, RequestStatus::Assigned);
}

// A completed request occupying the cache is released on the next schedule before new admissions are
// considered.
TEST_F(SchedulerContractTest, DynamicDeallocatesCompletedRequests) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto request = Assigned(10);
  scheduler.AddRequest(request);
  scheduler.Schedule();  // admits and allocates the request
  ASSERT_EQ(cache->AllocatedCount(), 1u);

  request->status_ = RequestStatus::Completed;
  auto second = Assigned(20);
  scheduler.AddRequest(second);

  auto scheduled = scheduler.Schedule();

  EXPECT_GE(cache->deallocate_calls, 1);
  // Only the still-active second request remains allocated and scheduled.
  EXPECT_EQ(cache->AllocatedCount(), 1u);
  ASSERT_EQ(scheduled.size(), 1u);
  EXPECT_EQ(scheduled[0], second);
}

// Removing a request from the dynamic scheduler deallocates its cache resources immediately.
TEST_F(SchedulerContractTest, DynamicRemoveReleasesCacheResources) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto request = Assigned(10);
  scheduler.AddRequest(request);
  scheduler.Schedule();
  ASSERT_EQ(cache->AllocatedCount(), 1u);

  scheduler.RemoveRequest(request);

  EXPECT_GE(cache->deallocate_calls, 1);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
  EXPECT_FALSE(scheduler.HasPendingRequests());
}

// With no pending requests the dynamic scheduler surfaces the empty-batch condition rather than
// returning an invalid batch.
TEST_F(SchedulerContractTest, DynamicScheduleWithNoRequestsThrows) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  EXPECT_THROW(scheduler.Schedule(), std::runtime_error);
}

TEST_F(SchedulerContractTest, DynamicPlanningDoesNotMutateAdmissionState) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto first = Assigned(10);
  auto second = Assigned(20);
  scheduler.AddRequest(first);
  scheduler.AddRequest(second);
  StepPlan plan;
  plan.transaction_id = 17;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 2u);
  EXPECT_EQ(plan.requests[0].request, first);
  EXPECT_EQ(plan.requests[0].packed_token_offset, 0u);
  EXPECT_EQ(plan.requests[0].logits_row_index, 2u);
  EXPECT_EQ(plan.requests[1].request, second);
  EXPECT_EQ(plan.requests[1].packed_token_offset, 3u);
  EXPECT_EQ(plan.requests[1].logits_row_index, 5u);
  EXPECT_EQ(plan.token_count, 6u);
  EXPECT_FALSE(plan.graph_capture_eligible);
  EXPECT_EQ(first->status_, RequestStatus::Assigned);
  EXPECT_EQ(second->status_, RequestStatus::Assigned);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
  EXPECT_EQ(cache->allocate_calls, 0);
}

TEST_F(SchedulerContractTest, DynamicPlanningKeepsActiveWorkWhenNewAdmissionIsDeferred) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto active = Assigned(10);
  scheduler.AddRequest(active);
  scheduler.Schedule();
  ASSERT_EQ(active->status_, RequestStatus::InProgress);

  auto deferred = Assigned(20);
  scheduler.AddRequest(deferred);
  cache->SetCanAllocate(false);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, active);
  EXPECT_FALSE(plan.requests[0].newly_admitted);
  EXPECT_EQ(deferred->status_, RequestStatus::Assigned);
  EXPECT_EQ(cache->AllocatedCount(), 1u);
}

TEST_F(SchedulerContractTest, DynamicPlanningReportsNoWorkWithoutMutation) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  StepPlan plan;
  plan.transaction_id = 9;

  const auto result = scheduler.PlanStep(plan);

  EXPECT_FALSE(result.executable);
  EXPECT_EQ(result.outcome.kind, StepOutcomeKind::NoWork);
  EXPECT_EQ(result.outcome.transaction_id, 9u);
  EXPECT_TRUE(plan.Empty());
}

TEST_F(SchedulerContractTest, UnserviceableCandidateDoesNotBlockSmallerWork) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto too_large = Assigned(10);
  auto fitting = Assigned(20);
  scheduler.AddRequest(too_large);
  scheduler.AddRequest(fitting);
  cache->SetUnserviceableRequest(too_large);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_EQ(result.unserviceable_request_id, too_large.get());
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, fitting);
  EXPECT_EQ(too_large->status_, RequestStatus::Assigned);
}

TEST_F(SchedulerContractTest, UnserviceableActiveGrowthIsNotDeferred) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto active = Assigned(10);
  scheduler.AddRequest(active);
  scheduler.Schedule();
  cache->SetUnserviceableRequest(active);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  EXPECT_FALSE(result.executable);
  EXPECT_FALSE(result.capacity_deferred);
  EXPECT_EQ(result.outcome.kind,
            StepOutcomeKind::UnserviceableRequest);
  EXPECT_EQ(result.outcome.request_id, active.get());
}

}  // namespace
}  // namespace test
}  // namespace Generators
