// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>

#include <gtest/gtest.h>

#include "engine/engine_invariants.h"
#include "engine/paged_key_value_cache.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"

namespace Generators {
namespace test {
namespace {

class PagedKeyValueCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadDummyDecoderModel();
    Config::Engine::DynamicBatching dynamic_batching;
    dynamic_batching.block_size = 4;
    dynamic_batching.num_blocks = 3;
    dynamic_batching.max_batch_size = 2;
    model_->config_->engine.dynamic_batching = dynamic_batching;
    assign_target_ =
        MakeDoublesEngine(model_, /*capacity=*/2, EosToken(*model_)).engine;
    cache_ = std::make_unique<PagedKeyValueCache>(model_);
  }

  std::shared_ptr<Request> AddCommittedRequest(
      std::array<int32_t, 4> prompt) {
    auto request =
        MintAssignedRequest(assign_target_, *model_, prompt);
    cache_->Add(request);
    cache_->AppendTokens(request);
    return request;
  }

  static RequestStepPlan PlanEntry(
      const std::shared_ptr<Request>& request,
      size_t target_cache_slots,
      bool newly_admitted = false,
      size_t whole_sequence_cache_slots = 0) {
    RequestStepPlan entry;
    entry.request = request;
    entry.request_id = request.get();
    entry.target_cache_slots = target_cache_slots;
    entry.whole_sequence_cache_slots = whole_sequence_cache_slots;
    entry.newly_admitted = newly_admitted;
    return entry;
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<Engine> assign_target_;
  std::unique_ptr<PagedKeyValueCache> cache_;
};

TEST_F(PagedKeyValueCacheTest, DeferredActiveRequestRunsAfterCapacityIsReleased) {
  auto first = AddCommittedRequest({2, 3, 4, 5});
  auto second = AddCommittedRequest({6, 7, 8, 9});

  StepPlan plan;
  plan.requests.push_back(PlanEntry(first, 5));
  plan.requests.push_back(PlanEntry(second, 5));

  const auto result =
      cache_->PlanStepResources(plan, /*committed_request_count=*/2);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  EXPECT_EQ(result.unserviceable_request_id, nullptr);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, first);

  const std::array reservation_requests{
      PagedCacheReservationRequest{first.get(), 5, false},
  };
  auto reservation = cache_->Reserve(reservation_requests);
  EXPECT_TRUE(
      ValidateCacheInvariants(cache_->Snapshot(reservation)).empty());

  reservation.Commit();
  const auto snapshot = cache_->Snapshot();
  EXPECT_TRUE(ValidateCacheInvariants(snapshot).empty());
  ASSERT_EQ(snapshot.requests.size(), 2u);
  EXPECT_EQ(snapshot.requests[0].request_id, first.get());
  EXPECT_EQ(snapshot.requests[0].used_slots, 5u);
  EXPECT_EQ(snapshot.requests[0].block_ids.size(), 2u);
  EXPECT_EQ(snapshot.requests[1].request_id, second.get());
  EXPECT_EQ(snapshot.requests[1].used_slots, 4u);
  EXPECT_EQ(snapshot.requests[1].block_ids.size(), 1u);

  cache_->Remove(first);
  StepPlan next_plan;
  next_plan.requests.push_back(PlanEntry(second, 5));

  const auto next_result =
      cache_->PlanStepResources(next_plan, /*committed_request_count=*/1);

  ASSERT_TRUE(next_result.executable);
  EXPECT_FALSE(next_result.capacity_deferred);
  EXPECT_EQ(next_result.unserviceable_request_id, nullptr);
  ASSERT_EQ(next_plan.requests.size(), 1u);
  EXPECT_EQ(next_plan.requests[0].request, second);

  const std::array next_reservation_requests{
      PagedCacheReservationRequest{second.get(), 5, false},
  };
  auto next_reservation = cache_->Reserve(next_reservation_requests);
  next_reservation.Commit();

  const auto next_snapshot = cache_->Snapshot();
  EXPECT_TRUE(ValidateCacheInvariants(next_snapshot).empty());
  ASSERT_EQ(next_snapshot.requests.size(), 1u);
  EXPECT_EQ(next_snapshot.requests[0].request_id, second.get());
  EXPECT_EQ(next_snapshot.requests[0].used_slots, 5u);
  EXPECT_EQ(next_snapshot.requests[0].block_ids.size(), 2u);
}

TEST_F(PagedKeyValueCacheTest, DeferredActiveRequestsStillConsumeAdmissionCapacity) {
  auto unserviceable = AddCommittedRequest({2, 3, 4, 5});
  auto fitting = AddCommittedRequest({6, 7, 8, 9});
  auto pending = MintAssignedRequest(
      assign_target_, *model_, std::array<int32_t, 1>{10});

  StepPlan plan;
  plan.requests.push_back(PlanEntry(unserviceable, 13));
  plan.requests.push_back(PlanEntry(fitting, 4));
  plan.requests.push_back(PlanEntry(pending, 1, true));

  const auto result =
      cache_->PlanStepResources(plan, /*committed_request_count=*/2);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  EXPECT_EQ(result.unserviceable_request_id, unserviceable.get());
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, fitting);
  EXPECT_FALSE(plan.requests[0].newly_admitted);
}

// A chunked prefill asks for one chunk at a time, but admission has to be decided on the whole
// prompt: the pool is three blocks of four slots, so a prompt of thirteen slots can never fit even
// though its first chunk would.
TEST_F(PagedKeyValueCacheTest, PromptTooLargeForThePoolIsUnserviceableEvenWhenItsChunkFits) {
  auto pending = MintAssignedRequest(
      assign_target_, *model_, std::array<int32_t, 1>{10});

  StepPlan plan;
  plan.requests.push_back(PlanEntry(pending, /*target_cache_slots=*/1, /*newly_admitted=*/true,
                                    /*whole_sequence_cache_slots=*/13));

  const auto result =
      cache_->PlanStepResources(plan, /*committed_request_count=*/0);

  EXPECT_FALSE(result.executable);
  EXPECT_EQ(result.unserviceable_request_id, pending.get());
  EXPECT_TRUE(plan.requests.empty());
}

// Admission also has to wait for enough free blocks to hold the whole prompt, so a request never
// starts a chunked prefill it cannot finish.
TEST_F(PagedKeyValueCacheTest, AdmissionWaitsUntilTheWholePromptFits) {
  auto committed = AddCommittedRequest({2, 3, 4, 5});
  auto pending = MintAssignedRequest(
      assign_target_, *model_, std::array<int32_t, 1>{10});

  StepPlan plan;
  plan.requests.push_back(PlanEntry(committed, /*target_cache_slots=*/4));
  // One block is already taken, leaving two of the three: the chunk needs one, the prompt needs
  // three.
  plan.requests.push_back(PlanEntry(pending, /*target_cache_slots=*/1, /*newly_admitted=*/true,
                                    /*whole_sequence_cache_slots=*/9));

  const auto result =
      cache_->PlanStepResources(plan, /*committed_request_count=*/1);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  EXPECT_EQ(result.unserviceable_request_id, nullptr);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, committed);
}

}  // namespace
}  // namespace test
}  // namespace Generators
