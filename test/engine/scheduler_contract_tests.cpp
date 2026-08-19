// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Component-contract tests for the continuous-batching DynamicBatchScheduler. They drive a real
// scheduler against a recording CacheManager double and a tiny CPU fixture model that is used only
// to mint real Request objects: the model never runs, so these tests need no GPU and no real
// inference. They assert the scheduler's admission contract -- which requests it admits, in what
// order, how it honors capacity backpressure, and how it releases completed and removed requests --
// by observing the calls it makes to the cache manager.

#include <array>
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
    model_->config_->engine.dynamic_batching =
        Config::Engine::DynamicBatching{};
    // A permissive engine used purely as the assignment target when minting requests.
    assign_target_ = MakeDoublesEngine(model_, /*capacity=*/1024, EosToken(*model_)).engine;
  }

  std::shared_ptr<Request> Assigned(int32_t seed) {
    auto prompt = Prompt(seed);
    return MintAssignedRequest(assign_target_, *model_, prompt);
  }

  void MakePrefillResident(
      DynamicBatchScheduler& scheduler,
      CacheManager& cache,
      const std::shared_ptr<Request>& request) {
    scheduler.AddRequest(request);
    cache.Allocate({request});
    request->Schedule();
  }

  void MakeDecodeResident(
      DynamicBatchScheduler& scheduler,
      CacheManager& cache,
      const std::shared_ptr<Request>& request) {
    MakePrefillResident(scheduler, cache, request);
    auto logits = model_->p_device_inputs_->Allocate<float>(
        static_cast<size_t>(model_->config_->model.vocab_size));
    auto cpu_logits = logits.CpuSpan();
    std::fill(cpu_logits.begin(), cpu_logits.end(), 0.0f);
    cpu_logits[5] = 100.0f;
    logits.CopyCpuToDevice();

    const auto before = request->Snapshot();
    RequestStepPlan plan;
    plan.request = request;
    plan.request_id = request.get();
    plan.sequence_length_before = before.current_sequence_length;
    plan.target_cache_slots =
        static_cast<size_t>(before.current_sequence_length);
    RequestTestAccess::PrepareForStep(*request, 1);
    request->SaveStateForTransaction();
    const auto result = request->ApplyLogitsForTransaction(logits);
    request->CommitStateForTransaction();
    request->CommitStep(plan, result);
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<Engine> assign_target_;
};

TEST_F(SchedulerContractTest, DynamicPlansSingleAssignedRequestWithoutAdmissionMutation) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto request = Assigned(10);
  scheduler.AddRequest(request);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, request);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 3u);
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
}

TEST_F(SchedulerContractTest, DynamicScheduledRequestsPreflightBoundedUnseenCapacity) {
  constexpr int kLargeMaxLength = 100'000;
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  request->Params()->search.max_length = kLargeMaxLength;
  request->Assign(assign_target_);
  scheduler.AddRequest(request);
  StepPlan plan;
  ASSERT_TRUE(scheduler.PlanStep(plan).executable);
  ASSERT_EQ(RequestTestAccess::UnseenTokenIndexCapacity(*request), 0u);

  auto scheduled_requests = scheduler.CreateScheduledRequests(plan);

  EXPECT_EQ(scheduled_requests.size(), 1u);
  EXPECT_GE(RequestTestAccess::UnseenTokenIndexCapacity(*request), 1u);
  EXPECT_LT(RequestTestAccess::UnseenTokenIndexCapacity(*request), 1024u);
}

TEST_F(SchedulerContractTest, StaticScheduledRequestsPreflightPreservesGenerationAppend) {
  constexpr int kLargeMaxLength = 100'000;
  model_->config_->engine.dynamic_batching.reset();
  auto cache = std::make_shared<RecordingCacheManager>(
      model_, /*capacity=*/4, nullptr, /*supports_dynamic_batching=*/false);
  StaticBatchScheduler scheduler(model_, cache);
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  request->Params()->search.max_length = kLargeMaxLength;
  request->Assign(assign_target_);
  scheduler.AddRequest(request);
  ASSERT_EQ(RequestTestAccess::UnseenTokenIndexCapacity(*request), 0u);

  auto scheduled_requests = scheduler.Schedule();

  EXPECT_GE(RequestTestAccess::UnseenTokenIndexCapacity(*request), 1u);
  EXPECT_LT(RequestTestAccess::UnseenTokenIndexCapacity(*request), 1024u);
  scheduled_requests.AddDecoderState(std::make_unique<ScriptedDecoderIO>(
      model_, scheduled_requests, cache, /*forced_token=*/5));
  scheduled_requests.GenerateNextTokens();
  ASSERT_TRUE(request->HasUnseenTokens());
  EXPECT_EQ(request->UnseenToken(), 5);
}

TEST_F(SchedulerContractTest, DynamicPreservesRequestOrder) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto first = Assigned(10);
  auto second = Assigned(20);
  auto third = Assigned(30);
  scheduler.AddRequest(first);
  scheduler.AddRequest(second);
  scheduler.AddRequest(third);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 3u);
  EXPECT_EQ(plan.requests[0].request, first);
  EXPECT_EQ(plan.requests[1].request, second);
  EXPECT_EQ(plan.requests[2].request, third);
}

TEST_F(SchedulerContractTest, DynamicHonorsCapacityBackpressure) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/2);
  DynamicBatchScheduler scheduler(model_, cache);

  auto first = Assigned(10);
  auto second = Assigned(20);
  auto third = Assigned(30);
  scheduler.AddRequest(first);
  scheduler.AddRequest(second);
  scheduler.AddRequest(third);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  ASSERT_EQ(plan.requests.size(), 2u);
  EXPECT_EQ(plan.requests[0].request, first);
  EXPECT_EQ(plan.requests[1].request, second);
  EXPECT_EQ(first->status_, RequestStatus::Assigned);
  EXPECT_EQ(second->status_, RequestStatus::Assigned);
  EXPECT_EQ(third->status_, RequestStatus::Assigned);
}

TEST_F(SchedulerContractTest, DynamicRetainsTurnCompleteRequestsUntilExplicitRemoval) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto request = Assigned(10);
  MakePrefillResident(scheduler, *cache, request);
  ASSERT_EQ(cache->AllocatedCount(), 1u);

  request->status_ = RequestStatus::TurnComplete;
  auto second = Assigned(20);
  scheduler.AddRequest(second);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_EQ(cache->deallocate_calls, 0);
  EXPECT_EQ(cache->AllocatedCount(), 1u);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, second);

  scheduler.RemoveRequest(request);

  EXPECT_EQ(cache->deallocate_calls, 1);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
  EXPECT_FALSE(cache->IsResident(request));
}

TEST_F(SchedulerContractTest, DynamicTurnCompleteResidencyAppliesCapacityBackpressure) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/1);
  DynamicBatchScheduler scheduler(model_, cache);

  auto completed = Assigned(10);
  MakePrefillResident(scheduler, *cache, completed);
  completed->status_ = RequestStatus::TurnComplete;

  auto waiting = Assigned(20);
  scheduler.AddRequest(waiting);
  StepPlan plan;

  const auto blocked = scheduler.PlanStep(plan);

  EXPECT_FALSE(blocked.executable);
  EXPECT_TRUE(blocked.capacity_deferred);
  EXPECT_EQ(cache->deallocate_calls, 0);
  EXPECT_EQ(cache->AllocatedCount(), 1u);
  EXPECT_TRUE(plan.requests.empty());

  scheduler.RemoveRequest(completed);
  const auto admitted = scheduler.PlanStep(plan);

  ASSERT_TRUE(admitted.executable);
  EXPECT_EQ(cache->deallocate_calls, 1);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, waiting);
}

TEST_F(SchedulerContractTest, DynamicResidentQueuedRequestIsNotReadmitted) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto request = Assigned(10);
  MakeDecodeResident(scheduler, *cache, request);
  request->status_ = RequestStatus::Assigned;
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, request);
  EXPECT_FALSE(plan.requests[0].newly_admitted);
  EXPECT_EQ(cache->AllocatedCount(), 1u);
}

// Removing a request from the dynamic scheduler deallocates its cache resources immediately.
TEST_F(SchedulerContractTest, DynamicRemoveReleasesCacheResources) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  auto request = Assigned(10);
  MakePrefillResident(scheduler, *cache, request);
  ASSERT_EQ(cache->AllocatedCount(), 1u);

  scheduler.RemoveRequest(request);

  EXPECT_GE(cache->deallocate_calls, 1);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
  EXPECT_FALSE(scheduler.HasPendingRequests());
}

TEST_F(SchedulerContractTest, LegacyDynamicScheduleCannotBypassStepPlan) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);

  EXPECT_THROW(scheduler.Schedule(), std::logic_error);
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
  MakePrefillResident(scheduler, *cache, active);
  ASSERT_EQ(active->status_, RequestStatus::Active);

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
  MakePrefillResident(scheduler, *cache, active);
  cache->SetUnserviceableRequest(active);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  EXPECT_FALSE(result.executable);
  EXPECT_FALSE(result.capacity_deferred);
  EXPECT_EQ(result.outcome.kind,
            StepOutcomeKind::UnserviceableRequest);
  EXPECT_EQ(result.outcome.request_id, active.get());
}

TEST_F(SchedulerContractTest, DecodeFirstOrderingUsesTheExactGlobalBudget) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 5;
  model_->config_->engine.dynamic_batching->max_batch_size = 8;
  DynamicBatchScheduler scheduler(model_, cache);
  auto resident_prefill = Assigned(10);
  auto decode = Assigned(20);
  auto new_prefill = Assigned(30);
  MakePrefillResident(scheduler, *cache, resident_prefill);
  MakeDecodeResident(scheduler, *cache, decode);
  scheduler.AddRequest(new_prefill);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 3u);
  EXPECT_EQ(plan.requests[0].request, decode);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 1u);
  EXPECT_EQ(plan.requests[0].packed_token_offset, 0u);
  EXPECT_EQ(plan.requests[1].request, resident_prefill);
  EXPECT_EQ(plan.requests[1].unprocessed_token_count, 3u);
  EXPECT_EQ(plan.requests[1].packed_token_offset, 1u);
  EXPECT_EQ(plan.requests[2].request, new_prefill);
  EXPECT_EQ(plan.requests[2].unprocessed_token_count, 1u);
  EXPECT_EQ(plan.requests[2].packed_token_offset, 4u);
  EXPECT_EQ(plan.token_count, 5u);
  EXPECT_FALSE(plan.graph_capture_eligible);
}

TEST_F(SchedulerContractTest, DecodeDemandCanExhaustTheGlobalBudget) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 2;
  model_->config_->engine.dynamic_batching->max_batch_size = 8;
  DynamicBatchScheduler scheduler(model_, cache);
  auto first = Assigned(10);
  auto second = Assigned(20);
  auto third = Assigned(30);
  auto prefill = Assigned(40);
  MakeDecodeResident(scheduler, *cache, first);
  MakeDecodeResident(scheduler, *cache, second);
  MakeDecodeResident(scheduler, *cache, third);
  scheduler.AddRequest(prefill);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  ASSERT_EQ(plan.requests.size(), 2u);
  EXPECT_EQ(plan.requests[0].request, first);
  EXPECT_EQ(plan.requests[1].request, second);
  EXPECT_EQ(plan.token_count, 2u);
  EXPECT_TRUE(plan.graph_capture_eligible);
}

TEST_F(SchedulerContractTest, PrefillRespectsChunkAndGlobalCaps) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 2;
  DynamicBatchScheduler scheduler(model_, cache);
  auto request = Assigned(10);
  request->Params()->search.chunk_size = 1;
  scheduler.AddRequest(request);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 1u);
  EXPECT_EQ(plan.token_count, 1u);
  EXPECT_FALSE(plan.graph_capture_eligible);
}

TEST_F(SchedulerContractTest, CacheQueryCapBoundsOversizedRequestChunk) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  cache->SetMaxQueryTokensPerRequest(2);
  DynamicBatchScheduler scheduler(model_, cache);
  auto request = Assigned(10);
  request->Params()->search.chunk_size = 7;
  scheduler.AddRequest(request);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 2u);
  EXPECT_EQ(plan.token_count, 2u);
}

TEST_F(SchedulerContractTest, CacheQueryCapBoundsUnchunkedRequest) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  cache->SetMaxQueryTokensPerRequest(2);
  DynamicBatchScheduler scheduler(model_, cache);
  auto request = Assigned(10);
  request->Params()->search.chunk_size = 0;
  scheduler.AddRequest(request);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 2u);
  EXPECT_EQ(plan.token_count, 2u);
}

TEST_F(SchedulerContractTest, GlobalCapChunksPromptWithoutSearchChunkSize) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 2;
  DynamicBatchScheduler scheduler(model_, cache);
  auto request = Assigned(10);
  ASSERT_FALSE(request->SearchOptions().chunk_size.has_value());
  scheduler.AddRequest(request);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 2u);
  EXPECT_EQ(plan.requests[0].whole_sequence_cache_slots, 3u);
  EXPECT_EQ(plan.requests[0].target_cache_slots, 2u);
  EXPECT_FALSE(plan.graph_capture_eligible);
}

TEST_F(SchedulerContractTest, BlockedPrefillDoesNotHideLaterFittingRequest) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto blocked = Assigned(10);
  auto fitting = Assigned(20);
  scheduler.AddRequest(blocked);
  scheduler.AddRequest(fitting);
  cache->SetCapacityDeferredRequest(blocked);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  EXPECT_TRUE(result.capacity_deferred);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, fitting);
  EXPECT_EQ(plan.token_count, 3u);
}

TEST_F(SchedulerContractTest, DynamicScheduledRequestsRejectInvalidPlanTokenCount) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  DynamicBatchScheduler scheduler(model_, cache);
  auto request = Assigned(10);
  scheduler.AddRequest(request);
  StepPlan plan;
  ASSERT_TRUE(scheduler.PlanStep(plan).executable);
  ASSERT_EQ(plan.requests.size(), 1u);

  plan.requests[0].unprocessed_token_count = 0;
  EXPECT_THROW(scheduler.CreateScheduledRequests(plan), std::runtime_error);

  plan.requests[0].unprocessed_token_count = 4;
  EXPECT_THROW(scheduler.CreateScheduledRequests(plan), std::runtime_error);
}

TEST_F(SchedulerContractTest, ProductionCachePreservesOmittedResident) {
  auto& config = *model_->config_->engine.dynamic_batching;
  config.block_size = 4;
  config.num_blocks = 4;
  config.max_batch_size = 2;
  config.max_scheduled_tokens = 1;
  auto cache = std::make_shared<PagedCacheManager>(model_);
  DynamicBatchScheduler scheduler(model_, cache);

  auto omitted = Assigned(10);
  auto decode = Assigned(20);
  MakePrefillResident(scheduler, *cache, omitted);
  MakeDecodeResident(scheduler, *cache, decode);
  const auto before = cache->Snapshot();
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, decode);
  EXPECT_TRUE(plan.graph_capture_eligible);

  auto reservation = cache->ReserveStep(plan);
  reservation->Commit();
  const auto after = cache->Snapshot();
  ASSERT_EQ(before.requests.size(), 2u);
  ASSERT_EQ(after.requests.size(), 2u);
  EXPECT_EQ(after.requests[0].request_id, omitted.get());
  EXPECT_EQ(after.requests[0].used_slots, before.requests[0].used_slots);
  EXPECT_EQ(after.requests[0].block_ids, before.requests[0].block_ids);
}

TEST_F(SchedulerContractTest, ProductionCacheReservesGloballyChunkedPrompt) {
  auto& config = *model_->config_->engine.dynamic_batching;
  config.block_size = 4;
  config.num_blocks = 3;
  config.max_batch_size = 1;
  config.max_scheduled_tokens = 2;
  auto cache = std::make_shared<PagedCacheManager>(model_);
  DynamicBatchScheduler scheduler(model_, cache);
  auto request = MintAssignedRequest(
      assign_target_, *model_,
      std::array<int32_t, 9>{2, 3, 4, 5, 6, 7, 8, 9, 10});
  scheduler.AddRequest(request);
  StepPlan plan;

  const auto result = scheduler.PlanStep(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].unprocessed_token_count, 2u);
  EXPECT_EQ(plan.requests[0].whole_sequence_cache_slots, 9u);
  auto reservation = cache->ReserveStep(plan);
  ASSERT_NE(reservation->PagedReservation(), nullptr);
  EXPECT_EQ(reservation->PagedReservation()->ReservedBlockCount(), 3u);
}

}  // namespace
}  // namespace test
}  // namespace Generators
