// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Component-contract tests for Engine::Step. They wire an Engine with recording cache-manager
// and model-executor doubles over a tiny CPU fixture model. The doubles never run the model: the
// executor fabricates end-of-stream logits so each scheduled request's real greedy search completes
// deterministically. This lets the tests assert how Step orchestrates its collaborators -- that it
// allocates a batch before decoding it, decodes once per internal cycle, drains ready requests
// without redundant model runs, and forms a fresh batch across steps under capacity backpressure.

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_helpers.h"
#include "engine_test_doubles.h"

namespace Generators {
namespace test {
namespace {

std::vector<int32_t> Prompt(int32_t seed) {
  const int32_t base = 2 + (seed % 8);
  return {base, base + 1, base + 2};
}

// Index of the first occurrence of `entry` in the trace, or -1 if absent.
int IndexOf(const CallTrace& trace, const std::string& entry) {
  auto it = std::find(trace.entries.begin(), trace.entries.end(), entry);
  return it == trace.entries.end() ? -1 : static_cast<int>(it - trace.entries.begin());
}

class ExternalRequestReference {
 public:
  explicit ExternalRequestReference(Request& request) : request_{&request} {
    request_->ExternalAddRef();
  }

  ExternalRequestReference(const ExternalRequestReference&) = delete;
  ExternalRequestReference& operator=(const ExternalRequestReference&) = delete;

  ~ExternalRequestReference() {
    Release();
  }

  void Release() noexcept {
    if (request_) {
      request_->ExternalRelease();
      request_ = nullptr;
    }
  }

 private:
  Request* request_;
};

static_assert(noexcept(std::declval<Request&>().ExternalRelease()));

class EngineStepTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadDummyDecoderModel();
    model_->config_->engine.dynamic_batching =
        Config::Engine::DynamicBatching{};
  }

  std::shared_ptr<Model> model_;
};

// One request: Step decodes the proposed batch exactly once, commits its cache allocation, and
// returns the request.
TEST_F(EngineStepTest, SingleRequestSchedulesThenDecodesThenReturns) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));

  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  engine.engine->AddRequest(request);
  ASSERT_TRUE(engine.engine->HasPendingRequests());

  auto ready = engine.engine->Step();

  ASSERT_NE(ready, nullptr);
  EXPECT_EQ(ready, request);
  EXPECT_TRUE(request->IsTurnComplete());
  EXPECT_EQ(engine.executor->decode_calls, 1);
  ASSERT_EQ(engine.executor->decoded_batch_sizes.size(), 1u);
  EXPECT_EQ(engine.executor->decoded_batch_sizes[0], 1u);

  const int allocate_at = IndexOf(*engine.trace, "Allocate");
  const int decode_at = IndexOf(*engine.trace, "Decode");
  ASSERT_GE(allocate_at, 0);
  ASSERT_GE(decode_at, 0);
  EXPECT_LT(decode_at, allocate_at);
}

// Several requests that all fit are decoded together in a single batch, and the remaining ready
// requests are drained without any further model execution.
TEST_F(EngineStepTest, FittingRequestsShareOneDecodeAndDrainWithoutReexecuting) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));

  // These test-only shared_ptr requests never acquire an external reference. Repeated Step
  // boundaries must not mistake them for abandoned public handles.
  std::vector<std::shared_ptr<Request>> requests;
  for (int32_t seed : {10, 20, 30}) {
    auto prompt = Prompt(seed);
    auto request = MintRequest(*model_, prompt);
    engine.engine->AddRequest(request);
    requests.push_back(request);
  }

  std::vector<std::shared_ptr<Request>> returned;
  while (auto ready = engine.engine->Step()) {
    returned.push_back(ready);
  }

  EXPECT_EQ(returned.size(), 3u);
  auto sorted_requests = requests;
  std::sort(sorted_requests.begin(), sorted_requests.end());
  std::sort(returned.begin(), returned.end());
  EXPECT_EQ(returned, sorted_requests);
  for (const auto& request : requests) {
    EXPECT_TRUE(request->IsTurnComplete());
  }
  EXPECT_EQ(engine.executor->decode_calls, 1);
  ASSERT_EQ(engine.executor->decoded_batch_sizes.size(), 1u);
  EXPECT_EQ(engine.executor->decoded_batch_sizes[0], 3u);
  EXPECT_FALSE(engine.engine->HasPendingRequests());
}

TEST_F(EngineStepTest, AddRequestReclaimsAbandonedTurnCompleteAtCapacity) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/1, EosToken(*model_));
  auto first_prompt = Prompt(10);
  auto first = MintRequest(*model_, first_prompt);
  ExternalRequestReference first_external{*first};
  engine.engine->AddRequest(first);

  ASSERT_EQ(engine.engine->Step(), first);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(engine.cache->AllocatedCount(), 1u);

  first_external.Release();

  auto second_prompt = Prompt(20);
  auto second = MintRequest(*model_, second_prompt);
  ExternalRequestReference second_external{*second};
  engine.engine->AddRequest(second);

  EXPECT_EQ(first->status_, RequestStatus::Closed);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
  EXPECT_EQ(engine.cache->deallocate_calls, 1);

  EXPECT_EQ(engine.engine->Step(), second);
  EXPECT_EQ(second->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);

  engine.engine->RemoveRequest(second);
  second_external.Release();
}

TEST_F(EngineStepTest, StepReclaimsAbandonedTurnCompleteBeforePlanningAtCapacity) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/1, EosToken(*model_));
  auto first_prompt = Prompt(10);
  auto second_prompt = Prompt(20);
  auto first = MintRequest(*model_, first_prompt);
  auto second = MintRequest(*model_, second_prompt);
  ExternalRequestReference first_external{*first};
  ExternalRequestReference second_external{*second};
  engine.engine->AddRequest(first);
  engine.engine->AddRequest(second);

  ASSERT_EQ(engine.engine->Step(), first);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(second->status_, RequestStatus::Assigned);
  ASSERT_EQ(engine.cache->AllocatedCount(), 1u);
  first_external.Release();

  EXPECT_EQ(engine.engine->Step(), second);
  EXPECT_EQ(first->status_, RequestStatus::Closed);
  EXPECT_EQ(second->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);
  EXPECT_EQ(engine.cache->deallocate_calls, 1);
  EXPECT_EQ(engine.executor->decode_calls, 2);

  engine.engine->RemoveRequest(second);
  second_external.Release();
}

TEST_F(EngineStepTest, StepPurgesAbandonedReadyAndQueuedRequestsExactlyOnce) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/2, EosToken(*model_));
  auto survivor_prompt = Prompt(10);
  auto ready_orphan_prompt = Prompt(20);
  auto queued_orphan_prompt = Prompt(30);
  auto survivor = MintRequest(*model_, survivor_prompt);
  auto ready_orphan = MintRequest(*model_, ready_orphan_prompt);
  auto queued_orphan = MintRequest(*model_, queued_orphan_prompt);
  ExternalRequestReference survivor_external{*survivor};
  ExternalRequestReference ready_orphan_external{*ready_orphan};
  ExternalRequestReference queued_orphan_external{*queued_orphan};
  engine.engine->AddRequest(survivor);
  engine.engine->AddRequest(ready_orphan);
  engine.engine->AddRequest(queued_orphan);

  ASSERT_EQ(engine.engine->Step(), survivor);
  ASSERT_EQ(survivor->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(ready_orphan->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(queued_orphan->status_, RequestStatus::Assigned);
  ASSERT_EQ(engine.cache->AllocatedCount(), 2u);

  ready_orphan_external.Release();
  queued_orphan_external.Release();
  ASSERT_EQ(engine.cache->deallocate_calls, 0);

  // Cleanup runs before Step can drain the orphan's ready notification or plan the queued orphan.
  EXPECT_EQ(engine.engine->Step(), nullptr);
  EXPECT_EQ(ready_orphan->status_, RequestStatus::Closed);
  EXPECT_EQ(queued_orphan->status_, RequestStatus::Closed);
  EXPECT_EQ(survivor->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);
  EXPECT_EQ(engine.cache->deallocate_calls, 2);
  EXPECT_EQ(engine.executor->decode_calls, 1);

  // Closed requests were removed from Engine ownership, so another boundary cannot clean them twice.
  EXPECT_EQ(engine.engine->Step(), nullptr);
  EXPECT_EQ(engine.cache->deallocate_calls, 2);

  engine.engine->RemoveRequest(survivor);
  survivor_external.Release();
}

TEST_F(EngineStepTest, ReacquiringExternalReferenceCancelsDeferredAbandonment) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/1, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  ExternalRequestReference initial_external{*request};
  engine.engine->AddRequest(request);
  ASSERT_EQ(engine.engine->Step(), request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);

  initial_external.Release();
  ExternalRequestReference reacquired_external{*request};

  EXPECT_EQ(engine.engine->Step(), nullptr);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);
  EXPECT_EQ(engine.cache->deallocate_calls, 0);

  engine.engine->RemoveRequest(request);
  reacquired_external.Release();
}

TEST_F(EngineStepTest, ContinueRejectsUndrainedReadyNotificationWithoutMutation) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto first_prompt = Prompt(10);
  auto second_prompt = Prompt(20);
  auto first = MintRequest(*model_, first_prompt);
  auto second = MintRequest(*model_, second_prompt);
  engine.engine->AddRequest(first);
  engine.engine->AddRequest(second);

  ASSERT_EQ(engine.engine->Step(), first);
  ASSERT_EQ(engine.executor->decode_calls, 1);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(second->status_, RequestStatus::TurnComplete);

  const auto before = second->Snapshot();
  const std::vector<int32_t> continuation{5, 6};
  try {
    second->Continue(continuation);
    FAIL() << "Expected Continue to reject the undrained ready notification.";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("Engine::Step()"), std::string::npos);
  }

  const auto rejected = second->Snapshot();
  EXPECT_EQ(rejected.status, before.status);
  EXPECT_EQ(rejected.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(rejected.processed_sequence_length, before.processed_sequence_length);

  EXPECT_EQ(engine.engine->Step(), second);
  EXPECT_EQ(engine.executor->decode_calls, 1);

  EXPECT_NO_THROW(second->Continue(continuation));
  const auto continued = second->Snapshot();
  EXPECT_EQ(continued.status, RequestStatus::Assigned);
  EXPECT_EQ(continued.current_sequence_length,
            before.current_sequence_length + static_cast<int64_t>(continuation.size()));
}

TEST_F(EngineStepTest, UnseenPreflightCompactsContinuedOutputWithoutReordering) {
  constexpr int kLargeMaxLength = 100'000;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  request->Params()->search.max_length = kLargeMaxLength;
  engine.engine->AddRequest(request);
  ASSERT_EQ(RequestTestAccess::UnseenTokenIndexCapacity(*request), 0u);

  // The initial scheduled-request construction reserves bounded append capacity before decode.
  ASSERT_EQ(engine.engine->Step(), request);
  EXPECT_GE(RequestTestAccess::UnseenTokenIndexCapacity(*request), 1u);
  EXPECT_LT(RequestTestAccess::UnseenTokenIndexCapacity(*request), 1024u);

  engine.executor->SetForcedToken(6);
  ASSERT_EQ(engine.engine->Step(), request);
  engine.executor->SetForcedToken(7);
  ASSERT_EQ(engine.engine->Step(), request);
  engine.executor->SetForcedToken(EosToken(*model_));
  ASSERT_EQ(engine.engine->Step(), request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(RequestTestAccess::UnseenTokenIndexCount(*request), 3u);

  EXPECT_EQ(request->UnseenToken(), 5);
  EXPECT_EQ(request->UnseenToken(), 6);
  ASSERT_EQ(RequestTestAccess::NextUnseenTokenIndex(*request), 2u);

  const std::vector<int32_t> continuation{8, 9};
  request->Continue(continuation);
  engine.executor->SetForcedToken(10);
  ASSERT_EQ(engine.engine->Step(), request);

  // Continued-step preflight removed the consumed prefix before CommitStep appended token 10.
  EXPECT_EQ(RequestTestAccess::NextUnseenTokenIndex(*request), 0u);
  EXPECT_EQ(RequestTestAccess::UnseenTokenIndexCount(*request), 2u);
  engine.executor->SetForcedToken(EosToken(*model_));
  ASSERT_EQ(engine.engine->Step(), request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);

  EXPECT_EQ(request->UnseenToken(), 7);
  EXPECT_EQ(request->UnseenToken(), 10);
  EXPECT_FALSE(request->HasUnseenTokens());
}

// Under capacity backpressure Step decodes only the requests that fit, then forms a fresh batch for
// the deferred request on a later step -- one decode per internal cycle, never an over-capacity run.
//
// These test requests use only internal shared_ptrs and have never had public handles, so they are
// not abandoned automatically. A finished request keeps its slot until explicitly removed. With
// capacity for two requests, the third is admitted only after a completed request has been removed.
// The assertions pin both halves of that contract -- the slot is still held when the request is
// handed back, and it is released exactly at removal.
TEST_F(EngineStepTest, BackpressureFormsAFreshBatchAcrossSteps) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/2, EosToken(*model_));

  std::vector<std::shared_ptr<Request>> requests;
  for (int32_t seed : {10, 20, 30}) {
    auto prompt = Prompt(seed);
    auto request = MintRequest(*model_, prompt);
    engine.engine->AddRequest(request);
    requests.push_back(request);
  }

  std::vector<std::shared_ptr<Request>> returned;
  int removals = 0;
  while (auto ready = engine.engine->Step()) {
    // The fixture's executor forces end-of-stream, so every request Step hands back is finished.
    EXPECT_TRUE(ready->IsTurnComplete());
    // The finished request still owns its cache slot: nothing is reclaimed implicitly.
    EXPECT_EQ(engine.cache->deallocate_calls, removals);
    returned.push_back(ready);

    engine.engine->RemoveRequest(ready);
    ++removals;
    EXPECT_EQ(engine.cache->deallocate_calls, removals);
  }

  EXPECT_EQ(returned.size(), 3u);
  auto sorted_requests = requests;
  std::sort(sorted_requests.begin(), sorted_requests.end());
  std::sort(returned.begin(), returned.end());
  EXPECT_EQ(returned, sorted_requests);
  for (const auto& request : requests) {
    EXPECT_EQ(request->status_, RequestStatus::Closed);
  }
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
  ASSERT_EQ(engine.executor->decoded_batch_sizes.size(), 2u);
  EXPECT_EQ(engine.executor->decoded_batch_sizes[0], 2u);
  EXPECT_EQ(engine.executor->decoded_batch_sizes[1], 1u);
}

// With nothing queued, Step reports no work and returns cleanly.
TEST_F(EngineStepTest, StepWithNoRequestsReturnsNull) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));

  EXPECT_FALSE(engine.engine->HasPendingRequests());
  EXPECT_EQ(engine.engine->Step(), nullptr);
  EXPECT_EQ(engine.executor->decode_calls, 0);
}

TEST_F(EngineStepTest, StaticBatchingPreservesOrderingAndReusesResidentContinuation) {
  model_->config_->engine.dynamic_batching.reset();
  auto trace = std::make_shared<CallTrace>();
  auto cache = std::make_shared<RecordingCacheManager>(
      model_, /*capacity=*/4, trace, /*supports_dynamic_batching=*/false);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, EosToken(*model_), trace);
  auto* cache_observer = cache.get();
  auto* executor_observer = executor.get();
  EngineDependencies dependencies{cache, std::move(scheduler),
                                  std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  engine->AddRequest(request);

  EXPECT_EQ(engine->Step(), request);
  EXPECT_EQ(executor_observer->decode_calls, 1);
  EXPECT_LT(IndexOf(*trace, "Allocate"), IndexOf(*trace, "Decode"));

  const int allocations_before = cache_observer->allocate_calls;
  const std::vector<int32_t> continuation{5, 6};
  request->Continue(continuation);
  ASSERT_EQ(request->status_, RequestStatus::Assigned);

  EXPECT_EQ(engine->Step(), request);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(executor_observer->decode_calls, 2);
  EXPECT_EQ(cache_observer->allocate_calls, allocations_before);
}

TEST_F(EngineStepTest, StaticStepClosesAbandonedResidentWithoutIndividualDeallocation) {
  model_->config_->engine.dynamic_batching.reset();
  auto cache = std::make_shared<RecordingCacheManager>(
      model_, /*capacity=*/1, nullptr, /*supports_dynamic_batching=*/false);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, EosToken(*model_));
  auto* executor_observer = executor.get();
  EngineDependencies dependencies{cache, std::move(scheduler),
                                  std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  ExternalRequestReference external{*request};
  engine->AddRequest(request);

  ASSERT_EQ(engine->Step(), request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(cache->AllocatedCount(), 1u);
  external.Release();

  EXPECT_EQ(engine->Step(), nullptr);
  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(executor_observer->decode_calls, 1);
  EXPECT_EQ(cache->AllocatedCount(), 1u);
  EXPECT_EQ(cache->deallocate_calls, 0);

  // A static row is logically closed once but remains physically resident until batch recycling.
  EXPECT_EQ(engine->Step(), nullptr);
  EXPECT_EQ(cache->deallocate_calls, 0);
}

TEST_F(EngineStepTest, StaticContinueFailsAfterBatchRecycling) {
  model_->config_->engine.dynamic_batching.reset();
  auto cache = std::make_shared<StaticCacheManager>(model_);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, EosToken(*model_));
  EngineDependencies dependencies{cache, std::move(scheduler),
                                  std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));

  auto first_prompt = Prompt(10);
  auto first = MintRequest(*model_, first_prompt);
  engine->AddRequest(first);
  ASSERT_EQ(engine->Step(), first);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);

  auto second_prompt = Prompt(20);
  auto second = MintRequest(*model_, second_prompt);
  engine->AddRequest(second);
  for (int step = 0; step < 2 && cache->IsResident(first); ++step) {
    ASSERT_NE(engine->Step(), nullptr);
  }
  ASSERT_FALSE(cache->IsResident(first));

  const std::vector<int32_t> continuation{5, 6};
  EXPECT_THROW(first->Continue(continuation), std::runtime_error);
  EXPECT_EQ(first->status_, RequestStatus::TurnComplete);
}

TEST_F(EngineStepTest, StaticContinueRejectsMultiRowBatchAfterPeerCloses) {
  model_->config_->engine.dynamic_batching.reset();
  auto cache = std::make_shared<RecordingCacheManager>(
      model_, /*capacity=*/4, nullptr, /*supports_dynamic_batching=*/false);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, EosToken(*model_));
  EngineDependencies dependencies{cache, std::move(scheduler),
                                  std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));

  auto first_prompt = Prompt(10);
  auto second_prompt = Prompt(20);
  auto first = MintRequest(*model_, first_prompt);
  auto second = MintRequest(*model_, second_prompt);
  engine->AddRequest(first);
  engine->AddRequest(second);
  ASSERT_EQ(engine->Step(), first);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(second->status_, RequestStatus::TurnComplete);

  engine->RemoveRequest(second);
  ASSERT_EQ(second->status_, RequestStatus::Closed);
  const std::vector<int32_t> continuation{5, 6};
  EXPECT_THROW(first->Continue(continuation), std::runtime_error);
  EXPECT_EQ(first->status_, RequestStatus::TurnComplete);
}

TEST_F(EngineStepTest, StepDoesNotReturnNullWhenCapacityDefersPendingWork) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  engine.cache->SetCanAllocate(false);
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  engine.engine->AddRequest(request);

  try {
    EXPECT_NE(engine.engine->Step(), nullptr);
    FAIL() << "Expected a capacity deferral error.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::CapacityDeferred);
  }

  EXPECT_TRUE(engine.engine->HasPendingRequests());
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
}

TEST_F(EngineStepTest, RetryableExecutionFailureRollsBackAndCanRetry) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  engine.engine->AddRequest(request);
  const auto before = request->Snapshot();
  engine.executor->SetNextFailure(
      ScriptedExecutionFailure::RetryableDuringExecution);

  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected retryable execution failure.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::RetryableBatchAbort);
  }

  const auto rolled_back = request->Snapshot();
  EXPECT_EQ(rolled_back.status, before.status);
  EXPECT_EQ(rolled_back.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
  EXPECT_TRUE(engine.engine->HasPendingRequests());

  auto ready = engine.engine->Step();
  EXPECT_EQ(ready, request);
  EXPECT_TRUE(request->IsTurnComplete());
}

TEST_F(EngineStepTest, ContinuedResidentRollsBackToQueuedAndCanRetry) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  engine.engine->AddRequest(request);
  ASSERT_EQ(engine.engine->Step(), request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(engine.cache->AllocatedCount(), 1u);

  const std::vector<int32_t> continuation{5, 6};
  request->Continue(continuation);
  const auto before = request->Snapshot();
  ASSERT_EQ(before.status, RequestStatus::Assigned);
  engine.executor->SetNextFailure(
      ScriptedExecutionFailure::RetryableDuringExecution);

  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected retryable execution failure.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::RetryableBatchAbort);
  }

  const auto rolled_back = request->Snapshot();
  EXPECT_EQ(rolled_back.status, RequestStatus::Assigned);
  EXPECT_EQ(rolled_back.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(rolled_back.processed_sequence_length,
            before.processed_sequence_length);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);

  auto ready = engine.engine->Step();
  EXPECT_EQ(ready, request);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
}

TEST_F(EngineStepTest, ContinuedResidentUsesChunkedPrefillBeforeSampling) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  request->Params()->search.chunk_size = 2;
  engine.engine->AddRequest(request);
  ASSERT_EQ(engine.engine->Step(), request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);

  const size_t calls_before = engine.executor->decoded_token_counts.size();
  const std::vector<int32_t> continuation{5, 6, 7, 8, 9};
  request->Continue(continuation);
  ASSERT_EQ(engine.engine->Step(), request);

  ASSERT_EQ(engine.executor->decoded_token_counts.size(), calls_before + 3);
  EXPECT_EQ(engine.executor->decoded_token_counts[calls_before], 2u);
  EXPECT_EQ(engine.executor->decoded_token_counts[calls_before + 1], 2u);
  EXPECT_EQ(engine.executor->decoded_token_counts[calls_before + 2], 1u);
  EXPECT_EQ(request->ProcessedSequenceLength(),
            request->CurrentSequenceLength());
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
}

TEST_F(EngineStepTest, UnserviceableContinuationRemainsQueuedUntilClosed) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  engine.engine->AddRequest(request);
  ASSERT_EQ(engine.engine->Step(), request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);

  const std::vector<int32_t> continuation{5, 6};
  request->Continue(continuation);
  engine.cache->SetUnserviceableRequest(request);

  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected an unserviceable continuation error.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::UnserviceableRequest);
  }

  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);
  EXPECT_NO_THROW(engine.engine->RemoveRequest(request));
  EXPECT_EQ(request->status_, RequestStatus::Closed);
}

TEST_F(EngineStepTest, ExecutionCapacityFailureRollsBackWithoutPoisoningEngine) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  engine.engine->AddRequest(request);
  const auto before = request->Snapshot();
  engine.executor->SetNextFailure(ScriptedExecutionFailure::CapacityExceeded);

  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected execution capacity failure.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind,
              StepOutcomeKind::ExecutionCapacityExceeded);
  }

  const auto rolled_back = request->Snapshot();
  EXPECT_EQ(rolled_back.status, before.status);
  EXPECT_EQ(rolled_back.current_sequence_length,
            before.current_sequence_length);
  EXPECT_EQ(rolled_back.processed_sequence_length,
            before.processed_sequence_length);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
  EXPECT_TRUE(engine.engine->HasPendingRequests());

  auto ready = engine.engine->Step();
  EXPECT_EQ(ready, request);
  EXPECT_TRUE(request->IsTurnComplete());
}

TEST_F(EngineStepTest, PostProcessingFailureRestoresSearchAndCanRetry) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  engine.engine->AddRequest(request);
  const auto before = request->Snapshot();
  engine.executor->SetNextFailure(ScriptedExecutionFailure::PostProcessing);

  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected post-processing failure.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::RetryableBatchAbort);
    EXPECT_NE(std::string{error.what()}.find(
                  "Injected post-processing failure."),
              std::string::npos);
  }

  const auto rolled_back = request->Snapshot();
  EXPECT_EQ(rolled_back.status, before.status);
  EXPECT_EQ(rolled_back.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(rolled_back.processed_sequence_length,
            before.processed_sequence_length);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);

  auto ready = engine.engine->Step();
  EXPECT_EQ(ready, request);
  EXPECT_TRUE(request->IsTurnComplete());
}

TEST_F(EngineStepTest, LaterRequestFailureRestoresEarlierSample) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto first_prompt = Prompt(10);
  auto second_prompt = Prompt(20);
  auto first = MintRequest(*model_, first_prompt);
  auto second = MintRequest(*model_, second_prompt);
  engine.engine->AddRequest(first);
  engine.engine->AddRequest(second);
  const auto first_before = first->Snapshot();
  const auto second_before = second->Snapshot();

  auto second_params = second->Params();
  second_params->search.do_sample = true;
  second_params->search.top_k = 2;
  second_params->search.top_p = -1.0f;

  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected the second request to reject invalid sampling parameters.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::RetryableBatchAbort);
  }

  EXPECT_EQ(first->Snapshot().current_sequence_length,
            first_before.current_sequence_length);
  EXPECT_EQ(second->Snapshot().current_sequence_length,
            second_before.current_sequence_length);
  EXPECT_EQ(first->status_, RequestStatus::Assigned);
  EXPECT_EQ(second->status_, RequestStatus::Assigned);
  EXPECT_FALSE(first->HasUnseenTokens());
  EXPECT_FALSE(second->HasUnseenTokens());
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);

  second_params->search.top_p = 1.0f;
  auto ready = engine.engine->Step();
  EXPECT_EQ(ready, first);
  EXPECT_EQ(first->CurrentSequenceLength(),
            first_before.current_sequence_length + 1);
  EXPECT_EQ(second->CurrentSequenceLength(),
            second_before.current_sequence_length + 1);
}

TEST_F(EngineStepTest, RemovingUndrainedReadyRequestPurgesItFromQueue) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto first_prompt = Prompt(10);
  auto second_prompt = Prompt(20);
  auto first = MintRequest(*model_, first_prompt);
  auto second = MintRequest(*model_, second_prompt);
  engine.engine->AddRequest(first);
  engine.engine->AddRequest(second);

  ASSERT_EQ(engine.engine->Step(), first);
  ASSERT_TRUE(second->HasUnseenTokens());
  engine.engine->RemoveRequest(second);
  ASSERT_EQ(second->status_, RequestStatus::Closed);

  EXPECT_EQ(engine.engine->Step(), first);
}

TEST_F(EngineStepTest, RemovingOnlyDrainedReadyRequestClearsQueue) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  engine.engine->AddRequest(request);

  ASSERT_EQ(engine.engine->Step(), request);
  engine.engine->RemoveRequest(request);

  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(engine.engine->Step(), nullptr);
}

TEST_F(EngineStepTest, StaticTurnCompleteRowIsNotRepublishedWhilePeerRuns) {
  model_->config_->engine.dynamic_batching.reset();
  auto cache = std::make_shared<RecordingCacheManager>(
      model_, /*capacity=*/4, nullptr, /*supports_dynamic_batching=*/false);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, /*forced_token=*/5);
  auto* executor_observer = executor.get();
  EngineDependencies dependencies{cache, std::move(scheduler),
                                  std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));

  auto first_prompt = Prompt(10);
  auto second_prompt = Prompt(20);
  auto first = MintRequest(*model_, first_prompt);
  auto second = MintRequest(*model_, second_prompt);
  first->Params()->search.max_length =
      static_cast<int>(first_prompt.size() + 1);
  second->Params()->search.max_length =
      static_cast<int>(second_prompt.size() + 3);
  engine->AddRequest(first);
  engine->AddRequest(second);

  ASSERT_EQ(engine->Step(), first);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);
  while (first->HasUnseenTokens()) {
    static_cast<void>(first->UnseenToken());
  }
  ASSERT_EQ(engine->Step(), second);  // Drain the other result from the same model run.
  while (second->HasUnseenTokens()) {
    static_cast<void>(second->UnseenToken());
  }
  ASSERT_EQ(executor_observer->decode_calls, 1);

  EXPECT_EQ(engine->Step(), second);
  EXPECT_EQ(executor_observer->decode_calls, 2);
}

TEST_F(EngineStepTest, FatalExecutionFailureMarksEngineUnhealthy) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  engine.engine->AddRequest(request);
  engine.executor->SetNextFailure(ScriptedExecutionFailure::Fatal);

  for (int attempt = 0; attempt < 2; ++attempt) {
    try {
      static_cast<void>(engine.engine->Step());
      FAIL() << "Expected fatal execution failure.";
    } catch (const EngineStepError& error) {
      EXPECT_EQ(error.Outcome().kind,
                StepOutcomeKind::FatalExecutionFailure);
    }
  }

  EXPECT_EQ(engine.executor->decode_calls, 1);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
  request->Remove();
  EXPECT_EQ(request->status_, RequestStatus::Closed);
}

TEST_F(EngineStepTest, ContinueIsRejectedAfterEngineBecomesUnhealthy) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto first_prompt = Prompt(10);
  auto first = MintRequest(*model_, first_prompt);
  engine.engine->AddRequest(first);
  ASSERT_EQ(engine.engine->Step(), first);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);

  auto second_prompt = Prompt(20);
  auto second = MintRequest(*model_, second_prompt);
  engine.engine->AddRequest(second);
  engine.executor->SetNextFailure(ScriptedExecutionFailure::Fatal);
  EXPECT_THROW(static_cast<void>(engine.engine->Step()), EngineStepError);

  const auto before = first->Snapshot();
  const std::vector<int32_t> continuation{5, 6};
  EXPECT_THROW(first->Continue(continuation), EngineStepError);
  const auto after = first->Snapshot();
  EXPECT_EQ(after.status, RequestStatus::TurnComplete);
  EXPECT_EQ(after.current_sequence_length, before.current_sequence_length);
}

TEST_F(EngineStepTest, UnserviceableRequestDoesNotBlockFittingRequest) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto large_prompt = Prompt(10);
  auto fitting_prompt = Prompt(20);
  auto too_large = MintRequest(*model_, large_prompt);
  auto fitting = MintRequest(*model_, fitting_prompt);
  engine.engine->AddRequest(too_large);
  engine.engine->AddRequest(fitting);
  engine.cache->SetUnserviceableRequest(too_large);

  EXPECT_EQ(engine.engine->Step(), fitting);
  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected the remaining unserviceable request to fail.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind,
              StepOutcomeKind::UnserviceableRequest);
    EXPECT_EQ(error.Outcome().request_id, too_large.get());
  }
}

TEST_F(EngineStepTest, LaterFailurePreservesEarlierCommittedCycle) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/1, /*forced_token=*/5);
  auto first_prompt = Prompt(10);
  auto second_prompt = Prompt(20);
  auto first = MintRequest(*model_, first_prompt);
  auto second = MintRequest(*model_, second_prompt);
  engine.engine->AddRequest(first);
  engine.engine->AddRequest(second);

  EXPECT_EQ(engine.engine->Step(), first);
  const auto committed = first->Snapshot();
  ASSERT_EQ(engine.cache->AllocatedCount(), 1u);
  engine.executor->SetNextFailure(
      ScriptedExecutionFailure::RetryableBeforeExecution);

  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected the later cycle to abort.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::RetryableBatchAbort);
  }

  const auto after_abort = first->Snapshot();
  EXPECT_EQ(after_abort.status, committed.status);
  EXPECT_EQ(after_abort.current_sequence_length,
            committed.current_sequence_length);
  EXPECT_EQ(after_abort.processed_sequence_length,
            committed.processed_sequence_length);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);
}

TEST_F(EngineStepTest, MixedDecodeAndPrefillCommitPlanOwnedTokenCounts) {
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 3;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto first_prompt = Prompt(10);
  auto decode = MintRequest(*model_, first_prompt);
  engine.engine->AddRequest(decode);
  ASSERT_EQ(engine.engine->Step(), decode);
  const auto decode_before_mixed = decode->Snapshot();

  const std::vector<int32_t> long_prompt{2, 3, 4, 5, 6};
  auto prefill = MintRequest(*model_, long_prompt);
  engine.engine->AddRequest(prefill);

  EXPECT_EQ(engine.engine->Step(), decode);

  ASSERT_EQ(engine.executor->decoded_batch_sizes.size(), 2u);
  EXPECT_EQ(engine.executor->decoded_batch_sizes[1], 2u);
  ASSERT_EQ(engine.executor->decoded_token_counts.size(), 2u);
  EXPECT_EQ(engine.executor->decoded_token_counts[1], 3u);
  ASSERT_EQ(engine.executor->decoded_request_ids[1].size(), 2u);
  EXPECT_EQ(engine.executor->decoded_request_ids[1][0], decode.get());
  EXPECT_EQ(engine.executor->decoded_request_ids[1][1], prefill.get());
  EXPECT_EQ(decode->ProcessedSequenceLength(),
            decode_before_mixed.processed_sequence_length + 1);
  EXPECT_EQ(prefill->ProcessedSequenceLength(), 2);
  EXPECT_TRUE(prefill->IsPrefill());
}

TEST_F(EngineStepTest, MixedStepRollbackPreservesProgressAndCacheResidents) {
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 3;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto first_prompt = Prompt(10);
  auto decode = MintRequest(*model_, first_prompt);
  engine.engine->AddRequest(decode);
  ASSERT_EQ(engine.engine->Step(), decode);

  const std::vector<int32_t> long_prompt{2, 3, 4, 5, 6};
  auto prefill = MintRequest(*model_, long_prompt);
  engine.engine->AddRequest(prefill);
  const auto decode_before = decode->Snapshot();
  const auto prefill_before = prefill->Snapshot();
  ASSERT_EQ(engine.cache->AllocatedCount(), 1u);
  engine.executor->SetNextFailure(
      ScriptedExecutionFailure::RetryableDuringExecution);

  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected the mixed step to roll back.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::RetryableBatchAbort);
  }

  const auto decode_after = decode->Snapshot();
  const auto prefill_after = prefill->Snapshot();
  EXPECT_EQ(decode_after.current_sequence_length,
            decode_before.current_sequence_length);
  EXPECT_EQ(decode_after.processed_sequence_length,
            decode_before.processed_sequence_length);
  EXPECT_EQ(prefill_after.status, prefill_before.status);
  EXPECT_EQ(prefill_after.current_sequence_length,
            prefill_before.current_sequence_length);
  EXPECT_EQ(prefill_after.processed_sequence_length,
            prefill_before.processed_sequence_length);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);

  EXPECT_EQ(engine.engine->Step(), decode);
  EXPECT_EQ(prefill->ProcessedSequenceLength(), 2);
  EXPECT_EQ(engine.cache->AllocatedCount(), 2u);
}

}  // namespace
}  // namespace test
}  // namespace Generators
