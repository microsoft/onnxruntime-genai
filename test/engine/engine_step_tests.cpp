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

class EngineStepTest : public ::testing::Test {
 protected:
  void SetUp() override { model_ = LoadDummyDecoderModel(); }

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
  EXPECT_TRUE(request->IsDone());
  EXPECT_EQ(engine.executor->decode_calls, 1);
  ASSERT_EQ(engine.executor->decoded_batch_sizes.size(), 1u);
  EXPECT_EQ(engine.executor->decoded_batch_sizes[0], 1u);

  const int allocate_at = IndexOf(*engine.trace, "Allocate");
  const int decode_at = IndexOf(*engine.trace, "Decode");
  ASSERT_GE(allocate_at, 0);
  ASSERT_GE(decode_at, 0);
  EXPECT_LT(decode_at, allocate_at);
}

TEST_F(EngineStepTest, CompletedRequestContinuesAfterToolResponseTokens) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);

  ASSERT_EQ(engine.engine->Step(), request);
  ASSERT_TRUE(request->IsDone());
  while (request->HasUnseenTokens()) {
    static_cast<void>(request->UnseenToken());
  }

  const auto completed = request->Snapshot();
  const std::vector<int32_t> tool_response_tokens{7, 8};
  request->AddTokens(tool_response_tokens);

  EXPECT_EQ(request->status_, RequestStatus::InProgress);
  EXPECT_FALSE(request->HasUnseenTokens());
  EXPECT_TRUE(engine.engine->HasPendingRequests());
  EXPECT_EQ(engine.cache->deallocate_calls, 0);

  engine.executor->SetForcedToken(5);
  ASSERT_EQ(engine.engine->Step(), request);
  EXPECT_FALSE(request->IsDone());
  EXPECT_TRUE(request->HasUnseenTokens());
  EXPECT_EQ(request->UnseenToken(), 5);
  EXPECT_EQ(engine.executor->decode_calls, 2);
  EXPECT_EQ(engine.cache->deallocate_calls, 0);
  EXPECT_GT(request->CurrentSequenceLength(), completed.current_sequence_length);
}

// Several requests that all fit are decoded together in a single batch, and the remaining ready
// requests are drained without any further model execution.
TEST_F(EngineStepTest, FittingRequestsShareOneDecodeAndDrainWithoutReexecuting) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));

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
    EXPECT_TRUE(request->IsDone());
  }
  EXPECT_EQ(engine.executor->decode_calls, 1);
  ASSERT_EQ(engine.executor->decoded_batch_sizes.size(), 1u);
  EXPECT_EQ(engine.executor->decoded_batch_sizes[0], 3u);
  EXPECT_FALSE(engine.engine->HasPendingRequests());
}

// Under capacity backpressure Step decodes only the requests that fit, then forms a fresh batch for
// the deferred request on a later step -- one decode per internal cycle, never an over-capacity run.
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
  while (auto ready = engine.engine->Step()) {
    returned.push_back(ready);
  }

  EXPECT_EQ(returned.size(), 3u);
  auto sorted_requests = requests;
  std::sort(sorted_requests.begin(), sorted_requests.end());
  std::sort(returned.begin(), returned.end());
  EXPECT_EQ(returned, sorted_requests);
  for (const auto& request : requests) {
    EXPECT_TRUE(request->IsDone());
  }
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

TEST_F(EngineStepTest, StaticBatchingRetainsLegacyCommitOrdering) {
  auto trace = std::make_shared<CallTrace>();
  auto cache = std::make_shared<RecordingCacheManager>(
      model_, /*capacity=*/4, trace, /*supports_dynamic_batching=*/false);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, EosToken(*model_), trace);
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
  EXPECT_TRUE(request->IsDone());
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
  EXPECT_TRUE(request->IsDone());
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
  EXPECT_EQ(request->status_, RequestStatus::Unassigned);
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

}  // namespace
}  // namespace test
}  // namespace Generators
