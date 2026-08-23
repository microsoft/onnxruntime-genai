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
#include <unordered_map>
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

// Number of per-request elements (product of every non-batch axis) in a fixed-state tensor row.
size_t FixedRowElements(const OrtValue& value) {
  const auto shape = value.GetTensorTypeAndShapeInfo()->GetShape();
  size_t elements = 1;
  for (size_t axis = 1; axis < shape.size(); ++axis) {
    elements *= static_cast<size_t>(shape[axis]);
  }
  return elements;
}

// Asserts every element of one gathered fixed-state input row equals `expected`. The composite
// tests run on CPU, so the staging tensors are directly addressable.
void ExpectFixedInputRow(const FixedStateBinding& binding, size_t row, float expected) {
  const size_t row_elements = FixedRowElements(*binding.input);
  const auto* data = binding.input->GetTensorData<float>();
  for (size_t index = 0; index < row_elements; ++index) {
    EXPECT_FLOAT_EQ(data[row * row_elements + index], expected);
  }
}

// Fills one staged fixed-state output row with `value`, simulating a model writing its next state.
void FillFixedOutputRow(const FixedStateBinding& binding, size_t row, float value) {
  const size_t row_elements = FixedRowElements(*binding.output);
  auto* data = binding.output->GetTensorMutableData<float>();
  std::fill_n(data + row * row_elements, row_elements, value);
}

const FixedStateSlotSnapshot& FixedSlotFor(const FixedStatePoolSnapshot& snapshot,
                                           const void* request_id) {
  const auto slot = std::find_if(
      snapshot.slots.begin(), snapshot.slots.end(),
      [request_id](const FixedStateSlotSnapshot& candidate) {
        return candidate.request_id == request_id;
      });
  if (slot == snapshot.slots.end()) {
    throw std::logic_error("Fixed state slot was not found.");
  }
  return *slot;
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

MtpDoublesEngine MakeMtpDoublesEngine(
    std::shared_ptr<Model> model, size_t capacity,
    int32_t target_token, int32_t draft_token) {
  auto mtp_model = std::dynamic_pointer_cast<DecoderOnly_Model>(LoadSyntheticPagedModel());
  if (!mtp_model) {
    throw std::logic_error("MTP Engine tests require a decoder-only auxiliary model.");
  }
  mtp_model->config_->engine.dynamic_batching = Config::Engine::DynamicBatching{};
  mtp_model->config_->model.decoder.inputs.hidden_states = "past_key_values.1.key";

  auto device = std::make_unique<CountingCudaDevice>();
  auto device_state = device->state;
  mtp_model->p_device_ = device.get();
  mtp_model->p_device_scoring_ = device.get();

  auto cache = std::make_shared<RecordingCacheManager>(model, capacity);
  cache->SetMaxDraftTokensPerStep(3);
  auto scheduler = Scheduler::Create(model, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(model, cache, target_token);
  executor->EnableHiddenStatesOutput(model->config_->model.decoder.hidden_size);

  auto mtp_cache = std::make_shared<RecordingCacheManager>(mtp_model, capacity);
  auto mtp_executor = std::make_unique<RecordingModelExecutor>(
      mtp_model, mtp_cache, draft_token);
  mtp_executor->EnableHiddenStatesOutput(model->config_->model.decoder.hidden_size);

  auto* cache_observer = cache.get();
  auto* executor_observer = executor.get();
  auto* mtp_cache_observer = mtp_cache.get();
  auto* mtp_executor_observer = mtp_executor.get();

  EngineDependencies dependencies{
      std::move(cache), std::move(scheduler), std::move(executor),
      std::move(mtp_model), std::move(mtp_cache), std::move(mtp_executor)};
  auto engine = std::make_shared<Engine>(std::move(model), std::move(dependencies));
  return MtpDoublesEngine{
      std::move(device), std::move(engine), cache_observer, executor_observer,
      mtp_cache_observer, mtp_executor_observer, std::move(device_state)};
}

void AdvanceUntilTargetDecodeCount(MtpDoublesEngine& engine, int decode_calls) {
  while (engine.executor->decode_calls < decode_calls) {
    ASSERT_TRUE(engine.engine->HasPendingRequests());
    static_cast<void>(engine.engine->Step());
  }
}

std::vector<int32_t> DrainTokens(const std::shared_ptr<Request>& request) {
  std::vector<int32_t> tokens;
  while (request->HasUnseenTokens()) {
    tokens.push_back(request->UnseenToken());
  }
  return tokens;
}

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

TEST_F(EngineStepTest, ContinuedUnreadOutputPreservesOrder) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  engine.engine->AddRequest(request);

  ASSERT_EQ(engine.engine->Step(), request);

  engine.executor->SetForcedToken(6);
  ASSERT_EQ(engine.engine->Step(), request);
  engine.executor->SetForcedToken(7);
  ASSERT_EQ(engine.engine->Step(), request);
  engine.executor->SetForcedToken(EosToken(*model_));
  ASSERT_EQ(engine.engine->Step(), request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);

  EXPECT_EQ(request->UnseenToken(), 5);
  EXPECT_EQ(request->UnseenToken(), 6);

  const std::vector<int32_t> continuation{8, 9};
  request->Continue(continuation);
  engine.executor->SetForcedToken(10);
  ASSERT_EQ(engine.engine->Step(), request);

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

// ---------------------------------------------------------------------------------------------
// Composite decoder-state transactions (paged KV + fixed state pool)
//
// These wire the real PagedCacheManager (real PagedKeyValueCache and real FixedStatePool) over the
// recording model-executor double. The executor fabricates logits and never runs the ONNX graph, so
// the fixed staging tensors are written by the execution callback in scheduled row order.
// ---------------------------------------------------------------------------------------------

TEST_F(EngineStepTest, DensePagedModelHasNoFixedStateReservation) {
  // A paged model with no fixed groups keeps the dense path: the composite manager owns no fixed
  // pool, plans no fixed rows, and the reservation exposes no fixed state.
  model_ = LoadSyntheticPagedModel();
  auto engine = MakeCompositeDoublesEngine(model_, EosToken(*model_));
  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  engine.executor->SetExecutionCallback([](ExecutionContext& context) {
    ASSERT_NE(context.plan, nullptr);
    EXPECT_FALSE(context.plan->fixed_state.required);
    EXPECT_EQ(context.plan->fixed_state.row_count, 0u);
    EXPECT_EQ(context.plan->fixed_state.new_slot_count, 0u);
    EXPECT_EQ(context.plan->fixed_state.staging_bytes, 0u);
    EXPECT_TRUE(context.fixed_state_slots.empty());
    EXPECT_TRUE(context.fixed_state_bindings.empty());
    EXPECT_EQ(context.fixed_state_staging_bytes, 0u);
  });

  EXPECT_EQ(engine.engine->Step(), request);
  EXPECT_FALSE(engine.cache->FixedStateSnapshot().has_value());
}

TEST_F(EngineStepTest, CompositeReservationExposesRowsAndCommitsBothStates) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto first = MintRequest(*model_, Prompt(10));
  auto second = MintRequest(*model_, Prompt(20));
  engine.engine->AddRequest(first);
  engine.engine->AddRequest(second);

  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    ASSERT_NE(context.plan, nullptr);
    ASSERT_TRUE(context.plan->fixed_state.required);
    EXPECT_EQ(context.plan->fixed_state.row_count, 2u);
    EXPECT_EQ(context.plan->fixed_state.new_slot_count, 2u);
    EXPECT_GT(context.fixed_state_staging_bytes, 0u);
    EXPECT_EQ(context.fixed_state_staging_bytes, context.plan->fixed_state.staging_bytes);
    ASSERT_EQ(context.fixed_state_slots.size(), 2u);
    // conv layers [0, 3] plus recurrent layers [2, 5] => four fixed tensors.
    ASSERT_EQ(context.fixed_state_bindings.size(), 4u);
    for (size_t row = 0; row < context.plan->requests.size(); ++row) {
      EXPECT_EQ(context.fixed_state_slots[row].request_id, context.plan->requests[row].request_id);
      for (const auto& binding : context.fixed_state_bindings) {
        ExpectFixedInputRow(binding, row, 0.0f);  // fresh admissions gather the zero row
        FillFixedOutputRow(binding, row, row == 0 ? 11.0f : 22.0f);
      }
    }
    // Mid-transaction nothing is published yet, so committed ownership on both sides is empty and the
    // combined (committed + reserved) snapshot still satisfies the composite invariants.
    const auto fixed = engine.cache->FixedStateSnapshot();
    ASSERT_TRUE(fixed.has_value());
    EXPECT_TRUE(ValidateCompositeStateInvariants(
                    engine.cache->Snapshot(context.cache_reservation), *fixed,
                    {first->Snapshot(), second->Snapshot()})
                    .empty());
  });

  EXPECT_EQ(engine.engine->Step(), first);
  const auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(fixed->committed_slots, 2u);
  EXPECT_EQ(fixed->reserved_slots, 0u);
  EXPECT_EQ(fixed->free_slots, fixed->capacity - 2);
  EXPECT_EQ(FixedSlotFor(*fixed, first.get()).state_generation, 1u);
  EXPECT_EQ(FixedSlotFor(*fixed, first.get()).committed_tokens, 3u);
  EXPECT_EQ(FixedSlotFor(*fixed, second.get()).state_generation, 1u);
  EXPECT_EQ(FixedSlotFor(*fixed, second.get()).committed_tokens, 3u);
  EXPECT_TRUE(ValidateCompositeStateInvariants(
                  engine.cache->Snapshot(), *fixed,
                  {first->Snapshot(), second->Snapshot()})
                  .empty());
}

TEST_F(EngineStepTest, CompositeExecutionFailureDiscardsBothAndRetryMatches) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  int callback_calls = 0;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    ++callback_calls;
    ASSERT_EQ(context.fixed_state_slots.size(), 1u);
    for (const auto& binding : context.fixed_state_bindings) {
      ExpectFixedInputRow(binding, 0, 0.0f);
      FillFixedOutputRow(binding, 0, 7.0f);
    }
  });
  engine.executor->SetNextFailure(ScriptedExecutionFailure::RetryableDuringExecution);

  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected a retryable execution failure.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::RetryableBatchAbort);
  }

  // Execution failed before prepare, so both the provisional fixed slot and the reserved paged
  // blocks are discarded and the request made no progress.
  auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(fixed->committed_slots, 0u);
  EXPECT_EQ(fixed->reserved_slots, 0u);
  EXPECT_EQ(fixed->free_slots, fixed->capacity);
  EXPECT_TRUE(engine.cache->Snapshot().requests.empty());
  EXPECT_EQ(request->Snapshot().processed_sequence_length, 0);

  EXPECT_EQ(engine.engine->Step(), request);
  fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(callback_calls, 2);
  EXPECT_EQ(fixed->committed_slots, 1u);
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).state_generation, 1u);
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).committed_tokens, 3u);
}

TEST_F(EngineStepTest, CompositePostProcessingFailurePreservesResidentState) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  float expected_input = 0.0f;
  float staged_output = 5.0f;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    for (const auto& binding : context.fixed_state_bindings) {
      ExpectFixedInputRow(binding, 0, expected_input);
      FillFixedOutputRow(binding, 0, staged_output);
    }
  });

  ASSERT_EQ(engine.engine->Step(), request);  // prefill commits: state_gen 1, committed 3, bank = 5
  expected_input = 5.0f;                      // the decode step gathers the published value
  staged_output = 99.0f;
  engine.executor->SetNextFailure(ScriptedExecutionFailure::PostProcessing);
  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected a post-processing failure.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::RetryableBatchAbort);
  }

  // The failure happened before prepare, so the resident's committed fixed state and paged progress
  // are untouched: the 99.0 staged output was never published.
  auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).state_generation, 1u);
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).committed_tokens, 3u);
  ASSERT_EQ(engine.cache->Snapshot().requests.size(), 1u);
  EXPECT_EQ(engine.cache->Snapshot().requests[0].used_slots, 3u);

  staged_output = 6.0f;  // the retry still gathers 5.0 and now publishes 6.0
  EXPECT_EQ(engine.engine->Step(), request);
  fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).state_generation, 2u);
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).committed_tokens, 4u);
}

TEST_F(EngineStepTest, CompositeReservationRequiredMismatchIsFatal) {
  // The plan claims fixed state is required, but the reservation exposes none. The Engine must catch
  // the divergence at the reservation boundary, treat it as fatal, and stay unhealthy afterwards.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/4, EosToken(*model_));
  engine.cache->ScriptFixedStateMismatch(
      FixedStateResourcePlan{true, 1, 1, 256}, /*slots=*/{}, /*staging_bytes=*/0);
  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);

  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected a fatal plan/reservation mismatch.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::ExecutionContractFailure);
  }
  EXPECT_THROW(static_cast<void>(engine.engine->Step()), EngineStepError);
}

TEST_F(EngineStepTest, CompositeReservationOverreportedRowsAreFatal) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/4, EosToken(*model_));
  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  static const char extra_request_storage{};
  // A buggy cache manager reports two fixed rows for a one-request plan. The Engine must reject the
  // row count before indexing the plan with either reservation row.
  engine.cache->ScriptFixedStateMismatch(
      FixedStateResourcePlan{true, 2, 1, 0},
      {
          FixedStateSlotHandle{nullptr, request.get(), 0, 0},
          FixedStateSlotHandle{nullptr, &extra_request_storage, 1, 0},
      },
      /*staging_bytes=*/0);

  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected a fatal fixed-state row-count mismatch.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::ExecutionContractFailure);
  }
}

TEST_F(EngineStepTest, CompositeReservationRowOrderMismatchIsFatal) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/4, EosToken(*model_));
  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  static const char other_storage{};
  // Row count and staging bytes match the plan, but the single fixed slot names a different request:
  // the per-row identity guard must fail fatally.
  engine.cache->ScriptFixedStateMismatch(
      FixedStateResourcePlan{true, 1, 1, 0},
      {FixedStateSlotHandle{nullptr, &other_storage, 0, 0}},
      /*staging_bytes=*/0);

  try {
    static_cast<void>(engine.engine->Step());
    FAIL() << "Expected a fatal fixed-state row-order mismatch.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::ExecutionContractFailure);
  }
}

TEST_F(EngineStepTest, CompositeCapacityBackpressureDefersNewAdmission) {
  model_ = LoadSyntheticCompositeModel();
  model_->config_->engine.dynamic_batching->max_batch_size = 2;
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto first = MintRequest(*model_, Prompt(10));
  auto second = MintRequest(*model_, Prompt(20));
  auto third = MintRequest(*model_, Prompt(30));
  engine.engine->AddRequest(first);
  engine.engine->AddRequest(second);
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    for (size_t row = 0; row < context.fixed_state_slots.size(); ++row) {
      for (const auto& binding : context.fixed_state_bindings) {
        FillFixedOutputRow(binding, row, 1.0f);
      }
    }
  });

  // Fill the batch (== fixed capacity of 2) with two committed residents.
  ASSERT_EQ(engine.engine->Step(), first);
  ASSERT_EQ(engine.engine->Step(), second);
  auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  ASSERT_EQ(fixed->capacity, 2u);
  EXPECT_EQ(fixed->committed_slots, 2u);
  EXPECT_EQ(fixed->free_slots, 0u);

  // A third request cannot be admitted while the fixed pool (== batch) is full: planning defers it,
  // so the step reserves no new fixed slot and the committed count is unchanged.
  engine.engine->AddRequest(third);
  size_t observed_new_slots = 999;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    observed_new_slots = context.plan->fixed_state.new_slot_count;
    for (size_t row = 0; row < context.fixed_state_slots.size(); ++row) {
      for (const auto& binding : context.fixed_state_bindings) {
        FillFixedOutputRow(binding, row, 2.0f);
      }
    }
  });
  ASSERT_EQ(engine.engine->Step(), first);
  EXPECT_EQ(observed_new_slots, 0u);
  fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(fixed->committed_slots, 2u);
  EXPECT_EQ(fixed->free_slots, 0u);
  EXPECT_FALSE(engine.cache->IsResident(third));
}

TEST_F(EngineStepTest, CompositeRemovalReleasesBothAndIsolatesSibling) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto first = MintRequest(*model_, Prompt(10));
  auto second = MintRequest(*model_, Prompt(20));
  engine.engine->AddRequest(first);
  engine.engine->AddRequest(second);
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    for (size_t row = 0; row < context.fixed_state_slots.size(); ++row) {
      for (const auto& binding : context.fixed_state_bindings) {
        FillFixedOutputRow(binding, row, 10.0f + static_cast<float>(row));
      }
    }
  });

  ASSERT_EQ(engine.engine->Step(), first);  // both committed in one batch
  auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  ASSERT_EQ(fixed->committed_slots, 2u);
  const auto second_generation = FixedSlotFor(*fixed, second.get()).state_generation;

  first->Remove();
  EXPECT_FALSE(engine.cache->IsResident(first));

  // Removing one request releases both its paged and fixed ownership and leaves the sibling intact.
  fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(fixed->committed_slots, 1u);
  EXPECT_EQ(fixed->free_slots, fixed->capacity - 1);
  EXPECT_EQ(FixedSlotFor(*fixed, second.get()).state_generation, second_generation);
  ASSERT_EQ(engine.cache->Snapshot().requests.size(), 1u);
  EXPECT_EQ(engine.cache->Snapshot().requests[0].request_id, second.get());
  EXPECT_TRUE(ValidateCompositeStateInvariants(
                  engine.cache->Snapshot(), *fixed, {second->Snapshot()})
                  .empty());
}

TEST_F(EngineStepTest, CompositeStagedOutputInvisibleUntilPublish) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  // Each step gathers the value the previous commit published, never the value being staged in the
  // current step. That proves a staged fixed output stays invisible (in the inactive bank) until the
  // transaction publishes it. The pool-level bank-flip invisibility is covered directly in
  // fixed_state_pool_tests.cpp; this asserts the same property across the Engine transaction.
  float committed_value = 0.0f;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    for (const auto& binding : context.fixed_state_bindings) {
      ExpectFixedInputRow(binding, 0, committed_value);
      FillFixedOutputRow(binding, 0, committed_value + 1.0f);
    }
    committed_value += 1.0f;
  });

  EXPECT_EQ(engine.engine->Step(), request);  // gather 0, stage 1, publish 1
  auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).state_generation, 1u);
  EXPECT_EQ(engine.engine->Step(), request);  // gather 1 (published), stage 2, publish 2
  fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).state_generation, 2u);
}

TEST_F(EngineStepTest, CompositeAggregateAdmissionCommitsEveryNewTable) {
  // One step admits three fresh requests in a single composite reservation. Its single paged
  // sub-reservation publishes all three new block tables in one CommitValidated without reallocating
  // committed_tables_ (the constructor reserved that aggregate headroom), and all three fixed slots
  // publish together. This exercises the aggregate-admission guarantee the split-commit contract
  // relies on for a single reservation.
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  std::vector<std::shared_ptr<Request>> requests;
  for (int i = 0; i < 3; ++i) {
    requests.push_back(MintRequest(*model_, Prompt(10 * (i + 1))));
    engine.engine->AddRequest(requests.back());
  }
  size_t observed_rows = 0;
  size_t observed_new = 0;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    observed_rows = context.fixed_state_slots.size();
    observed_new = context.plan->fixed_state.new_slot_count;
    for (size_t row = 0; row < context.fixed_state_slots.size(); ++row) {
      for (const auto& binding : context.fixed_state_bindings) {
        FillFixedOutputRow(binding, row, 1.0f);
      }
    }
  });

  EXPECT_EQ(engine.engine->Step(), requests[0]);
  EXPECT_EQ(observed_rows, 3u);
  EXPECT_EQ(observed_new, 3u);
  const auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(fixed->committed_slots, 3u);
  std::vector<RequestStateSnapshot> snapshots;
  for (const auto& request : requests) {
    EXPECT_TRUE(engine.cache->IsResident(request));
    EXPECT_EQ(FixedSlotFor(*fixed, request.get()).committed_tokens, 3u);
    snapshots.push_back(request->Snapshot());
  }
  EXPECT_EQ(engine.cache->Snapshot().requests.size(), 3u);
  EXPECT_TRUE(
      ValidateCompositeStateInvariants(engine.cache->Snapshot(), *fixed, snapshots).empty());
}

// ---------------------------------------------------------------------------------------------
// Speculative decoding: a request proposes drafts, the step runs 1 + drafts rows, and the Engine
// keeps the prefix the target model would have produced on its own.
// ---------------------------------------------------------------------------------------------

TEST_F(EngineStepTest, MtpCoordinatorCompactsMixedDraftLengthsWithoutIntermediateReadback) {
  constexpr int32_t target_token = 5;
  constexpr int32_t draft_token = 11;
  auto engine = MakeMtpDoublesEngine(
      model_, /*capacity=*/8, target_token, draft_token);

  std::vector<std::shared_ptr<Request>> requests;
  for (size_t draft_count : {size_t{1}, size_t{2}, size_t{3}}) {
    auto request = MintRequest(*model_, Prompt(static_cast<int32_t>(draft_count * 10)));
    request->Params()->speculative.max_draft_tokens = static_cast<int>(draft_count);
    engine.engine->AddRequest(request);
    requests.push_back(std::move(request));
  }

  ASSERT_NE(engine.engine->Step(), nullptr);

  ASSERT_EQ(engine.mtp_executor->decoded_batch_sizes,
            (std::vector<size_t>{3, 2, 1}));
  ASSERT_EQ(engine.mtp_executor->used_device_input_ids,
            (std::vector<bool>{false, true, true}));
  ASSERT_EQ(engine.device_state->argmax_rows,
            (std::vector<int>{3, 2, 1}));
  EXPECT_EQ(engine.device_state->device_to_host_copies, 1u);
  EXPECT_EQ(engine.device_state->synchronize_calls, 0u);
  for (size_t i = 0; i < requests.size(); ++i) {
    EXPECT_EQ(requests[i]->PendingDraftTokenCount(), i + 1);
  }
}

TEST_F(EngineStepTest, MtpCoordinatorRecordsFullPartialAndZeroAcceptance) {
  constexpr int32_t draft_token = 11;
  auto engine = MakeMtpDoublesEngine(
      model_, /*capacity=*/8, /*target_token=*/5, draft_token);

  std::vector<std::shared_ptr<Request>> requests;
  for (int32_t seed : {10, 20, 30}) {
    auto request = MintRequest(*model_, Prompt(seed));
    request->Params()->speculative.max_draft_tokens = 3;
    engine.engine->AddRequest(request);
    requests.push_back(std::move(request));
  }
  ASSERT_NE(engine.engine->Step(), nullptr);
  for (const auto& request : requests) {
    ASSERT_EQ(request->PendingDraftTokenCount(), 3u);
    static_cast<void>(DrainTokens(request));
  }

  engine.executor->SetVerifyRowTokens({
      draft_token,
      draft_token,
      draft_token,
      14,
      draft_token,
      12,
      13,
      14,
      12,
      13,
      14,
      15,
  });
  AdvanceUntilTargetDecodeCount(engine, 2);

  const auto stats = engine.engine->GetSpeculativeStats();
  EXPECT_EQ(stats.rounds, 3u);
  EXPECT_EQ(stats.draft_tokens_proposed, 9u);
  EXPECT_EQ(stats.draft_tokens_evaluated, 6u);
  EXPECT_EQ(stats.draft_tokens_accepted, 4u);
  EXPECT_EQ(stats.full_accept_rounds, 1u);
  EXPECT_EQ(stats.partial_accept_rounds, 1u);
  EXPECT_EQ(stats.zero_accept_rounds, 1u);
  EXPECT_EQ(stats.acceptance_length_histogram[0], 1u);
  EXPECT_EQ(stats.acceptance_length_histogram[1], 1u);
  EXPECT_EQ(stats.acceptance_length_histogram[3], 1u);
}

TEST_F(EngineStepTest, MtpHeadFailureRollsBackTargetAndAuxiliaryTransactions) {
  constexpr int32_t draft_token = 11;
  auto engine = MakeMtpDoublesEngine(
      model_, /*capacity=*/8, /*target_token=*/5, draft_token);
  auto request = MintRequest(*model_, Prompt(10));
  request->Params()->speculative.max_draft_tokens = 3;
  engine.engine->AddRequest(request);

  ASSERT_EQ(engine.engine->Step(), request);
  static_cast<void>(DrainTokens(request));
  const auto before = request->Snapshot();
  ASSERT_EQ(request->PendingDraftTokenCount(), 3u);
  const int target_releases_before = engine.cache->reservation_release_calls;
  const int releases_before = engine.mtp_cache->reservation_release_calls;

  engine.executor->SetVerifyRowTokens(
      {draft_token, draft_token, draft_token, 14});
  engine.mtp_executor->SetNextFailure(
      ScriptedExecutionFailure::RetryableDuringExecution);
  EXPECT_THROW(static_cast<void>(engine.engine->Step()), EngineStepError);

  const auto after_failure = request->Snapshot();
  EXPECT_EQ(after_failure.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(after_failure.processed_sequence_length, before.processed_sequence_length);
  EXPECT_EQ(request->PendingDraftTokenCount(), 3u);
  EXPECT_EQ(engine.cache->reservation_release_calls, target_releases_before + 1);
  EXPECT_EQ(engine.mtp_cache->reservation_release_calls, releases_before + 1);
  EXPECT_EQ(engine.mtp_cache->AllocatedCount(), 1u);
  EXPECT_EQ(engine.engine->GetSpeculativeStats().rounds, 0u);

  ASSERT_EQ(engine.engine->Step(), request);
  const auto stats = engine.engine->GetSpeculativeStats();
  EXPECT_EQ(stats.rounds, 1u);
  EXPECT_EQ(stats.full_accept_rounds, 1u);
  EXPECT_EQ(stats.draft_tokens_accepted, 3u);
}

TEST_F(EngineStepTest, MtpCoordinatorClampsDraftsAtMaxLength) {
  constexpr int32_t draft_token = 11;
  const auto prompt = Prompt(10);
  auto engine = MakeMtpDoublesEngine(
      model_, /*capacity=*/8, /*target_token=*/5, draft_token);
  auto request = MintRequest(*model_, prompt);
  request->Params()->speculative.max_draft_tokens = 3;
  request->Params()->search.max_length = static_cast<int>(prompt.size() + 4);
  engine.engine->AddRequest(request);

  ASSERT_EQ(engine.engine->Step(), request);
  EXPECT_EQ(request->PendingDraftTokenCount(), 2u);
  EXPECT_EQ(engine.mtp_executor->decoded_batch_sizes,
            (std::vector<size_t>{1, 1}));
  static_cast<void>(DrainTokens(request));

  engine.executor->SetVerifyRowTokens({draft_token, draft_token, 12});
  ASSERT_EQ(engine.engine->Step(), request);
  EXPECT_TRUE(request->IsTurnComplete());
  EXPECT_EQ(request->CurrentSequenceLength(), request->Params()->search.max_length);
  EXPECT_EQ(engine.mtp_executor->decode_calls, 2);
  EXPECT_EQ(engine.engine->GetSpeculativeStats().full_accept_rounds, 1u);
}

TEST_F(EngineStepTest, MtpCoordinatorReusesCacheAfterDiscardingSpeculativeRows) {
  constexpr int32_t draft_token = 11;
  auto engine = MakeMtpDoublesEngine(
      model_, /*capacity=*/8, /*target_token=*/5, draft_token);
  auto request = MintRequest(*model_, Prompt(10));
  request->Params()->speculative.max_draft_tokens = 3;
  engine.engine->AddRequest(request);

  ASSERT_EQ(engine.engine->Step(), request);
  static_cast<void>(DrainTokens(request));
  for (int round = 0; round < 2; ++round) {
    engine.executor->SetVerifyRowTokens({12, 13, 14, 15});
    ASSERT_EQ(engine.engine->Step(), request);
    static_cast<void>(DrainTokens(request));
  }

  ASSERT_EQ(engine.mtp_cache->reserved_new_request_counts,
            (std::vector<size_t>{1, 0, 0}));
  EXPECT_EQ(engine.mtp_cache->AllocatedCount(), 1u);
  EXPECT_EQ(engine.mtp_cache->deallocate_calls, 0);
  ASSERT_EQ(engine.mtp_executor->decoded_sequence_lengths_before.size(), 9u);
  EXPECT_EQ(engine.mtp_executor->decoded_sequence_lengths_before[0],
            (std::vector<int64_t>{1}));
  EXPECT_EQ(engine.mtp_executor->decoded_sequence_lengths_before[3],
            (std::vector<int64_t>{2}));
  EXPECT_EQ(engine.mtp_executor->decoded_sequence_lengths_before[6],
            (std::vector<int64_t>{3}));
  EXPECT_EQ(engine.device_state->device_to_host_copies, 3u);
}

TEST_F(EngineStepTest, SpeculativeStepKeepsTheAcceptedPrefixAndReplacesTheRejectedDraft) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  ASSERT_EQ(engine.engine->Step(), request);
  ASSERT_EQ(request->status_, RequestStatus::Active);
  const int64_t length_after_prefill = request->CurrentSequenceLength();
  while (request->HasUnseenTokens()) request->UnseenToken();

  const std::vector<int32_t> drafts{11, 12, 13};
  request->SetDraftTokens(drafts);
  // Rows 0 and 1 confirm the first two drafts; row 2 disagrees, so its own token replaces draft 2
  // and row 3 is never read.
  engine.executor->SetVerifyRowTokens({11, 12, 21, 22});

  ASSERT_EQ(engine.engine->Step(), request);

  ASSERT_EQ(engine.executor->decoded_token_counts.size(), 2u);
  EXPECT_EQ(engine.executor->decoded_token_counts[1], 4u);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].row, 0u);
  EXPECT_EQ(engine.cache->prefix_commits[0].request_id, request.get());
  EXPECT_EQ(engine.cache->prefix_commits[0].step_tokens, 4u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 3u);

  std::vector<int32_t> produced;
  while (request->HasUnseenTokens()) produced.push_back(request->UnseenToken());
  EXPECT_EQ(produced, (std::vector<int32_t>{11, 12, 21}));
  EXPECT_EQ(request->CurrentSequenceLength(), length_after_prefill + 3);
  EXPECT_EQ(request->ProcessedSequenceLength(), length_after_prefill + 2);
  EXPECT_EQ(request->PendingDraftTokenCount(), 0u);
}

TEST_F(EngineStepTest, SpeculativeStepRejectingTheFirstDraftAdvancesByOneToken) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  ASSERT_EQ(engine.engine->Step(), request);
  const int64_t length_after_prefill = request->CurrentSequenceLength();
  while (request->HasUnseenTokens()) request->UnseenToken();

  request->SetDraftTokens(std::vector<int32_t>{11, 12});
  engine.executor->SetVerifyRowTokens({21, 22, 23});

  ASSERT_EQ(engine.engine->Step(), request);

  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].step_tokens, 3u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 1u);
  std::vector<int32_t> produced;
  while (request->HasUnseenTokens()) produced.push_back(request->UnseenToken());
  EXPECT_EQ(produced, (std::vector<int32_t>{21}));
  EXPECT_EQ(request->CurrentSequenceLength(), length_after_prefill + 1);
}

TEST_F(EngineStepTest, SpeculativeStepAcceptingEveryDraftAlsoTakesTheBonusToken) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  ASSERT_EQ(engine.engine->Step(), request);
  const int64_t length_after_prefill = request->CurrentSequenceLength();
  while (request->HasUnseenTokens()) request->UnseenToken();

  request->SetDraftTokens(std::vector<int32_t>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({11, 12, 13, 25});

  ASSERT_EQ(engine.engine->Step(), request);

  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 4u);
  std::vector<int32_t> produced;
  while (request->HasUnseenTokens()) produced.push_back(request->UnseenToken());
  EXPECT_EQ(produced, (std::vector<int32_t>{11, 12, 13, 25}));
  EXPECT_EQ(request->CurrentSequenceLength(), length_after_prefill + 4);
  EXPECT_EQ(request->ProcessedSequenceLength(), length_after_prefill + 3);
}

TEST_F(EngineStepTest, SpeculativeTelemetryAggregatesAcceptanceLengths) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  ASSERT_EQ(engine.engine->Step(), request);
  while (request->HasUnseenTokens()) request->UnseenToken();

  request->SetDraftTokens(std::vector<int32_t>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({11, 12, 21, 22});
  ASSERT_EQ(engine.engine->Step(), request);
  while (request->HasUnseenTokens()) request->UnseenToken();

  request->SetDraftTokens(std::vector<int32_t>{14, 15});
  engine.executor->SetVerifyRowTokens({24, 25, 26});
  ASSERT_EQ(engine.engine->Step(), request);
  while (request->HasUnseenTokens()) request->UnseenToken();

  request->SetDraftTokens(std::vector<int32_t>{16});
  engine.executor->SetVerifyRowTokens({16, 26});
  ASSERT_EQ(engine.engine->Step(), request);

  const auto stats = engine.engine->GetSpeculativeStats();
  EXPECT_EQ(stats.target_forward_passes, 4u);
  EXPECT_EQ(stats.draft_forward_passes, 0u);
  EXPECT_EQ(stats.rounds, 3u);
  EXPECT_EQ(stats.draft_tokens_proposed, 6u);
  EXPECT_EQ(stats.draft_tokens_evaluated, 5u);
  EXPECT_EQ(stats.draft_tokens_accepted, 3u);
  EXPECT_EQ(stats.zero_accept_rounds, 1u);
  EXPECT_EQ(stats.partial_accept_rounds, 1u);
  EXPECT_EQ(stats.full_accept_rounds, 1u);
  EXPECT_EQ(stats.acceptance_length_histogram[0], 1u);
  EXPECT_EQ(stats.acceptance_length_histogram[1], 1u);
  EXPECT_EQ(stats.acceptance_length_histogram[2], 1u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 3.0f / 5.0f);
  EXPECT_FLOAT_EQ(stats.avg_draft_tokens_per_round, 2.0f);
}

TEST_F(EngineStepTest, RolledBackSpeculativeStepLeavesTheProposalPendingAndRetryable) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  ASSERT_EQ(engine.engine->Step(), request);
  const int64_t length_after_prefill = request->CurrentSequenceLength();
  while (request->HasUnseenTokens()) request->UnseenToken();

  request->SetDraftTokens(std::vector<int32_t>{11, 12});
  engine.executor->SetNextFailure(ScriptedExecutionFailure::PostProcessing);
  EXPECT_THROW(engine.engine->Step(), EngineStepError);

  EXPECT_EQ(request->CurrentSequenceLength(), length_after_prefill);
  EXPECT_EQ(request->PendingDraftTokenCount(), 2u);
  EXPECT_EQ(request->AcceptedDraftTokenCount(), 0u);
  EXPECT_TRUE(engine.cache->prefix_commits.empty());
  auto stats = engine.engine->GetSpeculativeStats();
  EXPECT_EQ(stats.target_forward_passes, 2u);
  EXPECT_EQ(stats.rounds, 0u);
  EXPECT_EQ(stats.draft_tokens_proposed, 0u);
  EXPECT_EQ(stats.draft_tokens_accepted, 0u);

  engine.executor->SetVerifyRowTokens({11, 12, 25});
  ASSERT_EQ(engine.engine->Step(), request);
  std::vector<int32_t> produced;
  while (request->HasUnseenTokens()) produced.push_back(request->UnseenToken());
  EXPECT_EQ(produced, (std::vector<int32_t>{11, 12, 25}));
  stats = engine.engine->GetSpeculativeStats();
  EXPECT_EQ(stats.target_forward_passes, 3u);
  EXPECT_EQ(stats.rounds, 1u);
  EXPECT_EQ(stats.draft_tokens_proposed, 2u);
  EXPECT_EQ(stats.draft_tokens_accepted, 2u);
  EXPECT_EQ(stats.acceptance_length_histogram[2], 1u);
}

TEST_F(EngineStepTest, DraftsAreRejectedWhenTheCacheCannotRollThemBack) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);

  EXPECT_EQ(engine.engine->MaxDraftTokensPerStep(), 0u);
  EXPECT_THROW(request->SetDraftTokens(std::vector<int32_t>{11}), std::runtime_error);
}

TEST_F(EngineStepTest, DraftsAreRejectedBeyondTheCacheWindowAndForSampledRequests) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  engine.cache->SetMaxDraftTokensPerStep(2);

  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  EXPECT_THROW(request->SetDraftTokens(std::vector<int32_t>{11, 12, 13}), std::runtime_error);

  auto sampled_params = MakeGreedyParams(*model_);
  sampled_params->search.do_sample = true;
  sampled_params->search.top_k = 40;
  sampled_params->search.temperature = 1.0f;
  auto sampled = std::make_shared<Request>(sampled_params);
  sampled->AddTokens(Prompt(20));
  engine.engine->AddRequest(sampled);
  EXPECT_THROW(sampled->SetDraftTokens(std::vector<int32_t>{11}), std::runtime_error);
}

TEST_F(EngineStepTest, PrefillingRequestDoesNotVerifyDrafts) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto params = MakeGreedyParams(*model_);
  params->search.chunk_size = 2;
  auto request = std::make_shared<Request>(params);
  request->AddTokens(Prompt(10));  // three tokens, so the first chunk stops short of the tail
  engine.engine->AddRequest(request);
  request->SetDraftTokens(std::vector<int32_t>{11, 12});

  ASSERT_NE(engine.engine->Step(), nullptr);
  // The first chunk stops short of the tail, and the second is still prompt, so neither can verify.
  ASSERT_EQ(engine.executor->decoded_token_counts.size(), 2u);
  EXPECT_EQ(engine.executor->decoded_token_counts[0], 2u);
  EXPECT_EQ(engine.executor->decoded_token_counts[1], 1u);
  EXPECT_TRUE(engine.cache->prefix_commits.empty());
  // A proposal only ever applies to the next step, so a committed step consumes it either way.
  EXPECT_EQ(request->PendingDraftTokenCount(), 0u);
}

// A drafted stop token must not short-circuit the search's end-of-sequence handling: the tokens
// the application sees have to be the same ones a plain decode would have produced.
TEST_F(EngineStepTest, SpeculativeStepEndsTheTurnOnADraftedStopToken) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;

  const auto drain = [](const std::shared_ptr<Request>& request) {
    std::vector<int32_t> tokens;
    while (request->HasUnseenTokens()) tokens.push_back(request->UnseenToken());
    return tokens;
  };

  // Reference: the model predicts 11 and then the stop token, one token per step.
  auto plain_engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  auto plain = MintRequest(*model_, Prompt(10));
  plain_engine.engine->AddRequest(plain);
  ASSERT_EQ(plain_engine.engine->Step(), plain);
  drain(plain);
  plain_engine.executor->SetVerifyRowTokens({11});
  ASSERT_EQ(plain_engine.engine->Step(), plain);
  auto expected = drain(plain);
  plain_engine.executor->SetVerifyRowTokens({eos});
  ASSERT_EQ(plain_engine.engine->Step(), plain);
  for (int32_t token : drain(plain)) expected.push_back(token);
  ASSERT_TRUE(plain->IsTurnComplete());

  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  ASSERT_EQ(engine.engine->Step(), request);
  drain(request);

  request->SetDraftTokens(std::vector<int32_t>{11, eos, 13});
  engine.executor->SetVerifyRowTokens({11, eos, eos, eos});
  ASSERT_EQ(engine.engine->Step(), request);

  EXPECT_EQ(drain(request), expected);
  EXPECT_TRUE(request->IsTurnComplete());
  // Only the first draft was accepted; the stop token came from the sampler on the next row.
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 2u);
}

TEST_F(EngineStepTest, CompositeSpeculativeStepPublishesTheAcceptedCheckpoint) {
  model_ = LoadSyntheticCompositeModel();
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeCompositeDoublesEngine(model_, filler);
  ASSERT_EQ(engine.cache->MaxDraftTokensPerStep(), 3u);

  auto request = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(request);
  engine.executor->SetExecutionCallback([](ExecutionContext& context) {
    for (const auto& binding : context.fixed_state_bindings) {
      EXPECT_EQ(binding.checkpoints, nullptr);  // no drafts, so no series is captured
      FillFixedOutputRow(binding, 0, 10.0f);
    }
  });
  ASSERT_EQ(engine.engine->Step(), request);
  const int64_t length_after_prefill = request->CurrentSequenceLength();
  while (request->HasUnseenTokens()) request->UnseenToken();

  request->SetDraftTokens(std::vector<int32_t>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({11, 12, 21, 22});
  engine.executor->SetExecutionCallback([](ExecutionContext& context) {
    // The drafts push this step past its first 4-slot block, and the cache plans block-table
    // columns before the token budget fixes the step length, so the plan has to have counted them.
    ASSERT_NE(context.cache_reservation, nullptr);
    EXPECT_LE(context.cache_reservation->RequiredBlockTableColumns(),
              context.plan->proposed_block_table_columns);
    for (const auto& binding : context.fixed_state_bindings) {
      ASSERT_NE(binding.checkpoints, nullptr);
      FillFixedOutputRow(binding, 0, 99.0f);  // the step's final state, which must NOT be committed
      const auto shape = binding.checkpoints->GetTensorTypeAndShapeInfo()->GetShape();
      size_t row_elements = 1;
      for (size_t axis = 2; axis < shape.size(); ++axis) {
        row_elements *= static_cast<size_t>(shape[axis]);
      }
      auto* data = binding.checkpoints->GetTensorMutableData<float>();
      for (size_t slot = 0; slot < static_cast<size_t>(shape[0]); ++slot) {
        std::fill_n(data + slot * row_elements, row_elements, 20.0f + static_cast<float>(slot));
      }
    }
  });

  ASSERT_EQ(engine.engine->Step(), request);

  std::vector<int32_t> produced;
  while (request->HasUnseenTokens()) produced.push_back(request->UnseenToken());
  EXPECT_EQ(produced, (std::vector<int32_t>{11, 12, 21}));

  // Two accepted drafts plus the token the step started from: both states stop at the same token.
  const auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  const auto expected_boundary = static_cast<uint64_t>(length_after_prefill + 2);
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).committed_tokens, expected_boundary);
  const auto paged = engine.cache->Snapshot();
  ASSERT_EQ(paged.requests.size(), 1u);
  EXPECT_EQ(paged.requests[0].used_slots, expected_boundary);
  EXPECT_TRUE(ValidateCompositeStateInvariants(paged, *fixed, {request->Snapshot()}).empty());

  // A three-token step of which three were kept: the conv group is left-aligned so it publishes
  // slot 2, and the right-aligned recurrent group publishes slot 4 - 4 + 3 - 1 = 2 as well.
  engine.executor->SetVerifyRowTokens({});
  engine.executor->SetExecutionCallback([](ExecutionContext& context) {
    for (const auto& binding : context.fixed_state_bindings) {
      ExpectFixedInputRow(binding, 0, 22.0f);
      FillFixedOutputRow(binding, 0, 30.0f);
    }
  });
  EXPECT_EQ(engine.engine->Step(), request);
}

TEST_F(EngineStepTest, CompositeDefersDraftsWhilePrefillSharesTheStep) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto decode = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(decode);
  engine.executor->SetExecutionCallback([](ExecutionContext& context) {
    for (const auto& binding : context.fixed_state_bindings) {
      FillFixedOutputRow(binding, 0, 10.0f);
    }
  });
  ASSERT_EQ(engine.engine->Step(), decode);
  while (decode->HasUnseenTokens()) decode->UnseenToken();

  decode->SetDraftTokens(std::array<int32_t, 3>{11, 12, 13});
  auto prefill = MintRequest(*model_, Prompt(20));
  engine.engine->AddRequest(prefill);
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    ASSERT_EQ(context.plan->requests.size(), 2u);
    EXPECT_EQ(context.plan->requests[0].request, decode);
    EXPECT_EQ(context.plan->requests[0].draft_token_count, 0u);
    EXPECT_EQ(context.plan->requests[0].unprocessed_token_count, 1u);
    EXPECT_EQ(context.plan->requests[1].request, prefill);
    EXPECT_TRUE(context.plan->requests[1].is_prefill);
    EXPECT_FALSE(context.plan->fixed_state.capture_checkpoints);
    for (size_t row = 0; row < context.fixed_state_slots.size(); ++row) {
      for (const auto& binding : context.fixed_state_bindings) {
        EXPECT_EQ(binding.checkpoints, nullptr);
        FillFixedOutputRow(binding, row, 20.0f + static_cast<float>(row));
      }
    }
  });

  ASSERT_EQ(engine.engine->Step(), decode);
  EXPECT_EQ(decode->PendingDraftTokenCount(), 0u);
  EXPECT_TRUE(decode->HasUnseenTokens());
  EXPECT_TRUE(prefill->HasUnseenTokens());
}

TEST_F(EngineStepTest, CompositeCompletionRemovalFreesSlotForReadmission) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, EosToken(*model_));  // force EOS to complete
  auto first = MintRequest(*model_, Prompt(10));
  engine.engine->AddRequest(first);
  engine.executor->SetExecutionCallback([](ExecutionContext& context) {
    for (const auto& binding : context.fixed_state_bindings) {
      FillFixedOutputRow(binding, 0, 31.0f);
    }
  });

  ASSERT_EQ(engine.engine->Step(), first);
  ASSERT_TRUE(first->IsTurnComplete());
  const auto completed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(completed.has_value());
  const auto released_slot = FixedSlotFor(*completed, first.get()).slot;
  const auto first_generation = FixedSlotFor(*completed, first.get()).generation;

  // A completed turn keeps its committed state until it is removed; removal frees both states.
  first->Remove();
  auto after_removal = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(after_removal.has_value());
  EXPECT_EQ(after_removal->committed_slots, 0u);

  auto second = MintRequest(*model_, Prompt(20));
  engine.engine->AddRequest(second);
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    ASSERT_EQ(context.fixed_state_slots.size(), 1u);
    EXPECT_EQ(context.fixed_state_slots[0].slot, released_slot);  // reuses the freed slot
    for (const auto& binding : context.fixed_state_bindings) {
      ExpectFixedInputRow(binding, 0, 0.0f);  // fresh admission gathers zeros, not the stale 31.0
      FillFixedOutputRow(binding, 0, 41.0f);
    }
  });

  EXPECT_EQ(engine.engine->Step(), second);
  const auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(fixed->committed_slots, 1u);
  EXPECT_EQ(FixedSlotFor(*fixed, second.get()).slot, released_slot);
  EXPECT_GT(FixedSlotFor(*fixed, second.get()).generation, first_generation);
  ASSERT_EQ(engine.cache->Snapshot().requests.size(), 1u);
  EXPECT_EQ(engine.cache->Snapshot().requests[0].request_id, second.get());
}

}  // namespace
}  // namespace test
}  // namespace Generators
