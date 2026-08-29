// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Component-contract tests for Engine::Run. They wire an Engine with recording cache-manager
// and model-executor doubles over a tiny CPU fixture model. The doubles never run the model: the
// executor fabricates end-of-stream logits so each scheduled request's real greedy search completes
// deterministically. This lets the tests assert how Run orchestrates its collaborators -- that it
// allocates a batch before decoding it, decodes once per internal cycle, drains ready requests
// without redundant model runs, and forms a fresh batch across runs under capacity backpressure.

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
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

struct RequestPostProcessingControl {
  bool fail{};
};

class FailingPostProcessingSearch final : public GreedySearch_Cpu {
 public:
  FailingPostProcessingSearch(
      const GeneratorParams& params,
      std::shared_ptr<RequestPostProcessingControl> control)
      : GreedySearch_Cpu(params), control_{std::move(control)} {}

  void ApplyRepetitionPenalty(float penalty) override {
    if (control_->fail) {
      throw std::runtime_error("Injected request post-processing failure.");
    }
    GreedySearch_Cpu::ApplyRepetitionPenalty(penalty);
  }

 private:
  std::shared_ptr<RequestPostProcessingControl> control_;
};

class FailingPostProcessingDevice final : public DeviceInterface {
 public:
  FailingPostProcessingDevice(
      DeviceInterface& inner,
      std::shared_ptr<RequestPostProcessingControl> control)
      : inner_{inner}, control_{std::move(control)} {}

  DeviceType GetType() const override { return inner_.GetType(); }
  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override {
    inner_.InitOrt(api, allocator);
  }
  Ort::Allocator& GetAllocator() override { return inner_.GetAllocator(); }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override {
    return inner_.GetMemoryInfo();
  }
  std::string GetExecutionProviderName() const override {
    return inner_.GetExecutionProviderName();
  }
  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return inner_.AllocateBase(size);
  }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* memory, size_t size) override {
    return inner_.WrapMemoryBase(memory, size);
  }
  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    return std::make_unique<FailingPostProcessingSearch>(params, control_);
  }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return inner_.CreateBeam(params);
  }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) override {
    return inner_.CreateKeyValueCache(state);
  }
  void Synchronize() override { inner_.Synchronize(); }

 private:
  DeviceInterface& inner_;
  std::shared_ptr<RequestPostProcessingControl> control_;
};

class ThrowingStaticScheduler final : public Scheduler {
 public:
  explicit ThrowingStaticScheduler(std::shared_ptr<Model> model)
      : Scheduler{std::move(model)} {}

  void AddRequest(std::shared_ptr<Request> request) override {
    request_ = std::move(request);
  }

  void RemoveRequest(std::shared_ptr<Request> request) override {
    if (request_ == request) {
      request_.reset();
    }
  }

  ScheduledRequests Schedule() override {
    throw std::runtime_error("Injected static scheduler failure.");
  }

  bool HasPendingRequests() const override {
    return request_ != nullptr;
  }

 private:
  std::shared_ptr<Request> request_;
};

class EngineRunTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadDummyDecoderModel();
    model_->config_->engine.dynamic_batching =
        Config::Engine::DynamicBatching{};
  }

  std::shared_ptr<Model> model_;
};

TEST(EngineLifetimeTest, EngineRetainsModelForItsLifetime) {
  auto model = LoadSyntheticPagedModel();
  std::weak_ptr<Model> model_observer = model;
  auto engine = std::make_shared<Engine>(model);

  model.reset();
  EXPECT_FALSE(model_observer.expired());

  engine.reset();
  EXPECT_TRUE(model_observer.expired());
}

// One request: Run decodes the proposed batch exactly once, commits its cache allocation, and
// returns the request.
TEST_F(EngineRunTest, SingleRequestSchedulesThenDecodesThenReturns) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));

  auto prompt = Prompt(10);
  auto request = CreateRequestWithPrompt(engine.engine, *model_, prompt);
  ASSERT_TRUE(engine.engine->HasPendingRequests());

  auto ready = RunOne(*engine.engine);

  EXPECT_EQ(ready.request, request);
  EXPECT_TRUE(request->IsTurnComplete());
  EXPECT_EQ(engine.executor->decode_calls, 1);
  ASSERT_EQ(engine.executor->decoded_batch_sizes.size(), 1u);
  EXPECT_EQ(engine.executor->decoded_batch_sizes[0], 1u);

  const int allocate_at = IndexOf(*engine.trace, "Allocate");
  const int decode_at = IndexOf(*engine.trace, "Decode");
  ASSERT_GE(allocate_at, 0);
  ASSERT_GE(decode_at, 0);
  EXPECT_LT(decode_at, allocate_at);
  // EOS completes without a visible token, but its terminal ready notification is still published
  // exactly once.
  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagNone);
}

// Several requests that all fit are decoded together in a single batch and returned by one bulk
// operation when caller capacity is sufficient.
TEST_F(EngineRunTest, FittingRequestsShareOneDecodeAndReturnAllEvents) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));

  // These test-only shared_ptr requests never acquire an external reference. Repeated Run
  // boundaries must not mistake them for abandoned public handles.
  std::vector<std::shared_ptr<Request>> requests;
  for (int32_t seed : {10, 20, 30}) {
    auto prompt = Prompt(seed);
    auto request = CreateRequestWithPrompt(engine.engine, *model_, prompt);
    requests.push_back(request);
  }

  std::array<EngineEvent, 3> storage;
  const size_t event_count = engine.engine->Run(storage);
  ASSERT_EQ(event_count, storage.size());

  std::vector<std::shared_ptr<Request>> returned;
  for (const auto& event : storage) {
    returned.push_back(event.request);
  }
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

TEST_F(EngineRunTest, CapacityOneExecutesOnceThenDrainsOverflow) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  std::vector<std::shared_ptr<Request>> requests;
  for (int32_t seed : {10, 20, 30}) {
    auto prompt = Prompt(seed);
    requests.push_back(
        CreateRequestWithPrompt(engine.engine, *model_, prompt));
  }

  std::array<EngineEvent, 1> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].request, requests[0]);
  EXPECT_EQ(engine.executor->decode_calls, 1);

  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].request, requests[1]);
  EXPECT_EQ(engine.executor->decode_calls, 1);

  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].request, requests[2]);
  EXPECT_EQ(engine.executor->decode_calls, 1);
}

TEST_F(EngineRunTest, RetainedEventsDrainWithoutExecutingIntoSpareCapacity) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  std::vector<std::shared_ptr<Request>> requests;
  for (int32_t seed : {10, 20, 30}) {
    auto prompt = Prompt(seed);
    requests.push_back(
        CreateRequestWithPrompt(engine.engine, *model_, prompt));
  }

  std::array<EngineEvent, 1> first_storage;
  ASSERT_EQ(engine.engine->Run(first_storage), 1u);
  EXPECT_EQ(first_storage[0].request, requests[0]);
  EXPECT_EQ(engine.executor->decode_calls, 1);

  requests[0]->BeginTurn(
      std::array<int32_t, 1>{5}, std::optional<size_t>{1});

  std::array<EngineEvent, 4> drain_storage;
  ASSERT_EQ(engine.engine->Run(drain_storage), 2u);
  EXPECT_EQ(drain_storage[0].request, requests[1]);
  EXPECT_EQ(drain_storage[1].request, requests[2]);
  EXPECT_EQ(engine.executor->decode_calls, 1);
  EXPECT_TRUE(engine.engine->HasPendingRequests());

  ASSERT_EQ(engine.engine->Run(drain_storage), 1u);
  EXPECT_EQ(drain_storage[0].request, requests[0]);
  EXPECT_EQ(engine.executor->decode_calls, 2);
}

TEST_F(EngineRunTest, CreateRequestReclaimsAbandonedTurnCompleteAtCapacity) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/1, EosToken(*model_));
  auto first_prompt = Prompt(10);
  auto first = CreateEngineRequest(engine.engine, *model_);
  ExternalRequestReference first_external{*first};
  first->BeginTurn(first_prompt);

  ASSERT_EQ(RunOne(*engine.engine).request, first);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(engine.cache->AllocatedCount(), 1u);

  first_external.Release();

  auto second_prompt = Prompt(20);
  auto second = CreateEngineRequest(engine.engine, *model_);
  ExternalRequestReference second_external{*second};
  second->BeginTurn(second_prompt);

  EXPECT_EQ(first->status_, RequestStatus::Closed);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
  EXPECT_EQ(engine.cache->deallocate_calls, 1);

  EXPECT_EQ(RunOne(*engine.engine).request, second);
  EXPECT_EQ(second->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);

  second->Close();
  second_external.Release();
}

TEST_F(EngineRunTest, RunReclaimsAbandonedTurnCompleteBeforePlanningAtCapacity) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/1, EosToken(*model_));
  auto first_prompt = Prompt(10);
  auto second_prompt = Prompt(20);
  auto first = CreateEngineRequest(engine.engine, *model_);
  auto second = CreateEngineRequest(engine.engine, *model_);
  ExternalRequestReference first_external{*first};
  ExternalRequestReference second_external{*second};
  first->BeginTurn(first_prompt);
  second->BeginTurn(second_prompt);

  ASSERT_EQ(RunOne(*engine.engine).request, first);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(second->status_, RequestStatus::Assigned);
  ASSERT_EQ(engine.cache->AllocatedCount(), 1u);
  first_external.Release();

  EXPECT_EQ(RunOne(*engine.engine).request, second);
  EXPECT_EQ(first->status_, RequestStatus::Closed);
  EXPECT_EQ(second->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);
  EXPECT_EQ(engine.cache->deallocate_calls, 1);
  EXPECT_EQ(engine.executor->decode_calls, 2);

  second->Close();
  second_external.Release();
}

TEST_F(EngineRunTest, RunPurgesAbandonedReadyAndQueuedRequestsExactlyOnce) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/2, EosToken(*model_));
  auto survivor_prompt = Prompt(10);
  auto ready_orphan_prompt = Prompt(20);
  auto queued_orphan_prompt = Prompt(30);
  auto survivor = CreateEngineRequest(engine.engine, *model_);
  auto ready_orphan = CreateEngineRequest(engine.engine, *model_);
  auto queued_orphan = CreateEngineRequest(engine.engine, *model_);
  ExternalRequestReference survivor_external{*survivor};
  ExternalRequestReference ready_orphan_external{*ready_orphan};
  ExternalRequestReference queued_orphan_external{*queued_orphan};
  survivor->BeginTurn(survivor_prompt);
  ready_orphan->BeginTurn(ready_orphan_prompt);
  queued_orphan->BeginTurn(queued_orphan_prompt);

  ASSERT_EQ(RunOne(*engine.engine).request, survivor);
  ASSERT_EQ(survivor->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(ready_orphan->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(queued_orphan->status_, RequestStatus::Assigned);
  ASSERT_EQ(engine.cache->AllocatedCount(), 2u);

  ready_orphan_external.Release();
  queued_orphan_external.Release();
  ASSERT_EQ(engine.cache->deallocate_calls, 0);

  // Cleanup runs before Run can drain the orphan's ready notification or plan the queued orphan.
  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagNone);
  EXPECT_EQ(ready_orphan->status_, RequestStatus::Closed);
  EXPECT_EQ(queued_orphan->status_, RequestStatus::Closed);
  EXPECT_EQ(survivor->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);
  EXPECT_EQ(engine.cache->deallocate_calls, 2);
  EXPECT_EQ(engine.executor->decode_calls, 1);

  // Closed requests were removed from Engine ownership, so another boundary cannot clean them twice.
  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagNone);
  EXPECT_EQ(engine.cache->deallocate_calls, 2);

  survivor->Close();
  survivor_external.Release();
}

TEST_F(EngineRunTest, ReacquiringExternalReferenceCancelsDeferredAbandonment) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/1, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = CreateEngineRequest(engine.engine, *model_);
  ExternalRequestReference initial_external{*request};
  request->BeginTurn(prompt);
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);

  initial_external.Release();
  ExternalRequestReference reacquired_external{*request};

  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagNone);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);
  EXPECT_EQ(engine.cache->deallocate_calls, 0);

  request->Close();
  reacquired_external.Release();
}

TEST_F(EngineRunTest, BeginTurnRejectsUndrainedReadyNotificationWithoutMutation) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto first_prompt = Prompt(10);
  auto second_prompt = Prompt(20);
  auto first = CreateRequestWithPrompt(engine.engine, *model_, first_prompt);
  auto second = CreateRequestWithPrompt(engine.engine, *model_, second_prompt);

  ASSERT_EQ(RunOne(*engine.engine).request, first);
  ASSERT_EQ(engine.executor->decode_calls, 1);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(second->status_, RequestStatus::TurnComplete);

  const auto before = second->Snapshot();
  const std::vector<int32_t> continuation{5, 6};
  try {
    second->BeginTurn(continuation);
    FAIL() << "Expected BeginTurn to reject the undrained ready notification.";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("Engine::Run()"), std::string::npos);
  }

  const auto rejected = second->Snapshot();
  EXPECT_EQ(rejected.status, before.status);
  EXPECT_EQ(rejected.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(rejected.processed_sequence_length, before.processed_sequence_length);

  EXPECT_EQ(RunOne(*engine.engine).request, second);
  EXPECT_EQ(engine.executor->decode_calls, 1);

  EXPECT_NO_THROW(second->BeginTurn(continuation));
  const auto continued = second->Snapshot();
  EXPECT_EQ(continued.status, RequestStatus::Assigned);
  EXPECT_EQ(continued.current_sequence_length,
            before.current_sequence_length + static_cast<int64_t>(continuation.size()));
}

TEST_F(EngineRunTest, ContinuedUnreadOutputPreservesOrder) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto prompt = Prompt(10);
  auto request = CreateRequestWithPrompt(engine.engine, *model_, prompt);

  auto event = RunOne(*engine.engine);
  ASSERT_EQ(event.request, request);
  EXPECT_EQ(event.token, 5);

  engine.executor->SetForcedToken(6);
  event = RunOne(*engine.engine);
  ASSERT_EQ(event.request, request);
  EXPECT_EQ(event.token, 6);
  engine.executor->SetForcedToken(7);
  event = RunOne(*engine.engine);
  ASSERT_EQ(event.request, request);
  EXPECT_EQ(event.token, 7);
  engine.executor->SetForcedToken(EosToken(*model_));
  event = RunOne(*engine.engine);
  ASSERT_EQ(event.request, request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);

  const std::vector<int32_t> continuation{8, 9};
  request->BeginTurn(continuation);
  engine.executor->SetForcedToken(10);
  event = RunOne(*engine.engine);
  ASSERT_EQ(event.request, request);
  EXPECT_EQ(event.token, 10);

  engine.executor->SetForcedToken(EosToken(*model_));
  event = RunOne(*engine.engine);
  ASSERT_EQ(event.request, request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);
}

TEST_F(EngineRunTest, PerTurnBudgetsPublishOneTerminalNotificationAcrossContinuations) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/1, /*forced_token=*/5);
  const auto prompt = Prompt(10);
  auto params = MakeGreedyParams(*model_);
  params->search.max_length =
      static_cast<int>(prompt.size() + 9);
  auto request = CreateEngineRequest(engine.engine, *params);

  const auto run_turn = [&](std::span<const int32_t> input,
                            size_t max_generated_tokens) {
    request->BeginTurn(
        input,
        std::optional<size_t>{max_generated_tokens});

    size_t terminal_notifications = 0;
    while (!request->IsTurnComplete()) {
      auto event = RunOne(*engine.engine);
      ASSERT_EQ(event.request, request);
      if ((event.flags & EngineEventFlagTurnFinished) != 0) {
        ++terminal_notifications;
      }
    }

    EXPECT_EQ(terminal_notifications, 1u);
    EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagNone);
  };

  run_turn(prompt, /*max_generated_tokens=*/2);
  const std::vector<int32_t> second_input{6};
  run_turn(second_input, /*max_generated_tokens=*/1);
  const std::vector<int32_t> third_input{7};
  run_turn(third_input, /*max_generated_tokens=*/1);

  request->Close();
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
}

// Under capacity backpressure Run decodes only the requests that fit, then forms a fresh batch for
// the deferred request on a later run -- one decode per internal cycle, never an over-capacity run.
//
// These test requests use only internal shared_ptrs and have never had public handles, so they are
// not abandoned automatically. A finished request keeps its slot until explicitly removed. With
// capacity for two requests, the third is admitted only after a completed request has been removed.
// The assertions pin both halves of that contract -- the slot is still held when the request is
// handed back, and it is released exactly at removal.
TEST_F(EngineRunTest, BackpressureFormsAFreshBatchAcrossRuns) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/2, EosToken(*model_));

  std::vector<std::shared_ptr<Request>> requests;
  for (int32_t seed : {10, 20, 30}) {
    auto prompt = Prompt(seed);
    auto request = CreateRequestWithPrompt(engine.engine, *model_, prompt);
    requests.push_back(request);
  }

  std::vector<std::shared_ptr<Request>> returned;
  int removals = 0;
  for (auto event = RunOne(*engine.engine);
       event.flags != EngineEventFlagNone;
       event = RunOne(*engine.engine)) {
    auto ready = event.request;
    // The fixture's executor forces end-of-stream, so every request Run hands back is finished.
    EXPECT_TRUE(ready->IsTurnComplete());
    // The finished request still owns its cache slot: nothing is reclaimed implicitly.
    EXPECT_EQ(engine.cache->deallocate_calls, removals);
    returned.push_back(ready);

    ready->Close();
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

// With nothing queued, Run reports no work and returns cleanly.
TEST_F(EngineRunTest, RunWithNoRequestsReturnsNull) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));

  EXPECT_FALSE(engine.engine->HasPendingRequests());
  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagNone);
  EXPECT_EQ(engine.executor->decode_calls, 0);
}

TEST_F(EngineRunTest, ZeroCapacityValidatesOwnerThreadWithoutReclaimingOrProgressing) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto request = CreateRequestWithPrompt(
      engine.engine, *model_, Prompt(10));
  ExternalRequestReference public_handle{*request};
  public_handle.Release();

  std::exception_ptr off_thread_error;
  std::thread off_owner_thread([&] {
    try {
      static_cast<void>(engine.engine->Run({}));
    } catch (...) {
      off_thread_error = std::current_exception();
    }
  });
  off_owner_thread.join();

  ASSERT_NE(off_thread_error, nullptr);
  EXPECT_THROW(std::rethrow_exception(off_thread_error),
               std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_EQ(engine.executor->decode_calls, 0);

  EXPECT_EQ(engine.engine->Run({}), 0u);
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_EQ(engine.executor->decode_calls, 0);

  EXPECT_FALSE(engine.engine->HasPendingRequests());
  EXPECT_EQ(request->status_, RequestStatus::Closed);
}

TEST_F(EngineRunTest, HasPendingRequestsReclaimsPubliclyDestroyedUnassignedRequest) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto request = CreateEngineRequest(engine.engine, *model_);
  ExternalRequestReference public_handle{*request};

  public_handle.Release();

  EXPECT_FALSE(engine.engine->HasPendingRequests());
  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(engine.cache->deallocate_calls, 1);
  EXPECT_EQ(engine.executor->decode_calls, 0);
}

TEST_F(EngineRunTest, HasPendingRequestsReclaimsPubliclyDestroyedQueuedRequest) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto request = CreateRequestWithPrompt(
      engine.engine, *model_, Prompt(10));
  ExternalRequestReference public_handle{*request};

  public_handle.Release();

  EXPECT_FALSE(engine.engine->HasPendingRequests());
  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(engine.cache->deallocate_calls, 1);
  EXPECT_EQ(engine.executor->decode_calls, 0);
}

TEST_F(EngineRunTest, HasPendingRequestsValidatesOwnerThreadBeforeReclamation) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto request = CreateRequestWithPrompt(
      engine.engine, *model_, Prompt(10));
  ExternalRequestReference public_handle{*request};
  public_handle.Release();

  std::exception_ptr off_thread_error;
  std::thread off_owner_thread([&] {
    try {
      static_cast<void>(engine.engine->HasPendingRequests());
    } catch (...) {
      off_thread_error = std::current_exception();
    }
  });
  off_owner_thread.join();

  ASSERT_NE(off_thread_error, nullptr);
  EXPECT_THROW(std::rethrow_exception(off_thread_error),
               std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Assigned);

  EXPECT_FALSE(engine.engine->HasPendingRequests());
  EXPECT_EQ(request->status_, RequestStatus::Closed);
}

TEST_F(EngineRunTest, HasPendingRequestsReclaimsPubliclyDestroyedActiveResidentRequest) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto request = CreateRequestWithPrompt(
      engine.engine, *model_, Prompt(10));
  ExternalRequestReference public_handle{*request};
  engine.cache->Allocate({request});
  request->Schedule();
  ASSERT_EQ(request->status_, RequestStatus::Active);
  ASSERT_EQ(engine.cache->AllocatedCount(), 1u);

  public_handle.Release();

  EXPECT_FALSE(engine.engine->HasPendingRequests());
  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
  EXPECT_EQ(engine.cache->deallocate_calls, 1);
  EXPECT_EQ(engine.executor->decode_calls, 0);
}

TEST_F(EngineRunTest, HasPendingRequestsPurgesDestroyedTurnCompleteEventWithoutAffectingPeer) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/2, EosToken(*model_));
  auto survivor = CreateRequestWithPrompt(
      engine.engine, *model_, Prompt(10));
  auto abandoned = CreateRequestWithPrompt(
      engine.engine, *model_, Prompt(20));
  ExternalRequestReference survivor_handle{*survivor};
  ExternalRequestReference abandoned_handle{*abandoned};

  ASSERT_EQ(RunOne(*engine.engine).request, survivor);
  ASSERT_EQ(survivor->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(abandoned->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(engine.cache->AllocatedCount(), 2u);

  abandoned_handle.Release();

  EXPECT_FALSE(engine.engine->HasPendingRequests());
  EXPECT_EQ(abandoned->status_, RequestStatus::Closed);
  EXPECT_EQ(survivor->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);
  EXPECT_EQ(engine.cache->deallocate_calls, 1);
  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagNone);

  survivor->BeginTurn(std::array<int32_t, 1>{5},
                      std::optional<size_t>{1});
  EXPECT_TRUE(engine.engine->HasPendingRequests());
  EXPECT_EQ(RunOne(*engine.engine).request, survivor);
  EXPECT_EQ(survivor->status_, RequestStatus::TurnComplete);

  survivor->Close();
  survivor_handle.Release();
}

TEST_F(EngineRunTest, DestroyingEngineClosesRequestWhosePublicHandleSurvives) {
  auto cache = std::make_shared<RecordingCacheManager>(
      model_, /*capacity=*/8);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, EosToken(*model_));
  EngineDependencies dependencies{
      cache, std::move(scheduler), std::move(executor)};
  auto engine = std::make_shared<Engine>(
      model_, std::move(dependencies));
  auto request = CreateRequestWithPrompt(
      engine, *model_, Prompt(10));
  ExternalRequestReference public_request{*request};

  engine.reset();

  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_NO_THROW(public_request.Release());
}

TEST_F(EngineRunTest, StaticBatchingPreservesOrderingAndReusesResidentContinuation) {
  model_->config_->engine.dynamic_batching.reset();
  auto trace = std::make_shared<CallTrace>();
  auto cache = std::make_shared<RecordingCacheManager>(
      model_, /*capacity=*/4, trace, /*supports_dynamic_batching=*/false);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, /*forced_token=*/5, trace);
  auto* cache_observer = cache.get();
  auto* executor_observer = executor.get();
  EngineDependencies dependencies{cache, std::move(scheduler),
                                  std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));
  auto prompt = Prompt(10);
  auto request = CreateEngineRequest(engine, *model_);
  request->BeginTurn(prompt, std::optional<size_t>{1});

  EXPECT_EQ(RunOne(*engine).request, request);
  EXPECT_EQ(executor_observer->decode_calls, 1);
  EXPECT_LT(IndexOf(*trace, "Allocate"), IndexOf(*trace, "Decode"));

  const int allocations_before = cache_observer->allocate_calls;
  const std::vector<int32_t> continuation{5, 6};
  request->BeginTurn(continuation, std::optional<size_t>{1});
  ASSERT_EQ(request->status_, RequestStatus::Assigned);

  EXPECT_EQ(RunOne(*engine).request, request);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(executor_observer->decode_calls, 2);
  EXPECT_EQ(cache_observer->allocate_calls, allocations_before);
}

TEST_F(EngineRunTest, StaticBatchReturnsAllRowEventsWhenCapacitySuffices) {
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
  auto first = CreateEngineRequest(engine, *model_);
  auto second = CreateEngineRequest(engine, *model_);
  first->BeginTurn(Prompt(10), std::optional<size_t>{1});
  second->BeginTurn(Prompt(20), std::optional<size_t>{1});

  std::array<EngineEvent, 2> storage;
  ASSERT_EQ(engine->Run(storage), storage.size());
  EXPECT_EQ(storage[0].request, first);
  EXPECT_EQ(storage[1].request, second);
  EXPECT_EQ(executor_observer->decode_calls, 1);
  EXPECT_EQ(storage[0].flags,
            EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[1].flags,
            EngineEventFlagToken | EngineEventFlagTurnFinished);
}

TEST_F(EngineRunTest, StaticExecutionFailureTerminatesRequestsAndMarksEngineUnhealthy) {
  model_->config_->engine.dynamic_batching.reset();
  auto cache = std::make_shared<RecordingCacheManager>(
      model_, /*capacity=*/4, nullptr, /*supports_dynamic_batching=*/false);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, EosToken(*model_));
  auto* executor_observer = executor.get();
  EngineDependencies dependencies{cache, std::move(scheduler),
                                  std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));
  auto request = CreateRequestWithPrompt(engine, *model_, Prompt(10));
  executor_observer->SetNextFailure(ScriptedExecutionFailure::Fatal);

  const auto failure = RunOne(*engine);
  EXPECT_EQ(failure.flags, EngineEventFlagFailed);
  EXPECT_EQ(failure.error_code, EngineErrorCode::EngineExecutionFailure);
  EXPECT_THROW(static_cast<void>(RunOne(*engine)), EngineStepError);

  EXPECT_EQ(executor_observer->decode_calls, 1);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::Failed);
  EXPECT_FALSE(engine->HasPendingRequests());
}

TEST_F(EngineRunTest, StaticHasPendingClosesAbandonedResidentWithoutIndividualDeallocation) {
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
  auto request = CreateEngineRequest(engine, *model_);
  ExternalRequestReference external{*request};
  request->BeginTurn(prompt);

  ASSERT_EQ(RunOne(*engine).request, request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(cache->AllocatedCount(), 1u);
  external.Release();

  EXPECT_FALSE(engine->HasPendingRequests());
  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(executor_observer->decode_calls, 1);
  EXPECT_EQ(cache->AllocatedCount(), 1u);
  EXPECT_EQ(cache->deallocate_calls, 0);

  // A static row is logically closed once but remains physically resident until batch recycling.
  EXPECT_EQ(RunOne(*engine).flags, EngineEventFlagNone);
  EXPECT_EQ(cache->deallocate_calls, 0);
}

TEST_F(EngineRunTest, StaticContinuationFailsAfterBatchRecycling) {
  model_->config_->engine.dynamic_batching.reset();
  auto cache = std::make_shared<StaticCacheManager>(model_);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, EosToken(*model_));
  EngineDependencies dependencies{cache, std::move(scheduler),
                                  std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));

  auto first_prompt = Prompt(10);
  auto first = CreateRequestWithPrompt(engine, *model_, first_prompt);
  ASSERT_EQ(RunOne(*engine).request, first);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);

  auto second_prompt = Prompt(20);
  auto second = CreateRequestWithPrompt(engine, *model_, second_prompt);
  for (int run = 0; run < 2 && cache->IsResident(first); ++run) {
    ASSERT_NE(RunOne(*engine).flags, EngineEventFlagNone);
  }
  ASSERT_FALSE(cache->IsResident(first));

  const std::vector<int32_t> continuation{5, 6};
  EXPECT_THROW(first->BeginTurn(continuation), std::runtime_error);
  EXPECT_EQ(first->status_, RequestStatus::TurnComplete);
}

TEST_F(EngineRunTest, StaticContinuationRejectsMultiRowBatchAfterPeerCloses) {
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
  auto first = CreateRequestWithPrompt(engine, *model_, first_prompt);
  auto second = CreateRequestWithPrompt(engine, *model_, second_prompt);
  ASSERT_EQ(RunOne(*engine).request, first);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(second->status_, RequestStatus::TurnComplete);

  second->Close();
  ASSERT_EQ(second->status_, RequestStatus::Closed);
  const std::vector<int32_t> continuation{5, 6};
  EXPECT_THROW(first->BeginTurn(continuation), std::runtime_error);
  EXPECT_EQ(first->status_, RequestStatus::TurnComplete);
}

TEST_F(EngineRunTest, RunDoesNotReturnNullWhenCapacityDefersPendingWork) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  engine.cache->SetCanAllocate(false);
  auto prompt = Prompt(10);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(prompt, std::optional<size_t>{1});

  const auto deferred = RunOne(*engine.engine);
  EXPECT_EQ(deferred.flags, EngineEventFlagCapacityBlocked);
  EXPECT_EQ(deferred.error_code, EngineErrorCode::CapacityDeferred);

  EXPECT_TRUE(engine.engine->HasPendingRequests());
  EXPECT_EQ(request->status_, RequestStatus::Assigned);

  engine.cache->SetCanAllocate(true);
  EXPECT_EQ(RunOne(*engine.engine).request, request);
}

TEST_F(EngineRunTest, RetryableExecutionFailureRollsBackAndCanRetry) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto prompt = Prompt(10);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(prompt, std::optional<size_t>{1});
  const auto before = request->Snapshot();
  engine.executor->SetNextFailure(
      ScriptedExecutionFailure::RetryableDuringExecution);

  const auto retryable = RunOne(*engine.engine);
  EXPECT_EQ(retryable.flags, EngineEventFlagRetryable);
  EXPECT_EQ(retryable.error_code, EngineErrorCode::RetryableExecution);

  const auto rolled_back = request->Snapshot();
  EXPECT_EQ(rolled_back.status, before.status);
  EXPECT_EQ(rolled_back.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
  EXPECT_TRUE(engine.engine->HasPendingRequests());

  auto ready = RunOne(*engine.engine);
  EXPECT_EQ(ready.request, request);
  EXPECT_TRUE(request->IsTurnComplete());
  EXPECT_EQ(ready.token, 5);
  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagNone);
}

TEST_F(EngineRunTest, ContinuedResidentRollsBackToQueuedAndCanRetry) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = CreateRequestWithPrompt(engine.engine, *model_, prompt);
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(engine.cache->AllocatedCount(), 1u);

  const std::vector<int32_t> continuation{5, 6};
  request->BeginTurn(continuation);
  const auto before = request->Snapshot();
  ASSERT_EQ(before.status, RequestStatus::Assigned);
  engine.executor->SetNextFailure(
      ScriptedExecutionFailure::RetryableDuringExecution);

  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagRetryable);

  const auto rolled_back = request->Snapshot();
  EXPECT_EQ(rolled_back.status, RequestStatus::Assigned);
  EXPECT_EQ(rolled_back.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(rolled_back.processed_sequence_length,
            before.processed_sequence_length);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);

  auto ready = RunOne(*engine.engine);
  EXPECT_EQ(ready.request, request);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
}

TEST_F(EngineRunTest, PartialPrefillCommitsOneTransactionAndReturnsNoEvents) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt(10);
  auto params = MakeGreedyParams(*model_);
  params->search.chunk_size = 2;
  auto request = CreateRequestWithPrompt(engine.engine, *params, prompt);

  std::array<EngineEvent, 1> storage;
  EXPECT_EQ(engine.engine->Run(storage), 0u);
  EXPECT_EQ(engine.executor->decode_calls, 1);
  EXPECT_EQ(request->ProcessedSequenceLength(), 2);
  EXPECT_TRUE(request->IsPrefill());
  EXPECT_TRUE(engine.engine->HasPendingRequests());

  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].request, request);
  EXPECT_EQ(engine.executor->decode_calls, 2);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);

  const size_t calls_before = engine.executor->decoded_token_counts.size();
  const std::vector<int32_t> continuation{5, 6, 7, 8, 9};
  request->BeginTurn(continuation);
  EXPECT_EQ(engine.engine->Run(storage), 0u);
  EXPECT_TRUE(engine.engine->HasPendingRequests());
  EXPECT_EQ(engine.engine->Run(storage), 0u);
  EXPECT_TRUE(engine.engine->HasPendingRequests());
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].request, request);

  ASSERT_EQ(engine.executor->decoded_token_counts.size(), calls_before + 3);
  EXPECT_EQ(engine.executor->decoded_token_counts[calls_before], 2u);
  EXPECT_EQ(engine.executor->decoded_token_counts[calls_before + 1], 2u);
  EXPECT_EQ(engine.executor->decoded_token_counts[calls_before + 2], 1u);
  EXPECT_EQ(request->ProcessedSequenceLength(),
            request->CurrentSequenceLength());
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
}

TEST_F(EngineRunTest, UnserviceableContinuationPublishesTerminalFailure) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = CreateRequestWithPrompt(engine.engine, *model_, prompt);
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);

  const std::vector<int32_t> continuation{5, 6};
  request->BeginTurn(continuation);
  engine.cache->SetUnserviceableRequest(request);

  const auto failure = RunOne(*engine.engine);
  EXPECT_EQ(failure.request, request);
  EXPECT_EQ(failure.flags,
            EngineEventFlagTurnFinished | EngineEventFlagFailed);
  EXPECT_EQ(failure.error_code, EngineErrorCode::RequestUnserviceable);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::Failed);
  EXPECT_NO_THROW(request->Close());
  EXPECT_EQ(request->status_, RequestStatus::Closed);
}

TEST_F(EngineRunTest, ExecutionCapacityFailureRollsBackWithoutPoisoningEngine) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = CreateRequestWithPrompt(engine.engine, *model_, prompt);
  const auto before = request->Snapshot();
  engine.executor->SetNextFailure(ScriptedExecutionFailure::CapacityExceeded);

  const auto capacity = RunOne(*engine.engine);
  EXPECT_EQ(capacity.flags, EngineEventFlagCapacityBlocked);
  EXPECT_EQ(capacity.error_code,
            EngineErrorCode::ExecutionCapacityExceeded);

  const auto rolled_back = request->Snapshot();
  EXPECT_EQ(rolled_back.status, before.status);
  EXPECT_EQ(rolled_back.current_sequence_length,
            before.current_sequence_length);
  EXPECT_EQ(rolled_back.processed_sequence_length,
            before.processed_sequence_length);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
  EXPECT_TRUE(engine.engine->HasPendingRequests());

  auto ready = RunOne(*engine.engine);
  EXPECT_EQ(ready.request, request);
  EXPECT_TRUE(request->IsTurnComplete());
}

TEST_F(EngineRunTest, PostProcessingFailureRestoresSearchAndCanRetry) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = CreateRequestWithPrompt(engine.engine, *model_, prompt);
  const auto before = request->Snapshot();
  engine.executor->SetNextFailure(ScriptedExecutionFailure::PostProcessing);

  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagRetryable);

  const auto rolled_back = request->Snapshot();
  EXPECT_EQ(rolled_back.status, before.status);
  EXPECT_EQ(rolled_back.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(rolled_back.processed_sequence_length,
            before.processed_sequence_length);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);

  auto ready = RunOne(*engine.engine);
  EXPECT_EQ(ready.request, request);
  EXPECT_TRUE(request->IsTurnComplete());
}

TEST_F(EngineRunTest, LaterRequestFailureRestoresEarlierSample) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto first_prompt = Prompt(10);
  auto second_prompt = Prompt(20);
  auto first = CreateRequestWithPrompt(engine.engine, *model_, first_prompt);
  auto control = std::make_shared<RequestPostProcessingControl>();
  auto second_params = MakeGreedyParams(*model_);
  FailingPostProcessingDevice device{*second_params->p_device, control};
  second_params->p_device = &device;
  auto second =
      CreateRequestWithPrompt(engine.engine, *second_params, second_prompt);
  const auto first_before = first->Snapshot();
  const auto second_before = second->Snapshot();

  control->fail = true;

  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagRetryable);

  EXPECT_EQ(first->Snapshot().current_sequence_length,
            first_before.current_sequence_length);
  EXPECT_EQ(second->Snapshot().current_sequence_length,
            second_before.current_sequence_length);
  EXPECT_EQ(first->status_, RequestStatus::Assigned);
  EXPECT_EQ(second->status_, RequestStatus::Assigned);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);

  control->fail = false;
  auto ready = RunOne(*engine.engine);
  EXPECT_EQ(ready.request, first);
  EXPECT_EQ(first->CurrentSequenceLength(),
            first_before.current_sequence_length + 1);
  EXPECT_EQ(second->CurrentSequenceLength(),
            second_before.current_sequence_length + 1);
}

TEST_F(EngineRunTest, ClosingUndrainedReadyRequestPurgesItFromQueue) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto first_prompt = Prompt(10);
  auto second_prompt = Prompt(20);
  auto first = CreateRequestWithPrompt(engine.engine, *model_, first_prompt);
  auto second = CreateRequestWithPrompt(engine.engine, *model_, second_prompt);

  ASSERT_EQ(RunOne(*engine.engine).request, first);
  second->Close();
  ASSERT_EQ(second->status_, RequestStatus::Closed);

  EXPECT_EQ(RunOne(*engine.engine).request, first);
}

TEST_F(EngineRunTest, ClosingOnlyDrainedReadyRequestClearsQueue) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto prompt = Prompt(10);
  auto request = CreateRequestWithPrompt(engine.engine, *model_, prompt);

  ASSERT_EQ(RunOne(*engine.engine).request, request);
  request->Close();

  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagNone);
}

TEST_F(EngineRunTest, StaticTurnCompleteRowIsNotRepublishedWhilePeerRuns) {
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
  auto first_params = MakeGreedyParams(*model_);
  first_params->search.max_length =
      static_cast<int>(first_prompt.size() + 1);
  auto second_params = MakeGreedyParams(*model_);
  second_params->search.max_length =
      static_cast<int>(second_prompt.size() + 3);
  auto first = CreateRequestWithPrompt(engine, *first_params, first_prompt);
  auto second = CreateRequestWithPrompt(engine, *second_params, second_prompt);

  ASSERT_EQ(RunOne(*engine).request, first);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(RunOne(*engine).request, second);  // Drain the other event from the same model run.
  ASSERT_EQ(executor_observer->decode_calls, 1);

  EXPECT_EQ(RunOne(*engine).request, second);
  EXPECT_EQ(executor_observer->decode_calls, 2);
}

TEST_F(EngineRunTest, FatalExecutionFailureMarksEngineUnhealthy) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt(10);
  auto request = CreateRequestWithPrompt(engine.engine, *model_, prompt);
  engine.executor->SetNextFailure(ScriptedExecutionFailure::Fatal);

  const auto failure = RunOne(*engine.engine);
  EXPECT_EQ(failure.flags, EngineEventFlagFailed);
  EXPECT_EQ(failure.error_code, EngineErrorCode::EngineExecutionFailure);
  EXPECT_THROW(
      static_cast<void>(engine.engine->Run({})), EngineStepError);
  EXPECT_THROW(static_cast<void>(RunOne(*engine.engine)), EngineStepError);

  EXPECT_EQ(engine.executor->decode_calls, 1);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
  request->Close();
  EXPECT_EQ(request->status_, RequestStatus::Closed);
}

TEST_F(EngineRunTest, StaticSchedulerFailureMarksEngineUnhealthy) {
  model_->config_->engine.dynamic_batching.reset();
  auto cache = std::make_shared<RecordingCacheManager>(
      model_, /*capacity=*/8, nullptr, /*supports_dynamic_batching=*/false);
  auto scheduler = std::make_unique<ThrowingStaticScheduler>(model_);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, EosToken(*model_));
  auto* executor_observer = executor.get();
  EngineDependencies dependencies{
      cache, std::move(scheduler), std::move(executor)};
  auto engine = std::make_shared<Engine>(
      model_, std::move(dependencies));

  auto request = CreateRequestWithPrompt(engine, *model_, Prompt(10));

  const auto failure = RunOne(*engine);
  EXPECT_EQ(failure.flags, EngineEventFlagFailed);
  EXPECT_EQ(failure.error_code, EngineErrorCode::EngineContractFailure);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::Failed);
  EXPECT_EQ(executor_observer->decode_calls, 0);
  EXPECT_EQ(cache->AllocatedCount(), 0u);

  EXPECT_THROW(static_cast<void>(engine->Run({})), EngineStepError);
  EXPECT_THROW(static_cast<void>(RunOne(*engine)), EngineStepError);
}

TEST_F(EngineRunTest, BeginTurnIsRejectedAfterEngineBecomesUnhealthy) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto first_prompt = Prompt(10);
  auto first = CreateRequestWithPrompt(engine.engine, *model_, first_prompt);
  ASSERT_EQ(RunOne(*engine.engine).request, first);
  ASSERT_EQ(first->status_, RequestStatus::TurnComplete);

  auto second_prompt = Prompt(20);
  auto second = CreateRequestWithPrompt(engine.engine, *model_, second_prompt);
  engine.executor->SetNextFailure(ScriptedExecutionFailure::Fatal);
  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagFailed);

  const auto before = first->Snapshot();
  const std::vector<int32_t> continuation{5, 6};
  EXPECT_THROW(first->BeginTurn(continuation), EngineStepError);
  const auto after = first->Snapshot();
  EXPECT_EQ(after.status, RequestStatus::TurnComplete);
  EXPECT_EQ(after.current_sequence_length, before.current_sequence_length);
}

TEST_F(EngineRunTest, UnserviceableRequestDoesNotBlockFittingRequest) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto large_prompt = Prompt(10);
  auto fitting_prompt = Prompt(20);
  auto too_large =
      CreateRequestWithPrompt(engine.engine, *model_, large_prompt);
  auto fitting =
      CreateRequestWithPrompt(engine.engine, *model_, fitting_prompt);
  engine.cache->SetUnserviceableRequest(too_large);

  const auto failed = RunOne(*engine.engine);
  EXPECT_EQ(failed.request, too_large);
  EXPECT_EQ(failed.error_code, EngineErrorCode::RequestUnserviceable);
  EXPECT_EQ(RunOne(*engine.engine).request, fitting);
}

TEST_F(EngineRunTest, LaterFailurePreservesEarlierCommittedCycle) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/1, /*forced_token=*/5);
  auto first_prompt = Prompt(10);
  auto second_prompt = Prompt(20);
  auto first = CreateRequestWithPrompt(engine.engine, *model_, first_prompt);
  auto second = CreateRequestWithPrompt(engine.engine, *model_, second_prompt);

  EXPECT_EQ(RunOne(*engine.engine).request, first);
  const auto committed = first->Snapshot();
  ASSERT_EQ(engine.cache->AllocatedCount(), 1u);
  engine.executor->SetNextFailure(
      ScriptedExecutionFailure::RetryableBeforeExecution);

  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagRetryable);

  const auto after_abort = first->Snapshot();
  EXPECT_EQ(after_abort.status, committed.status);
  EXPECT_EQ(after_abort.current_sequence_length,
            committed.current_sequence_length);
  EXPECT_EQ(after_abort.processed_sequence_length,
            committed.processed_sequence_length);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);
}

TEST_F(EngineRunTest, MixedDecodeAndPrefillCommitPlanOwnedTokenCounts) {
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 3;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto first_prompt = Prompt(10);
  auto decode = CreateRequestWithPrompt(engine.engine, *model_, first_prompt);
  ASSERT_EQ(RunOne(*engine.engine).request, decode);
  const auto decode_before_mixed = decode->Snapshot();

  const std::vector<int32_t> long_prompt{2, 3, 4, 5, 6};
  auto prefill =
      CreateRequestWithPrompt(engine.engine, *model_, long_prompt);

  EXPECT_EQ(RunOne(*engine.engine).request, decode);

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

TEST_F(EngineRunTest, MixedRunRollbackPreservesProgressAndCacheResidents) {
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 3;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto first_prompt = Prompt(10);
  auto decode = CreateRequestWithPrompt(engine.engine, *model_, first_prompt);
  ASSERT_EQ(RunOne(*engine.engine).request, decode);

  const std::vector<int32_t> long_prompt{2, 3, 4, 5, 6};
  auto prefill =
      CreateRequestWithPrompt(engine.engine, *model_, long_prompt);
  const auto decode_before = decode->Snapshot();
  const auto prefill_before = prefill->Snapshot();
  ASSERT_EQ(engine.cache->AllocatedCount(), 1u);
  engine.executor->SetNextFailure(
      ScriptedExecutionFailure::RetryableDuringExecution);

  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagRetryable);

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

  EXPECT_EQ(RunOne(*engine.engine).request, decode);
  EXPECT_EQ(prefill->ProcessedSequenceLength(), 2);
  EXPECT_EQ(engine.cache->AllocatedCount(), 2u);
}

}  // namespace
}  // namespace test
}  // namespace Generators
