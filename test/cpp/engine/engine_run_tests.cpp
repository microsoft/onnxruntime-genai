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
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
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

class TestBarrier {
 public:
  explicit TestBarrier(size_t participant_count)
      : remaining_{participant_count} {}

  void ArriveAndWait() {
    std::unique_lock lock{mutex_};
    if (--remaining_ == 0) {
      condition_.notify_all();
      return;
    }
    condition_.wait(lock, [this] { return remaining_ == 0; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  size_t remaining_;
};

struct ExternalRequestRaceProbe
    : std::enable_shared_from_this<ExternalRequestRaceProbe>,
      ExternalRefCounted<ExternalRequestRaceProbe> {
  explicit ExternalRequestRaceProbe(
      std::shared_ptr<std::atomic<bool>> destroyed)
      : destroyed_{std::move(destroyed)} {}

  ~ExternalRequestRaceProbe() {
    destroyed_->store(true, std::memory_order_release);
  }

  std::shared_ptr<std::atomic<bool>> destroyed_;
};

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

// Asserts every element of one fixed-state input row equals `expected`. The composite tests run on
// CPU, so both direct bank views and fallback staging tensors are directly addressable.
void ExpectFixedInputRow(const FixedStateBinding& binding, size_t row, float expected) {
  const size_t row_elements = FixedRowElements(*binding.input);
  const auto* data = binding.input->GetTensorData<float>();
  for (size_t index = 0; index < row_elements; ++index) {
    EXPECT_FLOAT_EQ(data[row * row_elements + index], expected);
  }
}

// Fills one fixed-state output row with `value`, simulating a model writing its next state.
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
static_assert(
    noexcept(std::declval<ExternalRequestRaceProbe&>().ExternalRelease()));

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

  SchedulerAdmissionPreparation PrepareAddRequest(
      const std::shared_ptr<Request>& /*request*/) override {
    return {};
  }

  void CommitAddRequest(
      std::shared_ptr<Request> request,
      SchedulerAdmissionPreparation&& /*preparation*/) noexcept override {
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

TEST(ExternalRefCountedTest,
     DistinguishesNeverHeldHeldAbandonedAndReacquiredStates) {
  auto destroyed = std::make_shared<std::atomic<bool>>(false);
  auto probe = std::make_shared<ExternalRequestRaceProbe>(destroyed);

  EXPECT_FALSE(probe->ExternalReferencesAbandoned());
  probe->ExternalAddRef();
  EXPECT_FALSE(probe->ExternalReferencesAbandoned());
  probe->ExternalRelease();
  EXPECT_TRUE(probe->ExternalReferencesAbandoned());
  probe->ExternalAddRef();
  EXPECT_FALSE(probe->ExternalReferencesAbandoned());
  probe->ExternalRelease();
  EXPECT_TRUE(probe->ExternalReferencesAbandoned());
}

TEST(ExternalRefCountedTest,
     ConcurrentFinalReleaseAndReacquirePreserveOwnerLifetime) {
  auto destroyed = std::make_shared<std::atomic<bool>>(false);
  auto engine_owner =
      std::make_shared<ExternalRequestRaceProbe>(destroyed);
  auto* raw = engine_owner.get();
  raw->ExternalAddRef();

  TestBarrier transition_start{3};
  std::thread release_thread([raw, &transition_start] {
    transition_start.ArriveAndWait();
    raw->ExternalRelease();
  });
  std::thread reacquire_thread([engine_owner, &transition_start] {
    transition_start.ArriveAndWait();
    engine_owner->ExternalAddRef();
  });
  transition_start.ArriveAndWait();
  release_thread.join();
  reacquire_thread.join();

  EXPECT_FALSE(raw->ExternalReferencesAbandoned());
  EXPECT_FALSE(destroyed->load(std::memory_order_acquire));

  engine_owner.reset();
  EXPECT_FALSE(destroyed->load(std::memory_order_acquire));
  raw->ExternalRelease();
  EXPECT_TRUE(destroyed->load(std::memory_order_acquire));
}

TEST(EngineLifetimeTest, EngineRetainsModelForItsLifetime) {
  auto model = LoadSyntheticPagedModel();
  std::weak_ptr<Model> model_observer = model;
  auto engine = std::make_shared<Engine>(model);

  model.reset();
  EXPECT_FALSE(model_observer.expired());

  engine.reset();
  EXPECT_TRUE(model_observer.expired());
}

TEST(EngineLifetimeTest, DestroyingEngineReleasesCompositeCacheStorage) {
  auto model = LoadSyntheticCompositeModel();
  auto cache = std::make_shared<PagedCacheManager>(model);
  auto scheduler = Scheduler::Create(model, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model, cache, /*forced_token=*/5);
  EngineDependencies dependencies{
      cache, std::move(scheduler), std::move(executor)};
  auto engine = std::make_shared<Engine>(model, std::move(dependencies));
  auto first = CreateRequestWithPrompt(engine, *model, Prompt(10));
  auto second = CreateRequestWithPrompt(engine, *model, Prompt(20));

  EXPECT_EQ(RunOne(*engine).request, first);
  EXPECT_EQ(RunOne(*engine).request, second);
  ASSERT_EQ(cache->ResidentRequestCount(), 2u);
  ASSERT_TRUE(cache->FixedStateSnapshot().has_value());

  engine.reset();

  EXPECT_EQ(cache->ResidentRequestCount(), 0u);
  EXPECT_FALSE(cache->FixedStateSnapshot().has_value());
  EXPECT_EQ(first->status_, RequestStatus::Closed);
  EXPECT_EQ(second->status_, RequestStatus::Closed);
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

TEST_F(EngineRunTest,
       ConcurrentFinalReleaseAndReacquireCancelsRequestAbandonment) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/1, EosToken(*model_));
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->ExternalAddRef();
  request->BeginTurn(Prompt(10));
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);

  TestBarrier transition_start{3};
  std::thread release_thread([request, &transition_start] {
    transition_start.ArriveAndWait();
    request->ExternalRelease();
  });
  std::thread reacquire_thread([request, &transition_start] {
    transition_start.ArriveAndWait();
    request->ExternalAddRef();
  });
  transition_start.ArriveAndWait();
  release_thread.join();
  reacquire_thread.join();

  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagNone);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(engine.cache->AllocatedCount(), 1u);
  EXPECT_EQ(engine.cache->deallocate_calls, 0);

  request->Close();
  request->ExternalRelease();
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

TEST_F(EngineRunTest, OffThreadFinalReleaseBeforeBeginTurnDefersDestructionToOwnerThread) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto request = CreateEngineRequest(engine.engine, *model_);
  ExternalRequestReference public_handle{*request};
  std::weak_ptr<Request> request_lifetime = request;
  request.reset();

  std::thread final_release([&] {
    public_handle.Release();
  });
  final_release.join();

  // The Engine's strong tracking reference keeps Search and other runtime state alive until an
  // owner-thread boundary performs reclamation.
  EXPECT_FALSE(request_lifetime.expired());
  EXPECT_FALSE(engine.engine->HasPendingRequests());
  EXPECT_TRUE(request_lifetime.expired());
  EXPECT_EQ(engine.cache->deallocate_calls, 1);
}

TEST_F(EngineRunTest, OffThreadFinalReleaseDoesNotKeepDestroyedEngineAlive) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto request = CreateEngineRequest(engine.engine, *model_);
  ExternalRequestReference public_handle{*request};
  std::weak_ptr<Engine> engine_lifetime = engine.engine;

  engine.engine.reset();
  EXPECT_TRUE(engine_lifetime.expired());

  std::thread final_release([&] {
    public_handle.Release();
  });
  final_release.join();
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
      model_, cache, /*forced_token=*/5);
  EngineDependencies dependencies{
      cache, std::move(scheduler), std::move(executor)};
  auto engine = std::make_shared<Engine>(
      model_, std::move(dependencies));
  auto request = CreateRequestWithPrompt(
      engine, *model_, Prompt(10));
  ExternalRequestReference public_request{*request};
  ASSERT_EQ(RunOne(*engine).request, request);
  ASSERT_EQ(cache->AllocatedCount(), 1u);
  ASSERT_EQ(request->status_, RequestStatus::Active);

  engine.reset();

  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
  EXPECT_EQ(cache->deallocate_calls, 1);
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

TEST_F(EngineRunTest, StaticCloseLogicallyRemovesActiveRowWhilePeerContinues) {
  model_->config_->engine.dynamic_batching.reset();
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto cache = std::make_shared<StaticCacheManager>(model_);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, filler);
  auto* executor_observer = executor.get();
  EngineDependencies dependencies{cache, std::move(scheduler),
                                  std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));
  auto first = CreateEngineRequest(engine, *model_);
  auto second = CreateEngineRequest(engine, *model_);
  first->BeginTurn(Prompt(10), std::optional<size_t>{2});
  second->BeginTurn(Prompt(20), std::optional<size_t>{2});

  // Capacity one retains the second row's first token event inside the Engine.
  ASSERT_EQ(RunOne(*engine).request, first);
  ASSERT_EQ(first->status_, RequestStatus::Active);
  ASSERT_EQ(second->status_, RequestStatus::Active);
  ASSERT_EQ(executor_observer->decode_calls, 1);
  ASSERT_EQ(cache->ResidentRequestCount(), 2u);
  const int64_t closed_length = second->CurrentSequenceLength();
  const int64_t closed_processed = second->ProcessedSequenceLength();
  const size_t closed_generated = second->TurnGeneratedTokens();
  const int searches_while_resident = LeakChecked<Search>::Count();

  EXPECT_NO_THROW(second->Close());
  EXPECT_EQ(second->status_, RequestStatus::Closed);
  EXPECT_TRUE(second->BelongsTo(*engine));
  EXPECT_EQ(cache->ResidentRequestCount(), 2u);
  EXPECT_EQ(LeakChecked<Search>::Count(), searches_while_resident);

  // Close purged the retained event, so Run executes the next batch step instead of returning the
  // closed row. Only the peer samples and publishes an event.
  const auto peer_event = RunOne(*engine);
  EXPECT_EQ(peer_event.request, first);
  EXPECT_EQ(peer_event.flags,
            EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(executor_observer->decode_calls, 2);
  EXPECT_EQ(first->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(second->status_, RequestStatus::Closed);
  EXPECT_EQ(second->CurrentSequenceLength(), closed_length);
  EXPECT_EQ(second->ProcessedSequenceLength(), closed_processed);
  EXPECT_EQ(second->TurnGeneratedTokens(), closed_generated);
  EXPECT_EQ(LeakChecked<Search>::Count(), searches_while_resident);

  // New work recycles the all-terminal static batch. The Engine completes the retained close on
  // its owner thread before executing the replacement row, leaving only a lightweight tombstone.
  auto replacement = CreateEngineRequest(engine, *model_);
  replacement->BeginTurn(Prompt(30), std::optional<size_t>{1});
  const int searches_before_recycle = LeakChecked<Search>::Count();
  EXPECT_EQ(RunOne(*engine).request, replacement);
  EXPECT_FALSE(second->BelongsTo(*engine));
  EXPECT_FALSE(cache->IsResident(second));
  EXPECT_EQ(cache->ResidentRequestCount(), 1u);
  EXPECT_EQ(LeakChecked<Search>::Count(), searches_before_recycle - 1);
}

TEST_F(EngineRunTest, StaticAbandonmentLogicallyRemovesActiveRowAtNextBoundary) {
  model_->config_->engine.dynamic_batching.reset();
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto cache = std::make_shared<StaticCacheManager>(model_);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, filler);
  auto* executor_observer = executor.get();
  EngineDependencies dependencies{cache, std::move(scheduler),
                                  std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));
  auto first = CreateEngineRequest(engine, *model_);
  auto second = CreateEngineRequest(engine, *model_);
  ExternalRequestReference second_handle{*second};
  first->BeginTurn(Prompt(10), std::optional<size_t>{2});
  second->BeginTurn(Prompt(20), std::optional<size_t>{2});

  // Capacity one retains the second row's first token event inside the Engine.
  ASSERT_EQ(RunOne(*engine).request, first);
  ASSERT_EQ(first->status_, RequestStatus::Active);
  ASSERT_EQ(second->status_, RequestStatus::Active);
  ASSERT_EQ(executor_observer->decode_calls, 1);
  const int64_t closed_length = second->CurrentSequenceLength();
  const int64_t closed_processed = second->ProcessedSequenceLength();
  const size_t closed_generated = second->TurnGeneratedTokens();
  const int searches_while_resident = LeakChecked<Search>::Count();

  second_handle.Release();
  EXPECT_TRUE(engine->HasPendingRequests());
  EXPECT_EQ(second->status_, RequestStatus::Closed);
  EXPECT_TRUE(second->BelongsTo(*engine));
  EXPECT_EQ(cache->ResidentRequestCount(), 2u);
  EXPECT_EQ(LeakChecked<Search>::Count(), searches_while_resident);

  // The abandonment boundary purged the retained event. Only the peer advances and publishes.
  const auto peer_event = RunOne(*engine);
  EXPECT_EQ(peer_event.request, first);
  EXPECT_EQ(peer_event.flags,
            EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(executor_observer->decode_calls, 2);
  EXPECT_EQ(first->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(second->status_, RequestStatus::Closed);
  EXPECT_EQ(second->CurrentSequenceLength(), closed_length);
  EXPECT_EQ(second->ProcessedSequenceLength(), closed_processed);
  EXPECT_EQ(second->TurnGeneratedTokens(), closed_generated);
  EXPECT_EQ(LeakChecked<Search>::Count(), searches_while_resident);

  auto replacement = CreateEngineRequest(engine, *model_);
  replacement->BeginTurn(Prompt(30), std::optional<size_t>{1});
  const int searches_before_recycle = LeakChecked<Search>::Count();
  EXPECT_EQ(RunOne(*engine).request, replacement);
  EXPECT_FALSE(second->BelongsTo(*engine));
  EXPECT_FALSE(cache->IsResident(second));
  EXPECT_EQ(cache->ResidentRequestCount(), 1u);
  EXPECT_EQ(LeakChecked<Search>::Count(), searches_before_recycle - 1);
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
  EXPECT_EQ(failure.request, request);
  EXPECT_EQ(failure.turn_id, request->CurrentTurnId());
  EXPECT_EQ(failure.flags,
            EngineEventFlagTurnFinished | EngineEventFlagFailed);
  EXPECT_EQ(failure.finish_reason, GenerationFinishReason::Failed);
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
  EXPECT_EQ(failure.request, request);
  EXPECT_EQ(failure.turn_id, request->CurrentTurnId());
  EXPECT_EQ(failure.flags,
            EngineEventFlagTurnFinished | EngineEventFlagFailed);
  EXPECT_EQ(failure.finish_reason, GenerationFinishReason::Failed);
  EXPECT_EQ(failure.error_code, EngineErrorCode::EngineExecutionFailure);
  EXPECT_THROW(
      static_cast<void>(engine.engine->Run({})), EngineStepError);
  EXPECT_THROW(static_cast<void>(RunOne(*engine.engine)), EngineStepError);

  EXPECT_EQ(engine.executor->decode_calls, 1);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
  request->Close();
  EXPECT_EQ(request->status_, RequestStatus::Closed);
}

TEST_F(EngineRunTest, FatalExecutionFailurePublishesEveryAffectedTurn) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto first = CreateRequestWithPrompt(
      engine.engine, *model_, Prompt(10));
  auto second = CreateRequestWithPrompt(
      engine.engine, *model_, Prompt(20));
  engine.executor->SetNextFailure(ScriptedExecutionFailure::Fatal);

  std::array<EngineEvent, 2> events;
  ASSERT_EQ(engine.engine->Run(events), events.size());
  EXPECT_EQ(events[0].request, first);
  EXPECT_EQ(events[1].request, second);
  for (const auto& event : events) {
    ASSERT_NE(event.request, nullptr);
    EXPECT_EQ(event.turn_id, event.request->CurrentTurnId());
    EXPECT_EQ(event.flags,
              EngineEventFlagTurnFinished | EngineEventFlagFailed);
    EXPECT_EQ(event.finish_reason, GenerationFinishReason::Failed);
    EXPECT_EQ(event.error_code, EngineErrorCode::EngineExecutionFailure);
    EXPECT_EQ(event.usage.prompt_tokens, event.request->TurnPromptTokens());
    EXPECT_EQ(event.usage.generated_tokens,
              event.request->TurnGeneratedTokens());
  }
  EXPECT_EQ(first->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(second->status_, RequestStatus::TurnComplete);
  EXPECT_FALSE(engine.engine->HasPendingRequests());
  EXPECT_THROW(static_cast<void>(RunOne(*engine.engine)), EngineStepError);
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
  EXPECT_EQ(failure.request, request);
  EXPECT_EQ(failure.turn_id, request->CurrentTurnId());
  EXPECT_EQ(failure.flags,
            EngineEventFlagTurnFinished | EngineEventFlagFailed);
  EXPECT_EQ(failure.finish_reason, GenerationFinishReason::Failed);
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
  EXPECT_EQ(
      RunOne(*engine.engine).flags,
      EngineEventFlagTurnFinished | EngineEventFlagFailed);

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

TEST_F(EngineRunTest, SpeculativeRunKeepsAcceptedPrefixAndEmitsAllTokens) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto request =
      CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  ASSERT_EQ(request->status_, RequestStatus::Active);
  const int64_t length_after_prefill = request->CurrentSequenceLength();
  const size_t generated_after_prefill = request->TurnGeneratedTokens();

  request->SetDraftTokens(std::vector<int32_t>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({11, 12, 21, 22});

  std::array<EngineEvent, 3> events;
  ASSERT_EQ(engine.engine->Run(events), 3u);

  EXPECT_EQ(events[0].request, request);
  EXPECT_EQ(events[0].flags, EngineEventFlagToken);
  EXPECT_EQ(events[0].token, 11);
  EXPECT_EQ(events[1].request, request);
  EXPECT_EQ(events[1].flags, EngineEventFlagToken);
  EXPECT_EQ(events[1].token, 12);
  EXPECT_EQ(events[2].request, request);
  EXPECT_EQ(events[2].flags, EngineEventFlagToken);
  EXPECT_EQ(events[2].token, 21);
  ASSERT_EQ(engine.executor->decoded_token_counts.size(), 2u);
  EXPECT_EQ(engine.executor->decoded_token_counts[1], 4u);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].row, 0u);
  EXPECT_EQ(engine.cache->prefix_commits[0].request_id, request.get());
  EXPECT_EQ(engine.cache->prefix_commits[0].step_tokens, 4u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 3u);
  EXPECT_EQ(request->CurrentSequenceLength(), length_after_prefill + 3);
  EXPECT_EQ(request->ProcessedSequenceLength(), length_after_prefill + 2);
  EXPECT_EQ(request->TurnGeneratedTokens(), generated_after_prefill + 3);
  EXPECT_EQ(request->PendingDraftTokenCount(), 0u);
}

// The bonus row predicting EOS ends the turn without appending it, so the step's only visible
// tokens are the accepted drafts.
TEST_F(EngineRunTest, SpeculativeRunWithEosBonusTokenEmitsOnlyAcceptedDrafts) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto request =
      CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  const int64_t length_after_prefill = request->CurrentSequenceLength();
  const size_t generated_after_prefill = request->TurnGeneratedTokens();

  request->SetDraftTokens(std::vector<int32_t>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({11, 12, 13, eos});

  std::array<EngineEvent, 4> events;
  ASSERT_EQ(engine.engine->Run(events), 3u);

  EXPECT_EQ(events[0].token, 11);
  EXPECT_EQ(events[1].token, 12);
  EXPECT_EQ(events[2].token, 13);
  EXPECT_EQ(events[0].flags, EngineEventFlagToken);
  EXPECT_EQ(events[1].flags, EngineEventFlagToken);
  EXPECT_EQ(
      events[2].flags,
      EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(events[2].finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(request->CurrentSequenceLength(), length_after_prefill + 3);
  EXPECT_EQ(request->TurnGeneratedTokens(), generated_after_prefill + 3);
}

TEST_F(EngineRunTest, SpeculativeRunAcceptingEveryDraftEmitsBonusToken) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto request =
      CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  const int64_t length_after_prefill = request->CurrentSequenceLength();
  const size_t generated_after_prefill = request->TurnGeneratedTokens();

  request->SetDraftTokens(std::vector<int32_t>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({11, 12, 13, 25});

  std::array<EngineEvent, 4> events;
  ASSERT_EQ(engine.engine->Run(events), 4u);

  EXPECT_EQ(events[0].token, 11);
  EXPECT_EQ(events[1].token, 12);
  EXPECT_EQ(events[2].token, 13);
  EXPECT_EQ(events[3].token, 25);
  for (const auto& event : events) {
    EXPECT_EQ(event.request, request);
    EXPECT_EQ(event.flags, EngineEventFlagToken);
  }
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 4u);
  EXPECT_EQ(request->CurrentSequenceLength(), length_after_prefill + 4);
  EXPECT_EQ(request->ProcessedSequenceLength(), length_after_prefill + 3);
  EXPECT_EQ(request->TurnGeneratedTokens(), generated_after_prefill + 4);
}

TEST_F(EngineRunTest, SpeculativeOverflowCancellationFinishesLastTokenEvent) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto request =
      CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  ASSERT_EQ(RunOne(*engine.engine).request, request);

  request->SetDraftTokens(std::vector<int32_t>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({11, 12, 13, 25});

  const auto first = RunOne(*engine.engine);
  EXPECT_EQ(first.request, request);
  EXPECT_EQ(first.flags, EngineEventFlagToken);
  EXPECT_EQ(first.token, 11);
  ASSERT_EQ(engine.executor->decoded_token_counts.size(), 2u);

  EXPECT_TRUE(request->Cancel(request->CurrentTurnId()));
  std::array<EngineEvent, 3> retained;
  ASSERT_EQ(engine.engine->Run(retained), 3u);
  EXPECT_EQ(retained[0].token, 12);
  EXPECT_EQ(retained[0].flags, EngineEventFlagToken);
  EXPECT_EQ(retained[1].token, 13);
  EXPECT_EQ(retained[1].flags, EngineEventFlagToken);
  EXPECT_EQ(retained[2].token, 25);
  EXPECT_EQ(
      retained[2].flags,
      EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(retained[2].finish_reason, GenerationFinishReason::Canceled);
  EXPECT_EQ(engine.executor->decoded_token_counts.size(), 2u);
}

TEST_F(EngineRunTest, CancelClearsPendingDraftsBeforeContinuation) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto request =
      CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  request->SetDraftTokens(std::vector<int32_t>{11, 12});
  ASSERT_EQ(request->PendingDraftTokenCount(), 2u);

  EXPECT_TRUE(request->Cancel(request->CurrentTurnId()));
  EXPECT_EQ(request->PendingDraftTokenCount(), 0u);
  const auto canceled = RunOne(*engine.engine);
  EXPECT_EQ(canceled.request, request);
  EXPECT_EQ(canceled.flags, EngineEventFlagTurnFinished);
  EXPECT_EQ(canceled.finish_reason, GenerationFinishReason::Canceled);

  request->BeginTurn(
      std::array<int32_t, 1>{5}, std::optional<size_t>{1});
  EXPECT_EQ(request->PendingDraftTokenCount(), 0u);
  const size_t decode_calls_before =
      engine.executor->decoded_token_counts.size();
  const auto continued = RunOne(*engine.engine);

  EXPECT_EQ(continued.request, request);
  EXPECT_EQ(
      continued.flags,
      EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(continued.finish_reason, GenerationFinishReason::TurnLimit);
  ASSERT_EQ(
      engine.executor->decoded_token_counts.size(),
      decode_calls_before + 1);
  EXPECT_EQ(engine.executor->decoded_token_counts.back(), 2u);
}

TEST_F(EngineRunTest, SpeculativeRunStopsAtAcceptedEos) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto request =
      CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  const int64_t length_after_prefill = request->CurrentSequenceLength();
  const size_t generated_after_prefill = request->TurnGeneratedTokens();

  request->SetDraftTokens(std::vector<int32_t>{11, eos, 13});
  engine.executor->SetVerifyRowTokens({11, eos, 13, 25});

  const auto event = RunOne(*engine.engine);

  EXPECT_EQ(event.request, request);
  EXPECT_EQ(
      event.flags,
      EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(event.token, 11);
  EXPECT_EQ(event.finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::EosToken);
  EXPECT_EQ(request->CurrentSequenceLength(), length_after_prefill + 1);
  EXPECT_EQ(request->ProcessedSequenceLength(), length_after_prefill + 1);
  EXPECT_EQ(request->TurnGeneratedTokens(), generated_after_prefill + 1);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 2u);
}

#if USE_CUDA
TEST_F(EngineRunTest, CudaSpeculativeRunStopsWhenTheFirstDraftIsEos) {
  model_ = LoadSyntheticPagedCudaModel();
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeCompositeDoublesEngine(model_, filler);

  auto request =
      CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  const int64_t length_after_prefill = request->CurrentSequenceLength();
  const size_t generated_after_prefill = request->TurnGeneratedTokens();

  request->SetDraftTokens(std::vector<int32_t>{eos, 12});
  engine.executor->SetVerifyRowTokens({eos, 12, 25});

  const auto event = RunOne(*engine.engine);

  EXPECT_EQ(event.request, request);
  EXPECT_EQ(event.flags, EngineEventFlagTurnFinished);
  EXPECT_EQ(event.flags & EngineEventFlagToken, 0u);
  EXPECT_EQ(event.finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::EosToken);
  EXPECT_EQ(request->CurrentSequenceLength(), length_after_prefill);
  EXPECT_EQ(request->ProcessedSequenceLength(), length_after_prefill);
  EXPECT_EQ(request->TurnGeneratedTokens(), generated_after_prefill);

  const auto cache = engine.cache->Snapshot();
  ASSERT_EQ(cache.requests.size(), 1u);
  EXPECT_EQ(cache.requests[0].request_id, request.get());
  EXPECT_EQ(cache.requests[0].used_slots,
            static_cast<size_t>(length_after_prefill));
}

TEST_F(EngineRunTest, CudaSpeculativeRunStopsAtAnAcceptedMiddleEos) {
  model_ = LoadSyntheticPagedCudaModel();
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeCompositeDoublesEngine(model_, filler);

  auto request =
      CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  const int64_t length_after_prefill = request->CurrentSequenceLength();
  const size_t generated_after_prefill = request->TurnGeneratedTokens();

  request->SetDraftTokens(std::vector<int32_t>{11, eos, 13});
  engine.executor->SetVerifyRowTokens({11, eos, 13, 25});

  const auto event = RunOne(*engine.engine);

  EXPECT_EQ(event.request, request);
  EXPECT_EQ(event.flags,
            EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(event.token, 11);
  EXPECT_EQ(event.finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::EosToken);
  EXPECT_EQ(request->CurrentSequenceLength(), length_after_prefill + 1);
  EXPECT_EQ(request->ProcessedSequenceLength(), length_after_prefill + 1);
  EXPECT_EQ(request->TurnGeneratedTokens(), generated_after_prefill + 1);

  const auto cache = engine.cache->Snapshot();
  ASSERT_EQ(cache.requests.size(), 1u);
  EXPECT_EQ(cache.requests[0].request_id, request.get());
  EXPECT_EQ(cache.requests[0].used_slots,
            static_cast<size_t>(length_after_prefill + 1));
}
#endif

TEST_F(EngineRunTest, RolledBackSpeculativeRunLeavesProposalPendingAndRetryable) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto request =
      CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  const auto committed = request->Snapshot();
  const size_t generated_after_prefill = request->TurnGeneratedTokens();

  request->SetDraftTokens(std::vector<int32_t>{11, 12});
  engine.executor->SetNextFailure(
      ScriptedExecutionFailure::PostProcessing);

  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagRetryable);
  EXPECT_EQ(request->Snapshot().current_sequence_length,
            committed.current_sequence_length);
  EXPECT_EQ(request->Snapshot().processed_sequence_length,
            committed.processed_sequence_length);
  EXPECT_EQ(request->TurnGeneratedTokens(), generated_after_prefill);
  EXPECT_EQ(request->PendingDraftTokenCount(), 2u);
  EXPECT_EQ(request->AcceptedDraftTokenCount(), 0u);
  EXPECT_TRUE(engine.cache->prefix_commits.empty());

  engine.executor->SetVerifyRowTokens({11, 12, 25});
  std::array<EngineEvent, 3> events;
  ASSERT_EQ(engine.engine->Run(events), 3u);
  EXPECT_EQ(events[0].token, 11);
  EXPECT_EQ(events[1].token, 12);
  EXPECT_EQ(events[2].token, 25);
  for (const auto& event : events) {
    EXPECT_EQ(event.request, request);
    EXPECT_EQ(event.flags, EngineEventFlagToken);
  }
  EXPECT_EQ(request->TurnGeneratedTokens(), generated_after_prefill + 3);
}

TEST_F(EngineRunTest, DraftsRequireRollbackAndPerTokenLogitsCapabilities) {
  auto engine =
      MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto request = CreateEngineRequest(engine.engine, *model_);

  EXPECT_EQ(engine.engine->MaxDraftTokensPerStep(), 0u);
  EXPECT_THROW(
      request->SetDraftTokens(std::vector<int32_t>{11}),
      std::runtime_error);

  engine.cache->SetMaxDraftTokensPerStep(3);
  engine.executor->SetSupportsDraftVerification(false);
  EXPECT_EQ(engine.engine->MaxDraftTokensPerStep(), 0u);
  EXPECT_THROW(
      request->SetDraftTokens(std::vector<int32_t>{11}),
      std::runtime_error);
}

TEST_F(EngineRunTest, DraftProposalRequiresDecodeReadyRequest) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto params = MakeGreedyParams(*model_);
  params->search.chunk_size = 2;
  auto request =
      CreateRequestWithPrompt(engine.engine, *params, Prompt(10));
  EXPECT_THROW(
      request->SetDraftTokens(std::vector<int32_t>{11, 12}),
      std::runtime_error);

  std::array<EngineEvent, 1> storage;
  EXPECT_EQ(engine.engine->Run(storage), 0u);
  EXPECT_EQ(request->PendingDraftTokenCount(), 0u);
  ASSERT_EQ(engine.executor->decoded_token_counts.size(), 1u);
  EXPECT_EQ(engine.executor->decoded_token_counts[0], 2u);
  EXPECT_TRUE(engine.cache->prefix_commits.empty());

  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].request, request);
  ASSERT_EQ(engine.executor->decoded_token_counts.size(), 2u);
  EXPECT_EQ(engine.executor->decoded_token_counts[1], 1u);
}

TEST_F(EngineRunTest, InvalidDraftReplacementPreservesPendingProposal) {
  const int32_t eos = EosToken(*model_);
  const int32_t filler = eos == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, filler);
  engine.cache->SetMaxDraftTokensPerStep(3);

  auto request =
      CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  request->SetDraftTokens(std::vector<int32_t>{11, 12});

  EXPECT_THROW(
      request->SetDraftTokens(std::vector<int32_t>{11, 12, 13, 14}),
      std::runtime_error);
  EXPECT_EQ(request->PendingDraftTokenCount(), 2u);

  engine.executor->SetVerifyRowTokens({11, 12, 25});
  std::array<EngineEvent, 3> events;
  ASSERT_EQ(engine.engine->Run(events), 3u);
  EXPECT_EQ(events[0].token, 11);
  EXPECT_EQ(events[1].token, 12);
  EXPECT_EQ(events[2].token, 25);
}

// ---------------------------------------------------------------------------------------------
// Composite decoder-state transactions (paged KV + fixed state pool)
//
// These wire the real PagedCacheManager (real PagedKeyValueCache and real FixedStatePool) over the
// recording model-executor double. The executor fabricates logits and never runs the ONNX graph, so
// the fixed state tensors are inspected and written by the execution callback in physical row order.
// ---------------------------------------------------------------------------------------------

TEST_F(EngineRunTest, DensePagedModelHasNoFixedStateReservation) {
  model_ = LoadSyntheticPagedModel();
  auto engine = MakeCompositeDoublesEngine(model_, EosToken(*model_));
  auto request = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
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

  EXPECT_EQ(RunOne(*engine.engine).request, request);
  EXPECT_FALSE(engine.cache->FixedStateSnapshot().has_value());
}

TEST_F(EngineRunTest, CompositeReservationExposesRowsAndCommitsBothStates) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto first = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  auto second = CreateRequestWithPrompt(engine.engine, *model_, Prompt(20));

  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    ASSERT_NE(context.plan, nullptr);
    ASSERT_TRUE(context.plan->fixed_state.required);
    EXPECT_EQ(context.plan->fixed_state.row_count, 2u);
    EXPECT_EQ(context.plan->fixed_state.new_slot_count, 2u);
    EXPECT_GT(context.fixed_state_staging_bytes, 0u);
    EXPECT_EQ(context.fixed_state_staging_bytes,
              context.plan->fixed_state.staging_bytes);
    ASSERT_EQ(context.fixed_state_slots.size(), 2u);
    // conv layers [0, 3] plus recurrent layers [2, 5] => four fixed tensors.
    ASSERT_EQ(context.fixed_state_bindings.size(), 4u);
    for (size_t row = 0; row < context.plan->requests.size(); ++row) {
      EXPECT_EQ(context.fixed_state_slots[row].request_id,
                context.plan->requests[row].request_id);
      for (const auto& binding : context.fixed_state_bindings) {
        ExpectFixedInputRow(binding, row, 0.0f);
        FillFixedOutputRow(binding, row, row == 0 ? 11.0f : 22.0f);
      }
    }
    const auto fixed = engine.cache->FixedStateSnapshot();
    ASSERT_TRUE(fixed.has_value());
    EXPECT_TRUE(ValidateCompositeStateInvariants(
                    engine.cache->Snapshot(context.cache_reservation), *fixed,
                    {first->Snapshot(), second->Snapshot()})
                    .empty());
  });

  EXPECT_EQ(RunOne(*engine.engine).request, first);
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

TEST_F(EngineRunTest, CompositeExecutionFailureDiscardsBothAndRetryMatches) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto request = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  int callback_calls = 0;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    ++callback_calls;
    ASSERT_EQ(context.fixed_state_slots.size(), 1u);
    for (const auto& binding : context.fixed_state_bindings) {
      ExpectFixedInputRow(binding, 0, 0.0f);
      FillFixedOutputRow(binding, 0, 7.0f);
    }
  });
  engine.executor->SetNextFailure(
      ScriptedExecutionFailure::RetryableDuringExecution);

  const auto retryable = RunOne(*engine.engine);
  EXPECT_EQ(retryable.flags, EngineEventFlagRetryable);
  EXPECT_EQ(retryable.error_code, EngineErrorCode::RetryableExecution);

  auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(fixed->committed_slots, 0u);
  EXPECT_EQ(fixed->reserved_slots, 0u);
  EXPECT_EQ(fixed->free_slots, fixed->capacity);
  EXPECT_TRUE(engine.cache->Snapshot().requests.empty());
  EXPECT_EQ(request->Snapshot().processed_sequence_length, 0);

  EXPECT_EQ(RunOne(*engine.engine).request, request);
  fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(callback_calls, 2);
  EXPECT_EQ(fixed->committed_slots, 1u);
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).state_generation, 1u);
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).committed_tokens, 3u);
}

TEST_F(EngineRunTest, CompositePostProcessingFailurePreservesResidentState) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto request = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  float expected_input = 0.0f;
  float staged_output = 5.0f;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    for (const auto& binding : context.fixed_state_bindings) {
      ExpectFixedInputRow(binding, 0, expected_input);
      FillFixedOutputRow(binding, 0, staged_output);
    }
  });

  ASSERT_EQ(RunOne(*engine.engine).request, request);
  expected_input = 5.0f;
  staged_output = 99.0f;
  engine.executor->SetNextFailure(ScriptedExecutionFailure::PostProcessing);
  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagRetryable);

  auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).state_generation, 1u);
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).committed_tokens, 3u);
  ASSERT_EQ(engine.cache->Snapshot().requests.size(), 1u);
  EXPECT_EQ(engine.cache->Snapshot().requests[0].used_slots, 3u);

  staged_output = 6.0f;
  EXPECT_EQ(RunOne(*engine.engine).request, request);
  fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).state_generation, 2u);
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).committed_tokens, 4u);
}

TEST_F(EngineRunTest, CompositeChunkFailureRetriesAndContinuesResidentState) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, EosToken(*model_));
  const std::vector<int32_t> prompt{2, 3, 4, 5, 6};
  auto params = MakeGreedyParams(*model_);
  params->search.chunk_size = 2;
  auto request = CreateRequestWithPrompt(engine.engine, *params, prompt);

  size_t execution_count = 0;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    ++execution_count;
    const float expected_input =
        execution_count == 1
            ? 0.0f
        : execution_count <= 3
            ? 1.0f
            : static_cast<float>(execution_count - 1);
    for (const auto& binding : context.fixed_state_bindings) {
      ExpectFixedInputRow(binding, 0, expected_input);
      FillFixedOutputRow(binding, 0, static_cast<float>(execution_count));
    }
    if (execution_count == 1) {
      engine.executor->SetNextFailure(
          ScriptedExecutionFailure::RetryableDuringExecution);
    }
  });

  std::array<EngineEvent, 1> storage;
  EXPECT_EQ(engine.engine->Run(storage), 0u);
  EXPECT_EQ(request->ProcessedSequenceLength(), 2);
  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagRetryable);

  auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(request->ProcessedSequenceLength(), 2);
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).committed_tokens, 2u);
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).state_generation, 1u);
  ASSERT_EQ(engine.cache->Snapshot().requests.size(), 1u);
  EXPECT_EQ(engine.cache->Snapshot().requests[0].used_slots, 2u);

  EXPECT_EQ(engine.engine->Run(storage), 0u);
  EXPECT_EQ(request->ProcessedSequenceLength(), 4);
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(request->ProcessedSequenceLength(), 5);
  fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).committed_tokens, 5u);

  const std::vector<int32_t> continuation{7, 8, 9};
  request->BeginTurn(continuation);
  EXPECT_EQ(engine.engine->Run(storage), 0u);
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(request->ProcessedSequenceLength(), 8);
  fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).committed_tokens, 8u);
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).state_generation,
            execution_count - 1);
}

TEST_F(EngineRunTest, CompositeReservationRequiredMismatchIsFatal) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/4, EosToken(*model_));
  engine.cache->ScriptFixedStateMismatch(
      FixedStateResourcePlan{true, 1, 1, 256}, /*slots=*/{},
      /*staging_bytes=*/0);
  auto request = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));

  const auto failure = RunOne(*engine.engine);
  EXPECT_EQ(failure.request, request);
  EXPECT_EQ(failure.flags,
            EngineEventFlagTurnFinished | EngineEventFlagFailed);
  EXPECT_EQ(failure.error_code, EngineErrorCode::EngineContractFailure);
  EXPECT_THROW(static_cast<void>(RunOne(*engine.engine)), EngineStepError);
}

TEST_F(EngineRunTest, PlanningAllocationFailureDoesNotPoisonEngine) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/4, EosToken(*model_));
  auto request = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  engine.cache->ThrowPlanningBadAllocOnce();

  EXPECT_THROW(static_cast<void>(RunOne(*engine.engine)), std::bad_alloc);
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
  EXPECT_EQ(RunOne(*engine.engine).request, request);
}

TEST_F(EngineRunTest, PlanningConsistencyFailurePoisonsEngine) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/4, EosToken(*model_));
  auto request = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  engine.cache->ThrowPlanningConsistencyOnce();

  const auto failure = RunOne(*engine.engine);
  EXPECT_EQ(failure.request, request);
  EXPECT_EQ(failure.flags,
            EngineEventFlagTurnFinished | EngineEventFlagFailed);
  EXPECT_EQ(failure.error_code, EngineErrorCode::EngineContractFailure);
  EXPECT_THROW(static_cast<void>(RunOne(*engine.engine)), EngineStepError);
}

TEST_F(EngineRunTest, OverselectedTokenBudgetPoisonsEngine) {
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 1;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/4, EosToken(*model_));
  auto first = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  auto second = CreateRequestWithPrompt(engine.engine, *model_, Prompt(11));
  engine.cache->OverselectPlanningOnce();

  std::array<EngineEvent, 2> failures;
  ASSERT_EQ(engine.engine->Run(failures), failures.size());
  EXPECT_EQ(failures[0].request, first);
  EXPECT_EQ(failures[1].request, second);
  for (const auto& failure : failures) {
    EXPECT_EQ(failure.flags,
              EngineEventFlagTurnFinished | EngineEventFlagFailed);
    EXPECT_EQ(failure.error_code, EngineErrorCode::EngineContractFailure);
  }
  EXPECT_THROW(static_cast<void>(RunOne(*engine.engine)), EngineStepError);
}

TEST_F(EngineRunTest, CommitPreparationFailureRestoresRequestStateBeforeFatal) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/4, /*forced_token=*/7);
  auto request = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  const auto sequence_length = request->CurrentSequenceLength();
  engine.cache->ThrowPrepareFailureOnce();

  const auto failure = RunOne(*engine.engine);
  EXPECT_EQ(failure.request, request);
  EXPECT_EQ(failure.flags,
            EngineEventFlagTurnFinished | EngineEventFlagFailed);
  EXPECT_EQ(failure.error_code, EngineErrorCode::EngineContractFailure);
  EXPECT_EQ(request->CurrentSequenceLength(), sequence_length);
  EXPECT_EQ(request->ProcessedSequenceLength(), 0);
  EXPECT_EQ(engine.cache->AllocatedCount(), 0u);
  EXPECT_THROW(static_cast<void>(RunOne(*engine.engine)), EngineStepError);
}

TEST_F(EngineRunTest, CompositeReservationOverreportedRowsAreFatal) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/4, EosToken(*model_));
  auto request = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  static const char extra_request_storage{};
  engine.cache->ScriptFixedStateMismatch(
      FixedStateResourcePlan{true, 2, 1, 0},
      {
          FixedStateSlotHandle{nullptr, request.get(), 0, 0},
          FixedStateSlotHandle{nullptr, &extra_request_storage, 1, 0},
      },
      /*staging_bytes=*/0);

  const auto failure = RunOne(*engine.engine);
  EXPECT_EQ(failure.request, request);
  EXPECT_EQ(failure.flags,
            EngineEventFlagTurnFinished | EngineEventFlagFailed);
  EXPECT_EQ(failure.error_code, EngineErrorCode::EngineContractFailure);
}

TEST_F(EngineRunTest, CompositeReservationRowOrderMismatchIsFatal) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/4, EosToken(*model_));
  auto request = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  static const char other_storage{};
  engine.cache->ScriptFixedStateMismatch(
      FixedStateResourcePlan{true, 1, 1, 0},
      {FixedStateSlotHandle{nullptr, &other_storage, 0, 0}},
      /*staging_bytes=*/0);

  const auto failure = RunOne(*engine.engine);
  EXPECT_EQ(failure.request, request);
  EXPECT_EQ(failure.flags,
            EngineEventFlagTurnFinished | EngineEventFlagFailed);
  EXPECT_EQ(failure.error_code, EngineErrorCode::EngineContractFailure);
}

TEST_F(EngineRunTest, CompositeReservationNewSlotCountMismatchIsFatal) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/4, EosToken(*model_));
  auto request = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  engine.cache->ScriptFixedStateMismatch(
      FixedStateResourcePlan{true, 1, 1, 0},
      {FixedStateSlotHandle{nullptr, request.get(), 0, 0}},
      /*staging_bytes=*/0,
      /*reservation_new_slot_count=*/0);

  const auto failure = RunOne(*engine.engine);
  EXPECT_EQ(failure.request, request);
  EXPECT_EQ(failure.flags,
            EngineEventFlagTurnFinished | EngineEventFlagFailed);
  EXPECT_EQ(failure.error_code, EngineErrorCode::EngineContractFailure);
}

TEST_F(EngineRunTest, CompositeCapacityBackpressureDefersNewAdmission) {
  model_ = LoadSyntheticCompositeModel();
  model_->config_->engine.dynamic_batching->max_batch_size = 2;
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto first = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  auto second = CreateRequestWithPrompt(engine.engine, *model_, Prompt(20));
  engine.executor->SetExecutionCallback([](ExecutionContext& context) {
    for (size_t row = 0; row < context.fixed_state_slots.size(); ++row) {
      for (const auto& binding : context.fixed_state_bindings) {
        FillFixedOutputRow(binding, row, 1.0f);
      }
    }
  });

  ASSERT_EQ(RunOne(*engine.engine).request, first);
  ASSERT_EQ(RunOne(*engine.engine).request, second);
  auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  ASSERT_EQ(fixed->capacity, 2u);
  EXPECT_EQ(fixed->committed_slots, 2u);
  EXPECT_EQ(fixed->free_slots, 0u);

  auto third = CreateRequestWithPrompt(engine.engine, *model_, Prompt(30));
  size_t observed_new_slots = 999;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    observed_new_slots = context.plan->fixed_state.new_slot_count;
    for (size_t row = 0; row < context.fixed_state_slots.size(); ++row) {
      for (const auto& binding : context.fixed_state_bindings) {
        FillFixedOutputRow(binding, row, 2.0f);
      }
    }
  });
  ASSERT_EQ(RunOne(*engine.engine).request, first);
  EXPECT_EQ(observed_new_slots, 0u);
  fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(fixed->committed_slots, 2u);
  EXPECT_EQ(fixed->free_slots, 0u);
  EXPECT_FALSE(engine.cache->IsResident(third));
  EXPECT_EQ(third->status_, RequestStatus::Assigned);
}

TEST_F(EngineRunTest, CompositeRemovalReleasesBothAndIsolatesSibling) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto first = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  auto second = CreateRequestWithPrompt(engine.engine, *model_, Prompt(20));
  engine.executor->SetExecutionCallback([](ExecutionContext& context) {
    for (size_t row = 0; row < context.fixed_state_slots.size(); ++row) {
      for (const auto& binding : context.fixed_state_bindings) {
        FillFixedOutputRow(binding, row, 10.0f + static_cast<float>(row));
      }
    }
  });

  ASSERT_EQ(RunOne(*engine.engine).request, first);
  auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  ASSERT_EQ(fixed->committed_slots, 2u);
  const auto second_generation =
      FixedSlotFor(*fixed, second.get()).state_generation;

  first->Close();
  EXPECT_FALSE(engine.cache->IsResident(first));

  fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(fixed->committed_slots, 1u);
  EXPECT_EQ(fixed->free_slots, fixed->capacity - 1);
  EXPECT_EQ(FixedSlotFor(*fixed, second.get()).state_generation,
            second_generation);
  ASSERT_EQ(engine.cache->Snapshot().requests.size(), 1u);
  EXPECT_EQ(engine.cache->Snapshot().requests[0].request_id, second.get());
  EXPECT_TRUE(ValidateCompositeStateInvariants(
                  engine.cache->Snapshot(), *fixed, {second->Snapshot()})
                  .empty());
}

TEST_F(EngineRunTest, CompositeStagedOutputInvisibleUntilPublish) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto request = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  float committed_value = 0.0f;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    for (const auto& binding : context.fixed_state_bindings) {
      ExpectFixedInputRow(binding, 0, committed_value);
      FillFixedOutputRow(binding, 0, committed_value + 1.0f);
    }
    committed_value += 1.0f;
  });

  EXPECT_EQ(RunOne(*engine.engine).request, request);
  auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).state_generation, 1u);
  EXPECT_EQ(RunOne(*engine.engine).request, request);
  fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(FixedSlotFor(*fixed, request.get()).state_generation, 2u);
}

TEST_F(EngineRunTest, CompositeAggregateAdmissionCommitsEveryNewTable) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  std::vector<std::shared_ptr<Request>> requests;
  for (int i = 0; i < 3; ++i) {
    requests.push_back(CreateRequestWithPrompt(
        engine.engine, *model_, Prompt(10 * (i + 1))));
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

  EXPECT_EQ(RunOne(*engine.engine).request, requests[0]);
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
      ValidateCompositeStateInvariants(
          engine.cache->Snapshot(), *fixed, snapshots)
          .empty());
}

TEST_F(EngineRunTest, CompositeCompletionRemovalFreesSlotForReadmission) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, EosToken(*model_));
  auto first = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  engine.executor->SetExecutionCallback([](ExecutionContext& context) {
    for (const auto& binding : context.fixed_state_bindings) {
      FillFixedOutputRow(binding, 0, 31.0f);
    }
  });

  ASSERT_EQ(RunOne(*engine.engine).request, first);
  ASSERT_TRUE(first->IsTurnComplete());
  const auto completed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(completed.has_value());
  const auto released_slot = FixedSlotFor(*completed, first.get()).slot;
  const auto first_generation =
      FixedSlotFor(*completed, first.get()).generation;

  first->Close();
  auto after_removal = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(after_removal.has_value());
  EXPECT_EQ(after_removal->committed_slots, 0u);

  auto second = CreateRequestWithPrompt(engine.engine, *model_, Prompt(20));
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    ASSERT_EQ(context.fixed_state_slots.size(), 1u);
    EXPECT_EQ(context.fixed_state_slots[0].slot, released_slot);
    for (const auto& binding : context.fixed_state_bindings) {
      ExpectFixedInputRow(binding, 0, 0.0f);
      FillFixedOutputRow(binding, 0, 41.0f);
    }
  });

  EXPECT_EQ(RunOne(*engine.engine).request, second);
  const auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  EXPECT_EQ(fixed->committed_slots, 1u);
  EXPECT_EQ(FixedSlotFor(*fixed, second.get()).slot, released_slot);
  EXPECT_GT(FixedSlotFor(*fixed, second.get()).generation, first_generation);
  ASSERT_EQ(engine.cache->Snapshot().requests.size(), 1u);
  EXPECT_EQ(engine.cache->Snapshot().requests[0].request_id, second.get());
}

TEST_F(EngineRunTest, CompositeOrdersResidentRowsByFixedSlotButEventsBySchedulerRank) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/5);
  auto first = CreateRequestWithPrompt(engine.engine, *model_, Prompt(10));
  auto second = CreateRequestWithPrompt(engine.engine, *model_, Prompt(20));
  engine.executor->SetExecutionCallback([](ExecutionContext& context) {
    for (size_t row = 0; row < context.fixed_state_slots.size(); ++row) {
      for (const auto& binding : context.fixed_state_bindings) {
        FillFixedOutputRow(binding, row, 10.0f + static_cast<float>(row));
      }
    }
  });
  EXPECT_EQ(RunOne(*engine.engine).request, first);
  EXPECT_EQ(RunOne(*engine.engine).request, second);

  auto fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  ASSERT_EQ(FixedSlotFor(*fixed, first.get()).slot, 0u);
  ASSERT_EQ(FixedSlotFor(*fixed, second.get()).slot, 1u);
  first->Close();

  auto replacement =
      CreateRequestWithPrompt(engine.engine, *model_, Prompt(30));
  engine.executor->SetExecutionCallback([](ExecutionContext& context) {
    for (size_t row = 0; row < context.fixed_state_slots.size(); ++row) {
      for (const auto& binding : context.fixed_state_bindings) {
        FillFixedOutputRow(binding, row, 20.0f + static_cast<float>(row));
      }
    }
  });
  EXPECT_EQ(RunOne(*engine.engine).request, second);
  EXPECT_EQ(RunOne(*engine.engine).request, replacement);

  fixed = engine.cache->FixedStateSnapshot();
  ASSERT_TRUE(fixed.has_value());
  ASSERT_EQ(FixedSlotFor(*fixed, replacement.get()).slot, 0u);
  ASSERT_EQ(FixedSlotFor(*fixed, second.get()).slot, 1u);

  engine.executor->SetExecutionCallback(
      [&](ExecutionContext& context) {
        ASSERT_EQ(context.plan->fixed_state.new_slot_count, 0u);
        ASSERT_EQ(context.plan->requests.size(), 2u);
        EXPECT_EQ(context.plan->requests[0].request_id, replacement.get());
        EXPECT_EQ(context.plan->requests[1].request_id, second.get());
        EXPECT_EQ(context.fixed_state_slots[0].slot, 0u);
        EXPECT_EQ(context.fixed_state_slots[1].slot, 1u);
        for (const auto& binding : context.fixed_state_bindings) {
          FillFixedOutputRow(binding, 0, 30.0f);
          FillFixedOutputRow(binding, 1, 31.0f);
        }
      });
  // Physical execution is fixed-slot ordered, but events retain the scheduler's decode order.
  EXPECT_EQ(RunOne(*engine.engine).request, second);
  EXPECT_EQ(RunOne(*engine.engine).request, replacement);
}

}  // namespace
}  // namespace test
}  // namespace Generators
