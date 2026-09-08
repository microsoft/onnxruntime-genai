// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Lifecycle tests for the engine Request state machine. Because a tiny real
// CPU fixture model is available, these tests drive genuine Request objects (rather than a mock
// Search) and pin the transition policy: which mutations each status permits, and how
// create/begin/schedule/continue/close move a request between Unassigned, Assigned, Active,
// TurnComplete, and Closed.

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_helpers.h"
#include "engine_test_doubles.h"
#include "engine/request_status.h"
#include "models/preprocessing/genai_tokenizer.h"

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

DeviceSpan<float> LogitsFavoringToken(Model& model, int32_t preferred_token,
                                      int32_t fallback_token) {
  auto logits = LogitsForToken(model, preferred_token);
  logits.CpuSpan()[fallback_token] = 50.0f;
  logits.CopyCpuToDevice();
  return logits;
}

struct FailingContinuationControl {
  bool fail_append{};
  bool fail_restore{};
};

class FailingContinuationSearch final : public GreedySearch_Cpu {
 public:
  FailingContinuationSearch(
      const GeneratorParams& params,
      std::shared_ptr<FailingContinuationControl> control)
      : GreedySearch_Cpu(params), control_{std::move(control)} {}

  void AppendTokens(DeviceSpan<int32_t>& tokens) override {
    if (control_->fail_append) {
      throw std::runtime_error("Injected continuation append failure.");
    }
    GreedySearch_Cpu::AppendTokens(tokens);
  }

 protected:
  void RestoreStateForTransactionImpl() override {
    if (control_->fail_restore) {
      throw std::runtime_error("Injected continuation restore failure.");
    }
    GreedySearch_Cpu::RestoreStateForTransactionImpl();
  }

 private:
  std::shared_ptr<FailingContinuationControl> control_;
};

class NoOpPositionInputs final : public PositionInputs {
 public:
  void Add() override {}
  void Update(DeviceSpan<int32_t> /*next_tokens*/, int /*total_length*/, int /*new_length*/) override {}
  void RewindTo(size_t /*index*/) override {}
};

class FailingContinuationDevice final : public DeviceInterface {
 public:
  FailingContinuationDevice(
      DeviceInterface& inner,
      std::shared_ptr<FailingContinuationControl> control)
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
    return std::make_unique<FailingContinuationSearch>(params, control_);
  }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return inner_.CreateBeam(params);
  }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) override {
    ++kv_cache_creation_count_;
    return {};
  }
  std::unique_ptr<PositionInputs> CreatePositionInputs(State& /*state*/,
                                                       DeviceSpan<int32_t> /*sequence_lengths*/,
                                                       const std::string& /*attention_mask_name*/) override {
    return std::make_unique<NoOpPositionInputs>();
  }
  size_t KvCacheCreationCount() const noexcept { return kv_cache_creation_count_; }
  void Synchronize() override { inner_.Synchronize(); }

 private:
  DeviceInterface& inner_;
  std::shared_ptr<FailingContinuationControl> control_;
  size_t kv_cache_creation_count_{};
};

class FailingAllocationDevice final : public DeviceInterface {
 public:
  explicit FailingAllocationDevice(DeviceInterface& inner) : inner_{inner} {}

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
    if (fail_allocation_) {
      throw std::runtime_error(
          "Injected request preparation allocation failure.");
    }
    return inner_.AllocateBase(size);
  }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* memory, size_t size) override {
    return inner_.WrapMemoryBase(memory, size);
  }
  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    return inner_.CreateGreedy(params);
  }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return inner_.CreateBeam(params);
  }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) override {
    return inner_.CreateKeyValueCache(state);
  }
  void Synchronize() override { inner_.Synchronize(); }

  void SetFailAllocation(bool fail) { fail_allocation_ = fail; }

 private:
  DeviceInterface& inner_;
  bool fail_allocation_{true};
};

class FailingAdmissionScheduler final : public DynamicBatchScheduler {
 public:
  FailingAdmissionScheduler(std::shared_ptr<Model> model,
                            std::shared_ptr<CacheManager> cache_manager)
      : DynamicBatchScheduler(std::move(model), std::move(cache_manager)) {}

  void AddRequest(std::shared_ptr<Request> request) override {
    if (fail_preparation_) {
      throw std::runtime_error(
          "Injected scheduler admission preparation failure.");
    }
    DynamicBatchScheduler::AddRequest(std::move(request));
  }

  void SetFailPreparation(bool fail) { fail_preparation_ = fail; }

 private:
  bool fail_preparation_{true};
};

static_assert(noexcept(std::declval<Request&>().CommitStep(
    std::declval<const RequestStepPlan&>(),
    std::declval<const RequestStepResult&>())));

DeviceSpan<float> SamplingLogits(Model& model) {
  auto logits = model.p_device_inputs_->Allocate<float>(
      static_cast<size_t>(model.config_->model.vocab_size));
  auto cpu_logits = logits.CpuSpan();
  std::fill(cpu_logits.begin(), cpu_logits.end(), -100.0f);
  cpu_logits[2] = 1.0f;
  cpu_logits[3] = 1.0f;
  cpu_logits[4] = 1.0f;
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
    return CreateEngineRequest(engine_.engine);
  }

  std::shared_ptr<Model> model_;
  DoublesEngine engine_;
};

TEST(ContinuousDecodingDeviceSupportTest, MatchesKvCacheCapabilityContract) {
  EXPECT_TRUE(SupportsContinuousDecoding(DeviceType::CPU));
  EXPECT_TRUE(SupportsContinuousDecoding(DeviceType::CUDA));
  EXPECT_FALSE(SupportsContinuousDecoding(DeviceType::DML));
  EXPECT_FALSE(SupportsContinuousDecoding(DeviceType::QnnHtp));
}

TEST(ConventionalKvCacheTest, CreationDispatchesThroughSelectedDevice) {
  auto model = LoadDummyDecoderModel();
  auto control = std::make_shared<FailingContinuationControl>();
  auto* const original_cache_device = model->p_device_kvcache_;
  FailingContinuationDevice device{*original_cache_device, control};
  model->p_device_kvcache_ = &device;
  struct CacheDeviceRestorer {
    Model& model;
    DeviceInterface* original;
    ~CacheDeviceRestorer() { model.p_device_kvcache_ = original; }
  } restore{*model, original_cache_device};

  auto params = CreateGeneratorParams(*model);
  class TestState final : public State {
   public:
    TestState(const GeneratorParams& params, const Model& model)
        : State{params, model} {}

    DeviceSpan<float> Run(int /*total_length*/, DeviceSpan<int32_t>& /*next_tokens*/, DeviceSpan<int32_t> /*next_indices*/) override {
      return {};
    }
  } state{*params, *model};
  static_cast<void>(model->p_device_kvcache_->CreateKeyValueCache(state));
  EXPECT_EQ(device.KvCacheCreationCount(), 1u);
}

TEST_F(RequestLifecycleTest, InvalidEosTokenIsRejected) {
  for (const int invalid_eos_token_id : {-1, 5}) {
    const std::string overlay =
        R"({ "model": { "vocab_size": 5, "eos_token_id": )" +
        std::to_string(invalid_eos_token_id) + " } }";
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder", nullptr, overlay);
    auto model = CreateModel(GetOrtEnv(), std::move(config));

    try {
      [[maybe_unused]] auto request = std::make_shared<Request>(
          *model, static_cast<size_t>(model->config_->search.max_length));
      FAIL() << "Expected invalid eos_token_id to be rejected";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find("eos_token_id"), std::string::npos);
    }
  }
}

TEST_F(RequestLifecycleTest, InvalidVocabSizeIsRejected) {
  for (const int invalid_vocab_size : {-1, 0}) {
    const std::string overlay =
        R"({ "model": { "vocab_size": )" + std::to_string(invalid_vocab_size) + " } }";
    auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder", nullptr, overlay);
    auto model = CreateModel(GetOrtEnv(), std::move(config));

    try {
      [[maybe_unused]] auto request = std::make_shared<Request>(
          *model, static_cast<size_t>(model->config_->search.max_length));
      FAIL() << "Expected invalid vocab_size to be rejected";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find("vocab_size must be 1 or greater"), std::string::npos);
    }
  }
}

// A turn must carry at least one input token.
TEST_F(RequestLifecycleTest, EmptyBeginTurnIsRejected) {
  auto request = NewRequest();
  EXPECT_THROW(request->BeginTurn({}), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Unassigned);
}

TEST_F(RequestLifecycleTest,
       DevicePreparationFailureLeavesFirstTurnUnassignedAndRetryAppendsPromptOnce) {
  const auto prompt = Prompt();
  FailingAllocationDevice failing_device{*model_->p_device_scoring_};
  ScopedScoringDevice scoped_device{*model_, failing_device};
  failing_device.SetFailAllocation(false);
  auto request = CreateEngineRequest(engine_.engine);

  failing_device.SetFailAllocation(true);
  EXPECT_THROW(request->BeginTurn(prompt), std::runtime_error);
  EXPECT_EQ(request->Status(), RequestStatus::Unassigned);
  EXPECT_EQ(request->CurrentSequenceLength(), 0);
  EXPECT_FALSE(engine_.engine->HasPendingRequests());
  EXPECT_EQ(engine_.cache->AllocatedCount(), 0u);

  failing_device.SetFailAllocation(false);
  EXPECT_EQ(request->BeginTurn(prompt), 1u);
  EXPECT_EQ(request->Status(), RequestStatus::Assigned);
  EXPECT_EQ(request->CurrentSequenceLength(),
            static_cast<int64_t>(prompt.size()));
  EXPECT_TRUE(engine_.engine->HasPendingRequests());
  EXPECT_EQ(engine_.cache->AllocatedCount(), 0u);

  request->Close();
}

TEST_F(RequestLifecycleTest,
       SchedulerPreparationFailureLeavesFirstTurnUnassignedAndRetryAppendsPromptOnce) {
  const auto prompt = Prompt();
  auto cache =
      std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  auto scheduler =
      std::make_unique<FailingAdmissionScheduler>(model_, cache);
  auto* scheduler_observer = scheduler.get();
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, EosToken(*model_));
  EngineDependencies dependencies{
      cache, std::move(scheduler), std::move(executor)};
  auto engine =
      std::make_shared<Engine>(model_, std::move(dependencies));
  auto request = CreateEngineRequest(engine);

  EXPECT_THROW(request->BeginTurn(prompt), std::runtime_error);
  EXPECT_EQ(request->Status(), RequestStatus::Unassigned);
  EXPECT_EQ(request->CurrentSequenceLength(), 0);
  EXPECT_FALSE(engine->HasPendingRequests());
  EXPECT_EQ(cache->AllocatedCount(), 0u);

  scheduler_observer->SetFailPreparation(false);
  EXPECT_EQ(request->BeginTurn(prompt), 1u);
  EXPECT_EQ(request->Status(), RequestStatus::Assigned);
  EXPECT_EQ(request->CurrentSequenceLength(),
            static_cast<int64_t>(prompt.size()));
  EXPECT_TRUE(engine->HasPendingRequests());
  EXPECT_EQ(cache->AllocatedCount(), 0u);

  request->Close();
}

TEST_F(RequestLifecycleTest, ZeroGeneratedTokenBudgetIsRejectedWithoutMutation) {
  auto request = NewRequest();
  const auto prompt = Prompt();

  EXPECT_THROW(
      request->BeginTurn(
          prompt, std::optional<size_t>{0}),
      std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Unassigned);

  EXPECT_NO_THROW(request->BeginTurn(
      prompt, std::optional<size_t>{1}));
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
}

TEST_F(RequestLifecycleTest, NewRequestIsUnassignedUntilBeginTurn) {
  auto request = NewRequest();

  EXPECT_EQ(request->status_, RequestStatus::Unassigned);
  EXPECT_EQ(request->CurrentTurnId(), 0u);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::None);
  EXPECT_FALSE(engine_.engine->HasPendingRequests());
}

TEST_F(RequestLifecycleTest, CancelRejectsRequestsWithoutAnActiveTurn) {
  auto request = NewRequest();
  EXPECT_THROW(request->Cancel(0), std::runtime_error);

  request->Close();
  EXPECT_THROW(request->Cancel(0), std::runtime_error);
}

TEST_F(RequestLifecycleTest, CancelQueuedTurnPublishesTerminalReadyAndCanContinue) {
  auto request = NewRequest();
  const auto prompt = Prompt();
  const auto turn_id = request->BeginTurn(prompt);
  ASSERT_EQ(turn_id, 1u);

  EXPECT_TRUE(request->Cancel(turn_id));
  EXPECT_FALSE(request->Cancel(turn_id));
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::Canceled);
  EXPECT_EQ(request->CurrentSequenceLength(),
            static_cast<int64_t>(prompt.size()));
  EXPECT_TRUE(engine_.engine->HasPendingRequests());

  const std::vector<int32_t> continuation{5};
  EXPECT_THROW(request->BeginTurn(continuation), std::runtime_error);
  EXPECT_EQ(RunOne(*engine_.engine).request, request);
  EXPECT_FALSE(engine_.engine->HasPendingRequests());

  EXPECT_EQ(request->BeginTurn(continuation), 2u);
  EXPECT_EQ(request->CurrentTurnId(), 2u);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::None);
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_EQ(request->CurrentSequenceLength(),
            static_cast<int64_t>(prompt.size() + continuation.size()));
}

TEST_F(RequestLifecycleTest, CancelActiveTurnPreservesResidentState) {
  auto request = CreateRequestWithPrompt(
      engine_.engine, Prompt());
  engine_.cache->Allocate({request});
  request->Schedule();
  ASSERT_EQ(request->status_, RequestStatus::Active);

  EXPECT_TRUE(request->Cancel(request->CurrentTurnId()));
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::Canceled);
  EXPECT_EQ(RunOne(*engine_.engine).request, request);

  request->BeginTurn(std::vector<int32_t>{5});
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_EQ(request->CurrentTurnId(), 2u);
}

TEST_F(RequestLifecycleTest, CancelCompletedTurnPreservesOriginalFinishReason) {
  auto request = NewRequest();
  engine_.executor->SetForcedToken(EosToken(*model_));
  request->BeginTurn(Prompt());
  ASSERT_EQ(RunOne(*engine_.engine).request, request);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);
  ASSERT_EQ(request->FinishReason(), GenerationFinishReason::EosToken);

  EXPECT_FALSE(request->Cancel(request->CurrentTurnId()));
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::EosToken);
  EXPECT_FALSE(engine_.engine->HasPendingRequests());
}

// Prefill chunking is model/Engine configuration, not per-Request policy. A Request takes the
// model's chunk size, and a static-batching Engine -- which rebuilds its contiguous cache from the
// whole sequence every step and so cannot resume a half-written prompt -- rejects a chunking model
// when it is constructed, long before any Request is admitted.
TEST_F(RequestLifecycleTest, ModelConfiguredChunkSizeAppliesToEveryRequest) {
  auto model = LoadDummyDecoderModelWithChunking(/*chunk_size=*/2);
  auto engine = MakeDoublesEngine(model, /*capacity=*/8, EosToken(*model));
  auto request = CreateEngineRequest(engine.engine);

  ASSERT_TRUE(request->PrefillChunkSize().has_value());
  EXPECT_EQ(*request->PrefillChunkSize(), 2u);
}

TEST_F(RequestLifecycleTest, StaticEngineRejectsModelConfiguredChunking) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  config->engine.dynamic_batching.reset();
  config->search.chunk_size = 2;
  auto model = CreateModel(GetOrtEnv(), std::move(config));

  try {
    static_cast<void>(Engine::CreateDependencies(model));
    FAIL() << "Expected model-configured chunking to be rejected for static batching.";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("chunk_size requires dynamic batching"),
              std::string::npos);
  }
}

// An Engine assembled from injected dependencies never runs Engine::CreateDependencies, so the
// static scheduler keeps its own admission guard. Admission is rejected before the batch is
// touched, leaving the Request retryable on a dynamic Engine.
TEST_F(RequestLifecycleTest, StaticSchedulerRejectsChunkedAdmissionWithInjectedDependencies) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  config->engine.dynamic_batching.reset();
  config->search.chunk_size = 2;
  auto model = CreateModel(GetOrtEnv(), std::move(config));

  auto cache = std::make_shared<RecordingCacheManager>(
      model, /*capacity=*/8, /*trace=*/nullptr, /*supports_dynamic_batching=*/false);
  EngineDependencies dependencies{
      cache, Scheduler::Create(model, cache),
      std::make_unique<RecordingModelExecutor>(model, cache, EosToken(*model))};
  auto engine = std::make_shared<Engine>(model, std::move(dependencies));

  auto request = CreateEngineRequest(engine);
  try {
    request->BeginTurn(Prompt());
    FAIL() << "Expected the static scheduler to reject a chunking Request.";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("chunk_size requires dynamic batching"),
              std::string::npos);
  }
  EXPECT_EQ(request->Status(), RequestStatus::Unassigned);
  EXPECT_EQ(request->CurrentSequenceLength(), 0);
  EXPECT_FALSE(engine->HasPendingRequests());
}

// Input that would exceed the model's max length is rejected before mutation, so a subsequent valid
// first turn reflects only the accepted tokens.
TEST_F(RequestLifecycleTest, BeginTurnBeyondContextIsRejectedBeforeMutation) {
  auto request = NewRequest();
  const size_t over_capacity = static_cast<size_t>(model_->config_->model.context_length) + 1;
  std::vector<int32_t> too_many(over_capacity, 2);
  EXPECT_THROW(request->BeginTurn(too_many), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Unassigned);

  auto prompt = Prompt();
  request->BeginTurn(prompt);
  EXPECT_EQ(request->CurrentSequenceLength(), static_cast<int64_t>(prompt.size()));
}

// A first turn can begin only while Unassigned; beginning another while Assigned throws and leaves
// the request untouched.
TEST_F(RequestLifecycleTest, BeginTurnIsRejectedWhenAlreadyAssigned) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, prompt);
  ASSERT_EQ(request->status_, RequestStatus::Assigned);
  const int64_t length_before = request->CurrentSequenceLength();

  EXPECT_THROW(request->BeginTurn(std::vector<int32_t>{5}), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_EQ(request->CurrentSequenceLength(), length_before);
}

// Scheduling requires a prior first turn; an unassigned request cannot be scheduled and stays
// Unassigned.
TEST_F(RequestLifecycleTest, ScheduleIsRejectedBeforeBeginTurn) {
  auto request = NewRequest();
  ASSERT_EQ(request->status_, RequestStatus::Unassigned);

  EXPECT_THROW(request->Schedule(), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Unassigned);
}

// An assigned, non-empty request schedules cleanly and moves to Active.
TEST_F(RequestLifecycleTest, ScheduleFromAssignedMovesToActive) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, prompt);
  request->Schedule();
  EXPECT_EQ(request->status_, RequestStatus::Active);
}

// While a request is active its token stream is owned by the engine, so BeginTurn is
// rejected without mutating the request.
TEST_F(RequestLifecycleTest, BeginTurnIsRejectedWhileActive) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, prompt);
  request->Schedule();
  ASSERT_EQ(request->status_, RequestStatus::Active);
  const int64_t length_before = request->CurrentSequenceLength();

  std::vector<int32_t> more{5, 6};
  EXPECT_THROW(request->BeginTurn(more), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Active);
  EXPECT_EQ(request->CurrentSequenceLength(), length_before);
}

// After a turn completes, BeginTurn appends another input fragment and queues the resident request.
TEST_F(RequestLifecycleTest, BeginTurnAfterTurnCompleteQueuesNextTurn) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, prompt);
  const int64_t assigned_length = static_cast<int64_t>(prompt.size());
  EXPECT_EQ(request->CurrentTurnId(), 1u);

  RunOne(*engine_.engine);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);
  std::vector<int32_t> more{5, 6};
  request->BeginTurn(more);

  EXPECT_EQ(request->CurrentTurnId(), 2u);
  EXPECT_EQ(request->CurrentSequenceLength(), assigned_length + static_cast<int64_t>(more.size()));
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_FALSE(request->IsTurnComplete());
  EXPECT_EQ(RunOne(*engine_.engine).request, request);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_TRUE(request->IsTurnComplete());
}

TEST_F(RequestLifecycleTest, ContinuationBeyondContextIsRejectedBeforeMutation) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, prompt);
  RunOne(*engine_.engine);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);
  const auto before = request->Snapshot();

  const size_t remaining =
      request->MaxSessionTokens() -
      static_cast<size_t>(request->CurrentSequenceLength());
  std::vector<int32_t> too_many(remaining, 5);
  EXPECT_THROW(request->BeginTurn(too_many), std::runtime_error);

  const auto after = request->Snapshot();
  EXPECT_EQ(after.status, before.status);
  EXPECT_EQ(after.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(after.processed_sequence_length, before.processed_sequence_length);
}

TEST_F(RequestLifecycleTest, FailedContinuationAppendPreservesCompletedTurnState) {
  auto control = std::make_shared<FailingContinuationControl>();
  FailingContinuationDevice device{*model_->p_device_scoring_, control};
  ScopedScoringDevice scoped_device{*model_, device};
  auto request = CreateEngineRequest(engine_.engine);
  engine_.executor->SetForcedToken(/*token=*/5);
  request->BeginTurn(Prompt(), std::optional<size_t>{1});
  ASSERT_EQ(RunOne(*engine_.engine).request, request);
  ASSERT_TRUE(request->IsTurnComplete());
  const auto before = request->Snapshot();

  control->fail_append = true;
  EXPECT_THROW(
      request->BeginTurn(
          std::vector<int32_t>{6},
          std::optional<size_t>{2}),
      std::runtime_error);

  EXPECT_EQ(request->Snapshot().status, before.status);
  EXPECT_EQ(request->Snapshot().current_sequence_length,
            before.current_sequence_length);

  control->fail_append = false;
  EXPECT_NO_THROW(request->BeginTurn(
      std::vector<int32_t>{6},
      std::optional<size_t>{2}));
}

TEST_F(RequestLifecycleTest, FailedContinuationRestoreClosesRequestAndPoisonsEngine) {
  auto control = std::make_shared<FailingContinuationControl>();
  FailingContinuationDevice device{*model_->p_device_scoring_, control};
  ScopedScoringDevice scoped_device{*model_, device};
  auto request = CreateEngineRequest(engine_.engine);
  request->BeginTurn(Prompt());
  ASSERT_EQ(RunOne(*engine_.engine).request, request);
  ASSERT_TRUE(request->IsTurnComplete());
  ASSERT_EQ(engine_.cache->AllocatedCount(), 1u);

  control->fail_append = true;
  control->fail_restore = true;
  try {
    request->BeginTurn(std::vector<int32_t>{5});
    FAIL() << "Expected continuation restore failure.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::FatalExecutionFailure);
  }

  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(engine_.cache->AllocatedCount(), 0u);
  EXPECT_THROW(request->BeginTurn(std::vector<int32_t>{6}), std::runtime_error);
  const auto failure = RunOne(*engine_.engine);
  EXPECT_EQ(failure.request, nullptr);
  EXPECT_EQ(failure.flags, EngineEventFlagFailed);
  EXPECT_EQ(failure.finish_reason, GenerationFinishReason::Failed);
  EXPECT_EQ(failure.error_code, EngineErrorCode::EngineExecutionFailure);
  EXPECT_THROW(static_cast<void>(RunOne(*engine_.engine)), EngineStepError);
}

TEST_F(RequestLifecycleTest,
       FailedContinuationCloseDetachesBeforeFatalPublication) {
  auto cache = std::make_shared<RecordingCacheManager>(
      model_, /*capacity=*/1);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(
      model_, cache, EosToken(*model_));
  EngineDependencies dependencies{
      cache,
      std::move(scheduler),
      std::move(executor)};
  auto engine = std::make_shared<Engine>(
      model_, std::move(dependencies));

  auto control = std::make_shared<FailingContinuationControl>();
  FailingContinuationDevice device{*model_->p_device_scoring_, control};
  ScopedScoringDevice scoped_device{*model_, device};
  auto request = CreateEngineRequest(engine);
  request->BeginTurn(Prompt());
  ASSERT_EQ(RunOne(*engine).request, request);
  ASSERT_TRUE(request->IsTurnComplete());
  ASSERT_EQ(cache->AllocatedCount(), 1u);

  auto drained_request = CreateEngineRequest(engine);
  auto pending_request = CreateEngineRequest(engine);
  const auto drained_turn = drained_request->BeginTurn(Prompt());
  const auto pending_turn = pending_request->BeginTurn(Prompt());
  ASSERT_TRUE(drained_request->Cancel(drained_turn));
  ASSERT_TRUE(pending_request->Cancel(pending_turn));
  ASSERT_EQ(RunOne(*engine).request, drained_request);

  control->fail_append = true;
  control->fail_restore = true;
  cache->ThrowDeallocateInvariantFailureOnce();
  try {
    request->BeginTurn(std::vector<int32_t>{5});
    FAIL() << "Expected continuation restore failure.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(
        error.Outcome().kind,
        StepOutcomeKind::FatalExecutionFailure);
    EXPECT_NE(
        std::string_view{error.what()}.find(
            "Closing the poisoned request also failed."),
        std::string_view::npos);
  }

  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(cache->AllocatedCount(), 0u);
  EXPECT_EQ(cache->deallocate_calls, 2);
  EXPECT_THROW(
      request->BeginTurn(std::vector<int32_t>{6}),
      std::runtime_error);
  std::array<EngineEvent, 2> failures;
  ASSERT_EQ(engine->Run(failures), failures.size());
  EXPECT_EQ(failures[0].request, pending_request);
  EXPECT_EQ(failures[0].flags, EngineEventFlagTurnFinished);
  EXPECT_EQ(
      failures[0].finish_reason,
      GenerationFinishReason::Canceled);
  EXPECT_EQ(failures[1].request, nullptr);
  EXPECT_EQ(failures[1].flags, EngineEventFlagFailed);
  EXPECT_EQ(
      failures[1].error_code,
      EngineErrorCode::EngineExecutionFailure);
  EXPECT_THROW(static_cast<void>(RunOne(*engine)), EngineStepError);
}

// Engine::BeginTurn builds the new turn's complete stop controller before touching the Request or
// its MTP shadow at all, and Request::CommitTurnAdmission is the only place that installs it as
// the live stop_controller_. A failure anywhere in between (here, the continuation append inside
// PrepareTurnAdmission, using the same FailingContinuationDevice/Control mechanism as
// FailedContinuationAppendPreservesCompletedTurnState above) must therefore leave the Request's
// previous turn's stop controller, its FinishReason(), and its MatchedStopStringIndex() completely
// untouched, and a subsequent retry must behave exactly like an ordinary BeginTurn.
TEST_F(RequestLifecycleTest, FailedBeginTurnRestoresThePriorTurnsStopController) {
  auto model = LoadSyntheticPagedModel();
  auto engine = MakeDoublesEngine(model, /*capacity=*/8, /*forced_token=*/5);

  auto control = std::make_shared<FailingContinuationControl>();
  FailingContinuationDevice device{*model->p_device_scoring_, control};
  ScopedScoringDevice scoped_device{*model, device};
  auto request = CreateEngineRequest(engine.engine);

  TurnOptions first_turn_options;
  first_turn_options.stop_strings = {"STOP"};
  request->BeginTurn(Prompt(), first_turn_options);

  // Step 1: "ST" (token 5), no match yet. Step 2: "OP" (token 6) completes "STOP".
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  engine.executor->SetForcedToken(6);
  const auto completed = RunOne(*engine.engine);
  EXPECT_EQ(completed.finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(completed.matched_stop_string_index, 0);
  ASSERT_EQ(request->FinishReason(), GenerationFinishReason::StopString);
  ASSERT_EQ(request->MatchedStopStringIndex(), 0);

  // Force the next turn's continuation append to fail. By this point Engine::BeginTurn has
  // already built the next turn's complete replacement stop controller (a different
  // configuration, to make any accidental leak of the new config observable) but
  // Request::CommitTurnAdmission -- the only place that installs it -- never runs.
  control->fail_append = true;
  TurnOptions second_turn_options;
  second_turn_options.stop_strings = {"UNREACHABLE_UNUSED_STOP"};
  try {
    request->BeginTurn(std::array<int32_t, 1>{2}, second_turn_options);
    FAIL() << "Expected continuation append failure";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "Injected continuation append failure.");
  }
  control->fail_append = false;

  // The failed admission attempt must not disturb the previous committed turn's public state.
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::StopString);
  EXPECT_EQ(request->MatchedStopStringIndex(), 0);

  // A retried BeginTurn must behave exactly like an ordinary call: turn state resets to a clean
  // slate and the retry's own stop configuration (not the discarded failed attempt's) takes
  // effect.
  TurnOptions retry_options;
  retry_options.stop_strings = {"STOP"};
  request->BeginTurn(std::array<int32_t, 1>{2}, retry_options);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::None);
  EXPECT_EQ(request->MatchedStopStringIndex(), -1);

  engine.executor->SetForcedToken(5);
  ASSERT_EQ(RunOne(*engine.engine).request, request);
  engine.executor->SetForcedToken(6);
  const auto retried_completed = RunOne(*engine.engine);
  EXPECT_EQ(retried_completed.finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(retried_completed.matched_stop_string_index, 0);
}

// The no-stop case must be just as transactional as the stop-enabled case: a turn that would have
// cleared a previous turn's stop controller must not do so when admission fails.
TEST_F(RequestLifecycleTest, FailedNoStopBeginTurnRestoresThePriorTurnsStopController) {
  auto model = LoadSyntheticPagedModel();
  auto engine = MakeDoublesEngine(model, /*capacity=*/8, /*forced_token=*/5);

  auto control = std::make_shared<FailingContinuationControl>();
  FailingContinuationDevice device{*model->p_device_scoring_, control};
  ScopedScoringDevice scoped_device{*model, device};
  auto request = CreateEngineRequest(engine.engine);

  TurnOptions first_turn_options;
  first_turn_options.stop_strings = {"STOP"};
  request->BeginTurn(Prompt(), first_turn_options);

  ASSERT_EQ(RunOne(*engine.engine).request, request);
  engine.executor->SetForcedToken(6);
  ASSERT_EQ(RunOne(*engine.engine).finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(request->MatchedStopStringIndex(), 0);

  // This attempt has no stop strings at all, so it would clear stop_controller_ on success.
  control->fail_append = true;
  try {
    request->BeginTurn(std::array<int32_t, 1>{2});
    FAIL() << "Expected continuation append failure";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "Injected continuation append failure.");
  }
  control->fail_append = false;

  // The failed attempt must preserve the prior turn's public completion state.
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::StopString);
  EXPECT_EQ(request->MatchedStopStringIndex(), 0);

  // A retried, corrected BeginTurn with no stop strings succeeds normally and does clear it.
  request->BeginTurn(std::array<int32_t, 1>{2});
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::None);
  EXPECT_EQ(request->MatchedStopStringIndex(), -1);
}

TEST_F(RequestLifecycleTest, ContinuationPreservesUnreadOutputAndHidesInputTokens) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, prompt);
  engine_.cache->Allocate({request});
  request->Schedule();

  RequestStepPlan first_plan;
  first_plan.request = request;
  first_plan.request_id = request.get();
  first_plan.sequence_length_before = request->CurrentSequenceLength();
  first_plan.target_cache_slots =
      static_cast<size_t>(first_plan.sequence_length_before);
  constexpr int32_t generated_token = 5;
  auto first_logits = LogitsForToken(*model_, generated_token);
  PrepareRequestStep(model_, first_plan);
  request->SaveStateForTransaction();
  const auto first_result = request->ApplyLogitsForTransaction(first_logits);
  request->CommitStateForTransaction();
  request->CommitStep(first_plan, first_result);

  RequestStepPlan completion_plan;
  completion_plan.request = request;
  completion_plan.request_id = request.get();
  completion_plan.sequence_length_before = request->CurrentSequenceLength();
  completion_plan.target_cache_slots =
      static_cast<size_t>(completion_plan.sequence_length_before);
  auto eos_logits = LogitsForToken(*model_, EosToken(*model_));
  PrepareRequestStep(model_, completion_plan);
  request->SaveStateForTransaction();
  const auto completion_result =
      request->ApplyLogitsForTransaction(eos_logits);
  request->CommitStateForTransaction();
  request->CommitStep(completion_plan, completion_result);
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);

  const std::vector<int32_t> continuation{6, 7};
  request->BeginTurn(continuation);

  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_EQ(first_result.token, generated_token);
}

// Closing a request releases it from the engine and makes it terminal.
TEST_F(RequestLifecycleTest, RequestCloseIsIdempotentAfterClose) {
  auto prompt = Prompt();
  const std::vector<int32_t> more{5};
  auto request = CreateRequestWithPrompt(engine_.engine, prompt);
  ASSERT_EQ(request->status_, RequestStatus::Assigned);

  request->Close();
  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(engine_.cache->deallocate_calls, 1);
  EXPECT_EQ(engine_.cache->AllocatedCount(), 0u);
  EXPECT_THROW(request->BeginTurn(more), std::runtime_error);
  EXPECT_NO_THROW(request->Close());
  EXPECT_EQ(engine_.cache->deallocate_calls, 1);
}

TEST_F(RequestLifecycleTest, CloseRoutesThroughOwningEngineAndIsIdempotent) {
  auto other_engine =
      MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto request =
      CreateRequestWithPrompt(engine_.engine, Prompt());
  ASSERT_EQ(request->status_, RequestStatus::Assigned);

  EXPECT_NO_THROW(request->Close());
  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(engine_.cache->deallocate_calls, 1);
  EXPECT_EQ(other_engine.cache->deallocate_calls, 0);

  EXPECT_NO_THROW(request->Close());
  EXPECT_EQ(engine_.cache->deallocate_calls, 1);
  EXPECT_EQ(other_engine.cache->deallocate_calls, 0);
}

TEST_F(RequestLifecycleTest, CloseIsIdempotentForNewRequest) {
  auto request = NewRequest();
  EXPECT_NO_THROW(request->Close());
  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_NO_THROW(request->Close());
  EXPECT_EQ(engine_.cache->deallocate_calls, 1);
}

TEST_F(RequestLifecycleTest, CloseRemainsIdempotentAfterEngineDestruction) {
  auto local_engine =
      MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(local_engine.engine, prompt);
  local_engine.engine.reset();

  EXPECT_NO_THROW(request->Close());
  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_NO_THROW(request->Close());
}

TEST_F(RequestLifecycleTest, TransactionalLogitsStageUntilCommit) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, prompt);
  const auto before = request->Snapshot();
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.target_cache_slots = static_cast<size_t>(before.current_sequence_length);
  const int32_t next_token = 5;
  auto logits = LogitsForToken(*model_, next_token);

  PrepareRequestStep(model_, plan);
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
  EXPECT_EQ(committed.status, RequestStatus::Active);
  EXPECT_EQ(committed.processed_sequence_length, before.current_sequence_length);
  EXPECT_FALSE(committed.is_prefill);
  EXPECT_EQ(result.token, next_token);
}

TEST_F(RequestLifecycleTest, PerTurnLimitStagesUntilTransactionCommit) {
  auto request = NewRequest();
  const auto prompt = Prompt();
  request->BeginTurn(prompt, std::optional<size_t>{1});
  const auto before = request->Snapshot();
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.target_cache_slots =
      static_cast<size_t>(before.current_sequence_length);
  auto logits = LogitsForToken(*model_, /*token=*/5);

  PrepareRequestStep(model_, plan);
  request->SaveStateForTransaction();
  const auto result = request->ApplyLogitsForTransaction(logits);

  EXPECT_TRUE(result.token_appended);
  EXPECT_TRUE(result.done);
  EXPECT_EQ(request->status_, RequestStatus::Assigned);

  request->CommitStateForTransaction();
  request->CommitStep(plan, result);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(result.token, 5);
}

TEST_F(RequestLifecycleTest, TransactionalLogitsRollbackRestoresSearchState) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, prompt);
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
}

TEST_F(RequestLifecycleTest, TransactionalRollbackRestoresSamplingState) {
  TurnOptions sampled;
  sampled.do_sample = true;
  sampled.top_k = 3;
  sampled.top_p = 1.0f;
  sampled.temperature = 1.0f;
  sampled.seed = 1234u;

  auto request = CreateRequestWithPrompt(engine_.engine, Prompt(), sampled);
  const auto before = request->Snapshot();
  auto logits = SamplingLogits(*model_);

  request->SaveStateForTransaction();
  // The turn's reseed is applied inside the step transaction, after the checkpoint, exactly as
  // ScheduledRequests does before it reaches any RNG consumer.
  request->ApplyPendingSeedForTransaction(
      /*sampler=*/nullptr, /*device_state_checkpointed=*/false);
  const auto first = request->ApplyLogitsForTransaction(logits);
  request->RestoreStateForTransaction();

  EXPECT_EQ(request->Snapshot().current_sequence_length,
            before.current_sequence_length);

  request->SaveStateForTransaction();
  // Rollback left the reseed pending, so the retry reseeds identically and reproduces the token.
  request->ApplyPendingSeedForTransaction(
      /*sampler=*/nullptr, /*device_state_checkpointed=*/false);
  const auto retried = request->ApplyLogitsForTransaction(logits);
  request->RestoreStateForTransaction();

  EXPECT_TRUE(first.token_appended);
  EXPECT_TRUE(retried.token_appended);
  EXPECT_EQ(retried.token, first.token);
}

TEST_F(RequestLifecycleTest, PartialPrefillAdvancesOnlyAtCommit) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, prompt);
  const auto before = request->Snapshot();
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.unprocessed_token_count = 2;
  plan.target_cache_slots = 2;

  PrepareRequestStep(model_, plan);
  request->SaveStateForTransaction();
  request->CommitStateForTransaction();
  request->CommitStep(plan, RequestStepResult{});

  const auto committed = request->Snapshot();
  EXPECT_EQ(committed.status, RequestStatus::Active);
  EXPECT_EQ(committed.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(committed.processed_sequence_length, 2);
  // Two of the three prompt tokens are in the cache, so the request is still prefilling.
  EXPECT_TRUE(committed.is_prefill);
}

TEST_F(RequestLifecycleTest, EosCompletesWithoutAppendingVisibleToken) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, prompt);
  const auto before = request->Snapshot();
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.target_cache_slots = static_cast<size_t>(before.current_sequence_length);
  auto logits = LogitsForToken(*model_, EosToken(*model_));

  PrepareRequestStep(model_, plan);
  request->SaveStateForTransaction();
  const auto result = request->ApplyLogitsForTransaction(logits);
  request->CommitStateForTransaction();
  request->CommitStep(plan, result);

  EXPECT_TRUE(result.done);
  EXPECT_FALSE(result.token_appended);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
}

// A Request is one sequence, whatever the model configuration says: the Engine derives its search
// from the model and forces batch size and beam count to one rather than trusting a caller.
TEST_F(RequestLifecycleTest, RequestForcesSingleSequenceSearch) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  config->search.batch_size = 2;
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  const int32_t forced_token = EosToken(*model) == 5 ? 6 : 5;
  auto engine = MakeDoublesEngine(model, /*capacity=*/8, forced_token);

  auto request = CreateRequestWithPrompt(engine.engine, Prompt());
  EXPECT_EQ(request->CurrentSequenceLength(),
            static_cast<int64_t>(Prompt().size()));

  // One step of a batch-size-2 model configuration still advances this Request by exactly one token
  // on row 0, which is the invariant every row-indexing assumption in Request depends on. A wider
  // batch means "the model was configured for the classic Generator"; the Engine batches Requests
  // instead, so it derives its own single-row search rather than honoring or rejecting that value.
  const auto event = RunOne(*engine.engine);
  ASSERT_EQ(event.request, request);
  EXPECT_EQ(event.token, forced_token);
  EXPECT_EQ(request->TurnGeneratedTokens(), 1u);
  EXPECT_EQ(request->CurrentSequenceLength(),
            static_cast<int64_t>(Prompt().size()) + 1);
}

// Beam search is different from a wider batch: silently decoding one beam would produce output the
// caller never asked for, so the model-configured value is rejected instead of forced. The message
// has to name a route the caller can actually take, so this also takes it.
TEST_F(RequestLifecycleTest, RequestRejectsModelConfiguredBeamSearch) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  config->search.num_beams = 4;
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  auto engine = MakeDoublesEngine(model, /*capacity=*/8, EosToken(*model));

  try {
    static_cast<void>(engine.engine->CreateRequest());
    FAIL() << "Expected a model-configured search.num_beams != 1 to be rejected.";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("search.num_beams"), std::string::npos) << message;
    EXPECT_NE(message.find("overlay"), std::string::npos) << message;
  }

  // Taking the route the message names makes the same model usable, and the Request it mints
  // decodes the single sequence the Engine promises.
  auto cleared_config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  cleared_config->search.num_beams = 4;
  OverlayConfig(*cleared_config, R"({"search": {"num_beams": 1}})");
  auto cleared_model = CreateModel(GetOrtEnv(), std::move(cleared_config));
  const int32_t forced_token = EosToken(*cleared_model) == 5 ? 6 : 5;
  auto cleared_engine =
      MakeDoublesEngine(cleared_model, /*capacity=*/8, forced_token);
  auto request = CreateRequestWithPrompt(cleared_engine.engine, Prompt());
  const auto event = RunOne(*cleared_engine.engine);
  ASSERT_EQ(event.request, request);
  EXPECT_EQ(event.token, forced_token);
  EXPECT_EQ(request->CurrentSequenceLength(),
            static_cast<int64_t>(Prompt().size()) + 1);
}

// A model that still carries the legacy session-absolute floor cannot be served by the Engine,
// whose minimum is per turn. The message has to name a route the caller can actually take, so this
// also takes it: the config overlay clears the value without editing the model directory.
TEST_F(RequestLifecycleTest, RequestRejectsLegacyModelMinLength) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  config->search.min_length = 4;
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  auto engine = MakeDoublesEngine(model, /*capacity=*/8, EosToken(*model));

  try {
    static_cast<void>(engine.engine->CreateRequest());
    FAIL() << "Expected a nonzero model search.min_length to be rejected.";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("search.min_length"), std::string::npos) << message;
    EXPECT_NE(message.find("OgaTurnOptionsSetMinGeneratedTokens"), std::string::npos)
        << message;
    EXPECT_NE(message.find("overlay"), std::string::npos) << message;
  }

  // The overlay the message points at is a real, supported route: applying it before the model is
  // created makes the same Engine admit the same Request.
  auto overlaid_config = CreateConfig(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  overlaid_config->search.min_length = 4;
  OverlayConfig(*overlaid_config, R"({"search": {"min_length": 0}})");
  ASSERT_EQ(overlaid_config->search.min_length, 0);
  auto overlaid_model = CreateModel(GetOrtEnv(), std::move(overlaid_config));
  auto overlaid_engine =
      MakeDoublesEngine(overlaid_model, /*capacity=*/8, EosToken(*overlaid_model));
  auto request = overlaid_engine.engine->CreateRequest();
  EXPECT_EQ(request->BeginTurn(Prompt()), 1u);
}

// Sampling limits are turn policy now, so they are rejected at admission -- before any Request
// mutation -- rather than at Request creation.
TEST_F(RequestLifecycleTest, TurnAdmissionRejectsInvalidSamplingLimits) {
  const auto expect_rejected = [&](const TurnOptions& options,
                                   std::string_view expected_fragment) {
    auto request = CreateEngineRequest(engine_.engine);
    try {
      request->BeginTurn(Prompt(), options);
      FAIL() << "Expected an invalid sampling policy to be rejected.";
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      EXPECT_NE(message.find(expected_fragment), std::string::npos) << message;
    }
    EXPECT_EQ(request->Status(), RequestStatus::Unassigned);
    EXPECT_EQ(request->CurrentSequenceLength(), 0);
    request->Close();
  };

  {
    TurnOptions options;
    options.do_sample = true;
    options.top_p = 1.5f;
    expect_rejected(options, "between 0.0 and 1.0");
  }
  {
    TurnOptions options;
    options.do_sample = true;
    options.top_p = std::numeric_limits<float>::quiet_NaN();
    expect_rejected(options, "top_p");
  }
  {
    TurnOptions options;
    options.do_sample = true;
    options.top_k = -1;
    expect_rejected(options, "top_k (-1)");
  }
  {
    // Explicitly greedy, so a top_k asking for a 40-candidate distribution contradicts it.
    TurnOptions options;
    options.do_sample = false;
    options.top_k = 40;
    expect_rejected(options, "contradict it: top_k");
  }
  {
    // top_k == 1 is itself a valid way to ask for greedy selection, but a nucleus top_p alongside
    // it is not.
    TurnOptions options;
    options.do_sample = true;
    options.top_k = 1;
    options.top_p = 0.5f;
    expect_rejected(options, "contradict it: top_p");
  }
  {
    TurnOptions options;
    options.repetition_penalty = 0.0f;
    expect_rejected(options, "repetition_penalty");
  }
  {
    TurnOptions options;
    options.min_generated_tokens = 5;
    options.max_generated_tokens = 2;
    expect_rejected(options, "must not exceed max_generated_tokens");
  }
}

// A guidance request that names only half of the pair is malformed, and admission rejects it
// before the Request is touched.
TEST_F(RequestLifecycleTest, TurnAdmissionRejectsIncompleteGuidanceConfiguration) {
  for (bool omit_type : {false, true}) {
    auto request = CreateEngineRequest(engine_.engine);
    TurnOptions options;
    options.guidance_type = omit_type ? "" : "regex";
    options.guidance_data = omit_type ? "[0-9]+" : "";

    try {
      request->BeginTurn(Prompt(), options);
      FAIL() << "Expected incomplete guidance configuration to be rejected";
    } catch (const std::runtime_error& error) {
      EXPECT_STREQ(error.what(), "Guidance type and data must be provided together.");
    }
    EXPECT_EQ(request->Status(), RequestStatus::Unassigned);
    EXPECT_EQ(request->CurrentSequenceLength(), 0);
    request->Close();
  }
}

#if USE_GUIDANCE
// A turn-scoped regex grammar. Guidance never carries over, so every guided turn names its own.
TurnOptions GuidedOptions(std::string grammar) {
  TurnOptions options;
  options.guidance_type = "regex";
  options.guidance_data = std::move(grammar);
  return options;
}

TEST_F(RequestLifecycleTest, TerminalGuidanceTakesPrecedenceOverMinimumGeneratedTokens) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto guidance_engine = MakeDoublesEngine(guidance_model, /*capacity=*/8,
                                           EosToken(*guidance_model));
  auto tokenizer = guidance_model->CreateTokenizer();
  const auto guided_tokens = tokenizer->Encode("!");
  ASSERT_EQ(guided_tokens.size(), 1u);

  auto request = CreateEngineRequest(guidance_engine.engine);
  auto options = GuidedOptions("!");
  options.min_generated_tokens = 4;
  options.max_generated_tokens = 8;
  request->BeginTurn(Prompt(), options);

  const auto token = RunOne(*guidance_engine.engine);
  ASSERT_EQ(token.request, request);
  ASSERT_NE(token.flags & EngineEventFlagToken, 0u);
  EXPECT_EQ(token.token, guided_tokens.front());
  EXPECT_FALSE(request->IsTurnComplete());

  const auto terminal = RunOne(*guidance_engine.engine);
  ASSERT_EQ(terminal.request, request);
  EXPECT_EQ(terminal.flags & EngineEventFlagToken, 0u);
  EXPECT_NE(terminal.flags & EngineEventFlagTurnFinished, 0u);
  EXPECT_EQ(terminal.finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(terminal.usage.generated_tokens, 1u);
  EXPECT_TRUE(request->IsTurnComplete());
}

TEST_F(RequestLifecycleTest, ExtendableGuidanceHonorsMinimumGeneratedTokens) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto guidance_engine = MakeDoublesEngine(guidance_model, /*capacity=*/8,
                                           EosToken(*guidance_model));

  auto request = CreateEngineRequest(guidance_engine.engine);
  auto options = GuidedOptions("[0-9]*");
  options.min_generated_tokens = 2;
  options.max_generated_tokens = 8;
  request->BeginTurn(Prompt(), options);

  for (size_t generated = 1; generated <= options.min_generated_tokens; ++generated) {
    const auto token = RunOne(*guidance_engine.engine);
    ASSERT_EQ(token.request, request);
    EXPECT_NE(token.flags & EngineEventFlagToken, 0u);
    EXPECT_EQ(token.flags & EngineEventFlagTurnFinished, 0u);
    EXPECT_EQ(request->TurnGeneratedTokens(), generated);
  }

  const auto terminal = RunOne(*guidance_engine.engine);
  ASSERT_EQ(terminal.request, request);
  EXPECT_EQ(terminal.flags & EngineEventFlagToken, 0u);
  EXPECT_NE(terminal.flags & EngineEventFlagTurnFinished, 0u);
  EXPECT_EQ(terminal.finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(terminal.usage.generated_tokens, options.min_generated_tokens);
}

TEST_F(RequestLifecycleTest, RequestRejectsDraftTokensWithGuidance) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto guidance_engine = MakeDoublesEngine(guidance_model, /*capacity=*/8,
                                           EosToken(*guidance_model));
  guidance_engine.cache->SetMaxDraftTokensPerStep(3);

  auto request = CreateEngineRequest(guidance_engine.engine);
  request->BeginTurn(Prompt(), GuidedOptions("!!"));

  EXPECT_THROW(request->SetDraftTokens(std::array<int32_t, 1>{11}),
               std::runtime_error);

  // Guidance is turn-scoped, so drafting eligibility comes back on the next unguided turn.
  guidance_engine.executor->SetForcedToken(EosToken(*guidance_model));
  // The grammar can mask the forced EOS until it accepts, so drive the turn to completion.
  constexpr size_t kMaxGuidedSteps = 8;
  for (size_t step = 0; step < kMaxGuidedSteps && !request->IsTurnComplete(); ++step) {
    ASSERT_EQ(RunOne(*guidance_engine.engine).request, request);
  }
  ASSERT_TRUE(request->IsTurnComplete());
  request->BeginTurn(std::array<int32_t, 1>{5});
  ASSERT_EQ(RunOne(*guidance_engine.engine).request, request);
  EXPECT_EQ(request->DraftTokenValidationError(), nullptr);
}

TEST_F(RequestLifecycleTest, GuidanceMasksTokensAndRollsBackWithSearchState) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto guidance_engine = MakeDoublesEngine(guidance_model, /*capacity=*/8,
                                           EosToken(*guidance_model));
  auto tokenizer = guidance_model->CreateTokenizer();
  const auto expected_tokens = tokenizer->Encode("!");
  const auto invalid_tokens = tokenizer->Encode("a");
  ASSERT_EQ(expected_tokens.size(), 1u);
  ASSERT_EQ(invalid_tokens.size(), 1u);

  auto request = CreateEngineRequest(guidance_engine.engine);
  auto options = GuidedOptions("!!");
  auto prompt = Prompt();
  request->BeginTurn(prompt, options);
  // The admitted turn snapshotted the grammar, so mutating the options afterwards cannot change it.
  options.guidance_data = "a";

  auto first_logits = LogitsFavoringToken(
      *guidance_model, invalid_tokens.front(), expected_tokens.front());
  request->SaveStateForTransaction();
  const auto staged_first = request->ApplyLogitsForTransaction(first_logits);
  EXPECT_EQ(staged_first.token, expected_tokens.front());
  request->RestoreStateForTransaction();

  request->SaveStateForTransaction();
  const auto retried_first = request->ApplyLogitsForTransaction(first_logits);
  EXPECT_EQ(retried_first.token, expected_tokens.front());
  request->CommitStateForTransaction();

  auto second_logits = LogitsFavoringToken(
      *guidance_model, invalid_tokens.front(), expected_tokens.front());
  request->SaveStateForTransaction();
  const auto staged_second = request->ApplyLogitsForTransaction(second_logits);
  EXPECT_EQ(staged_second.token, expected_tokens.front());
  request->RestoreStateForTransaction();
}

TEST_F(RequestLifecycleTest, BeginTurnAfterCancellationResetsGuidance) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto guidance_engine = MakeDoublesEngine(guidance_model, /*capacity=*/8,
                                           EosToken(*guidance_model));
  auto tokenizer = guidance_model->CreateTokenizer();
  const auto first_tokens = tokenizer->Encode("!");
  const auto second_tokens = tokenizer->Encode("a");
  ASSERT_EQ(first_tokens.size(), 1u);
  ASSERT_EQ(second_tokens.size(), 1u);

  auto request = CreateEngineRequest(guidance_engine.engine);
  const auto turn_id = request->BeginTurn(Prompt(), GuidedOptions("!a"));

  guidance_engine.executor->SetForcedToken(first_tokens.front());
  EXPECT_EQ(RunOne(*guidance_engine.engine).token, first_tokens.front());

  ASSERT_TRUE(request->Cancel(turn_id));
  EXPECT_EQ(
      RunOne(*guidance_engine.engine).finish_reason,
      GenerationFinishReason::Canceled);

  // A new turn asking for the same grammar starts that grammar over rather than resuming the
  // canceled turn's cursor.
  request->BeginTurn(std::array<int32_t, 1>{5}, GuidedOptions("!a"));
  auto first_logits = LogitsFavoringToken(
      *guidance_model, second_tokens.front(), first_tokens.front());
  request->SaveStateForTransaction();
  EXPECT_EQ(
      request->ApplyLogitsForTransaction(first_logits).token,
      first_tokens.front());
  request->RestoreStateForTransaction();
}

// A guided turn whose admission fails must leave the Request exactly as the previous turn left it,
// and a corrected retry must then behave like an ordinary guided turn.
TEST_F(RequestLifecycleTest, FailedGuidedBeginTurnLeavesTheRequestReusable) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto guidance_engine = MakeDoublesEngine(guidance_model, /*capacity=*/8,
                                           EosToken(*guidance_model));
  auto tokenizer = guidance_model->CreateTokenizer();
  const auto first_tokens = tokenizer->Encode("!");
  ASSERT_EQ(first_tokens.size(), 1u);
  const auto invalid_tokens = tokenizer->Encode("a");
  ASSERT_EQ(invalid_tokens.size(), 1u);

  auto control = std::make_shared<FailingContinuationControl>();
  FailingContinuationDevice device{*guidance_model->p_device_scoring_, control};
  ScopedScoringDevice scoped_device{*guidance_model, device};
  auto request = CreateEngineRequest(guidance_engine.engine);
  const auto turn_id = request->BeginTurn(Prompt(), GuidedOptions("!a"));

  guidance_engine.executor->SetForcedToken(first_tokens.front());
  EXPECT_EQ(RunOne(*guidance_engine.engine).token, first_tokens.front());

  ASSERT_TRUE(request->Cancel(turn_id));
  EXPECT_EQ(
      RunOne(*guidance_engine.engine).finish_reason,
      GenerationFinishReason::Canceled);
  const auto before = request->Snapshot();

  // An unusable grammar is rejected while building the turn's processor, before any mutation.
  EXPECT_THROW(request->BeginTurn(std::array<int32_t, 1>{5}, GuidedOptions("[")),
               std::runtime_error);
  EXPECT_EQ(request->Snapshot().current_sequence_length,
            before.current_sequence_length);

  // A failure after the grammar was built must roll back just as completely.
  control->fail_append = true;
  try {
    request->BeginTurn(std::array<int32_t, 1>{5}, GuidedOptions("!a"));
    FAIL() << "Expected continuation append failure";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "Injected continuation append failure.");
  }
  control->fail_append = false;
  EXPECT_EQ(request->Snapshot().current_sequence_length,
            before.current_sequence_length);

  // The corrected retry is guided from the start of the grammar.
  request->BeginTurn(std::array<int32_t, 1>{5}, GuidedOptions("!a"));
  auto logits = LogitsFavoringToken(
      *guidance_model, invalid_tokens.front(), first_tokens.front());
  request->SaveStateForTransaction();
  EXPECT_EQ(
      request->ApplyLogitsForTransaction(logits).token, first_tokens.front());
  request->RestoreStateForTransaction();
}

// Omitting guidance on a following turn leaves that turn unguided; nothing is inherited.
TEST_F(RequestLifecycleTest, OmittedTurnGuidanceContinuesUnguided) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto guidance_engine = MakeDoublesEngine(guidance_model, /*capacity=*/8,
                                           EosToken(*guidance_model));
  auto tokenizer = guidance_model->CreateTokenizer();
  const auto guided_tokens = tokenizer->Encode("!");
  const auto unguided_tokens = tokenizer->Encode("a");
  ASSERT_EQ(guided_tokens.size(), 1u);
  ASSERT_EQ(unguided_tokens.size(), 1u);

  auto request = CreateEngineRequest(guidance_engine.engine);
  request->BeginTurn(Prompt(), GuidedOptions("!a"));
  guidance_engine.executor->SetForcedToken(guided_tokens.front());
  EXPECT_EQ(RunOne(*guidance_engine.engine).token, guided_tokens.front());
  ASSERT_TRUE(request->Cancel(request->CurrentTurnId()));
  ASSERT_EQ(
      RunOne(*guidance_engine.engine).finish_reason,
      GenerationFinishReason::Canceled);

  // The grammar would still forbid this token; without guidance the argmax wins.
  request->BeginTurn(std::array<int32_t, 1>{5});
  EXPECT_FALSE(request->HasGuidance());
  auto logits = LogitsFavoringToken(
      *guidance_model, unguided_tokens.front(), guided_tokens.front());
  request->SaveStateForTransaction();
  EXPECT_EQ(
      request->ApplyLogitsForTransaction(logits).token, unguided_tokens.front());
  request->RestoreStateForTransaction();
}

TEST_F(RequestLifecycleTest, StaticCpuGenerationAdvancesGuidanceCursor) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto guidance_engine = MakeDoublesEngine(guidance_model, /*capacity=*/8,
                                           EosToken(*guidance_model));
  auto tokenizer = guidance_model->CreateTokenizer();
  const auto first_tokens = tokenizer->Encode("!");
  const auto second_tokens = tokenizer->Encode("a");
  ASSERT_EQ(first_tokens.size(), 1u);
  ASSERT_EQ(second_tokens.size(), 1u);

  auto request = CreateRequestWithPrompt(
      guidance_engine.engine, Prompt(), GuidedOptions("!a"));

  auto first_logits = LogitsFavoringToken(
      *guidance_model, second_tokens.front(), first_tokens.front());
  request->GenerateNextTokens(first_logits);
  EXPECT_EQ(request->CompleteGeneration().token, first_tokens.front());

  auto second_logits = LogitsFavoringToken(
      *guidance_model, first_tokens.front(), second_tokens.front());
  request->GenerateNextTokens(second_logits);
  EXPECT_EQ(request->CompleteGeneration().token, second_tokens.front());
}
#endif

}  // namespace
}  // namespace test
}  // namespace Generators
