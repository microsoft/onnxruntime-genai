// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Lifecycle tests for the engine Request state machine. Because a tiny real
// CPU fixture model is available, these tests drive genuine Request objects (rather than a mock
// Search) and pin the transition policy: which mutations each status permits, and how
// create/begin/schedule/continue/close move a request between Unassigned, Assigned, Active,
// TurnComplete, and Closed.

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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
    return CreateEngineRequest(engine_.engine, *model_);
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

  auto params = MakeGreedyParams(*model);
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
    auto params = MakeGreedyParams(*model);

    try {
      [[maybe_unused]] auto request = std::make_shared<Request>(params);
      FAIL() << "Expected invalid eos_token_id to be rejected";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find("eos_token_id"), std::string::npos);
    }
  }
}

TEST_F(RequestLifecycleTest, InvalidVocabSizeIsRejected) {
  for (const int invalid_vocab_size : {-1, 0}) {
    Config config;
    config.model.vocab_size = invalid_vocab_size;
    config.model.eos_token_id.clear();
    auto params = std::make_shared<GeneratorParams>(config);

    try {
      [[maybe_unused]] auto request = std::make_shared<Request>(params);
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
  EXPECT_FALSE(engine_.engine->HasPendingRequests());
}

TEST_F(RequestLifecycleTest, CreateRequestSnapshotsParameters) {
  auto params = MakeGreedyParams(*model_);
  params->search.chunk_size = 2;
  auto request = CreateEngineRequest(engine_.engine, *params);

  params->search.chunk_size = 7;

  ASSERT_TRUE(request->SearchOptions().chunk_size.has_value());
  EXPECT_EQ(*request->SearchOptions().chunk_size, 2);
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
  auto request = CreateRequestWithPrompt(engine_.engine, *model_, prompt);
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
  auto request = CreateRequestWithPrompt(engine_.engine, *model_, prompt);
  request->Schedule();
  EXPECT_EQ(request->status_, RequestStatus::Active);
}

// While a request is active its token stream is owned by the engine, so BeginTurn is
// rejected without mutating the request.
TEST_F(RequestLifecycleTest, BeginTurnIsRejectedWhileActive) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, *model_, prompt);
  request->Schedule();
  ASSERT_EQ(request->status_, RequestStatus::Active);
  const int64_t length_before = request->CurrentSequenceLength();

  std::vector<int32_t> more{5, 6};
  EXPECT_THROW(request->BeginTurn(more), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Active);
  EXPECT_EQ(request->CurrentSequenceLength(), length_before);
}

TEST_F(RequestLifecycleTest, BeginTurnIsRejectedWhileAssigned) {
  auto prompt = Prompt();
  const std::vector<int32_t> more{5};
  auto request = CreateRequestWithPrompt(engine_.engine, *model_, prompt);

  EXPECT_THROW(request->BeginTurn(more), std::runtime_error);
}

// After a turn completes, BeginTurn appends another input fragment and queues the resident request.
TEST_F(RequestLifecycleTest, BeginTurnAfterTurnCompleteQueuesNextTurn) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, *model_, prompt);
  const int64_t assigned_length = static_cast<int64_t>(prompt.size());

  engine_.engine->Run();
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);
  std::vector<int32_t> more{5, 6};
  request->BeginTurn(more);

  EXPECT_EQ(request->CurrentSequenceLength(), assigned_length + static_cast<int64_t>(more.size()));
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_FALSE(request->IsTurnComplete());
  EXPECT_EQ(engine_.engine->Run(), request);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_TRUE(request->IsTurnComplete());
}

TEST_F(RequestLifecycleTest, ContinuationBeyondContextIsRejectedBeforeMutation) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, *model_, prompt);
  engine_.engine->Run();
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);
  const auto before = request->Snapshot();

  const size_t remaining =
      static_cast<size_t>(request->SearchOptions().max_length -
                          request->CurrentSequenceLength());
  std::vector<int32_t> too_many(remaining, 5);
  EXPECT_THROW(request->BeginTurn(too_many), std::runtime_error);

  const auto after = request->Snapshot();
  EXPECT_EQ(after.status, before.status);
  EXPECT_EQ(after.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(after.processed_sequence_length, before.processed_sequence_length);
}

TEST_F(RequestLifecycleTest, FailedContinuationAppendPreservesCompletedTurnState) {
  auto params = MakeGreedyParams(*model_);
  auto control = std::make_shared<FailingContinuationControl>();
  FailingContinuationDevice device{*params->p_device, control};
  params->p_device = &device;
  auto request = CreateEngineRequest(engine_.engine, *params);
  engine_.executor->SetForcedToken(/*token=*/5);
  request->BeginTurn(Prompt(), std::optional<size_t>{1});
  ASSERT_EQ(engine_.engine->Run(), request);
  ASSERT_TRUE(request->IsTurnComplete());
  ASSERT_TRUE(request->HasUnseenTokens());
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
  EXPECT_TRUE(request->HasUnseenTokens());

  control->fail_append = false;
  EXPECT_NO_THROW(request->BeginTurn(
      std::vector<int32_t>{6},
      std::optional<size_t>{2}));
  EXPECT_TRUE(request->HasUnseenTokens());
}

TEST_F(RequestLifecycleTest, FailedContinuationRestoreClosesRequestAndPoisonsEngine) {
  auto params = MakeGreedyParams(*model_);
  auto control = std::make_shared<FailingContinuationControl>();
  FailingContinuationDevice device{*params->p_device, control};
  params->p_device = &device;
  auto request = CreateEngineRequest(engine_.engine, *params);
  request->BeginTurn(Prompt());
  ASSERT_EQ(engine_.engine->Run(), request);
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
  EXPECT_THROW(static_cast<void>(engine_.engine->Run()), EngineStepError);
}

TEST_F(RequestLifecycleTest, ContinuationPreservesUnreadOutputAndHidesInputTokens) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, *model_, prompt);
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
  ASSERT_TRUE(request->HasUnseenTokens());

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
  ASSERT_TRUE(request->HasUnseenTokens());
  EXPECT_EQ(request->UnseenToken(), generated_token);
  EXPECT_FALSE(request->HasUnseenTokens());
}

TEST_F(RequestLifecycleTest, ContinuationIsRejectedOutsideTurnComplete) {
  const std::vector<int32_t> more{5};
  auto request = NewRequest();
  request->BeginTurn(Prompt());
  EXPECT_THROW(request->BeginTurn(more), std::runtime_error);

  request->Schedule();
  EXPECT_THROW(request->BeginTurn(more), std::runtime_error);
}

// Closing a request releases it from the engine and makes it terminal.
TEST_F(RequestLifecycleTest, RequestCloseIsIdempotentAfterClose) {
  auto prompt = Prompt();
  const std::vector<int32_t> more{5};
  auto request = CreateRequestWithPrompt(engine_.engine, *model_, prompt);
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
      CreateRequestWithPrompt(engine_.engine, *model_, Prompt());
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
  auto request = CreateRequestWithPrompt(local_engine.engine, *model_, prompt);
  local_engine.engine.reset();

  EXPECT_NO_THROW(request->Close());
  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_NO_THROW(request->Close());
}

TEST_F(RequestLifecycleTest, TransactionalLogitsStageUntilCommit) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, *model_, prompt);
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
  ASSERT_TRUE(request->HasUnseenTokens());
  EXPECT_EQ(request->UnseenToken(), next_token);
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
  EXPECT_FALSE(request->HasUnseenTokens());

  request->CommitStateForTransaction();
  request->CommitStep(plan, result);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  ASSERT_TRUE(request->HasUnseenTokens());
  EXPECT_EQ(request->UnseenToken(), 5);
}

TEST_F(RequestLifecycleTest, TransactionalLogitsRollbackRestoresSearchState) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, *model_, prompt);
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
  EXPECT_FALSE(request->HasUnseenTokens());
}

TEST_F(RequestLifecycleTest, TransactionalRollbackRestoresSamplingState) {
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.top_p = 1.0f;
  params->search.temperature = 1.0f;
  params->search.random_seed = 1234;

  auto request = CreateRequestWithPrompt(engine_.engine, *params, Prompt());
  const auto before = request->Snapshot();
  auto logits = SamplingLogits(*model_);

  request->SaveStateForTransaction();
  const auto first = request->ApplyLogitsForTransaction(logits);
  request->RestoreStateForTransaction();

  EXPECT_EQ(request->Snapshot().current_sequence_length,
            before.current_sequence_length);

  request->SaveStateForTransaction();
  const auto retried = request->ApplyLogitsForTransaction(logits);
  request->RestoreStateForTransaction();

  EXPECT_TRUE(first.token_appended);
  EXPECT_TRUE(retried.token_appended);
  EXPECT_EQ(retried.token, first.token);
}

TEST_F(RequestLifecycleTest, PartialPrefillAdvancesOnlyAtCommit) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, *model_, prompt);
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
  EXPECT_FALSE(request->HasUnseenTokens());
}

TEST_F(RequestLifecycleTest, EosCompletesWithoutAppendingVisibleToken) {
  auto prompt = Prompt();
  auto request = CreateRequestWithPrompt(engine_.engine, *model_, prompt);
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
  EXPECT_FALSE(request->HasUnseenTokens());
}

TEST_F(RequestLifecycleTest, RequestRejectsMultiSequenceSearch) {
  auto params = MakeGreedyParams(*model_);
  params->search.batch_size = 2;
  EXPECT_THROW(
      {
        auto request = engine_.engine->CreateRequest(*params);
        static_cast<void>(request);
      },
      std::runtime_error);
}

TEST_F(RequestLifecycleTest, RequestRejectsGuidanceFastForwardTokens) {
  auto params = MakeGreedyParams(*model_);
  params->SetGuidance("regex", "[0-9]+", true);

  EXPECT_THROW(
      {
        auto request = engine_.engine->CreateRequest(*params);
        static_cast<void>(request);
      },
      std::runtime_error);
}

TEST_F(RequestLifecycleTest, RequestRejectsIncompleteGuidanceConfiguration) {
  for (bool omit_type : {false, true}) {
    auto params = MakeGreedyParams(*model_);
    params->guidance_type = omit_type ? "" : "regex";
    params->guidance_data = omit_type ? "[0-9]+" : "";

    try {
      auto request = engine_.engine->CreateRequest(*params);
      FAIL() << "Expected incomplete guidance configuration to be rejected";
    } catch (const std::runtime_error& error) {
      EXPECT_STREQ(error.what(), "Guidance type and data must be provided together.");
    }
  }
}

#if USE_GUIDANCE
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

  auto params = MakeGreedyParams(*guidance_model);
  params->SetGuidance("regex", "!!", false);
  auto request = CreateEngineRequest(guidance_engine.engine, *params);
  params->guidance_data = "a";
  auto prompt = Prompt();
  request->BeginTurn(prompt);

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

  auto params = MakeGreedyParams(*guidance_model);
  params->SetGuidance("regex", "!a", false);
  auto request = CreateRequestWithPrompt(
      guidance_engine.engine, *params, Prompt());

  auto first_logits = LogitsFavoringToken(
      *guidance_model, second_tokens.front(), first_tokens.front());
  request->GenerateNextTokens(first_logits);
  request->CompleteGeneration();
  EXPECT_EQ(request->UnseenToken(), first_tokens.front());

  auto second_logits = LogitsFavoringToken(
      *guidance_model, first_tokens.front(), second_tokens.front());
  request->GenerateNextTokens(second_logits);
  request->CompleteGeneration();
  EXPECT_EQ(request->UnseenToken(), second_tokens.front());
}
#endif

}  // namespace
}  // namespace test
}  // namespace Generators
