// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Lifecycle tests for the engine Request state machine. Because a tiny real
// CPU fixture model is available, these tests drive genuine Request objects (rather than a mock
// Search) and pin the transition policy: which mutations each status permits, and how
// create/assign/schedule/continue/remove move a request between Unassigned, Assigned, Active,
// TurnComplete, and Closed.

#include <barrier>
#include <array>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_helpers.h"
#include "engine_test_doubles.h"
#include "engine/request_status.h"
#include "constrained_logits_processor.h"
#include "models/preprocessing/genai_tokenizer.h"

namespace Generators {
namespace test {

#if USE_GUIDANCE
struct GuidanceProcessorTestAccess {
  static void InstallFailedMaskFuture(
      GuidanceLogitsProcessor& processor) {
    std::promise<std::vector<uint32_t>> promise;
    promise.set_exception(std::make_exception_ptr(
        std::runtime_error("Injected real guidance future failure.")));
    processor.pending_masks_ = promise.get_future().share();
    processor.mask_dirty_ = false;
  }

  static bool MaskDirty(
      const GuidanceLogitsProcessor& processor) {
    return processor.mask_dirty_;
  }
};
#endif

namespace {

std::vector<int32_t> Prompt() { return {2, 3, 4}; }

// LogitsForToken / LogitsFavoringToken now live in engine_test_helpers.h so guidance_processor_tests.cpp
// can share them.
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
    return std::make_shared<Request>(MakeGreedyParams(*model_));
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

// A request must carry at least one token per append; an empty batch is rejected.
TEST_F(RequestLifecycleTest, EmptyAppendIsRejected) {
  auto request = NewRequest();
  EXPECT_THROW(request->AddTokens({}), std::runtime_error);
  EXPECT_THROW(request->Continue({}), std::runtime_error);
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

TEST_F(RequestLifecycleTest, EmptyRequestIsRejectedBeforeAssignment) {
  auto request = NewRequest();

  EXPECT_THROW(engine_.engine->AddRequest(request), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Unassigned);
}

// An append that would exceed the model's max length is rejected before any tokens are buffered, so
// a subsequent valid append and assign reflect only the accepted tokens.
TEST_F(RequestLifecycleTest, AppendBeyondContextIsRejectedBeforeMutation) {
  auto request = NewRequest();
  const size_t over_capacity = static_cast<size_t>(model_->config_->model.context_length) + 1;
  std::vector<int32_t> too_many(over_capacity, 2);
  EXPECT_THROW(request->AddTokens(too_many), std::runtime_error);

  auto prompt = Prompt();
  request->AddTokens(prompt);
  request->Assign(engine_.engine);
  EXPECT_EQ(request->CurrentSequenceLength(), static_cast<int64_t>(prompt.size()));
}

// A request can only be assigned while Unassigned; assigning an already-assigned request throws and
// leaves the request untouched.
TEST_F(RequestLifecycleTest, AssignIsRejectedWhenAlreadyAssigned) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
  ASSERT_EQ(request->status_, RequestStatus::Assigned);
  const int64_t length_before = request->CurrentSequenceLength();

  EXPECT_THROW(request->Assign(engine_.engine), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_EQ(request->CurrentSequenceLength(), length_before);
}

// Scheduling requires a prior assign; an unassigned request cannot be scheduled and stays
// Unassigned.
TEST_F(RequestLifecycleTest, ScheduleIsRejectedBeforeAssign) {
  auto prompt = Prompt();
  auto request = MintRequest(*model_, prompt);
  ASSERT_EQ(request->status_, RequestStatus::Unassigned);

  EXPECT_THROW(request->Schedule(), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Unassigned);
}

// An assigned, non-empty request schedules cleanly and moves to Active.
TEST_F(RequestLifecycleTest, ScheduleFromAssignedMovesToActive) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
  request->Schedule();
  EXPECT_EQ(request->status_, RequestStatus::Active);
}

// While a request is active its token stream is owned by the engine, so external appends are
// rejected without mutating the request.
TEST_F(RequestLifecycleTest, AppendIsRejectedWhileActive) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
  request->Schedule();
  ASSERT_EQ(request->status_, RequestStatus::Active);
  const int64_t length_before = request->CurrentSequenceLength();

  std::vector<int32_t> more{5, 6};
  EXPECT_THROW(request->AddTokens(more), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Active);
  EXPECT_EQ(request->CurrentSequenceLength(), length_before);
}

TEST_F(RequestLifecycleTest, AddTokensIsRejectedWhileAssigned) {
  auto prompt = Prompt();
  const std::vector<int32_t> more{5};
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);

  EXPECT_THROW(request->AddTokens(more), std::runtime_error);
}

// After a turn completes, Continue appends another input fragment and queues the resident request.
TEST_F(RequestLifecycleTest, ContinueAfterTurnCompleteQueuesNextTurn) {
  auto prompt = Prompt();
  auto request = MintRequest(*model_, prompt);
  const int64_t assigned_length = static_cast<int64_t>(prompt.size());
  engine_.engine->AddRequest(request);

  engine_.engine->Step();
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);
  std::vector<int32_t> more{5, 6};
  EXPECT_THROW(request->AddTokens(more), std::runtime_error);
  request->Continue(more);

  EXPECT_EQ(request->CurrentSequenceLength(), assigned_length + static_cast<int64_t>(more.size()));
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_FALSE(request->IsTurnComplete());
  EXPECT_EQ(engine_.engine->Step(), request);
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
  EXPECT_TRUE(request->IsTurnComplete());
}

TEST_F(RequestLifecycleTest, ContinueBeyondContextIsRejectedBeforeMutation) {
  auto prompt = Prompt();
  auto request = MintRequest(*model_, prompt);
  engine_.engine->AddRequest(request);
  engine_.engine->Step();
  ASSERT_EQ(request->status_, RequestStatus::TurnComplete);
  const auto before = request->Snapshot();

  const size_t remaining =
      static_cast<size_t>(request->Params()->search.max_length -
                          request->CurrentSequenceLength());
  std::vector<int32_t> too_many(remaining, 5);
  EXPECT_THROW(request->Continue(too_many), std::runtime_error);

  const auto after = request->Snapshot();
  EXPECT_EQ(after.status, before.status);
  EXPECT_EQ(after.current_sequence_length, before.current_sequence_length);
  EXPECT_EQ(after.processed_sequence_length, before.processed_sequence_length);
}

TEST_F(RequestLifecycleTest, FailedContinuationRestoreClosesRequestAndPoisonsEngine) {
  auto params = MakeGreedyParams(*model_);
  auto control = std::make_shared<FailingContinuationControl>();
  FailingContinuationDevice device{*params->p_device, control};
  params->p_device = &device;
  auto request = std::make_shared<Request>(params);
  request->AddTokens(Prompt());
  engine_.engine->AddRequest(request);
  ASSERT_EQ(engine_.engine->Step(), request);
  ASSERT_TRUE(request->IsTurnComplete());
  ASSERT_EQ(engine_.cache->AllocatedCount(), 1u);

  control->fail_append = true;
  control->fail_restore = true;
  try {
    request->Continue(std::vector<int32_t>{5});
    FAIL() << "Expected continuation restore failure.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind, StepOutcomeKind::FatalExecutionFailure);
  }

  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(engine_.cache->AllocatedCount(), 0u);
  EXPECT_THROW(request->Continue(std::vector<int32_t>{6}), std::runtime_error);
  EXPECT_THROW(static_cast<void>(engine_.engine->Step()), EngineStepError);
}

TEST_F(RequestLifecycleTest, ContinuePreservesUnreadOutputAndHidesInputTokens) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
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
  request->Continue(continuation);

  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  ASSERT_TRUE(request->HasUnseenTokens());
  EXPECT_EQ(request->UnseenToken(), generated_token);
  EXPECT_FALSE(request->HasUnseenTokens());
}

TEST_F(RequestLifecycleTest, ContinueIsRejectedOutsideTurnComplete) {
  const std::vector<int32_t> more{5};
  auto request = NewRequest();
  EXPECT_THROW(request->Continue(more), std::runtime_error);

  auto prompt = Prompt();
  request->AddTokens(prompt);
  engine_.engine->AddRequest(request);
  EXPECT_THROW(request->Continue(more), std::runtime_error);

  request->Schedule();
  EXPECT_THROW(request->Continue(more), std::runtime_error);
}

// Removing a request releases it from the engine and makes it terminal.
TEST_F(RequestLifecycleTest, RequestRemoveIsIdempotentAfterClose) {
  auto prompt = Prompt();
  const std::vector<int32_t> more{5};
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
  ASSERT_EQ(request->status_, RequestStatus::Assigned);

  request->Remove();
  EXPECT_EQ(request->status_, RequestStatus::Closed);
  EXPECT_EQ(engine_.cache->deallocate_calls, 1);
  EXPECT_EQ(engine_.cache->AllocatedCount(), 0u);
  EXPECT_THROW(request->AddTokens(more), std::runtime_error);
  EXPECT_THROW(request->Continue(more), std::runtime_error);
  EXPECT_NO_THROW(request->Remove());
  EXPECT_EQ(engine_.cache->deallocate_calls, 1);
}

TEST_F(RequestLifecycleTest, EngineRemoveRequestIsIdempotentAfterClose) {
  auto other_engine =
      MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);

  engine_.engine->RemoveRequest(request);
  ASSERT_EQ(request->status_, RequestStatus::Closed);
  ASSERT_EQ(engine_.cache->deallocate_calls, 1);

  EXPECT_NO_THROW(engine_.engine->RemoveRequest(request));
  EXPECT_NO_THROW(other_engine.engine->RemoveRequest(request));
  EXPECT_EQ(engine_.cache->deallocate_calls, 1);
  EXPECT_EQ(other_engine.cache->deallocate_calls, 0);
}

TEST_F(RequestLifecycleTest, EngineRemoveRequestRejectsNonterminalRequestFromAnotherEngine) {
  auto other_engine =
      MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);

  EXPECT_THROW(other_engine.engine->RemoveRequest(request), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Assigned);
  EXPECT_EQ(engine_.cache->deallocate_calls, 0);
  EXPECT_EQ(other_engine.cache->deallocate_calls, 0);
}

TEST_F(RequestLifecycleTest, RemoveIsRejectedBeforeSubmission) {
  auto request = NewRequest();
  EXPECT_THROW(request->Remove(), std::runtime_error);
  EXPECT_EQ(request->status_, RequestStatus::Unassigned);
}

TEST_F(RequestLifecycleTest, RemoveClosesRequestAfterEngineDestruction) {
  auto local_engine =
      MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  auto prompt = Prompt();
  auto request = MintRequest(*model_, prompt);
  local_engine.engine->AddRequest(request);
  local_engine.engine.reset();

  EXPECT_NO_THROW(request->Remove());
  EXPECT_EQ(request->status_, RequestStatus::Closed);
}

TEST_F(RequestLifecycleTest, TransactionalLogitsStageUntilCommit) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
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

TEST_F(RequestLifecycleTest, TransactionalLogitsRollbackRestoresSearchState) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
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

  auto request = std::make_shared<Request>(params);
  request->AddTokens(Prompt());
  request->Assign(engine_.engine);
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
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
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

TEST_F(RequestLifecycleTest, FirstTransactionalStepCanCommitDirectlyToTurnComplete) {
  auto prompt = Prompt();
  auto request = MintAssignedRequest(engine_.engine, *model_, prompt);
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
  EXPECT_EQ(request->status_, RequestStatus::TurnComplete);
}

TEST_F(RequestLifecycleTest, RequestRejectsMultiSequenceSearch) {
  auto params = MakeGreedyParams(*model_);
  params->search.batch_size = 2;
  EXPECT_THROW(
      {
        auto request = std::make_shared<Request>(params);
        static_cast<void>(request);
      },
      std::runtime_error);
}

TEST_F(RequestLifecycleTest, RequestRejectsGuidanceFastForwardTokens) {
  auto params = MakeGreedyParams(*model_);
  params->SetGuidance("regex", "[0-9]+", true);

  EXPECT_THROW(
      {
        auto request = std::make_shared<Request>(params);
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
      auto request = std::make_shared<Request>(params);
      FAIL() << "Expected incomplete guidance configuration to be rejected";
    } catch (const std::runtime_error& error) {
      EXPECT_STREQ(error.what(), "Guidance type and data must be provided together.");
    }
  }
}

#if USE_GUIDANCE
TEST_F(RequestLifecycleTest, GuidanceCacheReusesTokenizerAndCompiledGrammar) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  auto first_params = MakeGreedyParams(*guidance_model);
  first_params->SetGuidance("regex", "!", false);
  auto first_request = std::make_shared<Request>(first_params);

  auto second_params = MakeGreedyParams(*guidance_model);
  second_params->SetGuidance("regex", "!", false);
  auto second_request = std::make_shared<Request>(second_params);

  const auto stats = GetGuidanceCacheStats(*guidance_model);
  EXPECT_EQ(stats.tokenizer_initializations, 1u);
  EXPECT_EQ(stats.grammar_misses, 1u);
  EXPECT_EQ(stats.grammar_hits, 1u);
  EXPECT_EQ(stats.cached_grammars, 1u);
  EXPECT_GT(stats.cached_key_bytes, 0u);
}

TEST_F(RequestLifecycleTest, GuidanceCacheSingleFlightsConcurrentIdenticalGrammars) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  constexpr size_t kRequestCount = 8;
  std::barrier gate(static_cast<std::ptrdiff_t>(kRequestCount));
  std::vector<std::exception_ptr> errors(kRequestCount);
  std::vector<std::thread> threads;
  threads.reserve(kRequestCount);

  for (size_t i = 0; i < kRequestCount; ++i) {
    threads.emplace_back([&, i] {
      try {
        gate.arrive_and_wait();
        auto params = MakeGreedyParams(*guidance_model);
        params->SetGuidance("regex", "!", false);
        auto request = std::make_shared<Request>(params);
        static_cast<void>(request);
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    EXPECT_EQ(error, nullptr);
  }

  const auto stats = GetGuidanceCacheStats(*guidance_model);
  EXPECT_EQ(stats.tokenizer_initializations, 1u);
  EXPECT_EQ(stats.grammar_misses, 1u);
  EXPECT_EQ(stats.grammar_hits + stats.grammar_waits, kRequestCount - 1);
}

TEST_F(RequestLifecycleTest, GuidanceCacheIsolatesConcurrentUniqueGrammars) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  constexpr size_t kRequestCount = 8;
  std::barrier gate(static_cast<std::ptrdiff_t>(kRequestCount));
  std::vector<std::exception_ptr> errors(kRequestCount);
  std::vector<std::thread> threads;
  threads.reserve(kRequestCount);

  for (size_t i = 0; i < kRequestCount; ++i) {
    threads.emplace_back([&, i] {
      try {
        gate.arrive_and_wait();
        auto params = MakeGreedyParams(*guidance_model);
        const std::string grammar(1, static_cast<char>('a' + i));
        params->SetGuidance("regex", grammar.c_str(), false);
        auto request = std::make_shared<Request>(params);
        static_cast<void>(request);
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    EXPECT_EQ(error, nullptr);
  }

  const auto stats = GetGuidanceCacheStats(*guidance_model);
  EXPECT_EQ(stats.tokenizer_initializations, 1u);
  EXPECT_EQ(stats.grammar_misses, kRequestCount);
  EXPECT_EQ(stats.cached_grammars, kRequestCount);
}

TEST_F(RequestLifecycleTest, GuidanceClonesScheduleDirtyMasksIndependently) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto tokenizer = guidance_model->CreateTokenizer();
  const auto expected_tokens = tokenizer->Encode("!");
  ASSERT_EQ(expected_tokens.size(), 1u);

  auto params = MakeGreedyParams(*guidance_model);
  params->SetGuidance("regex", "!!", false);
  auto processor = CreateGuidanceLogitsProcessor(*guidance_model, params);
  processor->ProcessLogits(LogitsForToken(*guidance_model, expected_tokens.front()));
  int32_t token = expected_tokens.front();
  processor->CommitTokens(std::span<int32_t>{&token, 1});
  auto clone = processor->Clone();

  std::barrier gate(2);
  std::exception_ptr errors[2];
  std::thread original_thread([&] {
    try {
      gate.arrive_and_wait();
      processor->ProcessLogits(LogitsForToken(*guidance_model, expected_tokens.front()));
    } catch (...) {
      errors[0] = std::current_exception();
    }
  });
  std::thread clone_thread([&] {
    try {
      gate.arrive_and_wait();
      clone->ProcessLogits(LogitsForToken(*guidance_model, expected_tokens.front()));
    } catch (...) {
      errors[1] = std::current_exception();
    }
  });
  original_thread.join();
  clone_thread.join();

  EXPECT_EQ(errors[0], nullptr);
  EXPECT_EQ(errors[1], nullptr);
}

TEST_F(RequestLifecycleTest, PendingGuidanceMaskOutlivesRemovedProcessor) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto tokenizer = guidance_model->CreateTokenizer();
  const auto expected_tokens = tokenizer->Encode("!");
  ASSERT_EQ(expected_tokens.size(), 1u);

  auto params = MakeGreedyParams(*guidance_model);
  params->SetGuidance("regex", "!!", false);
  auto processor = CreateGuidanceLogitsProcessor(*guidance_model, params);
  processor->ProcessLogits(
      LogitsForToken(*guidance_model, expected_tokens.front()));
  int32_t token = expected_tokens.front();
  processor->CommitTokens(std::span<int32_t>{&token, 1});
  std::array<ConstrainedLogitsProcessor*, 1> processors{processor.get()};
  ScheduleGuidanceMaskComputation(processors);

  auto observer = processor->Clone();
  processor.reset();

  const auto mask = observer->GetReadyMask();
  ASSERT_FALSE(mask.empty());
  const uint32_t expected = static_cast<uint32_t>(expected_tokens.front());
  EXPECT_NE(mask[expected / 32] & (uint32_t{1} << (expected % 32)), 0u);
}

TEST_F(RequestLifecycleTest, FailedGuidanceFutureReschedulesRealProcessor) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto params = MakeGreedyParams(*guidance_model);
  params->SetGuidance("regex", "!", false);
  auto processor =
      CreateGuidanceLogitsProcessor(*guidance_model, params);
  auto& guidance =
      dynamic_cast<GuidanceLogitsProcessor&>(*processor);
  GuidanceProcessorTestAccess::InstallFailedMaskFuture(guidance);

  EXPECT_THROW(processor->GetReadyMask(), std::runtime_error);
  EXPECT_TRUE(GuidanceProcessorTestAccess::MaskDirty(guidance));

  const auto retried_mask = processor->GetReadyMask();
  EXPECT_FALSE(retried_mask.empty());
  EXPECT_FALSE(GuidanceProcessorTestAccess::MaskDirty(guidance));
}

TEST_F(RequestLifecycleTest, GuidanceCacheEvictionKeepsLiveProcessorAssetsValid) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  auto first_params = MakeGreedyParams(*guidance_model);
  first_params->SetGuidance("regex", "first", false);
  auto first_processor = CreateGuidanceLogitsProcessor(*guidance_model, first_params);

  for (size_t i = 0; i < 65; ++i) {
    auto params = MakeGreedyParams(*guidance_model);
    const auto grammar = "value" + std::to_string(i);
    params->SetGuidance("regex", grammar.c_str(), false);
    auto processor = CreateGuidanceLogitsProcessor(*guidance_model, params);
    static_cast<void>(processor);
  }

  const auto stats = GetGuidanceCacheStats(*guidance_model);
  EXPECT_GT(stats.grammar_evictions, 0u);
  EXPECT_LE(stats.cached_grammars, 64u);
  EXPECT_LE(stats.cached_key_bytes, 16u * 1024u * 1024u);
  const auto first_tokens = guidance_model->CreateTokenizer()->Encode("first");
  ASSERT_FALSE(first_tokens.empty());
  int32_t first_token = first_tokens.front();
  EXPECT_NO_THROW(first_processor->CommitTokens(std::span<int32_t>{&first_token, 1}));
  EXPECT_NO_THROW(first_processor->GetReadyMask());
  first_processor.reset();
}

TEST_F(RequestLifecycleTest, OversizedGuidanceGrammarDoesNotEvictCachedEntries) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  auto cached_params = MakeGreedyParams(*guidance_model);
  cached_params->SetGuidance("regex", "!", false);
  auto cached_processor =
      CreateGuidanceLogitsProcessor(*guidance_model, cached_params);
  const auto before = GetGuidanceCacheStats(*guidance_model);
  ASSERT_EQ(before.cached_grammars, 1u);

  std::string oversized_schema(16u * 1024u * 1024u, ' ');
  oversized_schema.front() = 'x';
  auto oversized_params = MakeGreedyParams(*guidance_model);
  oversized_params->SetGuidance(
      "json_schema", oversized_schema.c_str(), false);
  EXPECT_THROW(
      CreateGuidanceLogitsProcessor(*guidance_model, oversized_params),
      std::runtime_error);

  const auto after = GetGuidanceCacheStats(*guidance_model);
  EXPECT_EQ(after.cached_grammars, before.cached_grammars);
  EXPECT_EQ(after.cached_key_bytes, before.cached_key_bytes);
  EXPECT_EQ(after.grammar_evictions, before.grammar_evictions);
}

TEST_F(RequestLifecycleTest, GuidanceCacheDoesNotRetainFailedCompilations) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  for (size_t attempt = 1; attempt <= 2; ++attempt) {
    auto params = MakeGreedyParams(*guidance_model);
    params->SetGuidance("regex", "[", false);
    EXPECT_THROW(CreateGuidanceLogitsProcessor(*guidance_model, params), std::runtime_error);

    const auto stats = GetGuidanceCacheStats(*guidance_model);
    EXPECT_EQ(stats.grammar_misses, attempt);
    EXPECT_EQ(stats.cached_grammars, 0u);
    EXPECT_EQ(stats.cached_key_bytes, 0u);
  }
}

TEST_F(RequestLifecycleTest, RequestRejectsDraftTokensWithGuidance) {
  auto guidance_model = CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto guidance_engine = MakeDoublesEngine(guidance_model, /*capacity=*/8,
                                           EosToken(*guidance_model));
  guidance_engine.cache->SetMaxDraftTokensPerStep(3);

  auto params = MakeGreedyParams(*guidance_model);
  params->SetGuidance("regex", "!!", false);
  auto request = std::make_shared<Request>(params);
  request->AddTokens(Prompt());
  guidance_engine.engine->AddRequest(request);

  EXPECT_THROW(request->SetDraftTokens(std::array<int32_t, 1>{11}),
               std::runtime_error);
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

  auto params = MakeGreedyParams(*guidance_model);
  params->SetGuidance("regex", "!!", false);
  auto request = std::make_shared<Request>(params);
  params->guidance_data = "a";
  auto prompt = Prompt();
  request->AddTokens(prompt);
  request->Assign(guidance_engine.engine);

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
  auto request = std::make_shared<Request>(params);
  request->AddTokens(Prompt());
  request->Assign(guidance_engine.engine);

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

TEST_F(RequestLifecycleTest, RealGuidanceContinuationRestartsGrammarAtFirstToken) {
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
  params->search.max_length = static_cast<int32_t>(Prompt().size()) + 8;
  params->SetGuidance("regex", "!", false);
  auto request = std::make_shared<Request>(params);
  request->AddTokens(Prompt());
  guidance_engine.engine->AddRequest(request);

  guidance_engine.executor->SetForcedToken(invalid_tokens.front());
  ASSERT_EQ(guidance_engine.engine->Step(), request);
  EXPECT_EQ(request->UnseenToken(), expected_tokens.front());

  // Once the one-token regex is complete, the fresh mask allows only EOS and completes the turn.
  guidance_engine.executor->SetForcedToken(invalid_tokens.front());
  ASSERT_EQ(guidance_engine.engine->Step(), request);
  ASSERT_TRUE(request->IsTurnComplete());

  request->Continue(std::vector<int32_t>{6});

  // The new turn must restart at the grammar's first token, not remain at the completed EOS-only cursor.
  guidance_engine.executor->SetForcedToken(invalid_tokens.front());
  ASSERT_EQ(guidance_engine.engine->Step(), request);
  EXPECT_EQ(request->UnseenToken(), expected_tokens.front());
}
#endif

}  // namespace
}  // namespace test
}  // namespace Generators
