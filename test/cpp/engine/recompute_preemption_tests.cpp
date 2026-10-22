// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engine/decoders/varlen_decoder_io.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"

namespace Generators::test {
namespace {

constexpr std::array<int32_t, 3> kPrompt{2, 3, 4};

std::shared_ptr<Model> PreemptionModel(
    size_t blocks = 1, bool enabled = true, size_t chunk = 0,
    size_t quantum = 1, size_t victims = 1, size_t block_size = 64) {
  auto model = LoadDummyDecoderModelWithChunking(chunk);
  Config::Engine::DynamicBatching batching;
  batching.block_size = block_size;
  batching.num_blocks = blocks;
  batching.max_batch_size = 4;
  batching.max_scheduled_tokens = chunk == 0 ? 16 : chunk;
  batching.enable_recompute_preemption = enabled;
  batching.min_decode_steps_before_preemption = quantum;
  batching.max_preemptions_per_step = victims;
  model->config_->engine.dynamic_batching = batching;
  return model;
}

std::shared_ptr<Request> Queue(
    const std::shared_ptr<Engine>& engine, std::span<const int32_t> prompt = kPrompt,
    size_t generated = 12) {
  TurnOptions options;
  options.max_generated_tokens = generated;
  return CreateRequestWithPrompt(engine, prompt, options);
}

void ModelSteps(CompositeDoublesEngine& test, int count) {
  const int target = test.executor->decode_calls + count;
  for (int guard = 0; test.executor->decode_calls < target && guard < 1000; ++guard) {
    RunOne(*test.engine);
  }
  ASSERT_EQ(test.executor->decode_calls, target);
}

void CheckState(const CompositeDoublesEngine& test,
                std::span<const std::shared_ptr<Request>> requests) {
  std::vector<RequestStateSnapshot> snapshots;
  for (const auto& request : requests) {
    snapshots.push_back(request->Snapshot());
  }
  const auto violations = ValidateInvariants(test.cache->Snapshot(), snapshots);
  for (const auto& violation : violations) {
    ADD_FAILURE() << violation.message;
  }
}

void Finish(Engine& engine) {
  for (int guard = 0; engine.HasPendingRequests() && guard < 2000; ++guard) {
    RunOne(engine);
  }
  ASSERT_FALSE(engine.HasPendingRequests());
}

RequestStepPlan Admission(const std::shared_ptr<Request>& request) {
  RequestStepPlan entry;
  entry.request = request;
  entry.request_id = request.get();
  entry.sequence_length_before = request->CurrentSequenceLength();
  entry.unprocessed_token_count = 1;
  entry.target_cache_slots = 1;
  entry.whole_sequence_cache_slots = static_cast<size_t>(request->CurrentSequenceLength());
  entry.newly_admitted = true;
  entry.is_prefill = true;
  return entry;
}

struct ScopedCacheDeviceOverride {
  ScopedCacheDeviceOverride(Model& model, DeviceInterface& device)
      : model_{model}, original_{model.p_device_kvcache_} {
    model_.p_device_kvcache_ = &device;
  }

  ~ScopedCacheDeviceOverride() {
    model_.p_device_kvcache_ = original_;
  }

  Model& model_;
  DeviceInterface* original_;
};

TEST(RecomputePreemptionConfigTest, DefaultsOffWithBoundedStepAndServiceQuantum) {
  Config config{fs::path{MODEL_PATH "engine/dummy-decoder"},
                R"({"engine":{"dynamic_batching":{"num_blocks":4}}})"};
  const auto& batching = *config.engine.dynamic_batching;
  EXPECT_FALSE(batching.enable_recompute_preemption);
  EXPECT_EQ(batching.max_preemptions_per_step, 1u);
  EXPECT_EQ(batching.min_decode_steps_before_preemption, 8u);
}

TEST(RecomputePreemptionConfigTest, RejectsInvalidLimitsAndBooleanTypes) {
  for (const char* name : {"max_preemptions_per_step", "min_decode_steps_before_preemption"}) {
    for (const char* value : {"0", "-1", "1.5", "4000000000", "true"}) {
      const std::string overlay =
          std::string{R"({"engine":{"dynamic_batching":{")"} + name + "\":" + value + "}}}";
      EXPECT_THROW((Config{fs::path{MODEL_PATH "engine/dummy-decoder"}, overlay}), std::runtime_error)
          << name << "=" << value;
    }
    const std::string valid =
        std::string{R"({"engine":{"dynamic_batching":{")"} + name + "\":3}}}";
    EXPECT_NO_THROW((Config{fs::path{MODEL_PATH "engine/dummy-decoder"}, valid}));
  }
  EXPECT_THROW((Config{fs::path{MODEL_PATH "engine/dummy-decoder"},
                       R"({"engine":{"dynamic_batching":{"enable_recompute_preemption":1}}})"}),
               std::runtime_error);
}

TEST(RecomputePreemptionConfigTest, RuntimeProfileSupportsEnableAndExplicitDisable) {
  Config config{fs::path{MODEL_PATH "engine/dummy-decoder"}, R"({
    "engine":{"dynamic_batching":{"num_blocks":4}},
    "runtime_profiles":[{"id":"preempt","eligibility":{"minimum_total_device_memory_bytes":1},
      "overlay":{"engine":{"dynamic_batching":{"enable_recompute_preemption":true,
        "max_preemptions_per_step":2,"min_decode_steps_before_preemption":3}}}}]
  })"};
  ApplyRuntimeProfile(config, uint64_t{100});
  EXPECT_TRUE(config.engine.dynamic_batching->enable_recompute_preemption);
  EXPECT_EQ(config.engine.dynamic_batching->max_preemptions_per_step, 2u);
  EXPECT_EQ(config.engine.dynamic_batching->min_decode_steps_before_preemption, 3u);
  Config disabled{fs::path{MODEL_PATH "engine/dummy-decoder"}, R"({
    "engine":{"dynamic_batching":{"num_blocks":4,"enable_recompute_preemption":true}},
    "runtime_profiles":[{"id":"wait","eligibility":{"minimum_total_device_memory_bytes":1},
      "overlay":{"engine":{"dynamic_batching":{"enable_recompute_preemption":false}}}}]
  })"};
  ApplyRuntimeProfile(disabled, uint64_t{100});
  EXPECT_FALSE(disabled.engine.dynamic_batching->enable_recompute_preemption);
}

TEST(RecomputePreemptionTest, DefaultOffPreservesWaitOnlyAdmission) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(1, false), 5);
  auto resident = Queue(test.engine);
  ModelSteps(test, 2);
  auto waiting = Queue(test.engine);
  ModelSteps(test, 1);
  EXPECT_EQ(waiting->Status(), RequestStatus::Assigned);
  EXPECT_TRUE(test.cache->IsResident(resident));
  EXPECT_EQ(resident->PreemptionCount(), 0u);
  CheckState(test, std::array{resident, waiting});
}

TEST(RecomputePreemptionTest, SuspensionPreservesTurnPolicyAndCompleteLogicalSequence) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(), 5);
  auto resident = Queue(test.engine);
  ModelSteps(test, 2);
  const auto before = resident->Snapshot();
  const std::vector<int32_t> tokens{resident->TokensCpu().begin(), resident->TokensCpu().end()};
  const auto generated = resident->TurnGeneratedTokens();
  auto waiting = Queue(test.engine);
  ModelSteps(test, 1);
  EXPECT_EQ(resident->Status(), RequestStatus::Suspended);
  EXPECT_FALSE(test.cache->IsResident(resident));
  EXPECT_EQ(resident->ProcessedSequenceLength(), 0);
  EXPECT_EQ(resident->CurrentTurnId(), before.current_turn_id);
  EXPECT_EQ(resident->TurnGeneratedTokens(), generated);
  EXPECT_EQ(resident->TurnPromptTokens(), kPrompt.size());
  EXPECT_EQ(resident->TurnPolicy().max_generated_tokens, 12u);
  EXPECT_EQ(std::vector<int32_t>(resident->TokensCpu().begin(), resident->TokensCpu().end()), tokens);
  EXPECT_EQ(resident->PreemptionCount(), 1u);
  EXPECT_TRUE(test.cache->IsResident(waiting));
  EXPECT_THROW(resident->BeginTurn(kPrompt), std::runtime_error);
  CheckState(test, std::array{resident, waiting});
}

TEST(RecomputePreemptionTest, RebuildIsChunkBoundedAndDoesNotResampleReplayedTokens) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(1, true, 2), 5);
  const std::array<int32_t, 5> prompt{2, 3, 4, 5, 6};
  auto victim = Queue(test.engine, prompt);
  ModelSteps(test, 4);
  const auto generated = victim->TurnGeneratedTokens();
  const auto processed = victim->ProcessedSequenceLength();
  auto beneficiary = Queue(test.engine);
  ModelSteps(test, 1);
  ASSERT_EQ(victim->Status(), RequestStatus::Suspended);
  size_t replay_steps = 0;
  test.executor->SetExecutionCallback([&](ExecutionContext& context) {
    for (const auto& entry : context.plan->requests) {
      if (entry.request == victim && entry.is_prefill && victim->PreemptionCount() != 0) {
        EXPECT_LE(entry.unprocessed_token_count, 2u);
        ++replay_steps;
      }
    }
  });
  for (int guard = 0; victim->TurnGeneratedTokens() == generated && guard < 100; ++guard) {
    RunOne(*test.engine);
  }
  EXPECT_GE(replay_steps, 3u);
  EXPECT_EQ(victim->TurnGeneratedTokens(), generated + 1);
  EXPECT_EQ(victim->RecomputedTokenCount(), static_cast<uint64_t>(processed));
  EXPECT_FALSE(victim->NeedsCacheReplay());
  CheckState(test, std::array{victim, beneficiary});
}

TEST(RecomputePreemptionTest, ServiceQuantumPreventsImmediatePingPong) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(1, true, 0, 3), 5);
  auto resident = Queue(test.engine);
  ModelSteps(test, 2);
  auto waiting = Queue(test.engine);
  ModelSteps(test, 2);
  EXPECT_EQ(resident->PreemptionCount(), 0u);
  ModelSteps(test, 1);
  ASSERT_EQ(resident->Status(), RequestStatus::Suspended);
  ModelSteps(test, 3);
  EXPECT_TRUE(test.cache->IsResident(waiting));
  EXPECT_EQ(waiting->PreemptionCount(), 0u);
  CheckState(test, std::array{resident, waiting});
}

TEST(RecomputePreemptionTest, VictimBoundDoesNotDiscardInsufficientCapacity) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(2, true, 0, 1, 1), 5);
  auto first = Queue(test.engine);
  auto second = Queue(test.engine);
  ModelSteps(test, 2);
  const std::vector<int32_t> prompt(70, 9);
  auto waiting = Queue(test.engine, prompt, 1);
  ModelSteps(test, 1);
  EXPECT_EQ(first->PreemptionCount(), 0u);
  EXPECT_EQ(second->PreemptionCount(), 0u);
  EXPECT_EQ(waiting->Status(), RequestStatus::Assigned);
  CheckState(test, std::array{first, second, waiting});
}

TEST(RecomputePreemptionTest, MultipleVictimsAreReleasedOnlyWhenTheWholeAdmissionFits) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(2, true, 0, 1, 2), 5);
  auto first = Queue(test.engine);
  auto second = Queue(test.engine);
  ModelSteps(test, 2);
  const std::vector<int32_t> prompt(70, 9);
  auto waiting = Queue(test.engine, prompt, 1);
  ModelSteps(test, 1);
  EXPECT_EQ(first->PreemptionCount(), 1u);
  EXPECT_EQ(second->PreemptionCount(), 1u);
  EXPECT_TRUE(test.cache->IsResident(waiting));
  CheckState(test, std::array{first, second, waiting});
}

TEST(RecomputePreemptionTest, TokenBudgetPressureDoesNotRequireKVReclamation) {
  auto model = PreemptionModel(4);
  model->config_->engine.dynamic_batching->max_scheduled_tokens = 1;
  auto test = MakeCompositeDoublesEngine(model, 5);
  auto first = Queue(test.engine);
  ModelSteps(test, 4);
  auto second = Queue(test.engine);
  ModelSteps(test, 4);
  EXPECT_GT(second->ProcessedSequenceLength(), 0);
  EXPECT_EQ(first->PreemptionCount(), 0u);
  EXPECT_EQ(second->PreemptionCount(), 0u);
  CheckState(test, std::array{first, second});
}

TEST(RecomputePreemptionTest, StalledResidentsCanProgressBeforePayingTheFullQuantum) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(6, true, 0, 8, 1, 1), 5);
  const std::array<int32_t, 2> prompt{2, 3};
  auto first = Queue(test.engine, prompt, 3);
  auto second = Queue(test.engine, prompt, 3);
  ModelSteps(test, 2);
  ASSERT_EQ(test.cache->Snapshot().free_blocks, 0u);
  ModelSteps(test, 1);
  EXPECT_EQ(first->PreemptionCount() + second->PreemptionCount(), 1u);
  CheckState(test, std::array{first, second});
  Finish(*test.engine);
  EXPECT_TRUE(first->IsTurnComplete());
  EXPECT_TRUE(second->IsTurnComplete());
}

TEST(RecomputePreemptionTest, IneffectiveIdleVictimDoesNotHideAFeasibleSingleton) {
  auto model = PreemptionModel(3, true, 0, 8, 1, 4);
  model->config_->engine.dynamic_batching->prefix_caching = true;
  auto test = MakeCompositeDoublesEngine(model, 5);
  const std::array<int32_t, 4> prompt{2, 3, 4, 5};
  auto idle = Queue(test.engine, prompt, 1);
  ModelSteps(test, 1);
  ASSERT_TRUE(idle->IsTurnComplete());
  ASSERT_EQ(idle->ProcessedSequenceLength(), 4);
  auto newer = Queue(test.engine, idle->TokensCpu(), 6);
  ModelSteps(test, 1);
  ASSERT_EQ(newer->TurnCachedPromptTokens(), 4u);
  const std::array<int32_t, 4> other_prompt{10, 11, 12, 13};
  auto older = Queue(test.engine, other_prompt, 4);
  ModelSteps(test, 3);
  ASSERT_EQ(newer->ProcessedSequenceLength(), 8);
  ASSERT_EQ(newer->CurrentSequenceLength(), 9);
  ASSERT_EQ(older->ProcessedSequenceLength(), 4);
  ASSERT_EQ(older->CurrentSequenceLength(), 5);
  ASSERT_EQ(test.cache->Snapshot().free_blocks, 0u);
  ASSERT_EQ(idle->PreemptionCount() + newer->PreemptionCount() + older->PreemptionCount(), 0u);

  const size_t generated = older->TurnGeneratedTokens();
  const int decode_calls = test.executor->decode_calls;
  std::array<EngineEvent, 8> events;
  const size_t count = test.engine->Run(events);

  EXPECT_EQ(test.executor->decode_calls, decode_calls + 1);
  EXPECT_EQ(older->TurnGeneratedTokens(), generated + 1);
  EXPECT_EQ(older->ProcessedSequenceLength(), 5);
  EXPECT_EQ(newer->Status(), RequestStatus::Suspended);
  EXPECT_EQ(newer->PreemptionCount(), 1u);
  EXPECT_EQ(idle->PreemptionCount(), 0u);
  EXPECT_TRUE(idle->IsTurnComplete());
  EXPECT_TRUE(test.cache->IsResident(idle));
  EXPECT_EQ(older->PreemptionCount(), 0u);
  for (const auto& event : std::span{events}.first(count)) {
    EXPECT_EQ(event.flags & (EngineEventFlagCapacityBlocked | EngineEventFlagFailed), 0u);
  }
  CheckState(test, std::array{idle, newer, older});
}

TEST(RecomputePreemptionTest, AlternativeVictimSetsPreserveJointSharedOwnerReclamation) {
  auto model = PreemptionModel(3, true, 0, 1, 2, 4);
  model->config_->engine.dynamic_batching->prefix_caching = true;
  auto test = MakeCompositeDoublesEngine(model, 5);
  const std::array<int32_t, 4> prompt{2, 3, 4, 5};
  auto first_idle = Queue(test.engine, prompt, 1);
  ModelSteps(test, 1);
  auto shared_owner = Queue(test.engine, first_idle->TokensCpu(), 4);
  ModelSteps(test, 4);
  ASSERT_TRUE(shared_owner->IsTurnComplete());
  ASSERT_EQ(shared_owner->ProcessedSequenceLength(), 8);
  auto active = Queue(test.engine, shared_owner->TokensCpu(), 4);
  ModelSteps(test, 2);
  ASSERT_EQ(active->TurnCachedPromptTokens(), 8u);
  ASSERT_EQ(test.cache->Snapshot().free_blocks, 0u);
  const std::array<int32_t, 5> other_prompt{10, 11, 12, 13, 14};
  auto waiting = Queue(test.engine, other_prompt, 1);

  StepPlan shared_only;
  shared_only.requests.push_back(Admission(waiting));
  EXPECT_FALSE(test.cache->PlanStepAfterReleasing(shared_only, std::array{shared_owner}).executable);
  StepPlan active_only;
  active_only.requests.push_back(Admission(waiting));
  EXPECT_FALSE(test.cache->PlanStepAfterReleasing(active_only, std::array{active}).executable);
  StepPlan jointly;
  jointly.requests.push_back(Admission(waiting));
  ASSERT_TRUE(test.cache->PlanStepAfterReleasing(jointly, std::array{shared_owner, active}).executable);

  const int decode_calls = test.executor->decode_calls;
  std::array<EngineEvent, 8> events;
  const size_t count = test.engine->Run(events);

  EXPECT_EQ(test.executor->decode_calls, decode_calls + 1);
  EXPECT_TRUE(waiting->IsTurnComplete());
  EXPECT_EQ(waiting->TurnGeneratedTokens(), 1u);
  EXPECT_EQ(first_idle->PreemptionCount(), 0u);
  EXPECT_TRUE(test.cache->IsResident(first_idle));
  EXPECT_EQ(shared_owner->PreemptionCount(), 1u);
  EXPECT_TRUE(shared_owner->IsTurnComplete());
  EXPECT_TRUE(shared_owner->NeedsCacheReplay());
  EXPECT_EQ(active->PreemptionCount(), 1u);
  EXPECT_EQ(active->Status(), RequestStatus::Suspended);
  for (const auto& event : std::span{events}.first(count)) {
    EXPECT_EQ(event.flags & (EngineEventFlagCapacityBlocked | EngineEventFlagFailed), 0u);
  }
  CheckState(test, std::array{first_idle, shared_owner, active, waiting});
}

struct CountingPreemptionProbeCache final : PagedCacheManager {
  using PagedCacheManager::PagedCacheManager;

  StepPlanningResult PlanStepAfterReleasing(
      StepPlan& plan, std::span<const std::shared_ptr<Request>> requests) const override {
    ++probe_calls;
    largest_release_set = std::max(largest_release_set, requests.size());
    return PagedCacheManager::PlanStepAfterReleasing(plan, requests);
  }

  mutable size_t probe_calls{};
  mutable size_t largest_release_set{};
};

TEST(RecomputePreemptionTest, InfeasibleAlternativeSearchHasAResidentBoundedProbeCount) {
  constexpr size_t kResidents = 4;
  constexpr size_t kBlockSize = 16;
  for (size_t limit : {1u, 2u, 3u}) {
    SCOPED_TRACE(limit);
    auto model = PreemptionModel(kResidents, true, 0, 1, limit, kBlockSize);
    auto cache = std::make_shared<CountingPreemptionProbeCache>(model);
    auto executor = std::make_unique<RecordingModelExecutor>(model, cache, 5);
    auto* recording = executor.get();
    auto engine = std::make_shared<Engine>(
        model, EngineDependencies{cache, Scheduler::Create(model, cache), std::move(executor)});
    CompositeDoublesEngine test{engine, cache.get(), recording};
    std::vector<std::shared_ptr<Request>> residents;
    for (size_t index = 0; index < kResidents; ++index) {
      residents.push_back(Queue(engine));
    }
    std::array<EngineEvent, 8> events;
    for (size_t step = 0; step < 2; ++step) {
      ASSERT_EQ(engine->Run(events), kResidents);
    }
    ASSERT_EQ(cache->Snapshot().free_blocks, 0u);
    ASSERT_EQ(cache->probe_calls, 0u);
    const std::vector<int32_t> prompt((kResidents - 1) * kBlockSize + 1, 9);
    auto waiting = Queue(engine, prompt, 1);

    EXPECT_EQ(engine->Run(events), kResidents);

    EXPECT_EQ(cache->probe_calls, kResidents * limit);
    EXPECT_EQ(cache->largest_release_set, limit);
    EXPECT_EQ(waiting->Status(), RequestStatus::Assigned);
    for (const auto& request : residents) {
      EXPECT_EQ(request->PreemptionCount(), 0u);
      EXPECT_TRUE(cache->IsResident(request));
    }
    residents.push_back(waiting);
    CheckState(test, residents);
  }
}

TEST(RecomputePreemptionTest, NewArrivalsDoNotStarveASuspendedTurn) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(1, true, 0, 2), 5);
  auto victim = Queue(test.engine, kPrompt, 12);
  ModelSteps(test, 3);
  auto beneficiary = Queue(test.engine, kPrompt, 12);
  ModelSteps(test, 1);
  ASSERT_EQ(victim->Status(), RequestStatus::Suspended);
  const size_t generated = victim->TurnGeneratedTokens();
  std::vector<std::shared_ptr<Request>> requests{victim, beneficiary};
  for (int index = 0; index < 12; ++index) {
    requests.push_back(Queue(test.engine, kPrompt, 1));
    ModelSteps(test, 1);
  }
  EXPECT_GT(victim->TurnGeneratedTokens(), generated);
  Finish(*test.engine);
  for (const auto& request : requests) {
    EXPECT_TRUE(request->IsTurnComplete());
  }
  CheckState(test, requests);
}

class PreemptionRollbackTest : public ::testing::TestWithParam<ScriptedExecutionFailure> {};

TEST_P(PreemptionRollbackTest, FailureRestoresTheBatchButRetainsTheSuspensionBoundary) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(), 5);
  auto victim = Queue(test.engine);
  ModelSteps(test, 2);
  auto beneficiary = Queue(test.engine);
  const auto length = victim->CurrentSequenceLength();
  const auto generated = victim->TurnGeneratedTokens();
  test.executor->SetNextFailure(GetParam());
  const auto event = RunOne(*test.engine);
  if (GetParam() == ScriptedExecutionFailure::CapacityExceeded) {
    EXPECT_EQ(event.flags, EngineEventFlagCapacityBlocked);
    EXPECT_EQ(event.error_code, EngineErrorCode::ExecutionCapacityExceeded);
  } else {
    EXPECT_EQ(event.flags, EngineEventFlagRetryable);
    EXPECT_EQ(event.error_code, EngineErrorCode::RetryableExecution);
  }
  EXPECT_EQ(victim->Status(), RequestStatus::Suspended);
  EXPECT_EQ(victim->CurrentSequenceLength(), length);
  EXPECT_EQ(victim->TurnGeneratedTokens(), generated);
  EXPECT_EQ(victim->RecomputedTokenCount(), 0u);
  EXPECT_EQ(beneficiary->Status(), RequestStatus::Assigned);
  EXPECT_EQ(beneficiary->ProcessedSequenceLength(), 0);
  EXPECT_EQ(test.cache->Snapshot().free_blocks, 1u);
  CheckState(test, std::array{victim, beneficiary});
  ModelSteps(test, 1);
  EXPECT_TRUE(test.cache->IsResident(beneficiary));
  Finish(*test.engine);
}

INSTANTIATE_TEST_SUITE_P(
    RecoverableFailures, PreemptionRollbackTest,
    ::testing::Values(ScriptedExecutionFailure::RetryableBeforeExecution,
                      ScriptedExecutionFailure::RetryableDuringExecution,
                      ScriptedExecutionFailure::CapacityExceeded,
                      ScriptedExecutionFailure::PostProcessing));

TEST(RecomputePreemptionTest, CancelSuspendedTurnThenContinueFromItsRetainedTokens) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(1, true, 2), 5);
  auto victim = Queue(test.engine);
  ModelSteps(test, 3);
  auto beneficiary = Queue(test.engine);
  ModelSteps(test, 1);
  ASSERT_EQ(victim->Status(), RequestStatus::Suspended);
  const auto old_length = victim->CurrentSequenceLength();
  const auto generated = victim->TurnGeneratedTokens();
  const auto turn = victim->CurrentTurnId();
  ASSERT_TRUE(victim->Cancel(turn));
  const auto event = RunOne(*test.engine);
  EXPECT_EQ(event.request, victim);
  EXPECT_EQ(event.finish_reason, GenerationFinishReason::Canceled);
  EXPECT_EQ(event.usage.generated_tokens, generated);
  EXPECT_TRUE(victim->NeedsCacheReplay());
  const std::array<int32_t, 2> continuation{8, 9};
  const auto new_turn = victim->BeginTurn(continuation, 3);
  EXPECT_EQ(new_turn, turn + 1);
  EXPECT_EQ(victim->CurrentSequenceLength(), old_length + 2);
  Finish(*test.engine);
  EXPECT_EQ(victim->TurnPromptTokens(), 2u);
  EXPECT_EQ(victim->TurnGeneratedTokens(), 3u);
  EXPECT_EQ(victim->CurrentSequenceLength(), old_length + 5);
  CheckState(test, std::array{victim, beneficiary});
}

TEST(RecomputePreemptionTest, CancelPartialReplayThenEvictPreservesDiscardedKVBoundary) {
  auto model = PreemptionModel(1, true, 4);
  model->config_->engine.dynamic_batching->prefix_caching = false;
  auto test = MakeCompositeDoublesEngine(model, 5);
  auto victim = Queue(test.engine);
  ModelSteps(test, 8);
  ASSERT_EQ(victim->ProcessedSequenceLength(), 10);
  ASSERT_EQ(victim->CurrentSequenceLength(), 11);
  const auto generated = victim->TurnGeneratedTokens();
  const auto turn = victim->CurrentTurnId();
  auto first_beneficiary = Queue(test.engine, kPrompt, 1);
  ModelSteps(test, 1);
  ASSERT_EQ(victim->Status(), RequestStatus::Suspended);
  ASSERT_EQ(victim->PreemptionCount(), 1u);
  ASSERT_TRUE(first_beneficiary->IsTurnComplete());
  first_beneficiary->Close();

  ModelSteps(test, 1);
  ASSERT_EQ(victim->ProcessedSequenceLength(), 4);
  ASSERT_EQ(victim->RecomputedTokenCount(), 4u);
  ASSERT_TRUE(victim->NeedsCacheReplay());
  EXPECT_EQ(victim->TurnGeneratedTokens(), generated);
  ASSERT_TRUE(victim->Cancel(turn));
  const auto canceled = RunOne(*test.engine);
  ASSERT_EQ(canceled.request, victim);
  EXPECT_EQ(canceled.finish_reason, GenerationFinishReason::Canceled);
  EXPECT_EQ(canceled.usage.generated_tokens, generated);
  ASSERT_TRUE(victim->IsTurnComplete());
  ASSERT_TRUE(test.cache->IsResident(victim));

  auto second_beneficiary = Queue(test.engine, kPrompt, 1);
  ModelSteps(test, 1);
  ASSERT_EQ(victim->PreemptionCount(), 2u);
  EXPECT_TRUE(victim->IsTurnComplete());
  EXPECT_EQ(victim->FinishReason(), GenerationFinishReason::Canceled);
  EXPECT_FALSE(test.cache->IsResident(victim));
  EXPECT_EQ(victim->ProcessedSequenceLength(), 0);
  EXPECT_EQ(victim->RecomputedTokenCount(), 4u);
  EXPECT_TRUE(victim->NeedsCacheReplay());
  ASSERT_TRUE(second_beneficiary->IsTurnComplete());
  second_beneficiary->Close();

  const std::array<int32_t, 2> continuation{8, 9};
  EXPECT_EQ(victim->BeginTurn(continuation, 1), turn + 1);
  Finish(*test.engine);

  EXPECT_TRUE(victim->IsTurnComplete());
  EXPECT_FALSE(victim->NeedsCacheReplay());
  EXPECT_EQ(victim->PreemptionCount(), 2u);
  EXPECT_EQ(victim->RecomputedTokenCount(), 14u);
  EXPECT_EQ(victim->TurnPromptTokens(), continuation.size());
  EXPECT_EQ(victim->TurnCachedPromptTokens(), 0u);
  EXPECT_EQ(victim->TurnGeneratedTokens(), 1u);
  EXPECT_EQ(victim->ProcessedSequenceLength(), 13);
  EXPECT_EQ(victim->CurrentSequenceLength(), 14);
  CheckState(test, std::array{victim});
}

TEST(RecomputePreemptionTest, CancelSuspendedTurnThenRewindAndRestart) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(), 5);
  auto victim = Queue(test.engine);
  ModelSteps(test, 2);
  auto beneficiary = Queue(test.engine);
  ModelSteps(test, 1);
  ASSERT_EQ(victim->Status(), RequestStatus::Suspended);
  const auto turn = victim->CurrentTurnId();
  ASSERT_TRUE(victim->Cancel(turn));
  RunOne(*test.engine);
  EXPECT_NO_THROW(victim->RewindToStartOfTurn(turn));
  EXPECT_EQ(victim->CurrentSequenceLength(), 0);
  EXPECT_EQ(victim->BeginTurn(kPrompt, 2), turn + 1);
  Finish(*test.engine);
  EXPECT_EQ(victim->CurrentSequenceLength(), 5);
  CheckState(test, std::array{victim, beneficiary});
}

TEST(RecomputePreemptionTest, CloseWhileSuspendedDoesNotReleaseAnotherRequestsBlocks) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(), 5);
  auto victim = Queue(test.engine);
  ModelSteps(test, 2);
  auto beneficiary = Queue(test.engine);
  ModelSteps(test, 1);
  ASSERT_EQ(victim->Status(), RequestStatus::Suspended);
  victim->Close();
  EXPECT_EQ(victim->Status(), RequestStatus::Closed);
  EXPECT_TRUE(test.cache->IsResident(beneficiary));
  Finish(*test.engine);
  CheckState(test, std::array{beneficiary});
}

TEST(RecomputePreemptionTest, CompletedResidentCanBeEvictedAndContinueLater) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(), 5);
  auto idle = Queue(test.engine, kPrompt, 1);
  ModelSteps(test, 1);
  ASSERT_TRUE(idle->IsTurnComplete());
  const auto length = idle->CurrentSequenceLength();
  auto beneficiary = Queue(test.engine, kPrompt, 1);
  ModelSteps(test, 1);
  EXPECT_TRUE(idle->IsTurnComplete());
  EXPECT_FALSE(test.cache->IsResident(idle));
  EXPECT_TRUE(idle->NeedsCacheReplay());
  EXPECT_EQ(idle->FinishReason(), GenerationFinishReason::TurnLimit);
  EXPECT_EQ(idle->CurrentSequenceLength(), length);
  EXPECT_EQ(idle->BeginTurn(kPrompt, 1), 2u);
  Finish(*test.engine);
  EXPECT_EQ(idle->CurrentSequenceLength(), length + 4);
  CheckState(test, std::array{idle, beneficiary});
}

TEST(RecomputePreemptionTest, SharedPrefixReferencesAreNotCountedAsIndependentlyFreeBlocks) {
  auto model = PreemptionModel(4, false, 0, 1, 2, 4);
  auto test = MakeCompositeDoublesEngine(model, 5);
  const std::array<int32_t, 5> first_prompt{2, 3, 4, 5, 6};
  const std::array<int32_t, 5> second_prompt{2, 3, 4, 5, 7};
  auto first = Queue(test.engine, first_prompt, 6);
  ModelSteps(test, 1);
  auto second = Queue(test.engine, second_prompt, 6);
  ModelSteps(test, 1);
  const auto before = test.cache->Snapshot();
  ASSERT_EQ(before.free_blocks, 1u);
  ASSERT_EQ(before.requests.size(), 2u);
  ASSERT_EQ(before.requests[0].block_ids[0], before.requests[1].block_ids[0]);
  const std::array<int32_t, 10> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  auto waiting = Queue(test.engine, prompt, 1);
  StepPlan one;
  one.requests.push_back(Admission(waiting));
  EXPECT_FALSE(test.cache->PlanStepAfterReleasing(one, std::array{first}).executable);
  StepPlan both;
  both.requests.push_back(Admission(waiting));
  EXPECT_TRUE(test.cache->PlanStepAfterReleasing(both, std::array{first, second}).executable);
  EXPECT_EQ(test.cache->Snapshot().free_blocks, before.free_blocks);
  EXPECT_TRUE(test.cache->IsResident(first));
  EXPECT_TRUE(test.cache->IsResident(second));
  CheckState(test, std::array{first, second, waiting});
}

TEST(RecomputePreemptionTest, ReleasingASharedPrefixOwnerPreservesTheOtherOwner) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(4, true, 0, 1, 1, 4), 5);
  const std::array<int32_t, 5> first_prompt{2, 3, 4, 5, 6};
  const std::array<int32_t, 5> second_prompt{2, 3, 4, 5, 7};
  auto first = Queue(test.engine, first_prompt, 6);
  ModelSteps(test, 1);
  auto second = Queue(test.engine, second_prompt, 6);
  ModelSteps(test, 1);
  const auto before = test.cache->Snapshot();
  const size_t shared_block = before.requests[0].block_ids[0];
  const std::array<int32_t, 6> prompt{10, 11, 12, 13, 14, 15};
  auto waiting = Queue(test.engine, prompt, 1);
  ModelSteps(test, 1);
  ASSERT_EQ(first->Status(), RequestStatus::Suspended);
  ASSERT_TRUE(test.cache->IsResident(second));
  const auto after = test.cache->Snapshot();
  const auto owner = std::find_if(after.requests.begin(), after.requests.end(), [&](const auto& request) {
    return request.request_id == second.get();
  });
  ASSERT_NE(owner, after.requests.end());
  EXPECT_EQ(owner->block_ids.front(), shared_block);
  const auto block = std::find_if(after.blocks.begin(), after.blocks.end(), [&](const auto& value) {
    return value.block_id == shared_block;
  });
  ASSERT_NE(block, after.blocks.end());
  EXPECT_EQ(block->ref_count, 2u);
  CheckState(test, std::array{first, second, waiting});
}

std::vector<std::vector<int32_t>> SampledStreams(bool preemption) {
  auto model = PreemptionModel(preemption ? 1 : 4, preemption, 2, 2);
  auto test = MakeCompositeDoublesEngine(model, 5);
  test.executor->SetSamplingCandidateTokens({5, 6, 7});
  std::vector<std::shared_ptr<Request>> requests;
  for (uint64_t seed : {1234u, 5678u}) {
    TurnOptions options;
    options.max_generated_tokens = 12;
    options.do_sample = true;
    options.top_k = 3;
    options.seed = seed;
    requests.push_back(CreateRequestWithPrompt(test.engine, kPrompt, options));
  }
  std::vector<std::vector<int32_t>> streams(requests.size());
  for (int guard = 0; test.engine->HasPendingRequests() && guard < 2000; ++guard) {
    const auto event = RunOne(*test.engine);
    if ((event.flags & EngineEventFlagToken) != 0) {
      const auto request = std::find(requests.begin(), requests.end(), event.request);
      streams[static_cast<size_t>(request - requests.begin())].push_back(event.token);
    }
  }
  EXPECT_FALSE(test.engine->HasPendingRequests());
  EXPECT_EQ(requests[0]->PreemptionCount() + requests[1]->PreemptionCount() > 0, preemption);
  CheckState(test, requests);
  return streams;
}

TEST(RecomputePreemptionTest, ReplayingChunksPreservesIndependentSeededRandomStreams) {
  const auto reference = SampledStreams(false);
  const auto preempted = SampledStreams(true);
  ASSERT_EQ(reference.size(), 2u);
  ASSERT_EQ(reference[0].size(), 12u);
  ASSERT_EQ(reference[1].size(), 12u);
  EXPECT_NE(reference[0], reference[1]);
  EXPECT_EQ(preempted, reference);
}

TEST(RecomputePreemptionTest, RejectsStaticAndNonReclaimingCacheDependencies) {
  auto model = PreemptionModel();
  EXPECT_THROW(MakeDoublesEngine(model, 4, 5), std::runtime_error);
  EXPECT_THROW(MakeStaticDoublesEngine(model, 4, 5), std::runtime_error);
}

TEST(RecomputePreemptionTest, RejectsFixedRingAndAuxiliaryStateBeforeExecution) {
  auto hybrid = LoadSyntheticCompositeModel();
  hybrid->config_->engine.dynamic_batching->enable_recompute_preemption = true;
  EXPECT_THROW((void)Engine{hybrid}, std::runtime_error);
  auto ring = PreemptionModel();
  ring->config_->model.decoder.sliding_window.emplace();
  EXPECT_THROW((void)Engine{ring}, std::runtime_error);
  auto mtp = PreemptionModel();
  mtp->config_->model.mtp.filename = "not-loaded.onnx";
  EXPECT_THROW((void)Engine{mtp}, std::runtime_error);
  auto drafter = PreemptionModel();
  drafter->config_->model.dflash2.filename = "not-loaded.onnx";
  EXPECT_THROW((void)Engine{drafter}, std::runtime_error);
}

TEST(RecomputePreemptionTest, AutomaticGraphCaptureReservationAllowsPreemption) {
  auto model = LoadSyntheticPagedModel();
  auto& batching = *model->config_->engine.dynamic_batching;
  batching.num_blocks.reset();
  batching.gpu_utilization_factor = 0.5f;
  batching.enable_recompute_preemption = true;
  SetProviderOption(*model->config_, "cuda", "enable_cuda_graph", "1");

  const size_t graph_reserved_bytes = VarlenGraphBufferBytes(
      *model, PackedPositionIdPlanes(*model), kMaxDraftTokensPerStep + 1);
  const size_t block_bytes = PagedKeyValueCacheBytesPerBlock(model);
  ASSERT_GT(graph_reserved_bytes, 0u);
  ASSERT_GT(block_bytes, 0u);

  size_t available_bytes =
      (graph_reserved_bytes + 8 * block_bytes) * 3;
  size_t expected_blocks = 0;
  size_t blocks_without_reservation = 0;
  for (size_t offset = 0; offset <= 4 * block_bytes; ++offset) {
    expected_blocks = ComputePagedBlockCapacityFromBytes(
        available_bytes, *batching.gpu_utilization_factor,
        graph_reserved_bytes, block_bytes, 0);
    blocks_without_reservation = ComputePagedBlockCapacityFromBytes(
        available_bytes, *batching.gpu_utilization_factor,
        0, block_bytes, 0);
    if (expected_blocks >= 4 && blocks_without_reservation > expected_blocks) {
      break;
    }
    ++available_bytes;
  }
  ASSERT_GT(blocks_without_reservation, expected_blocks);

  CountingCudaDevice device;
  device.state->total_memory_bytes = available_bytes;
  device.state->free_memory_bytes = available_bytes;
  ScopedCacheDeviceOverride cache_device{*model, device};

  EngineDependencies dependencies;
  ASSERT_NO_THROW(dependencies = Engine::CreateDependencies(model));
  ASSERT_NE(dependencies.cache_manager, nullptr);
  EXPECT_TRUE(dependencies.cache_manager->SupportsRecomputePreemption());
  EXPECT_EQ(dependencies.cache_manager->TargetBlockCount(), expected_blocks);
  EXPECT_EQ(device.state->memory_queries, 1u);
  EXPECT_NO_THROW({
    auto engine = std::make_shared<Engine>(model, std::move(dependencies));
  });
}

TEST(RecomputePreemptionTest, GenuineReservedAuxiliaryStateStillRejectsPreemption) {
  auto model = PreemptionModel(8);
  auto cache = CacheManager::Create(
      model, /*auxiliary_bytes_per_block=*/0,
      /*auxiliary_reserved_memory_bytes=*/1);
  ASSERT_NE(cache, nullptr);
  EXPECT_FALSE(cache->SupportsRecomputePreemption());
  auto shared_cache = std::shared_ptr<CacheManager>{std::move(cache)};
  EXPECT_THROW(Scheduler::Create(model, shared_cache), std::runtime_error);
}

TEST(RecomputePreemptionTest, LargeWaitingQueueBuildsProbesFromBoundedResidents) {
  constexpr size_t kWaitingCount = 4096;
  std::vector<int> identities(kWaitingCount + 2);
  std::vector<RequestStepPlan> candidates;
  candidates.reserve(identities.size());

  RequestStepPlan first_resident;
  first_resident.request_id = &identities[kWaitingCount];
  candidates.push_back(first_resident);

  for (size_t index = 0; index < kWaitingCount; ++index) {
    RequestStepPlan waiting;
    waiting.request_id = &identities[index];
    waiting.newly_admitted = true;
    candidates.push_back(waiting);
  }

  RequestStepPlan second_resident;
  second_resident.request_id = &identities[kWaitingCount + 1];
  candidates.insert(candidates.begin() + 2048, second_resident);
  PreemptionProbeBuilder builder{candidates, /*resident_capacity=*/2};

  StepPlan source;
  source.transaction_id = 7;
  source.scheduled_request_limit = 4;
  size_t probe_count = 0;
  for (const auto& candidate : candidates) {
    if (!candidate.newly_admitted) {
      continue;
    }
    const auto probe = builder.Build(source, candidate);
    ASSERT_EQ(probe.requests.size(), 3u);
    EXPECT_EQ(probe.requests[0].request_id, candidate.request_id);
    EXPECT_EQ(probe.requests[1].request_id, first_resident.request_id);
    EXPECT_EQ(probe.requests[2].request_id, second_resident.request_id);
    EXPECT_EQ(probe.transaction_id, source.transaction_id);
    EXPECT_EQ(probe.scheduled_request_limit, source.scheduled_request_limit);
    ++probe_count;
  }
  EXPECT_EQ(probe_count, kWaitingCount);
}

TEST(RecomputePreemptionTest, RejectsManualSpeculationWithoutDiscardingTheTurn) {
  auto test = MakeCompositeDoublesEngine(PreemptionModel(), 5);
  auto request = Queue(test.engine);
  ModelSteps(test, 1);
  const std::array<int32_t, 1> draft{6};
  EXPECT_THROW(request->SetDraftTokens(draft), std::runtime_error);
  EXPECT_EQ(request->PendingDraftTokenCount(), 0u);
  EXPECT_EQ(request->Status(), RequestStatus::Active);
  Finish(*test.engine);
}

struct FailingPreemptionValidationCache final : PagedCacheManager {
  using PagedCacheManager::PagedCacheManager;
  mutable bool fail_validation{};

  void ValidateRewind(const std::shared_ptr<Request>& request) const override {
    if (std::exchange(fail_validation, false)) {
      throw std::bad_alloc{};
    }
    PagedCacheManager::ValidateRewind(request);
  }
};

TEST(RecomputePreemptionTest, ValidationAllocationFailureDoesNotPublishAnySuspension) {
  auto model = PreemptionModel();
  auto cache = std::make_shared<FailingPreemptionValidationCache>(model);
  auto executor = std::make_unique<RecordingModelExecutor>(model, cache, 5);
  auto* recording = executor.get();
  auto engine = std::make_shared<Engine>(
      model, EngineDependencies{cache, Scheduler::Create(model, cache), std::move(executor)});
  CompositeDoublesEngine test{engine, cache.get(), recording};
  auto victim = Queue(engine);
  ModelSteps(test, 2);
  auto beneficiary = Queue(engine);
  cache->fail_validation = true;
  EXPECT_THROW(RunOne(*engine), std::bad_alloc);
  EXPECT_EQ(victim->Status(), RequestStatus::Active);
  EXPECT_EQ(victim->PreemptionCount(), 0u);
  EXPECT_TRUE(cache->IsResident(victim));
  EXPECT_EQ(beneficiary->Status(), RequestStatus::Assigned);
  CheckState(test, std::array{victim, beneficiary});
  ModelSteps(test, 1);
  EXPECT_EQ(victim->Status(), RequestStatus::Suspended);
}

std::vector<std::vector<int32_t>> SyntheticStreams(bool preemption) {
  auto model = LoadSyntheticPagedModel();
  auto& batching = *model->config_->engine.dynamic_batching;
  batching.enable_recompute_preemption = preemption;
  batching.min_decode_steps_before_preemption = 2;
  batching.max_scheduled_tokens = 512;
  auto engine = std::make_shared<Engine>(model);
  // This checked-in graph fixes both cache dimensions (128 blocks of four slots). Fill that
  // actual pool with long residents instead of changing its declared tensor geometry.
  std::vector<std::shared_ptr<Request>> requests;
  for (int32_t seed = 2; seed < 6; ++seed) {
    requests.push_back(Queue(engine, std::vector<int32_t>(96, seed), 8));
  }
  std::vector<std::vector<int32_t>> streams(5);
  std::array<EngineEvent, 8> events;
  uint64_t preemptions = 0;
  const auto consume = [&] {
    const size_t count = engine->Run(events);
    for (const auto& event : std::span{events}.first(count)) {
      if ((event.flags & (EngineEventFlagFailed | EngineEventFlagCapacityBlocked)) != 0) {
        throw std::runtime_error("Synthetic graph execution did not make valid progress.");
      }
      const auto request = std::find(requests.begin(), requests.end(), event.request);
      if (request == requests.end()) {
        throw std::runtime_error("Synthetic graph emitted an unknown Request.");
      }
      const size_t index = static_cast<size_t>(request - requests.begin());
      if ((event.flags & EngineEventFlagToken) != 0) {
        streams[index].push_back(event.token);
      }
      if ((event.flags & EngineEventFlagTurnFinished) != 0) {
        preemptions += event.request->PreemptionCount();
        event.request->Close();
      }
    }
  };
  for (int guard = 0; streams[3].size() < 4 && guard < 100; ++guard) {
    consume();
  }
  for (size_t index = 0; index < 4; ++index) {
    EXPECT_EQ(streams[index].size(), 4u);
  }
  requests.push_back(Queue(engine, std::vector<int32_t>(121, 9), 4));
  for (int guard = 0; engine->HasPendingRequests() && guard < 1000; ++guard) {
    consume();
  }
  EXPECT_FALSE(engine->HasPendingRequests());
  EXPECT_EQ(preemptions > 0, preemption);
  for (size_t index = 0; index < streams.size(); ++index) {
    EXPECT_EQ(streams[index].size(), index < 4 ? 8u : 4u);
  }
  return streams;
}

TEST(RecomputePreemptionTest, ExecutedSyntheticGraphPreservesGreedyOutputAcrossReplay) {
  const auto reference = SyntheticStreams(false);
  const auto replayed = SyntheticStreams(true);
  EXPECT_FALSE(reference[0].empty());
  EXPECT_FALSE(reference[1].empty());
  EXPECT_EQ(replayed, reference);
}

}  // namespace
}  // namespace Generators::test
