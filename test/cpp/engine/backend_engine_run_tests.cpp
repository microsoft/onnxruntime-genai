// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_doubles.h"
#include "engine_test_helpers.h"

namespace Generators::test {
namespace {

class ScriptedStaticScheduler final : public Scheduler {
 public:
  ScriptedStaticScheduler(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache,
                          std::vector<StepOutcomeKind> outcomes)
      : Scheduler{model}, inner_{model, std::move(cache)}, outcomes_{std::move(outcomes)} {}

  void AddRequest(std::shared_ptr<Request> request) override {
    requests_.push_back(request);
    inner_.AddRequest(std::move(request));
  }
  void RemoveRequest(std::shared_ptr<Request> request) override {
    inner_.RemoveRequest(request);
    requests_.erase(std::remove(requests_.begin(), requests_.end(), request), requests_.end());
  }
  ScheduledRequests Schedule() override {
    ++execute_plans;
    return inner_.Schedule();
  }
  NontransactionalSchedulingResult ScheduleStep() override {
    if (next_outcome_ == outcomes_.size()) return Scheduler::ScheduleStep();
    const auto kind = outcomes_[next_outcome_++];
    const void* request_id = kind == StepOutcomeKind::UnserviceableRequest && !requests_.empty()
                                 ? requests_.front().get()
                                 : nullptr;
    return {{kind, 0, request_id}, std::nullopt};
  }
  bool HasPendingRequests() const override { return inner_.HasPendingRequests(); }

  size_t execute_plans{};

 private:
  StaticBatchScheduler inner_;
  std::vector<StepOutcomeKind> outcomes_;
  std::vector<std::shared_ptr<Request>> requests_;
  size_t next_outcome_{};
};

struct ObservedExecutor final : RecordingModelExecutor {
  ObservedExecutor(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache,
                   bool selects_tokens, bool produces_tokens)
      : RecordingModelExecutor(std::move(model), std::move(cache), 5),
        selects_tokens_{selects_tokens},
        produces_tokens_{produces_tokens} {}

  bool SelectsOwnTokens() const override { return selects_tokens_; }
  void Decode(ScheduledRequests& scheduled, ExecutionContext& context) override {
    if (!produces_tokens_) {
      RecordingModelExecutor::Decode(scheduled, context);
      return;
    }
    ++decode_calls;
    std::vector<BackendPlannedRow> plan;
    for (size_t row = 0; row < scheduled.size(); ++row) {
      CompletionKey key{42, 1, {static_cast<uint32_t>(row), 1}, scheduled[row]->CurrentTurnId(), ++submission_};
      plan.push_back({row, key, 0, false});
    }
    scheduled.BeginBackendStep(plan);
    for (size_t row = 0; row < scheduled.size(); ++row) {
      BackendRowResult result;
      result.key = plan[row].key;
      SelectedTokenBlock tokens;
      for (size_t i = 0; i < candidate_width; ++i) {
        tokens.candidates[i] = static_cast<int32_t>(5 + i);
      }
      tokens.candidate_count = candidate_width;
      result.selected_tokens = tokens;
      scheduled.StageBackendRow(row, std::move(result));
    }
  }
  void OnStepCommitted(const ScheduledRequests& scheduled,
                       std::span<const RequestStepResult>) override {
    ++commit_calls;
    acknowledged_rows = scheduled.CommitAcks().size();
    if (!scheduled.CommitAcks().empty()) {
      last_staged_for_events = scheduled.CommitAcks().front().tokens_staged_for_events;
    }
    if (fail_acknowledgement) throw std::runtime_error("Injected backend acknowledgement failure.");
  }
  void OnExecutionFailure() noexcept override { ++failure_notifications; }

  bool fail_acknowledgement{};
  size_t candidate_width{1};
  size_t commit_calls{};
  size_t acknowledged_rows{};
  size_t last_staged_for_events{};
  size_t failure_notifications{};

 private:
  bool selects_tokens_;
  bool produces_tokens_;
  uint64_t submission_{};
};

struct StaticEngineFixture {
  std::shared_ptr<Engine> engine;
  ScriptedStaticScheduler* scheduler;
  ObservedExecutor* executor;
};

StaticEngineFixture MakeStaticEngine(std::vector<StepOutcomeKind> outcomes,
                                     bool selects_tokens = false, bool produces_tokens = false) {
  auto model = LoadDummyDecoderModel();
  auto cache = std::make_shared<RecordingCacheManager>(model, 2, nullptr, false);
  auto scheduler = std::make_unique<ScriptedStaticScheduler>(model, cache, std::move(outcomes));
  auto executor = std::make_unique<ObservedExecutor>(model, cache, selects_tokens, produces_tokens);
  auto* scheduler_observer = scheduler.get();
  auto* executor_observer = executor.get();
  auto engine = std::make_shared<Engine>(
      model, EngineDependencies{cache, std::move(scheduler), std::move(executor)});
  return {std::move(engine), scheduler_observer, executor_observer};
}

TEST(BackendEngineRunTest, NoWorkAndCapacityDeferralKeepEngineHealthyWithoutExecuting) {
  auto fixture = MakeStaticEngine({StepOutcomeKind::NoWork, StepOutcomeKind::CapacityDeferred});
  auto request = CreateEngineRequest(fixture.engine);
  request->BeginTurn(std::array<int32_t, 1>{3}, std::optional<size_t>{1});
  const auto prompt_length = request->CurrentSequenceLength();
  std::array<EngineEvent, 8> events;
  for (size_t step = 0; step < 2; ++step) {
    const size_t count = fixture.engine->Run(events);
    for (size_t event = 0; event < count; ++event) {
      EXPECT_EQ(events[event].flags & (EngineEventFlagToken | EngineEventFlagFailed), 0u);
    }
    EXPECT_EQ(request->CurrentSequenceLength(), prompt_length);
    EXPECT_EQ(fixture.executor->decode_calls, 0);
    EXPECT_EQ(fixture.executor->commit_calls, 0u);
    EXPECT_EQ(fixture.executor->failure_notifications, 0u);
    EXPECT_TRUE(fixture.engine->HasPendingRequests());
  }
  const size_t count = fixture.engine->Run(events);
  ASSERT_GT(count, 0u);
  EXPECT_EQ(events[0].request, request);
  EXPECT_NE(events[0].flags & EngineEventFlagToken, 0u);
  EXPECT_EQ(fixture.executor->decode_calls, 1);
  EXPECT_EQ(fixture.executor->failure_notifications, 0u);
}

TEST(BackendEngineRunTest, UnserviceableRequestFailsWithoutPoisoningEngineOrExecutingIt) {
  auto fixture = MakeStaticEngine({StepOutcomeKind::UnserviceableRequest});
  auto oversized = CreateEngineRequest(fixture.engine);
  oversized->BeginTurn(std::array<int32_t, 1>{3}, std::optional<size_t>{1});
  std::array<EngineEvent, 8> events;
  ASSERT_EQ(fixture.engine->Run(events), 1u);
  EXPECT_EQ(events[0].request, oversized);
  EXPECT_EQ(events[0].error_code, EngineErrorCode::RequestUnserviceable);
  EXPECT_EQ(events[0].flags & EngineEventFlagToken, 0u);
  EXPECT_EQ(fixture.executor->decode_calls, 0);
  EXPECT_EQ(fixture.executor->failure_notifications, 0u);

  auto successor = CreateEngineRequest(fixture.engine);
  successor->BeginTurn(std::array<int32_t, 1>{4}, std::optional<size_t>{1});
  ASSERT_GT(fixture.engine->Run(events), 0u);
  EXPECT_EQ(events[0].request, successor);
  EXPECT_NE(events[0].flags & EngineEventFlagToken, 0u);
  EXPECT_EQ(fixture.executor->decode_calls, 1);
}

TEST(BackendEngineRunTest, OutputProtocolMismatchFailsBeforeRequestCommit) {
  for (bool selects_tokens : {false, true}) {
    auto fixture = MakeStaticEngine({}, selects_tokens, !selects_tokens);
    auto request = CreateEngineRequest(fixture.engine);
    request->BeginTurn(std::array<int32_t, 1>{3}, std::optional<size_t>{2});
    const auto prompt_length = request->CurrentSequenceLength();
    std::array<EngineEvent, 8> events;
    const size_t count = fixture.engine->Run(events);
    ASSERT_GT(count, 0u);
    for (size_t event = 0; event < count; ++event) {
      EXPECT_EQ(events[event].flags & EngineEventFlagToken, 0u);
      EXPECT_NE(events[event].flags & EngineEventFlagFailed, 0u);
    }
    EXPECT_EQ(request->CurrentSequenceLength(), prompt_length);
    EXPECT_EQ(fixture.executor->commit_calls, 0u);
    EXPECT_EQ(fixture.executor->failure_notifications, 1u);
    EXPECT_THROW(fixture.engine->Run(events), EngineStepError);
    EXPECT_EQ(fixture.executor->failure_notifications, 1u);
  }
}

TEST(BackendEngineRunTest, AcknowledgementFailurePublishesNoSuccessAndPoisonsOnce) {
  auto fixture = MakeStaticEngine({}, true, true);
  fixture.executor->fail_acknowledgement = true;
  auto request = CreateEngineRequest(fixture.engine);
  request->BeginTurn(std::array<int32_t, 1>{3}, std::optional<size_t>{2});
  const auto prompt_length = request->CurrentSequenceLength();
  std::array<EngineEvent, 8> events;
  const size_t count = fixture.engine->Run(events);
  ASSERT_GT(count, 0u);
  for (size_t event = 0; event < count; ++event) {
    EXPECT_EQ(events[event].flags & EngineEventFlagToken, 0u);
    EXPECT_NE(events[event].flags & EngineEventFlagFailed, 0u);
  }
  EXPECT_EQ(request->CurrentSequenceLength(), prompt_length + 1);
  EXPECT_EQ(fixture.executor->commit_calls, 1u);
  EXPECT_EQ(fixture.executor->acknowledged_rows, 1u);
  EXPECT_EQ(fixture.executor->failure_notifications, 1u);
  EXPECT_THROW(fixture.engine->Run(events), EngineStepError);
  EXPECT_THROW(fixture.engine->CreateRequest(), EngineStepError);
  EXPECT_EQ(fixture.executor->commit_calls, 1u);
  EXPECT_EQ(fixture.executor->failure_notifications, 1u);
}

TEST(BackendEngineRunTest, StagedOutputAckDoesNotChangeWhenCallerDrainsEventsAcrossMultipleRuns) {
  auto fixture = MakeStaticEngine({}, true, true);
  fixture.executor->candidate_width = 3;
  auto request = CreateEngineRequest(fixture.engine);
  request->BeginTurn(std::array<int32_t, 1>{3}, std::optional<size_t>{3});

  std::array<EngineEvent, 1> single_event;
  ASSERT_EQ(fixture.engine->Run(single_event), 1u);
  EXPECT_EQ(fixture.executor->decode_calls, 1);
  EXPECT_EQ(fixture.executor->commit_calls, 1u);
  EXPECT_EQ(fixture.executor->last_staged_for_events, 3u);
  EXPECT_EQ(single_event[0].token, 5);
  EXPECT_EQ(single_event[0].flags & EngineEventFlagTurnFinished, 0u);

  ASSERT_EQ(fixture.engine->Run(single_event), 1u);
  EXPECT_EQ(fixture.executor->decode_calls, 1);
  EXPECT_EQ(fixture.executor->commit_calls, 1u);
  EXPECT_EQ(single_event[0].token, 6);
  EXPECT_EQ(single_event[0].flags & EngineEventFlagTurnFinished, 0u);

  ASSERT_EQ(fixture.engine->Run(single_event), 1u);
  EXPECT_EQ(fixture.executor->decode_calls, 1);
  EXPECT_EQ(fixture.executor->commit_calls, 1u);
  EXPECT_EQ(single_event[0].token, 7);
  EXPECT_NE(single_event[0].flags & EngineEventFlagTurnFinished, 0u);
}

}  // namespace
}  // namespace Generators::test
