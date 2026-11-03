// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Engine contract tests for grammar fast-forward tokens: the continuation a turn's grammar forces
// after a sampled token. These run over the synthetic-paged fixture model, whose checked-in
// tokenizer maps token 5 -> "ST" and token 6 -> "OP" (so the pair completes a "STOP" stop string)
// and tokens 11-63 to unique filler text. The recording executor never runs the model; it
// fabricates one-hot logits for the token a test scripts.

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_helpers.h"
#include "engine_test_doubles.h"
#include "guidance_test_access.h"
#include "constrained_logits_processor.h"
#include "engine/engine_invariants.h"
#include "engine/step_plan.h"

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

// Stands in for a grammar that, once `trigger` is committed, forces `continuation`. Like
// llguidance, the forced tokens are reported by the commit itself and are never committed back.
class ForcingGrammar final : public ConstrainedLogitsProcessor {
 public:
  ForcingGrammar(int32_t trigger, std::vector<int32_t> continuation)
      : trigger_{trigger}, continuation_{std::move(continuation)} {}

  void CommitTokens(std::span<int32_t> tokens) override {
    committed_tokens_.insert(committed_tokens_.end(), tokens.begin(), tokens.end());
    pending_forced_tokens_.clear();
    if (!tokens.empty() && tokens.back() == trigger_) {
      pending_forced_tokens_ = continuation_;
    }
  }
  void ProcessLogits(DeviceSpan<float>) override {}
  bool AllowsOnlyTokens(size_t, std::span<const int>) override { return false; }
  void Reset() override {
    committed_tokens_.clear();
    pending_forced_tokens_.clear();
  }
  std::vector<int32_t> GetFFTokens(size_t) override {
    return std::exchange(pending_forced_tokens_, {});
  }
  std::unique_ptr<ConstrainedLogitsProcessor> Clone() const override {
    auto clone = std::make_unique<ForcingGrammar>(trigger_, continuation_);
    clone->committed_tokens_ = committed_tokens_;
    clone->pending_forced_tokens_ = pending_forced_tokens_;
    return clone;
  }

  const std::vector<int32_t>& CommittedTokens() const { return committed_tokens_; }

 private:
  int32_t trigger_;
  std::vector<int32_t> continuation_;
  std::vector<int32_t> committed_tokens_;
  std::vector<int32_t> pending_forced_tokens_;
};

void InstallForcingGrammar(Request& request, int32_t trigger,
                           std::vector<int32_t> continuation) {
  RequestGuidanceTestAccess::Install(
      request, std::make_unique<ForcingGrammar>(trigger, std::move(continuation)));
}

std::vector<int32_t> EventTokens(std::span<const EngineEvent> events) {
  std::vector<int32_t> tokens;
  for (const auto& event : events) {
    if (event.flags & EngineEventFlagToken) {
      tokens.push_back(event.token);
    }
  }
  return tokens;
}

std::vector<int32_t> Concat(std::vector<int32_t> head, const std::vector<int32_t>& tail) {
  head.insert(head.end(), tail.begin(), tail.end());
  return head;
}

class GuidanceFastForwardTest : public ::testing::Test {
 protected:
  void SetUp() override { model_ = LoadSyntheticPagedModel(); }

  std::shared_ptr<Model> model_;
};

TEST_F(GuidanceFastForwardTest, ForcedTokensFollowTheSampledTokenAndReachTheModelInOneStep) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto request = CreateEngineRequest(engine.engine);
  request->BeginTurn(Prompt());
  InstallForcingGrammar(*request, /*trigger=*/11, {12, 13});

  std::array<EngineEvent, 8> storage;
  const size_t first_step_events = engine.engine->Run(storage);
  ASSERT_EQ(first_step_events, 3u);
  EXPECT_EQ(EventTokens(std::span{storage}.first(first_step_events)),
            (std::vector<int32_t>{11, 12, 13}));
  EXPECT_EQ(storage[2].flags & EngineEventFlagTurnFinished, 0u);
  EXPECT_EQ(request->TurnGeneratedTokens(), 3u);

  // The sampled token and both forced tokens reach the model together, and the next token is
  // sampled from that same step.
  engine.executor->SetForcedToken(EosToken(*model_));
  engine.executor->SetExecutionCallback([](ExecutionContext& context) {
    ASSERT_NE(context.plan, nullptr);
    ASSERT_EQ(context.plan->requests.size(), 1u);
    EXPECT_EQ(context.plan->requests[0].unprocessed_token_count, 3u);
    EXPECT_EQ(context.plan->requests[0].draft_token_count, 0u);
    EXPECT_FALSE(context.plan->graph_capture_eligible);
  });
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(engine.executor->decoded_token_counts, (std::vector<size_t>{Prompt().size(), 3}));
  const auto tokens = request->TokensCpu();
  EXPECT_EQ(std::vector<int32_t>(tokens.begin(), tokens.end()), Concat(Prompt(), {11, 12, 13}));
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
}

TEST_F(GuidanceFastForwardTest, StepBudgetSplitsForcedTokensAndSamplesOnlyAfterTheLastChunk) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 2;
  auto request = CreateEngineRequest(engine.engine);
  request->BeginTurn(Prompt());
  InstallForcingGrammar(*request, /*trigger=*/11, {12, 13});

  std::array<EngineEvent, 8> storage;
  ASSERT_EQ(engine.engine->Run(storage), 0u);  // first 2 prompt tokens
  ASSERT_EQ(engine.engine->Run(storage), 3u);  // last prompt token; samples 11, forces 12 and 13

  engine.executor->SetForcedToken(EosToken(*model_));
  EXPECT_EQ(engine.engine->Run(storage), 0u);  // 11 and 12: the row is not caught up yet
  ASSERT_EQ(engine.engine->Run(storage), 1u);  // 13, then samples EOS
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(engine.executor->decoded_token_counts, (std::vector<size_t>{2, 1, 2, 1}));
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
}

TEST_F(GuidanceFastForwardTest, TurnLimitCutsTheForcedSpanAndEndsTheTurn) {
  // A limit of 1 is reached by the sampled token itself, so no forced token may follow it.
  constexpr int32_t kEmitted[] = {11, 12};
  for (const size_t limit : {size_t{1}, size_t{2}}) {
    SCOPED_TRACE(::testing::Message() << "limit " << limit);
    auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
    auto request = CreateEngineRequest(engine.engine);
    TurnOptions options;
    options.max_generated_tokens = limit;
    request->BeginTurn(Prompt(), options);
    InstallForcingGrammar(*request, /*trigger=*/11, {12, 13, 14});

    std::array<EngineEvent, 8> storage;
    const size_t events = engine.engine->Run(storage);
    const std::vector<int32_t> expected(std::begin(kEmitted), std::begin(kEmitted) + limit);
    ASSERT_EQ(events, limit);
    EXPECT_EQ(EventTokens(std::span{storage}.first(events)), expected);
    EXPECT_NE(storage[limit - 1].flags & EngineEventFlagTurnFinished, 0u);
    EXPECT_EQ(storage[limit - 1].finish_reason, GenerationFinishReason::TurnLimit);
    EXPECT_EQ(request->TurnGeneratedTokens(), limit);
    const auto tokens = request->TokensCpu();
    EXPECT_EQ(std::vector<int32_t>(tokens.begin(), tokens.end()), Concat(Prompt(), expected));
    EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
  }
}

TEST_F(GuidanceFastForwardTest, ContextLimitCutsTheForcedSpanAndEndsTheTurn) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto request = CreateEngineRequest(engine.engine, Prompt().size() + 3);
  request->BeginTurn(Prompt());
  InstallForcingGrammar(*request, /*trigger=*/11, {12, 13, 14});

  std::array<EngineEvent, 8> storage;
  const size_t events = engine.engine->Run(storage);
  ASSERT_EQ(events, 3u);
  EXPECT_EQ(EventTokens(std::span{storage}.first(events)), (std::vector<int32_t>{11, 12, 13}));
  EXPECT_EQ(storage[2].finish_reason, GenerationFinishReason::ContextLimit);
  EXPECT_EQ(request->CurrentSequenceLength(), static_cast<int64_t>(Prompt().size() + 3));
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
}

TEST_F(GuidanceFastForwardTest, ForcedEosEndsTheTurnWithoutBeingEmitted) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto request = CreateEngineRequest(engine.engine);
  request->BeginTurn(Prompt());
  InstallForcingGrammar(*request, /*trigger=*/11, {12, EosToken(*model_), 13});

  std::array<EngineEvent, 8> storage;
  const size_t events = engine.engine->Run(storage);
  ASSERT_EQ(events, 2u);
  EXPECT_EQ(EventTokens(std::span{storage}.first(events)), (std::vector<int32_t>{11, 12}));
  EXPECT_NE(storage[1].flags & EngineEventFlagTurnFinished, 0u);
  EXPECT_EQ(storage[1].finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(request->TurnGeneratedTokens(), 2u);
  const auto tokens = request->TokensCpu();
  EXPECT_EQ(std::vector<int32_t>(tokens.begin(), tokens.end()), Concat(Prompt(), {11, 12}));
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
}

TEST_F(GuidanceFastForwardTest, StopStringCompletedByAForcedTokenEndsTheTurnAtThatToken) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5 /* "ST" */);
  auto request = CreateEngineRequest(engine.engine);
  TurnOptions options;
  options.stop_strings = {"STOP"};
  // "OP" also exhausts the turn budget; the stop string must still win.
  options.max_generated_tokens = 2;
  request->BeginTurn(Prompt(), options);
  InstallForcingGrammar(*request, /*trigger=*/5, {6 /* "OP" */, 12});

  std::array<EngineEvent, 8> storage;
  const size_t events = engine.engine->Run(storage);
  ASSERT_EQ(events, 2u);
  EXPECT_EQ(EventTokens(std::span{storage}.first(events)), (std::vector<int32_t>{5, 6}));
  EXPECT_EQ(storage[1].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[1].matched_stop_string_index, 0);
  EXPECT_EQ(request->TurnGeneratedTokens(), 2u);
  const auto tokens = request->TokensCpu();
  EXPECT_EQ(std::vector<int32_t>(tokens.begin(), tokens.end()), Concat(Prompt(), {5, 6}));
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
}

TEST_F(GuidanceFastForwardTest, HybridStepReadingForcedTokensDefersAnotherRequestsDrafts) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/11);
  auto drafting = CreateRequestWithPrompt(engine.engine, Prompt());
  auto forcing = CreateRequestWithPrompt(engine.engine, Prompt());
  InstallForcingGrammar(*forcing, /*trigger=*/11, {12, 13});
  std::array<EngineEvent, 8> storage;
  ASSERT_EQ(engine.engine->Run(storage), 4u);  // drafting: 11; forcing: 11, 12, 13
  drafting->SetDraftTokens(std::array<int32_t, 1>{11});

  // Recurrent state cannot capture draft checkpoints while a multi-token row shares the step.
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    ASSERT_NE(context.plan, nullptr);
    ASSERT_EQ(context.plan->requests.size(), 2u);
    for (const auto& entry : context.plan->requests) {
      const bool is_forcing = entry.request_id == forcing.get();
      EXPECT_EQ(entry.unprocessed_token_count, is_forcing ? 3u : 1u);
      EXPECT_EQ(entry.draft_token_count, 0u);
    }
  });
  ASSERT_GT(engine.engine->Run(storage), 0u);
}

TEST_F(GuidanceFastForwardTest, RollbackDiscardsForcedTokensAndTheRetryCommitsThemOnce) {
  for (const bool queued_restore : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "queued restore " << queued_restore);
    auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
    auto request = CreateEngineRequest(engine.engine);
    request->BeginTurn(Prompt());
    InstallForcingGrammar(*request, /*trigger=*/11, {12, 13});
    const auto before = request->Snapshot();

    RequestStepPlan plan;
    plan.request = request;
    plan.request_id = request.get();
    plan.sequence_length_before = before.current_sequence_length;
    plan.target_cache_slots = static_cast<size_t>(before.current_sequence_length);
    PrepareRequestStep(model_, plan);

    request->SaveStateForTransaction();
    const auto staged = request->ApplyLogitsForTransaction(LogitsForToken(*model_, 11));
    ASSERT_EQ(staged.visible_token_count, 3u);
    EXPECT_EQ(request->CurrentSequenceLength(), before.current_sequence_length + 3);
    if (queued_restore) {
      request->QueueStateRestoreForTransaction();
      request->CompleteStateRestoreForTransaction();
    } else {
      request->RestoreStateForTransaction();
    }
    EXPECT_EQ(request->CurrentSequenceLength(), before.current_sequence_length);

    request->SaveStateForTransaction();
    const auto retried = request->ApplyLogitsForTransaction(LogitsForToken(*model_, 11));
    ASSERT_EQ(retried.visible_token_count, 3u);
    request->CommitStateForTransaction();
    request->CommitStep(plan, retried);

    const auto tokens = request->TokensCpu();
    EXPECT_EQ(std::vector<int32_t>(tokens.begin(), tokens.end()), Concat(Prompt(), {11, 12, 13}));
    EXPECT_EQ(request->TurnGeneratedTokens(), 3u);
    auto* grammar = static_cast<ForcingGrammar*>(RequestGuidanceTestAccess::Get(*request));
    EXPECT_EQ(grammar->CommittedTokens(), (std::vector<int32_t>{11}));
  }
}

}  // namespace
}  // namespace test
}  // namespace Generators
