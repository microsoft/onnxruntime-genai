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

TEST_F(GuidanceFastForwardTest, ForcedTokensFollowTheSampledTokenAndReachTheModelBeforeTheNextSample) {
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

  // The next token is sampled only after the model has read every forced token.
  engine.executor->SetForcedToken(EosToken(*model_));
  size_t events = 0;
  for (int step = 0; step < 4 && events == 0; ++step) {
    events = engine.engine->Run(storage);
  }
  ASSERT_EQ(events, 1u);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::EosToken);
  size_t decoded_tokens = 0;
  for (const size_t count : engine.executor->decoded_token_counts) {
    decoded_tokens += count;
  }
  EXPECT_EQ(decoded_tokens, Prompt().size() + 3);
  const auto tokens = request->TokensCpu();
  EXPECT_EQ(std::vector<int32_t>(tokens.begin(), tokens.end()), Concat(Prompt(), {11, 12, 13}));
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
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
