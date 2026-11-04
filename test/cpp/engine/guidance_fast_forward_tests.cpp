// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Engine contract tests for grammar fast-forward tokens: the continuation a turn's grammar forces
// after a sampled token. These run over the synthetic-paged fixture model, whose checked-in
// tokenizer maps token 5 -> "ST" and token 6 -> "OP" (so the pair completes a "STOP" stop string)
// and tokens 11-63 to unique filler text; the hybrid case uses the synthetic-composite fixture. The
// recording executor never runs the model; it fabricates one-hot logits for the token a test scripts.

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_helpers.h"
#include "engine_test_doubles.h"
#include "guidance_test_access.h"
#include "constrained_logits_processor.h"
#include "engine/engine_invariants.h"
#include "engine/step_plan.h"
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

// The sampling step and each later step publish at most kMaxGeneratedTokensPerStep tokens. The rest
// of the span stays private, and the row publishes the next of it, instead of sampling, as it reads
// to its tail.
TEST_F(GuidanceFastForwardTest, LongForcedSpanIsPublishedAtMostEightTokensPerStepBeforeSampling) {
  static_assert(kMaxGeneratedTokensPerStep == 8);
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto request = CreateEngineRequest(engine.engine);
  request->BeginTurn(Prompt());
  const std::vector<int32_t> forced{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
  InstallForcingGrammar(*request, /*trigger=*/11, forced);

  std::array<EngineEvent, 16> storage;
  ASSERT_EQ(engine.engine->Run(storage), 8u);
  EXPECT_EQ(EventTokens(std::span{storage}.first(8)),
            (std::vector<int32_t>{11, 12, 13, 14, 15, 16, 17, 18}));

  // Sampling now would end the turn on EOS.
  engine.executor->SetForcedToken(EosToken(*model_));
  ASSERT_EQ(engine.engine->Run(storage), 4u);
  EXPECT_EQ(EventTokens(std::span{storage}.first(4)), (std::vector<int32_t>{19, 20, 21, 22}));
  EXPECT_EQ(storage[3].flags & EngineEventFlagTurnFinished, 0u);

  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(engine.executor->decoded_token_counts, (std::vector<size_t>{Prompt().size(), 8, 4}));
  const auto tokens = request->TokensCpu();
  EXPECT_EQ(std::vector<int32_t>(tokens.begin(), tokens.end()), Concat(Prompt(), Concat({11}, forced)));
  EXPECT_EQ(request->TurnGeneratedTokens(), forced.size() + 1);
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
}

TEST_F(GuidanceFastForwardTest, StepBudgetSplitsForcedTokensAndSamplesOnlyAfterTheLastChunk) {
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 2;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
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

// A later published batch carries no sampled token, so only its forced tokens count toward the limit.
TEST_F(GuidanceFastForwardTest, TurnLimitInsideTheUnpublishedSpanEndsTheTurnThere) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto request = CreateEngineRequest(engine.engine);
  TurnOptions options;
  options.max_generated_tokens = 10;
  request->BeginTurn(Prompt(), options);
  InstallForcingGrammar(*request, /*trigger=*/11, {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22});

  std::array<EngineEvent, 16> storage;
  ASSERT_EQ(engine.engine->Run(storage), 8u);
  ASSERT_EQ(engine.engine->Run(storage), 2u);
  EXPECT_EQ(EventTokens(std::span{storage}.first(2)), (std::vector<int32_t>{19, 20}));
  EXPECT_EQ(storage[1].finish_reason, GenerationFinishReason::TurnLimit);
  EXPECT_EQ(request->TurnGeneratedTokens(), 10u);
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

// Rolling back the sampling step restores the private span together with the grammar cursor, and
// rolling back a step that publishes from it restores what that step staged.
TEST_F(GuidanceFastForwardTest, RollbackRestoresTheUnpublishedSpanAndTheRetryPublishesItOnce) {
  for (const bool queued_restore : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "queued restore " << queued_restore);
    auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
    auto request = CreateEngineRequest(engine.engine);
    request->BeginTurn(Prompt());
    InstallForcingGrammar(*request, /*trigger=*/11, {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22});
    const auto restore = [&] {
      if (queued_restore) {
        request->QueueStateRestoreForTransaction();
        request->CompleteStateRestoreForTransaction();
      } else {
        request->RestoreStateForTransaction();
      }
    };
    const auto plan_step = [&] {
      RequestStepPlan plan;
      plan.request = request;
      plan.request_id = request.get();
      plan.sequence_length_before = request->CurrentSequenceLength();
      plan.target_cache_slots = static_cast<size_t>(plan.sequence_length_before);
      PrepareRequestStep(model_, plan);
      return plan;
    };

    auto plan = plan_step();
    request->SaveStateForTransaction();
    ASSERT_EQ(request->ApplyLogitsForTransaction(LogitsForToken(*model_, 11)).visible_token_count, 8u);
    restore();
    EXPECT_EQ(request->CurrentSequenceLength(), static_cast<int64_t>(Prompt().size()));
    request->SaveStateForTransaction();
    const auto sampled = request->ApplyLogitsForTransaction(LogitsForToken(*model_, 11));
    ASSERT_EQ(sampled.visible_token_count, 8u);
    request->CommitStateForTransaction();
    request->CommitStep(plan, sampled);

    plan = plan_step();
    ASSERT_TRUE(request->PublishesForcedTokens());
    request->SaveStateForTransaction();
    ASSERT_EQ(request->StageForcedTokensForTransaction().visible_token_count, 4u);
    restore();
    plan = plan_step();
    request->SaveStateForTransaction();
    const auto published = request->StageForcedTokensForTransaction();
    ASSERT_EQ(published.visible_token_count, 4u);
    request->CommitStateForTransaction();
    request->CommitStep(plan, published);

    const auto tokens = request->TokensCpu();
    EXPECT_EQ(std::vector<int32_t>(tokens.begin(), tokens.end()),
              Concat(Prompt(), {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}));
    EXPECT_EQ(request->TurnGeneratedTokens(), 12u);
    auto* grammar = static_cast<ForcingGrammar*>(RequestGuidanceTestAccess::Get(*request));
    EXPECT_EQ(grammar->CommittedTokens(), (std::vector<int32_t>{11}));
  }
}

// Cancellation ends the turn with what it already published and drops the rest of the span, so the
// next turn samples instead of publishing the canceled turn's leftovers.
TEST_F(GuidanceFastForwardTest, CancelKeepsPublishedTokensAndDropsTheUnpublishedSpan) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto request = CreateEngineRequest(engine.engine);
  request->BeginTurn(Prompt());
  InstallForcingGrammar(*request, /*trigger=*/11, {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22});

  std::array<EngineEvent, 16> storage;
  ASSERT_EQ(engine.engine->Run(storage), 8u);
  ASSERT_TRUE(request->Cancel(request->CurrentTurnId()));
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::Canceled);
  const auto published = Concat(Prompt(), {11, 12, 13, 14, 15, 16, 17, 18});
  const auto tokens = request->TokensCpu();
  EXPECT_EQ(std::vector<int32_t>(tokens.begin(), tokens.end()), published);

  engine.executor->SetForcedToken(30);
  request->BeginTurn(std::vector<int32_t>{5});
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].token, 30);
  const auto continued = request->TokensCpu();
  EXPECT_EQ(std::vector<int32_t>(continued.begin(), continued.end()), Concat(published, {5, 30}));
}

// A batched sampler hands its output slots to whichever requests sample next. A row publishing
// forced tokens commits them without sampling, so it must stop writing the slot it was last bound to
// before a sampling peer that now owns that slot reads it.
TEST_F(GuidanceFastForwardTest, PublishingRowLeavesThePeerOwnedSamplerSlotIntact) {
  SharedSlotDevice device{*model_->p_device_scoring_, /*reports_batched_sampling=*/true};
  ScopedScoringDevice scoring{*model_, device};
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto peer = CreateEngineRequest(engine.engine);
  auto forcing = CreateEngineRequest(engine.engine);
  peer->BeginTurn(Prompt());
  forcing->BeginTurn(Prompt());
  InstallForcingGrammar(*forcing, /*trigger=*/11, {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22});
  std::array<EngineEvent, 16> storage;
  ASSERT_EQ(engine.engine->Run(storage), 9u);  // peer: 11; forcing: 11 through 18

  auto slot = model_->p_device_scoring_->Allocate<int32_t>(1);
  ASSERT_TRUE(forcing->BindNextTokensSlot(slot));
  ASSERT_TRUE(peer->BindNextTokensSlot(slot));
  engine.executor->SetForcedToken(30);
  ASSERT_EQ(engine.engine->Run(storage), 5u);  // peer: 30; forcing: 19 through 22
  EXPECT_EQ(EventTokens(std::span{storage}.first(5)), (std::vector<int32_t>{30, 19, 20, 21, 22}));
  EXPECT_EQ(slot.CpuSpan()[0], 30);
}

TEST_F(GuidanceFastForwardTest, StaticBatchingRejectsAFastForwardTurnWithoutConsumingIt) {
  model_->config_->engine.dynamic_batching.reset();
  auto cache = std::make_shared<RecordingCacheManager>(
      model_, /*capacity=*/4, nullptr, /*supports_dynamic_batching=*/false);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(model_, cache, EosToken(*model_));
  EngineDependencies dependencies{cache, std::move(scheduler), std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));
  auto request = CreateEngineRequest(engine);
  TurnOptions options;
  options.guidance_type = "regex";
  options.guidance_data = "ok";
  options.guidance_fast_forward = true;

  try {
    request->BeginTurn(Prompt(), options);
    FAIL() << "Static batching admitted a fast-forward turn.";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("dynamic batching"), std::string::npos);
  }
  EXPECT_TRUE(request->IsAwaitingFirstTurn());
}

// Forced tokens skip the no-repeat-ngram mask, so admission must reject the combination whether the
// turn or the model's search defaults supply the size.
TEST_F(GuidanceFastForwardTest, NoRepeatNgramRejectsAFastForwardTurnWithoutConsumingIt) {
  for (const bool from_model_default : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "from model default " << from_model_default);
    model_ = LoadSyntheticPagedModel();
    TurnOptions options;
    options.guidance_type = "regex";
    options.guidance_data = "ok";
    options.guidance_fast_forward = true;
    if (from_model_default) {
      model_->config_->search.no_repeat_ngram_size = 1;
    } else {
      options.no_repeat_ngram_size = 1;
    }
    auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
    auto request = CreateEngineRequest(engine.engine);

    try {
      request->BeginTurn(Prompt(), options);
      FAIL() << "A fast-forward turn was admitted with no_repeat_ngram_size.";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string(error.what()).find("no_repeat_ngram_size"), std::string::npos);
    }
    EXPECT_TRUE(request->IsAwaitingFirstTurn());
  }
}

#if USE_GUIDANCE
// With a real llguidance grammar, step-by-step decoding reads the literal's tokens one model step
// each, while fast-forward forces everything after the first token and reads it in one step. Both
// emit the literal's exact token IDs.
TEST(GuidanceFastForwardRealGrammarTest, ForcedTextMatchesStepByStepDecodingInOneReadStep) {
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto tokenizer = model->CreateTokenizer();
  // Every token of this text is inside the fixture model's 1000-token vocabulary.
  constexpr const char* kText = "on and on and on";
  const auto text_tokens = tokenizer->Encode(kText);
  ASSERT_GT(text_tokens.size(), 2u);

  const auto run = [&](bool fast_forward) {
    auto engine = MakeDoublesEngine(model, /*capacity=*/8, text_tokens.front());
    auto request = CreateEngineRequest(engine.engine);
    TurnOptions options;
    options.guidance_type = "regex";
    options.guidance_data = kText;
    options.guidance_fast_forward = fast_forward;
    options.max_generated_tokens = 32;
    request->BeginTurn(Prompt(), options);

    std::array<EngineEvent, 8> storage;
    std::vector<int32_t> tokens;
    bool finished = false;
    for (int step = 0; step < 64 && !finished; ++step) {
      const size_t events = engine.engine->Run(storage);
      for (const auto& event : std::span{storage}.first(events)) {
        if (event.flags & EngineEventFlagToken) {
          tokens.push_back(event.token);
        }
        finished |= (event.flags & EngineEventFlagTurnFinished) != 0;
      }
    }
    EXPECT_TRUE(finished);
    EXPECT_EQ(request->FinishReason(), GenerationFinishReason::EosToken);
    return std::pair{tokens, engine.executor->decoded_token_counts};
  };

  const auto [step_tokens, step_reads] = run(false);
  const auto [forward_tokens, forward_reads] = run(true);
  EXPECT_EQ(step_tokens, text_tokens);
  EXPECT_EQ(forward_tokens, text_tokens);
  // Step-by-step: the prompt, then one model step per generated token before EOS is sampled.
  std::vector<size_t> expected_step_reads(text_tokens.size() + 1, 1);
  expected_step_reads.front() = Prompt().size();
  EXPECT_EQ(step_reads, expected_step_reads);
  // Fast-forward: the prompt, then the sampled token and every forced token in one step.
  EXPECT_EQ(forward_reads, (std::vector<size_t>{Prompt().size(), text_tokens.size()}));
}
#endif

}  // namespace
}  // namespace test
}  // namespace Generators
