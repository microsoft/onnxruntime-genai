// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Component-contract tests for ordinary (non-speculative) dynamic Engine decoded stop strings.
// These wire an Engine with the recording cache-manager and model-executor doubles over the
// synthetic-paged fixture model, which carries a tiny checked-in tokenizer (see
// test/models/engine/synthetic-paged/tokenizer.json) whose vocabulary is designed for exact,
// deterministic stop-string scenarios:
//
//   token 2 -> "A", token 3 -> "B", token 4 -> "C"
//   token 5 -> "ST", token 6 -> "OP"          (a stop match can span two tokens/steps)
//   token 7 -> "STOPX"                        (a single token whose bytes complete a match with
//                                               trailing bytes after it)
//   token 8 -> "AB", token 9 -> "CD", token 10 -> "Z"
//   tokens 0/1 are the special <bos>/<eos> tokens, which decode to nothing
//   tokens 11-63 -> "t11".."t63" (arbitrary unique filler)
//
// The doubles never run the model: the executor fabricates one-hot logits for whichever token(s)
// a test scripts, so search always selects that token deterministically without depending on the
// synthetic ONNX graph's actual output.

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_helpers.h"
#include "engine_test_doubles.h"
#include "guidance_test_access.h"
#include "constrained_logits_processor.h"
#include "engine/request_status.h"
#include "engine/step_plan.h"
#include "stop_string_controller.h"
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

TurnOptions StopOptions(std::vector<std::string> stop_strings,
                        std::optional<size_t> max_generated_tokens = std::nullopt) {
  TurnOptions options;
  options.stop_strings = std::move(stop_strings);
  options.max_generated_tokens = max_generated_tokens;
  return options;
}

// A minimal ConstrainedLogitsProcessor stand-in so stop-string tests can verify the two features
// coexist without depending on USE_GUIDANCE or a real grammar. Records every committed token; does
// not otherwise constrain anything.
class NoOpGuidanceProcessor final : public ConstrainedLogitsProcessor {
 public:
  void CommitTokens(std::span<int32_t> tokens) override {
    committed_tokens.insert(committed_tokens.end(), tokens.begin(), tokens.end());
  }
  void ProcessLogits(DeviceSpan<float>) override {}
  void Reset() override {
    committed_tokens.clear();
    ++reset_count;
  }
  std::vector<int32_t> GetFFTokens(size_t) override { return {}; }
  std::unique_ptr<ConstrainedLogitsProcessor> Clone() const override {
    auto clone = std::make_unique<NoOpGuidanceProcessor>();
    clone->committed_tokens = committed_tokens;
    clone->reset_count = reset_count;
    return clone;
  }

  std::vector<int32_t> committed_tokens;
  int reset_count{};
};

class StopStringEngineTest : public ::testing::Test {
 protected:
  void SetUp() override { model_ = LoadSyntheticPagedModel(); }

  std::shared_ptr<Model> model_;
};

// ---------------------------------------------------------------------------------------------
// Request-level transactional behavior (direct Search/logits driving, no Engine::Run involved).
// ---------------------------------------------------------------------------------------------

TEST_F(StopStringEngineTest, NoStopStringsPreserveOrdinaryGenerationBehavior) {
  // Exercise the no-stop path entirely through the public request and event contract.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/8 /* "AB" */);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({}, std::optional<size_t>{2}));

  std::array<EngineEvent, 4> storage;
  GenerationFinishReason finish_reason = GenerationFinishReason::None;
  int32_t matched_index = -999;
  bool turn_finished = false;
  for (int attempt = 0; attempt < 4 && !turn_finished; ++attempt) {
    const size_t count = engine.engine->Run(storage);
    for (size_t i = 0; i < count; ++i) {
      if (storage[i].flags & EngineEventFlagTurnFinished) {
        turn_finished = true;
        finish_reason = storage[i].finish_reason;
        matched_index = storage[i].matched_stop_string_index;
      }
    }
  }
  ASSERT_TRUE(turn_finished);
  EXPECT_EQ(finish_reason, GenerationFinishReason::TurnLimit);
  EXPECT_EQ(matched_index, -1);
  EXPECT_EQ(request->MatchedStopStringIndex(), -1);
}

TEST_F(StopStringEngineTest, MatchSpanningTwoTokensReportsEarliestCompletionAndIndex) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5 /* "ST" */);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));

  std::array<EngineEvent, 4> storage;
  // Step 1: prefill + first generated token ("ST"). No match yet.
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].flags & EngineEventFlagTurnFinished, 0u);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::None);

  // Step 2: second generated token ("OP") completes "STOP".
  engine.executor->SetForcedToken(6);
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[0].token, 6);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[0].matched_stop_string_index, 0);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::StopString);
  EXPECT_EQ(request->MatchedStopStringIndex(), 0);
  // The token whose decoded bytes complete the match is retained, emitted, and counted like any
  // other generated token.
  EXPECT_EQ(request->TurnGeneratedTokens(), 2u);
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
}

TEST_F(StopStringEngineTest, SecondOfThreePatternsReportsItsOwnNonZeroIndex) {
  // Guards against an off-by-one/always-index-0 regression: with several configured patterns, the
  // one that actually matches ("STOP", configured at index 1) must report its own index, not 0 or
  // the index of an unrelated pattern that never matches.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5 /* "ST" */);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"UNREACHABLE1", "STOP", "UNREACHABLE2"}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);  // "ST": no match yet
  EXPECT_EQ(storage[0].flags & EngineEventFlagTurnFinished, 0u);

  engine.executor->SetForcedToken(6);  // "OP": completes "STOP"
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[0].matched_stop_string_index, 1);
  EXPECT_EQ(request->MatchedStopStringIndex(), 1);
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
}

TEST_F(StopStringEngineTest, MatchWithinASingleTokenRetainsTrailingBytesInHistory) {
  // Token 7 decodes to "STOPX": the match completes inside the token's decoded bytes, but the
  // whole token is still retained, emitted, and counted -- the Engine never trims or rewrites a
  // committed token's raw bytes.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/7);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].token, 7);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[0].matched_stop_string_index, 0);
  EXPECT_EQ(request->TurnGeneratedTokens(), 1u);
}

TEST_F(StopStringEngineTest, StopStringTakesPrecedenceOverTurnLimitForTheSameToken) {
  // max_generated_tokens=1 and the first generated token both complete a stop match and exhaust
  // the turn's token budget. StopString must win.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/7 /* "STOPX" */);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}, std::optional<size_t>{1}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[0].matched_stop_string_index, 0);
}

TEST_F(StopStringEngineTest, RealEosTokensDecodeEmptyAndStopStringsDoNotDisturbEosCompletion) {
  // GreedySearch_Cpu never appends a sampled EOS token to the sequence for a single-sequence
  // Request (see GreedySearch_Cpu::SelectTop/SetNextToken): search decides this purely from
  // config.model.eos_token_id, before Request-level stop-string classification ever runs. That
  // token is therefore never observed by the stop controller at all, which is consistent with
  // GenAI's tokenizers defaulting to skip_special_tokens=true and registering EOS as a special
  // added token: a real EOS token always decodes to zero bytes and so could never independently
  // complete a stop match anyway. This test is the achievable, realistic half of the "StopString
  // precedes EosToken" contract: an active (non-matching-yet) stop-string configuration must not
  // disturb ordinary EOS termination. See StopStringTakesPrecedenceOverTurnLimitForTheSameToken
  // and StopStringTakesPrecedenceOverContextLimitForTheSameToken for the precedence cases that do
  // not depend on Search's append decision, which the Engine can and does resolve in the stop
  // string's favor.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5 /* "ST" */);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].flags & EngineEventFlagTurnFinished, 0u);

  engine.executor->SetForcedToken(EosToken(*model_));
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(storage[0].matched_stop_string_index, -1);
}

TEST_F(StopStringEngineTest, StopStringTakesPrecedenceOverContextLimitForTheSameToken) {
  auto params = MakeGreedyParams(*model_);
  params->search.max_length = static_cast<int32_t>(Prompt().size()) + 1;

  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/7 /* "STOPX" */);
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[0].matched_stop_string_index, 0);
}

TEST_F(StopStringEngineTest, TurnResetClearsMatchAndControllerForTheNextTurn) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/7 /* "STOPX" */);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  ASSERT_EQ(request->FinishReason(), GenerationFinishReason::StopString);
  ASSERT_EQ(request->MatchedStopStringIndex(), 0);

  // The next turn resets the finish reason and matched index immediately at admission, before any
  // new generation, and starts a fresh matcher/tokenizer stream even when it reuses the exact same
  // stop-string configuration.
  request->BeginTurn(std::array<int32_t, 1>{2}, StopOptions({"STOP"}));
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::None);
  EXPECT_EQ(request->MatchedStopStringIndex(), -1);

  engine.executor->SetForcedToken(7);
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[0].matched_stop_string_index, 0);
}

// ---------------------------------------------------------------------------------------------
// Generated-current-turn-only matching: the controller's stream starts fresh at the first
// *generated* token of the active turn (see "Decoded stop strings" in
// docs/paged_attention_engine.md). These three tests each isolate one way that boundary could be
// crossed by a regression, distinct from the tokenizer's own per-stream first-piece decoding
// behavior (already covered by StopStringMatcherTest and the real-tokenizer controller test): (1)
// the prompt itself is never fed to the controller even when it alone decodes to the full
// configured stop string; (2) a later turn's continuation input tokens are never fed to that
// turn's controller even when they are byte-identical to a prior turn's generated suffix; (3) a
// prior turn's controller automaton state is never carried into the next turn's fresh controller.
// ---------------------------------------------------------------------------------------------

TEST_F(StopStringEngineTest, PromptTokensAreNeverSeededIntoTheStopStringMatcher) {
  // The prompt alone ("ST" + "OP" = "STOP") decodes to exactly the configured stop string, but the
  // controller only ever observes *generated* tokens (Request::StageGeneration), never the prompt
  // used to admit the turn. A single nonmatching generated token ("A") must end the turn on the
  // turn limit, not StopString.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/2 /* "A" */);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(std::vector<int32_t>{5, 6} /* "ST" + "OP" = "STOP" */,
                     StopOptions({"STOP"}, std::optional<size_t>{1}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::TurnLimit);
  EXPECT_EQ(storage[0].matched_stop_string_index, -1);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::TurnLimit);
  EXPECT_EQ(request->MatchedStopStringIndex(), -1);
}

TEST_F(StopStringEngineTest, ContinuationInputTokensAreNeverSeededIntoTheNextTurnsStopStringMatcher) {
  // Turn 1 generates "ST" (token 5) and ends on the turn limit (not a match). Turn 2 re-provides
  // that exact same token, byte-identical to turn 1's generated suffix, as *continuation input* --
  // not as something turn 2 generates -- then generates "OP" (token 6) alone. If continuation input
  // were ever fed to the new turn's controller, "ST" (seeded) + "OP" (generated) would complete
  // "STOP"; since continuation input is never seeded, only "OP" reaches the fresh controller, which
  // does not complete "STOP" by itself.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5 /* "ST" */);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}, std::optional<size_t>{1}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  ASSERT_EQ(storage[0].finish_reason, GenerationFinishReason::TurnLimit);
  ASSERT_EQ(request->TurnGeneratedTokens(), 1u);

  // Continuation input is exactly token 5 ("ST"), the same bytes turn 1 just generated.
  request->BeginTurn(std::array<int32_t, 1>{5}, StopOptions({"STOP"}));
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::None);
  EXPECT_EQ(request->MatchedStopStringIndex(), -1);

  engine.executor->SetForcedToken(6);  // "OP"
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_NE(storage[0].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[0].matched_stop_string_index, -1);
  EXPECT_NE(request->FinishReason(), GenerationFinishReason::StopString);
  EXPECT_EQ(request->MatchedStopStringIndex(), -1);
}

TEST_F(StopStringEngineTest, PreviousTurnsStopControllerStateNeverLeaksIntoTheNextTurn) {
  // Turn 1 generates "ST" (token 5) and ends on the turn limit, leaving that turn's controller
  // mid-pattern (two bytes short of "STOP") if it were ever kept alive. Turn 2's continuation input
  // is token 3 ("B"), unrelated to "ST", so this scenario cannot be explained by continuation-input
  // seeding (see the previous test for that, isolated separately) -- only by turn 1's controller
  // automaton state surviving into turn 2. If it did, generating "OP" alone in turn 2 would
  // wrongly complete "STOP" using the stale prefix state; since a fresh controller is built for
  // every turn, it does not.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5 /* "ST" */);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}, std::optional<size_t>{1}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  ASSERT_EQ(storage[0].finish_reason, GenerationFinishReason::TurnLimit);
  ASSERT_EQ(request->TurnGeneratedTokens(), 1u);

  // Continuation input is token 3 ("B"), unrelated to "ST"/"STOP".
  request->BeginTurn(std::array<int32_t, 1>{3}, StopOptions({"STOP"}));
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::None);
  EXPECT_EQ(request->MatchedStopStringIndex(), -1);

  engine.executor->SetForcedToken(6);  // "OP"
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_NE(storage[0].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[0].matched_stop_string_index, -1);
  EXPECT_NE(request->FinishReason(), GenerationFinishReason::StopString);
  EXPECT_EQ(request->MatchedStopStringIndex(), -1);
}

TEST_F(StopStringEngineTest, MultiRequestIsolationKeepsIndependentControllersAndTiming) {
  // Two concurrent requests with different stop-string configurations and different per-row
  // scripted tokens must not observe each other's decoded text.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/2);
  auto first = CreateEngineRequest(engine.engine, *model_);
  auto second = CreateEngineRequest(engine.engine, *model_);
  first->BeginTurn(Prompt(), StopOptions({"STOP"}));
  second->BeginTurn(Prompt(), StopOptions({"AB"}));  // completes within one token (token 8)

  std::array<EngineEvent, 4> storage;
  // Step 1: first gets "ST" (no match yet); second gets "AB" (completes immediately). Both
  // requests' events are delivered in one Run() call: first's plain token event, and second's
  // completed (Token | TurnFinished) event.
  engine.executor->SetVerifyRowTokens({5, 8});
  size_t drained = engine.engine->Run(storage);
  ASSERT_EQ(drained, 2u);
  EXPECT_EQ(storage[0].request, first);
  EXPECT_EQ(storage[0].flags & EngineEventFlagTurnFinished, 0u);
  EXPECT_EQ(storage[1].request, second);
  EXPECT_EQ(storage[1].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[1].matched_stop_string_index, 0);
  EXPECT_EQ(second->MatchedStopStringIndex(), 0);
  EXPECT_EQ(first->FinishReason(), GenerationFinishReason::None);

  // Step 2: only `first` is still scheduled now; "OP" completes its independent "STOP" match.
  engine.executor->SetVerifyRowTokens({});
  engine.executor->SetForcedToken(6);
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].request, first);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[0].matched_stop_string_index, 0);
}

TEST_F(StopStringEngineTest, GuidanceAndStopStringsCoexistOnTheSameRequest) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  auto* guidance = new NoOpGuidanceProcessor();
  RequestGuidanceTestAccess::Install(
      *request, std::unique_ptr<ConstrainedLogitsProcessor>(guidance));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(guidance->committed_tokens, (std::vector<int32_t>{5}));
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::None);

  engine.executor->SetForcedToken(6);
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  // Guidance still saw both committed tokens before the turn's own completion reset its cursor --
  // exactly the same turn-completion behavior guidance already has with EOS/turn/context-limit
  // completions, now also exercised by a StopString completion.
  EXPECT_EQ(guidance->reset_count, 1);
  EXPECT_TRUE(guidance->committed_tokens.empty());
}

TEST_F(StopStringEngineTest, GuidanceAndStopRollbackCoexistOnTheSameRequest) {
  // Both guidance and the stop controller are mutated during staging (Request::StageGeneration
  // calls CommitGuidanceToken() -- and, since a StopString match finishes the turn, immediately
  // Resets() the *active* guidance processor in the same call -- and observes the stop controller
  // unconditionally once a token is appended, before the step commits), so a rollback must restore
  // both together, not just one. RequestGuidanceTestAccess::Get() is used throughout (rather than
  // holding onto the originally-installed raw pointer) because a rollback swaps the active
  // processor back to a different object: the pre-staging Clone() checkpoint.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5 /* "ST" */);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  auto* guidance = new NoOpGuidanceProcessor();
  RequestGuidanceTestAccess::Install(
      *request, std::unique_ptr<ConstrainedLogitsProcessor>(guidance));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);  // commits "ST" (token 5)
  ASSERT_EQ(guidance->committed_tokens, (std::vector<int32_t>{5}));

  const auto before = request->Snapshot();
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.target_cache_slots = static_cast<size_t>(before.current_sequence_length);
  PrepareRequestStep(model_, plan);

  // Stage a second step with "OP" (token 6), which completes "STOP". Staging both commits the
  // token to guidance and, because the step is now done, immediately Resets() it -- exactly the
  // same turn-completion behavior GuidanceAndStopStringsCoexistOnTheSameRequest exercises on the
  // committed path, but here it happens speculatively, before any Search/cache commit.
  request->SaveStateForTransaction();
  const auto staged = request->ApplyLogitsForTransaction(LogitsForToken(*model_, 6));
  ASSERT_EQ(staged.finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(guidance->reset_count, 1);
  EXPECT_TRUE(guidance->committed_tokens.empty());

  // Roll back without committing: both guidance and the stop controller must be restored together
  // to their pre-staging state. Guidance's checkpoint is a distinct Clone() taken before staging,
  // so the *active* processor after rollback is not `guidance` (whose Reset() call cannot be
  // undone in place) but the untouched pre-staging clone.
  request->RestoreStateForTransaction();
  auto* restored_guidance =
      static_cast<NoOpGuidanceProcessor*>(RequestGuidanceTestAccess::Get(*request));
  ASSERT_NE(restored_guidance, nullptr);
  EXPECT_EQ(restored_guidance->committed_tokens, (std::vector<int32_t>{5}));
  EXPECT_EQ(restored_guidance->reset_count, 0);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::None);

  // Mutation-resistant: re-stage and commit the *same* completing token ("OP", token 6) again,
  // rather than a different, non-matching token. A rollback that only truncated turn_tokens_ and
  // Reset() the matcher without actually replaying "ST" back through a fresh stream would leave the
  // controller holding just "OP" alone at this point -- which does not complete "STOP" -- so this
  // would fail to reproduce the match if the replay loop were ever deleted.
  request->SaveStateForTransaction();
  const auto retried = request->ApplyLogitsForTransaction(LogitsForToken(*model_, 6));
  ASSERT_EQ(retried.finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(retried.matched_stop_string_index, 0);
  request->CommitStateForTransaction();
  request->CommitStep(plan, retried);

  EXPECT_EQ(request->TurnGeneratedTokens(), 2u);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::StopString);
  EXPECT_EQ(request->MatchedStopStringIndex(), 0);
  // Guidance reaches the same terminal/reset state on the replayed-and-recompleted match as it did
  // on the original (now-discarded) completion: one more Reset() (turn completion resets guidance's
  // cursor), and no leftover committed tokens.
  EXPECT_EQ(restored_guidance->reset_count, 1);
  EXPECT_TRUE(restored_guidance->committed_tokens.empty());
}

TEST_F(StopStringEngineTest, DirectRollbackDiscardsTheStagedObservationCompletely) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);  // commits "ST" (token 5)
  ASSERT_EQ(request->TurnGeneratedTokens(), 1u);

  const auto before = request->Snapshot();
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.target_cache_slots = static_cast<size_t>(before.current_sequence_length);
  PrepareRequestStep(model_, plan);

  // Stage a second step with "OP" (token 6), which would complete "STOP", then roll it back
  // without committing.
  request->SaveStateForTransaction();
  const auto staged = request->ApplyLogitsForTransaction(LogitsForToken(*model_, 6));
  ASSERT_TRUE(staged.token_appended);
  ASSERT_EQ(staged.finish_reason, GenerationFinishReason::StopString);
  // Not committed yet: the Request's own finish reason and matched index are untouched until
  // CommitStep runs.
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::None);
  EXPECT_EQ(request->MatchedStopStringIndex(), -1);
  request->RestoreStateForTransaction();

  // Mutation-resistant: re-stage and commit the *same* completing token ("OP", token 6) again,
  // rather than a different, non-matching token. If rollback only truncated turn_tokens_ and
  // Reset() the matcher without actually replaying "ST" back through a fresh stream, the controller
  // would hold just "OP" alone here -- which does not complete "STOP" -- so this would fail to
  // reproduce the match if the replay loop were ever deleted.
  request->SaveStateForTransaction();
  const auto retried = request->ApplyLogitsForTransaction(LogitsForToken(*model_, 6));
  ASSERT_EQ(retried.finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(retried.matched_stop_string_index, 0);
  request->CommitStateForTransaction();
  request->CommitStep(plan, retried);

  // Exactly the committed tokens ("ST" then "OP") were observed: the discarded first "OP" attempt
  // left no trace, and the replayed "ST" correctly still lets the *second* "OP" complete the match.
  EXPECT_EQ(request->TurnGeneratedTokens(), 2u);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::StopString);
  EXPECT_EQ(request->MatchedStopStringIndex(), 0);
}

TEST_F(StopStringEngineTest, QueuedRollbackReplaysTheControllerInTheCompletionPhase) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  ASSERT_EQ(request->TurnGeneratedTokens(), 1u);

  const auto before = request->Snapshot();
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.target_cache_slots = static_cast<size_t>(before.current_sequence_length);
  PrepareRequestStep(model_, plan);

  request->SaveStateForTransaction();
  const auto staged = request->ApplyLogitsForTransaction(LogitsForToken(*model_, 6));
  ASSERT_EQ(staged.finish_reason, GenerationFinishReason::StopString);
  // Exercise the queued and completion phases through their production interface, then verify the
  // completed rollback from the next externally observable match.
  request->QueueStateRestoreForTransaction();
  request->CompleteStateRestoreForTransaction();

  // Mutation-resistant: re-stage and commit the *same* completing token ("OP", token 6) again. If
  // CompleteStateRestoreForTransaction() only truncated turn_tokens_ and Reset() the matcher
  // without actually replaying "ST" back through a fresh stream, the controller would hold just
  // "OP" alone here -- which does not complete "STOP" -- so this would fail to reproduce the match
  // if the replay loop were ever deleted.
  request->SaveStateForTransaction();
  const auto retried = request->ApplyLogitsForTransaction(LogitsForToken(*model_, 6));
  ASSERT_EQ(retried.finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(retried.matched_stop_string_index, 0);
  request->CommitStateForTransaction();
  request->CommitStep(plan, retried);

  EXPECT_EQ(request->TurnGeneratedTokens(), 2u);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::StopString);
  EXPECT_EQ(request->MatchedStopStringIndex(), 0);
}

TEST_F(StopStringEngineTest, QueuedRollbackReplaysControllerBeforeRetry) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);  // commits "ST" (token 5)

  const auto before = request->Snapshot();
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.target_cache_slots = static_cast<size_t>(before.current_sequence_length);
  PrepareRequestStep(model_, plan);

  request->SaveStateForTransaction();
  const auto staged = request->ApplyLogitsForTransaction(LogitsForToken(*model_, 6));  // "OP"
  ASSERT_EQ(staged.finish_reason, GenerationFinishReason::StopString);

  request->QueueStateRestoreForTransaction();
  request->CompleteStateRestoreForTransaction();

  // Re-staging the same completing token proves completion rebuilt the matcher from the committed
  // prefix. If queued rollback retained the staged token or lost "ST", this would not match once.
  request->SaveStateForTransaction();
  const auto retried = request->ApplyLogitsForTransaction(LogitsForToken(*model_, 6));
  EXPECT_EQ(retried.finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(retried.matched_stop_string_index, 0);
}

TEST_F(StopStringEngineTest, DirectRollbackReplaysMultipleCommittedTokensInOrder) {
  // The earlier direct-rollback test checkpoints after a single committed token; this strengthens
  // that to two committed tokens ("A" then "ST") so RollbackTo() actually has to replay more than
  // one token, in order, to reconstruct the correct matcher/stream state.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/2 /* "A" */);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);  // commits "A" (token 2)
  engine.executor->SetForcedToken(5);          // "ST"
  ASSERT_EQ(engine.engine->Run(storage), 1u);  // commits "ST" (token 5): history is now "A" + "ST"
  ASSERT_EQ(request->TurnGeneratedTokens(), 2u);

  const auto before = request->Snapshot();
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.target_cache_slots = static_cast<size_t>(before.current_sequence_length);
  PrepareRequestStep(model_, plan);

  // Stage a third step with "OP" (token 6): "A"+"ST"+"OP" ends with "STOP", a completed match. Roll
  // it back without committing -- this must replay both prior tokens ("A" then "ST"), in order, to
  // reconstruct exactly the pre-staging state.
  request->SaveStateForTransaction();
  const auto staged = request->ApplyLogitsForTransaction(LogitsForToken(*model_, 6));
  ASSERT_EQ(staged.finish_reason, GenerationFinishReason::StopString);
  request->RestoreStateForTransaction();
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::None);

  // Mutation-resistant: re-stage and commit the *same* completing token ("OP", token 6) again. If
  // the rollback only replayed one of the two retained tokens (or none at all), the controller
  // would not hold the full "A"+"ST" history here, and re-staging "OP" alone would not complete
  // "STOP" -- so this would fail to reproduce the match if the (in-order, multi-token) replay were
  // ever broken or partially deleted.
  request->SaveStateForTransaction();
  const auto retried = request->ApplyLogitsForTransaction(LogitsForToken(*model_, 6));
  ASSERT_EQ(retried.finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(retried.matched_stop_string_index, 0);
  request->CommitStateForTransaction();
  request->CommitStep(plan, retried);

  // Exactly the three committed tokens ("A", "ST", "OP") were observed: the discarded first "OP"
  // attempt left no trace even though the checkpoint it rolled back to was two tokens deep, and the
  // replayed "A"+"ST" correctly still lets the second "OP" complete the match.
  EXPECT_EQ(request->TurnGeneratedTokens(), 3u);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::StopString);
  EXPECT_EQ(request->MatchedStopStringIndex(), 0);
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
}

TEST_F(StopStringEngineTest, ControllerRollbackToDirectlyReproducesAMidPatternMatchAfterReplay) {
  // A direct, Request/Engine-free test of StopStringController::RollbackTo() itself: builds up
  // "ST" (one token), stages "OP" (would complete "STOP"), rolls back to the 1-token checkpoint
  // directly, then re-observes the same completing token and asserts the match reappears. This
  // isolates the controller's own replay contract from the Request-level transaction plumbing the
  // tests above already cover, without duplicating them.
  auto tokenizer = model_->CreateTokenizer();
  StopStringController controller(tokenizer, {"STOP"});

  controller.ObserveToken(5);  // "ST"
  const size_t checkpoint = controller.TurnTokens().size();
  ASSERT_EQ(checkpoint, 1u);

  const auto& staged_match = controller.ObserveToken(6);  // "OP": completes "STOP"
  ASSERT_TRUE(staged_match.has_value());
  ASSERT_TRUE(controller.Matched());

  controller.RollbackTo(checkpoint);
  EXPECT_FALSE(controller.Matched());
  EXPECT_FALSE(controller.Match().has_value());
  EXPECT_EQ(controller.TurnTokens().size(), checkpoint);

  // Mutation-resistant: re-observing the same completing token must reproduce the match. If
  // RollbackTo() only resized turn_tokens_ and Reset() the matcher without actually replaying "ST"
  // back through a fresh stream, the controller would hold nothing here, and this "OP" alone would
  // not complete "STOP".
  const auto& replayed_match = controller.ObserveToken(6);
  ASSERT_TRUE(replayed_match.has_value());
  EXPECT_EQ(replayed_match->index, 0u);
  EXPECT_EQ(replayed_match->start_offset, 0u);
  EXPECT_EQ(replayed_match->end_offset, 4u);
  EXPECT_TRUE(controller.Matched());
}

// ---------------------------------------------------------------------------------------------
// Static batching and speculative exclusion.
// ---------------------------------------------------------------------------------------------

TEST_F(StopStringEngineTest, StaticBatchingEngineRejectsStopEnabledTurnBeforeMutation) {
  model_->config_->engine.dynamic_batching.reset();
  auto cache = std::make_shared<RecordingCacheManager>(
      model_, /*capacity=*/8, /*trace=*/nullptr, /*supports_dynamic_batching=*/false);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(model_, cache, EosToken(*model_));
  EngineDependencies dependencies{cache, std::move(scheduler), std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));

  auto request = CreateEngineRequest(engine, *model_);
  EXPECT_THROW(
      request->BeginTurn(Prompt(), StopOptions({"STOP"})), std::runtime_error);
  // Rejected before any mutation: the request never left its pre-turn state.
  EXPECT_TRUE(request->IsAwaitingFirstTurn());

  // The same request and Engine still work for an ordinary (non-stop) turn.
  EXPECT_NO_THROW(request->BeginTurn(Prompt()));
}

TEST_F(StopStringEngineTest, ManualDraftTokensAreRejectedForAStopEnabledTurn) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  cache->SetMaxDraftTokensPerStep(3);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(model_, cache, /*forced_token=*/2);
  executor->SetSupportsDraftVerification(true);
  EngineDependencies dependencies{cache, std::move(scheduler), std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));
  ASSERT_GT(engine->MaxDraftTokensPerStep(), 0u);

  auto request = CreateEngineRequest(engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"UNREACHABLE"}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine->Run(storage), 1u);  // past prefill, ready to decode
  ASSERT_FALSE(request->IsPrefill());

  EXPECT_NE(request->DraftTokenValidationError(), nullptr);
  EXPECT_THROW(
      request->SetDraftTokens(std::array<int32_t, 1>{2}), std::runtime_error);
}

// ---------------------------------------------------------------------------------------------
// Event replacement: cancellation and fatal failure must never leave a stale matched index.
// ---------------------------------------------------------------------------------------------

// Cancel() only ever succeeds while a request is still executable (Assigned/Active); once a
// StopString match commits, the request becomes TurnComplete in the very same step that stages
// its terminal event, so a *committed StopString terminal* can never itself be the "existing"
// event Engine::CancelRequest merges into -- there is no window where it is both staged and still
// cancellable. The reachable manifestation of that merge is a still-executing request's retained,
// non-terminal token event; this proves the merge correctly leaves (or forces) the matched index
// at -1 rather than leaking a stale value into the newly-Canceled terminal.
TEST_F(StopStringEngineTest, CancelMergesIntoARetainedTokenEventWithoutLeakingAStopIndex) {
  // `plain` is created first so its event drains first (staged_event_order_ follows scheduling
  // order), leaving `stopping`'s token event retained/undrained while `stopping` is still Active.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/2 /* "A", never matches */);
  auto plain = CreateEngineRequest(engine.engine, *model_);
  auto stopping = CreateEngineRequest(engine.engine, *model_);
  plain->BeginTurn(Prompt());
  stopping->BeginTurn(Prompt(), StopOptions({"UNREACHABLE"}));

  std::array<EngineEvent, 1> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  ASSERT_EQ(storage[0].request, plain);
  ASSERT_TRUE(engine.engine->HasPendingRequests());
  ASSERT_EQ(stopping->Status(), RequestStatus::Active);

  EXPECT_TRUE(stopping->Cancel(stopping->CurrentTurnId()));

  std::array<EngineEvent, 4> drain_storage;
  const auto drained = engine.engine->Run(drain_storage);
  ASSERT_EQ(drained, 1u);
  EXPECT_EQ(drain_storage[0].request, stopping);
  EXPECT_EQ(drain_storage[0].flags & EngineEventFlagTurnFinished, EngineEventFlagTurnFinished);
  EXPECT_EQ(drain_storage[0].finish_reason, GenerationFinishReason::Canceled);
  EXPECT_EQ(drain_storage[0].matched_stop_string_index, -1);
  EXPECT_EQ(stopping->MatchedStopStringIndex(), -1);
}

// A fatal engine failure only forcibly fails requests that are still executable
// (Engine::MarkUnhealthyAndThrow's sweep skips TurnComplete ones entirely and copies their
// already-staged pending events through unchanged); an already-committed StopString terminal is
// therefore never touched by it. Combined with the previous test, this establishes that neither
// event-replacement path can downgrade or corrupt an already-committed matched index -- each only
// clears an unset (-1) one, which is a no-op.
TEST_F(StopStringEngineTest, FatalFailurePreservesAnAlreadyCommittedStopMatchOnAnUnrelatedRequest) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/7 /* "STOPX" */);
  auto completed = CreateEngineRequest(engine.engine, *model_);
  completed->BeginTurn(Prompt(), StopOptions({"STOP"}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  ASSERT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(storage[0].matched_stop_string_index, 0);
  ASSERT_EQ(completed->Status(), RequestStatus::TurnComplete);
  ASSERT_EQ(completed->FinishReason(), GenerationFinishReason::StopString);
  ASSERT_EQ(completed->MatchedStopStringIndex(), 0);

  // An unrelated request hits a fatal execution failure in a later, separate step.
  auto executing = CreateEngineRequest(engine.engine, *model_);
  executing->BeginTurn(Prompt());
  engine.executor->SetForcedToken(2);
  engine.executor->SetNextFailure(ScriptedExecutionFailure::Fatal);
  const auto failure = RunOne(*engine.engine);
  EXPECT_EQ(failure.request, executing);
  EXPECT_EQ(failure.flags, EngineEventFlagTurnFinished | EngineEventFlagFailed);
  EXPECT_EQ(failure.finish_reason, GenerationFinishReason::Failed);
  EXPECT_EQ(failure.matched_stop_string_index, -1);

  // The Engine is now unhealthy, but `completed` was not part of the failing batch: its own
  // committed stop-match state must be completely unaffected.
  EXPECT_EQ(completed->FinishReason(), GenerationFinishReason::StopString);
  EXPECT_EQ(completed->MatchedStopStringIndex(), 0);
}

TEST(StopStringMtpExclusionTest,
     StopEnabledRequestIsExcludedFromAutomaticMtpButEngineKeepsSpeculativeCapability) {
  auto model = LoadSyntheticPagedMtpModel();
  const int32_t eos = EosToken(*model);
  const int32_t filler = eos == 5 ? 6 : 5;  // never EOS, never matches "UNREACHABLE"
  auto engine = MakeMtpDoublesEngine(model, filler);

  auto stop_request = CreateEngineRequest(engine.engine, *model);
  stop_request->BeginTurn(Prompt(), StopOptions({"UNREACHABLE"}));
  auto plain_request = CreateRequestWithPrompt(engine.engine, *model, Prompt());

  std::array<EngineEvent, 8> storage;
  ASSERT_GT(engine.engine->Run(storage), 0u);

  // Only the plain request receives an MTP shadow: the stop-enabled request is excluded request-
  // locally (via the same DraftTokenValidationError() check SetDraftTokens uses), without
  // disabling speculative drafting for the rest of the Engine.
  EXPECT_EQ(engine.mtp_cache->AllocatedCount(), 1u);
}

// ---------------------------------------------------------------------------------------------
// Real-tokenizer coverage boundary: the synthetic-paged fixture's tokenizer has byte_fallback=false
// and a plain (non-byte-level) vocabulary purely of whole ASCII tokens, so it cannot represent a
// genuine multi-byte UTF-8 codepoint whose *tokens* (not just its raw bytes) straddle a boundary --
// every token it defines decodes to either zero or whole-ASCII-character bytes. Extending it to a
// byte-level vocabulary (the GPT-2-style byte<->unicode remapping needed to store an individual raw
// byte as one BPE token) would meaningfully complicate a fixture that is deliberately kept small and
// hand-verifiable for the rest of this test file.
//
// Rather than bloat that fixture, this reuses the already checked-in real GPT-2 byte-level tokenizer
// (hf-internal-testing/tiny-random-gpt2-fp32) directly against StopStringController, with no
// Engine/Model-executor doubles involved. That tokenizer's base vocabulary includes a single-byte
// token for every raw byte value (standard GPT-2 byte-level BPE), so token id 127 is the single raw
// byte 0xC3 and token id 102 is the single raw byte 0xA9 -- together the exact 2-byte UTF-8 encoding
// of U+00E9 ("é"). Feeding them as two separate tokens exercises the real ORT Extensions
// detokenizer: it withholds an incomplete UTF-8 sequence internally and only emits complete UTF-8 to
// the caller, so TokenizerStream::Decode(127) returns an *empty* string (verified directly below,
// with its own separate stream instance) and TokenizerStream::Decode(102) returns the complete
// 2-byte "é" in one call -- not a dangling lead byte followed by a completing continuation byte.
// StopStringController::ObserveToken() therefore never actually receives a split UTF-8 sequence from
// a real tokenizer stream in this scenario; what this test establishes is that integration property
// (a real TokenizerStream's own internal UTF-8 buffering, and the controller correctly handling an
// empty decoded chunk as a trivial no-op), not the matcher's own byte-level split-handling.
//
// StopStringMatcherTest already exhaustively covers UTF-8 byte-boundary correctness when the matcher
// itself is fed genuinely split raw bytes directly (MultiByteStopStringSplitInsideCodePoint,
// SafeOutputMayEndInsideACodePoint, IsValidUtf8*) -- that is where a *raw-byte* split is actually
// exercised, independent of whether any real tokenizer would ever produce one. This test is the
// separate, real-tokenizer-integration half: proving what a real tokenizer stream actually hands the
// controller across a two-token codepoint, and that the controller still reports the correct match
// once the (empty, then complete) chunks are fed through it.
TEST(StopStringControllerRealTokenizerTest, TokenizerStreamBuffersAnIncompleteCodepointUntilItIsComplete) {
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::shared_ptr<const Tokenizer> tokenizer = model->CreateTokenizer();

  // Established directly, with a stream instance dedicated to this assertion (not shared with the
  // controller below): the real ORT Extensions detokenizer buffers token 127's lone lead byte 0xC3
  // and returns nothing for it, then emits the complete 2-byte UTF-8 sequence 0xC3 0xA9 ("é") only
  // once token 102 arrives.
  {
    auto stream = tokenizer->CreateStream();
    const std::string& first = stream->Decode(127);
    EXPECT_TRUE(first.empty());
    const std::string& second = stream->Decode(102);
    EXPECT_EQ(second, "\xC3\xA9");
  }

  // The controller integration: an empty first chunk is a trivial no-op (nothing for the matcher to
  // see yet), and the second call is where the complete "é" actually reaches the matcher in one
  // piece, completing the configured match with the expected offsets.
  StopStringController controller(tokenizer, {"\xC3\xA9"} /* "é" */);
  EXPECT_FALSE(controller.ObserveToken(127).has_value());  // Buffered by the tokenizer: no match yet.
  EXPECT_FALSE(controller.Matched());

  const auto& match = controller.ObserveToken(102);  // Tokenizer now emits the complete "é".
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->index, 0u);
  EXPECT_EQ(match->start_offset, 0u);
  EXPECT_EQ(match->end_offset, 2u);
  EXPECT_TRUE(controller.Matched());
}

}  // namespace
}  // namespace test
}  // namespace Generators
