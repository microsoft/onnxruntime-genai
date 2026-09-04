// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Component-contract tests for dynamic Engine decoded stop strings, covering both the ordinary
// one-token-per-step path and speculative draft verification (manual SetDraftTokens and automatic
// MTP). These wire an Engine with the recording cache-manager and model-executor doubles over the
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

// ---------------------------------------------------------------------------------------------
// Speculative draft verification: a stop-enabled turn drafts and verifies exactly like an ordinary
// one, for both manual SetDraftTokens callers and the automatic MTP drafter. Only static batching's
// own rejection (above) still excludes a stop-enabled turn before mutation.
// ---------------------------------------------------------------------------------------------

TEST_F(StopStringEngineTest, ManualDraftTokensAreAcceptedAndVerifiedForAStopEnabledTurn) {
  auto cache = std::make_shared<RecordingCacheManager>(model_, /*capacity=*/8);
  cache->SetMaxDraftTokensPerStep(3);
  auto scheduler = Scheduler::Create(model_, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(model_, cache, /*forced_token=*/11);
  executor->SetSupportsDraftVerification(true);
  auto* executor_observer = executor.get();
  EngineDependencies dependencies{cache, std::move(scheduler), std::move(executor)};
  auto engine = std::make_shared<Engine>(model_, std::move(dependencies));
  ASSERT_GT(engine->MaxDraftTokensPerStep(), 0u);

  auto request = CreateEngineRequest(engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"UNREACHABLE"}));

  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine->Run(storage), 1u);  // past prefill, ready to decode
  ASSERT_FALSE(request->IsPrefill());

  // A stop-enabled turn no longer excludes draft verification: the request is ready to decode, so
  // proposing drafts succeeds exactly as it would without stop strings, and every non-matching
  // draft commits and is emitted normally.
  EXPECT_EQ(request->DraftTokenValidationError(), nullptr);
  EXPECT_NO_THROW(request->SetDraftTokens(std::array<int32_t, 3>{11, 12, 13}));
  executor_observer->SetVerifyRowTokens({11, 12, 13, 14});

  ASSERT_EQ(engine->Run(storage), 4u);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(storage[i].flags, EngineEventFlagToken);
  }
  EXPECT_EQ(request->AcceptedDraftTokenCount(), 0u);  // cleared by CommitStep after the round
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
}

TEST_F(StopStringEngineTest, GreedyDraftMatchAtTheFirstPositionDiscardsLaterDraftsAndTruncatesCache) {
  // Drafts "STOPX", "t11", "t12"; the target agrees with all three, but the match completes on the
  // very first committed draft, so verification must never even look at the later two.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{7, 11, 12});
  engine.executor->SetVerifyRowTokens({7, 11, 12, 13});

  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].token, 7);
  EXPECT_EQ(storage[0].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[0].matched_stop_string_index, 0);
  EXPECT_EQ(request->AcceptedDraftTokenCount(), 0u);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  // Only the one matching draft is kept in the paged cache; the two later drafts, though the
  // target agreed with them too, were never committed. kept_tokens also counts the one always-
  // consumed continuation token that seeded this round's first draft logits.
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 2u);
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
}

TEST_F(StopStringEngineTest, GreedyDraftMatchAtAMiddlePositionEmitsThePriorAcceptedTokenFirst) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 7, 12});
  engine.executor->SetVerifyRowTokens({11, 7, 12, 13});

  ASSERT_EQ(engine.engine->Run(storage), 2u);
  EXPECT_EQ(storage[0].token, 11);
  EXPECT_EQ(storage[0].flags, EngineEventFlagToken);
  EXPECT_EQ(storage[1].token, 7);
  EXPECT_EQ(storage[1].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[1].finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 3u);
}

TEST_F(StopStringEngineTest, GreedyDraftMatchAtTheLastDraftPositionDiscardsTheBonusRow) {
  // All three drafts are accepted, and the match completes on the last one -- the row after it
  // (the bonus/replacement row the target would otherwise have contributed) is never reached.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 12, 7});
  engine.executor->SetVerifyRowTokens({11, 12, 7, 13});

  ASSERT_EQ(engine.engine->Run(storage), 3u);
  EXPECT_EQ(storage[0].token, 11);
  EXPECT_EQ(storage[1].token, 12);
  EXPECT_EQ(storage[2].token, 7);
  EXPECT_EQ(storage[2].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[2].finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 4u);
}

TEST_F(StopStringEngineTest, GreedyBonusTokenCanCompleteAMatchAfterAllDraftsAccepted) {
  // None of the three drafts individually match; verification accepts all of them and, since
  // there was no early stop, falls through to the ordinary one-token bonus row, which does match.
  // This exercises the ordinary ApplyLogitsForTransaction path rather than the greedy draft loop's
  // own observation, confirming the bonus token needs no special handling.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({11, 12, 13, 7});

  ASSERT_EQ(engine.engine->Run(storage), 4u);
  EXPECT_EQ(storage[3].token, 7);
  EXPECT_EQ(storage[3].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[3].finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 4u);
}

TEST_F(StopStringEngineTest, GreedyRejectedDraftNeverReachesTheMatcherButItsReplacementCan) {
  // Draft position 0 ("t11") is rejected -- the target predicts "STOPX" instead. The matcher never
  // sees the rejected draft at all; it observes only the target's own replacement, which matches.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({7, 12, 13, 14});  // row 0 disagrees with draft 0

  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].token, 7);
  EXPECT_EQ(storage[0].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 1u);
}

TEST_F(StopStringEngineTest, GreedyMatchCanBeginInPriorCommittedOutputAndFinishInADraft) {
  // The ordinary first generated token ("ST") only partially matches; the very next round proposes
  // a draft whose first token ("OP") completes the match using the controller's carried-over
  // partial state, not anything reseeded from the draft proposal itself.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5 /* "ST" */);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].flags & EngineEventFlagTurnFinished, 0u);

  request->SetDraftTokens(std::array<int32_t, 3>{6, 11, 12});  // "OP", filler, filler
  engine.executor->SetVerifyRowTokens({6, 11, 12, 13});

  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].token, 6);
  EXPECT_EQ(storage[0].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[0].matched_stop_string_index, 0);
  EXPECT_EQ(request->TurnGeneratedTokens(), 2u);
}

TEST_F(StopStringEngineTest, GreedyDraftVerificationNeverObservesAnAcceptedEosDraftThatDoesNotAppend) {
  // GreedySearch_Cpu::CommitToken(eos) marks the search done without appending (sequence length
  // unchanged): draft position 1 is EOS, argmax-accepted, but must never reach the controller or be
  // able to produce a StopString result, exactly like an unappended EOS on the ordinary one-token
  // path. Verification must also stop there -- draft position 2 and the bonus/replacement row are
  // never reached.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11 /* filler */);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);  // commits "t11" (token 11)

  const int32_t eos = EosToken(*model_);
  request->SetDraftTokens(std::array<int32_t, 3>{11, eos, 12});
  engine.executor->SetVerifyRowTokens({11, eos, 12, 13});  // argmax accepts all three positions

  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].token, 11);
  EXPECT_EQ(storage[0].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(storage[0].matched_stop_string_index, -1);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::EosToken);
  EXPECT_EQ(request->MatchedStopStringIndex(), -1);

  // Only the appended draft is retained. The non-appended EOS, draft position 2, and the
  // bonus/replacement row are excluded from the public sequence and cache commit.
  EXPECT_EQ(request->AcceptedDraftTokenCount(), 0u);
  EXPECT_EQ(request->PendingDraftTokenCount(), 0u);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 2u);  // 1 base + 1 appended draft
  EXPECT_TRUE(ValidateRequestInvariants(request->Snapshot()).empty());
}

TEST_F(StopStringEngineTest, SampledDraftMatchAtTheFirstAcceptedPositionTruncates) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{7, 12, 13});
  engine.executor->SetVerifyRowTokens({7, 12, 13, 14});

  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].token, 7);
  EXPECT_EQ(storage[0].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  // The matching draft is itself target-confirmed (the target's own sample equals the proposed
  // draft), so it counts as accepted -- 1 confirmed draft plus the always-kept prior slot.
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 2u);
}

TEST_F(StopStringEngineTest, SampledDraftMatchAtAMiddleAcceptedPositionEmitsThePriorTokenFirst) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 7, 13});
  engine.executor->SetVerifyRowTokens({11, 7, 13, 14});

  ASSERT_EQ(engine.engine->Run(storage), 2u);
  EXPECT_EQ(storage[0].token, 11);
  EXPECT_EQ(storage[0].flags, EngineEventFlagToken);
  EXPECT_EQ(storage[1].token, 7);
  EXPECT_EQ(storage[1].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[1].finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  // 2 confirmed drafts (the prior one plus the matching one) plus the always-kept prior slot.
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 3u);
}

TEST_F(StopStringEngineTest, SampledDraftMatchAtTheLastAcceptedPositionDiscardsTheTrailingBonus) {
  // All three drafts are accepted and a bonus token is drawn too, but the match completes on the
  // last *draft* position -- the trailing bonus token must never be committed or emitted.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 12, 7});
  engine.executor->SetVerifyRowTokens({11, 12, 7, 14});

  ASSERT_EQ(engine.engine->Run(storage), 3u);
  EXPECT_EQ(storage[2].token, 7);
  EXPECT_EQ(storage[2].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[2].finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  // All 3 drafts are confirmed (including the matching one, which ends verification but was still
  // itself target-confirmed) plus the always-kept prior slot; the drawn-but-discarded bonus token
  // (14) is not part of this count.
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 4u);
}

TEST_F(StopStringEngineTest, SampledFullyAcceptedBonusTokenCanCompleteAMatch) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({11, 12, 13, 7});

  ASSERT_EQ(engine.engine->Run(storage), 4u);
  EXPECT_EQ(storage[3].token, 7);
  EXPECT_EQ(storage[3].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[3].finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 4u);
}

TEST_F(StopStringEngineTest, SampledRejectedDraftReplacementCanCompleteAMatch) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({7, 12, 13, 14});  // target replaces draft 0 with "STOPX"

  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].token, 7);
  EXPECT_EQ(storage[0].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 1u);
}

TEST_F(StopStringEngineTest, SampledDraftVerificationNeverObservesAnAcceptedEosDraftThatDoesNotAppend) {
  // Mirrors GreedyDraftVerificationNeverObservesAnAcceptedEosDraftThatDoesNotAppend for the sampled
  // path: an EOS draft never reaches the matcher and never gets promoted as an accepted draft,
  // because StageGeneration()'s own token_appended gate (shared unchanged with the ordinary path)
  // already excludes it -- both stage-loop branches reuse that same function unmodified for this.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);  // commits "t11"

  const int32_t eos = EosToken(*model_);
  request->SetDraftTokens(std::array<int32_t, 3>{11, eos, 12});
  engine.executor->SetVerifyRowTokens({11, eos, 12, 13});

  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].token, 11);
  EXPECT_EQ(storage[0].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(storage[0].matched_stop_string_index, -1);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 2u);  // 1 base + 1 appended draft
}

TEST_F(StopStringEngineTest, PromoteFinalStageAsAcceptedDraftRejectsAResultThatNeverAppended) {
  // Direct unit coverage for PromoteFinalStageAsAcceptedDraft's own precondition: it must never
  // silently mirror a token into tokens_host_ for a stage StageGeneration() itself reports as not
  // appended (which would diverge Request's host mirror from Search's own sequence). This can only
  // be reached today by a caller bug -- the stage loop only ever calls it when confirmed_draft_
  // counts and token_appended already agree -- so this exercises the guard directly rather than
  // trying to contrive a real end-to-end scenario that violates the invariant.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt());
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  RequestStepResult result{};
  result.token = 42;
  result.token_appended = false;
  EXPECT_THROW(request->PromoteFinalStageAsAcceptedDraft(result), std::logic_error);
}

// ---------------------------------------------------------------------------------------------
// Accepted/telemetry accounting: a sampled draft that itself completes the match is genuinely
// target-confirmed and must count as accepted -- for cache/fixed-state CommitPrefix (already
// covered by the kept_tokens assertions above) and for speculative telemetry.
// ---------------------------------------------------------------------------------------------

TEST_F(StopStringEngineTest, SampledAcceptedTelemetryCountsAMatchAtTheFirstPosition) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{7, 12, 13});
  engine.executor->SetVerifyRowTokens({7, 12, 13, 14});
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  const auto stats = engine.engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_proposed, 3u);
  // The match ends verification without evaluating (and rejecting) anything beyond the confirmed
  // prefix: evaluated is exactly the accepted count, not accepted + 1.
  EXPECT_EQ(stats.draft_tokens_evaluated, 1u);
  EXPECT_EQ(stats.draft_tokens_accepted, 1u);
  EXPECT_EQ(stats.acceptance_length_histogram[1], 1u);
  EXPECT_EQ(stats.partial_accept_rounds, 1u);
  EXPECT_EQ(stats.zero_accept_rounds, 0u);
  EXPECT_EQ(stats.full_accept_rounds, 0u);
}

TEST_F(StopStringEngineTest, SampledAcceptedTelemetryCountsAMatchAtAMiddlePosition) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 7, 13});
  engine.executor->SetVerifyRowTokens({11, 7, 13, 14});
  ASSERT_EQ(engine.engine->Run(storage), 2u);

  const auto stats = engine.engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_proposed, 3u);
  EXPECT_EQ(stats.draft_tokens_evaluated, 2u);
  EXPECT_EQ(stats.draft_tokens_accepted, 2u);
  EXPECT_EQ(stats.acceptance_length_histogram[2], 1u);
  EXPECT_EQ(stats.partial_accept_rounds, 1u);
}

TEST_F(StopStringEngineTest, SampledAcceptedTelemetryCountsAFullyConfirmedMatchAtTheLastPosition) {
  // All 3 drafts are genuinely confirmed and the match completes on the last one; a bonus token is
  // drawn (then discarded) but never counted. This is "full acceptance" for telemetry purposes --
  // every proposed draft was confirmed -- even though the round ended via a match, not a bonus.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 12, 7});
  engine.executor->SetVerifyRowTokens({11, 12, 7, 14});
  ASSERT_EQ(engine.engine->Run(storage), 3u);

  const auto stats = engine.engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_proposed, 3u);
  EXPECT_EQ(stats.draft_tokens_evaluated, 3u);
  EXPECT_EQ(stats.draft_tokens_accepted, 3u);
  EXPECT_EQ(stats.acceptance_length_histogram[3], 1u);
  EXPECT_EQ(stats.full_accept_rounds, 1u);
  EXPECT_EQ(stats.partial_accept_rounds, 0u);
}

TEST_F(StopStringEngineTest, NoStopBudgetExhaustedNaturalLastGenuineDraftKeepsValidAcceptedCacheWork) {
  // Regression coverage for the no-stop case review item 2 calls out explicitly: the turn token
  // budget (not a stop match) ends verification exactly on the last genuinely confirmed draft, with
  // no bonus/replacement token ever drawn. AcceptedDraftTokenCount() and cache kept_tokens must
  // still count that final draft as accepted -- this request has no stop controller at all.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), std::optional<size_t>{3});  // no stop strings
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  ASSERT_EQ(request->TurnGeneratedTokens(), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({11, 12, 13, 25});

  // Budget allows exactly 2 more tokens (3 total): the sequential draw loop confirms drafts 0 and
  // 1, then its own budget check (not a rejection) stops it before drawing a bonus/replacement.
  ASSERT_EQ(engine.engine->Run(storage), 2u);
  EXPECT_EQ(storage[1].finish_reason, GenerationFinishReason::TurnLimit);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  // Both confirmed drafts (11, 12) plus the always-kept prior slot -- not 2, which would silently
  // under-count the final genuine draft as though it were an unconfirmed replacement/bonus.
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 3u);
}

TEST_F(StopStringEngineTest, ManualDraftCountFarExceedingTheBudgetStillCountsTheFinalAcceptedDraftOnce) {
  // A manual SetDraftTokens caller proposes far more drafts (5) than the turn's remaining budget
  // (2) can ever use. The natural last selected stage this produces is still a genuinely confirmed
  // draft, not a replacement/bonus (there is no room left to draw one) -- the fix must not infer
  // "last stage => not accepted" from the stage index alone. Every stage must be counted exactly
  // once: the two prior/confirmed stages via AppendAcceptedSampledToken, and the final one via
  // PromoteFinalStageAsAcceptedDraft, never both for the same stage and never zero times.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(5);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), std::optional<size_t>{3});  // no stop strings; budget for 2 more
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  const auto length_after_prefill = request->CurrentSequenceLength();
  const auto processed_after_prefill = request->ProcessedSequenceLength();

  request->SetDraftTokens(std::array<int32_t, 5>{11, 12, 13, 14, 15});
  engine.executor->SetVerifyRowTokens({11, 12, 13, 14, 15, 25});

  ASSERT_EQ(engine.engine->Run(storage), 2u);
  EXPECT_EQ(storage[0].token, 11);
  EXPECT_EQ(storage[0].flags, EngineEventFlagToken);
  EXPECT_EQ(storage[1].token, 12);
  EXPECT_EQ(storage[1].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[1].finish_reason, GenerationFinishReason::TurnLimit);

  // Retained sequence/telemetry reflect exactly 2 accepted drafts out of 5 proposed -- partial,
  // not full, acceptance -- even though the budget (not a target rejection) is what stopped it.
  EXPECT_EQ(request->CurrentSequenceLength(), length_after_prefill + 2);
  const auto stats = engine.engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_proposed, 5u);
  EXPECT_EQ(stats.draft_tokens_accepted, 2u);
  // Exactly 2 proposal positions were logically resolved before the budget boundary (the
  // sequential sampled loop stops before a 3rd draw), not accepted + 1 = 3.
  EXPECT_EQ(stats.draft_tokens_evaluated, 2u);
  EXPECT_EQ(stats.partial_accept_rounds, 1u);
  EXPECT_EQ(stats.full_accept_rounds, 0u);
  EXPECT_EQ(stats.acceptance_length_histogram[2], 1u);

  // Cache/ProcessedSequenceLength count both confirmed drafts (11, 12), not just the first one:
  // the final stage is promoted exactly once, neither silently dropped nor double-counted.
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 3u);  // 1 base + 2 confirmed drafts
  EXPECT_EQ(request->ProcessedSequenceLength(), processed_after_prefill + 3);
  // Unlike an ordinary one-token round (which always leaves its own freshly sampled output 1 token
  // behind processed_sequence_length_, since that output's own KV is computed only when it becomes
  // a future round's input), a draft round's retained final token was itself one of this round's
  // fed positions, so its KV is already computed: processed catches all the way up to current here,
  // not merely close to it. Either way, processed must never exceed current.
  EXPECT_EQ(request->CurrentSequenceLength(), request->ProcessedSequenceLength());

  // A further continuation with new input succeeds as an ordinary decode -- it does not need to
  // reprocess "11" or "12" themselves, only the fresh token it is given, proving the sequence and
  // cache bookkeeping this round left behind is coherent, not a hidden divergence forcing a larger
  // re-prefill than an ordinary continuation would ever need.
  EXPECT_NO_THROW(request->BeginTurn(std::array<int32_t, 1>{25}));
  ASSERT_EQ(engine.engine->Run(storage), 1u);
}

TEST_F(StopStringEngineTest, GreedyDraftCountFarExceedingTheBudgetEvaluatesOnlyTheCommittedPrefix) {
  // Greedy counterpart: the turn/context-limit check inside CommitAcceptedDraftsForTransaction's
  // own loop (not a target rejection) breaks it after 2 of 5 argmax-confirmed drafts. Later target
  // comparisons may already be available, but only these 2 positions are logically resolved into
  // the committed outcome, so evaluated is 2 rather than an inferred accepted + 1 = 3.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(5);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), std::optional<size_t>{3});  // no stop strings; budget for 2 more
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 5>{11, 12, 13, 14, 15});
  engine.executor->SetVerifyRowTokens({11, 12, 13, 14, 15, 25});  // all 5 argmax-confirmed

  ASSERT_EQ(engine.engine->Run(storage), 2u);
  EXPECT_EQ(storage[1].finish_reason, GenerationFinishReason::TurnLimit);

  const auto stats = engine.engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_proposed, 5u);
  EXPECT_EQ(stats.draft_tokens_accepted, 2u);
  EXPECT_EQ(stats.draft_tokens_evaluated, 2u);
  EXPECT_EQ(stats.partial_accept_rounds, 1u);
}

// ---------------------------------------------------------------------------------------------
// Rollback/retry and RNG fidelity for sampled draft verification.
// ---------------------------------------------------------------------------------------------

TEST_F(StopStringEngineTest, DirectRollbackDuringSampledDraftVerificationRestoresControllerAndDrafts) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  const auto generated_before = request->TurnGeneratedTokens();

  request->SetDraftTokens(std::array<int32_t, 3>{11, 7, 12});
  engine.executor->SetVerifyRowTokens({11, 7, 12, 13});
  engine.executor->SetNextFailure(ScriptedExecutionFailure::PostProcessing);

  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagRetryable);
  EXPECT_EQ(request->TurnGeneratedTokens(), generated_before);
  EXPECT_EQ(request->PendingDraftTokenCount(), 3u);
  EXPECT_EQ(request->AcceptedDraftTokenCount(), 0u);
  EXPECT_TRUE(engine.cache->prefix_commits.empty());

  // Retry: re-feeding the exact same draft/verify script must reproduce the exact same match.
  engine.executor->SetVerifyRowTokens({11, 7, 12, 13});
  ASSERT_EQ(engine.engine->Run(storage), 2u);
  EXPECT_EQ(storage[1].token, 7);
  EXPECT_EQ(storage[1].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[1].matched_stop_string_index, 0);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 3u);  // both confirmed drafts counted
}

TEST_F(StopStringEngineTest, SampledStopTruncationPreservesFutureSampleSequence) {
  // A stop match on draft position 0 discards 3 later, already-drawn-but-now-invisible tokens (2
  // more drafts plus a bonus). The retained request's RNG state after rollback must exactly equal
  // a reference request's, which only ever drew the same 2 tokens (1 prefill token, 1 draft) via
  // the very same draft-verification sampling path -- not "advanced as if 4 draws happened".
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;

  // Reference: prefill's own sampled token, then exactly one more draft draw with no bonus (the
  // turn's budget is set to end exactly there), and no stop configured at all.
  auto engine_ref = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine_ref.cache->SetMaxDraftTokensPerStep(3);
  auto ref = CreateEngineRequest(engine_ref.engine, *params);
  ref->BeginTurn(Prompt(), std::optional<size_t>{2});
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine_ref.engine->Run(storage), 1u);

  // Speculative: the same seed and prefill token, then a 3-draft proposal whose first token
  // completes a stop match, discarding 3 later draws made before the match was detected.
  auto engine_spec = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine_spec.cache->SetMaxDraftTokensPerStep(3);
  auto spec = CreateEngineRequest(engine_spec.engine, *params);
  spec->BeginTurn(Prompt(), StopOptions({"STOP"}));
  ASSERT_EQ(engine_spec.engine->Run(storage), 1u);

  ref->SetDraftTokens(std::array<int32_t, 1>{7});
  engine_ref.executor->SetVerifyRowTokens({7, 14});
  ASSERT_EQ(engine_ref.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::TurnLimit);

  spec->SetDraftTokens(std::array<int32_t, 3>{7, 7, 7});
  engine_spec.executor->SetVerifyRowTokens({7, 7, 7, 7});
  ASSERT_EQ(engine_spec.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].token, 7);
  EXPECT_EQ(storage[0].finish_reason, GenerationFinishReason::StopString);

  // Continue both requests through the public Engine API with an injected three-way sampling
  // distribution. Equal future token streams demonstrate that discarded speculative candidates
  // did not advance the stop-truncated request's RNG beyond the retained prefix.
  const auto sample_continuation = [&](DoublesEngine& engine,
                                       const std::shared_ptr<Request>& request) {
    engine.executor->SetVerifyRowTokens({});
    engine.executor->SetSamplingCandidateTokens({20, 21, 22});
    request->BeginTurn(std::array<int32_t, 1>{2}, std::optional<size_t>{16});
    std::vector<int32_t> tokens;
    bool done = false;
    while (!done) {
      const size_t count = engine.engine->Run(storage);
      for (size_t i = 0; i < count; ++i) {
        if (storage[i].request != request) continue;
        if (storage[i].flags & EngineEventFlagToken) tokens.push_back(storage[i].token);
        done = (storage[i].flags & EngineEventFlagTurnFinished) != 0;
      }
    }
    return tokens;
  };

  const auto reference_tokens = sample_continuation(engine_ref, ref);
  const auto speculative_tokens = sample_continuation(engine_spec, spec);
  EXPECT_EQ(speculative_tokens, reference_tokens);
  EXPECT_EQ(speculative_tokens.size(), 16u);
}

TEST_F(StopStringEngineTest, MixedBatchBothSampledRequestsMatchAtStageZeroTerminatesTheStageLoopEarly) {
  // Both requests originally propose 3 drafts (so max_selected_tokens would reach 4 if either had
  // continued to a natural bonus), but both match on their very first draft, truncating both to
  // size 1. Every stage beyond 0 therefore has an empty active[] -- the loop must still produce
  // correct results for both requests despite breaking (or, on this CPU-only fallback branch,
  // simply never entering the per-row loop body) well before the original max_selected_tokens. The
  // can_batch_commits branch is CUDA-only and not reachable from these CPU test doubles, so this
  // does not itself demonstrate the avoided device transfer/sync the early break exists for; that
  // is a structural code-reasoning claim (see the loop's own comment), not something a CPU test can
  // observe directly.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  auto make_request = [&](unsigned int seed) {
    auto p = MakeGreedyParams(*model_);
    p->search.do_sample = true;
    p->search.top_k = 3;
    p->search.temperature = 0.01f;
    p->search.random_seed = seed;
    auto r = CreateEngineRequest(engine.engine, *p);
    r->BeginTurn(Prompt(), StopOptions({"STOP"}));
    return r;
  };
  auto first = make_request(1234);
  auto second = make_request(5678);
  std::array<EngineEvent, 2> prefill;
  ASSERT_EQ(engine.engine->Run(prefill), 2u);

  first->SetDraftTokens(std::array<int32_t, 3>{7, 11, 12});
  second->SetDraftTokens(std::array<int32_t, 3>{7, 13, 14});
  engine.executor->SetVerifyRowTokens({7, 11, 12, 15, 7, 13, 14, 16});

  std::array<EngineEvent, 4> events;
  ASSERT_EQ(engine.engine->Run(events), 2u);
  EXPECT_EQ(events[0].request, first);
  EXPECT_EQ(events[0].token, 7);
  EXPECT_EQ(events[0].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(events[0].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(events[1].request, second);
  EXPECT_EQ(events[1].token, 7);
  EXPECT_EQ(events[1].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(events[1].finish_reason, GenerationFinishReason::StopString);
}

TEST_F(StopStringEngineTest, StopStringPrecedesTurnLimitForTheSameCompletingDraftToken) {
  // The matching draft is also the exact token that would otherwise exhaust the turn's generated-
  // token budget; StopString must still be reported, not TurnLimit.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}, std::optional<size_t>{3}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);  // 1 of 3 generated tokens used

  request->SetDraftTokens(std::array<int32_t, 3>{11, 7, 12});
  engine.executor->SetVerifyRowTokens({11, 7, 12, 13});

  // Committing "OP" (offset 1) both completes "STOP" and would independently exhaust the 3-token
  // budget (1 prior + this round's 2 accepted so far == 3); the stop check runs first in the
  // greedy loop and breaks before the turn-limit check for this same token is ever reached.
  ASSERT_EQ(engine.engine->Run(storage), 2u);
  EXPECT_EQ(storage[1].token, 7);
  EXPECT_EQ(storage[1].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[1].finish_reason, GenerationFinishReason::StopString);
  EXPECT_NE(storage[1].finish_reason, GenerationFinishReason::TurnLimit);
}

TEST_F(StopStringEngineTest, SampledStopStringPrecedesTurnLimitForTheSameCompletingDraftToken) {
  // Sampled-path regression coverage for the same precedence: this reuses StageGeneration()'s own,
  // unchanged stop-before-turn-limit ordering (the same mechanism the ordinary and greedy paths
  // already rely on), so this specifically guards against a bookkeeping change to
  // accepted_draft_count_/confirmed_draft_counts/PromoteFinalStageAsAcceptedDraft accidentally
  // disturbing that ordering rather than exercising new precedence logic of its own.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}, std::optional<size_t>{3}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);  // 1 of 3 generated tokens used

  request->SetDraftTokens(std::array<int32_t, 3>{11, 7, 12});
  engine.executor->SetVerifyRowTokens({11, 7, 12, 13});

  // Stage 1 ("OP", token 7) both completes "STOP" and would independently exhaust the 3-token
  // budget (1 prior + 2 accepted this round == 3); stop precedence must still win.
  ASSERT_EQ(engine.engine->Run(storage), 2u);
  EXPECT_EQ(storage[1].token, 7);
  EXPECT_EQ(storage[1].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(storage[1].finish_reason, GenerationFinishReason::StopString);
  EXPECT_NE(storage[1].finish_reason, GenerationFinishReason::TurnLimit);
  ASSERT_EQ(engine.cache->prefix_commits.size(), 1u);
  EXPECT_EQ(engine.cache->prefix_commits[0].kept_tokens, 3u);  // both confirmed drafts counted
}

TEST_F(StopStringEngineTest, MixedBatchStopEnabledSpeculativeRequestTruncatesWhileSiblingCommitsNormally) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto stopping = CreateEngineRequest(engine.engine, *model_);
  stopping->BeginTurn(Prompt(), StopOptions({"STOP"}));
  auto plain = CreateEngineRequest(engine.engine, *model_);
  plain->BeginTurn(Prompt());

  std::array<EngineEvent, 2> prefill;
  ASSERT_EQ(engine.engine->Run(prefill), 2u);

  stopping->SetDraftTokens(std::array<int32_t, 3>{11, 7, 12});
  plain->SetDraftTokens(std::array<int32_t, 1>{14});
  engine.executor->SetVerifyRowTokens({11, 7, 12, 13, 14, 25});

  std::array<EngineEvent, 4> events;
  ASSERT_EQ(engine.engine->Run(events), 4u);
  EXPECT_EQ(events[0].request, stopping);
  EXPECT_EQ(events[0].token, 11);
  EXPECT_EQ(events[0].flags, EngineEventFlagToken);
  EXPECT_EQ(events[1].request, stopping);
  EXPECT_EQ(events[1].token, 7);
  EXPECT_EQ(events[1].flags, EngineEventFlagToken | EngineEventFlagTurnFinished);
  EXPECT_EQ(events[1].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(events[2].request, plain);
  EXPECT_EQ(events[2].token, 14);
  EXPECT_EQ(events[2].flags, EngineEventFlagToken);
  EXPECT_EQ(events[3].request, plain);
  EXPECT_EQ(events[3].token, 25);
  EXPECT_EQ(events[3].flags, EngineEventFlagToken);
  EXPECT_EQ(plain->Status(), RequestStatus::Active);
  EXPECT_EQ(stopping->Status(), RequestStatus::TurnComplete);
}

TEST_F(StopStringEngineTest, AcceptedDraftTokenCountAndTelemetryReflectATruncatedGreedyMatch) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 7, 12});
  engine.executor->SetVerifyRowTokens({11, 7, 12, 13});
  ASSERT_EQ(engine.engine->Run(storage), 2u);

  // The proposed count (3) and evaluated count still reflect the round the target actually saw.
  // Both the greedy and sampled/batched paths count the matching draft itself as accepted -- it
  // is genuinely target-confirmed (the argmax/sample agreed with it) -- just via different
  // mechanisms: greedy's loop increments accepted_draft_count_ before checking for a match, so the
  // match is already included the instant it is observed; the sampled/batched stage loop instead
  // promotes it after the fact via Request::PromoteFinalStageAsAcceptedDraft, once the caller knows
  // this stage is both the round's end and within the confirmed prefix. Because the match completes
  // on this confirmed draft (token_appended == false, from StageDraftCompletionForTransaction's
  // hardcoded result), evaluated is exactly the accepted count, not accepted + 1.
  const auto stats = engine.engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_proposed, 3u);
  EXPECT_EQ(stats.draft_tokens_accepted, 2u);
  EXPECT_EQ(stats.draft_tokens_evaluated, 2u);
}

TEST_F(StopStringEngineTest, GreedyRejectedDraftReplacementStopTelemetryCountsTheEvaluatedRejection) {
  // Mirrors GreedyRejectedDraftNeverReachesTheMatcherButItsReplacementCan's setup, adding the
  // telemetry assertions review correction #3 calls out: unlike a match on a confirmed draft, a
  // match on the ordinary replacement following a rejected draft still has token_appended == true
  // (it flows through the ordinary ApplyLogitsForTransaction path, not
  // StageDraftCompletionForTransaction), so verification genuinely evaluated and rejected one
  // draft to reach it -- evaluated must be accepted + 1 (= 1), not accepted (= 0).
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({7, 12, 13, 14});  // row 0 disagrees with draft 0
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  const auto stats = engine.engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_proposed, 3u);
  EXPECT_EQ(stats.draft_tokens_accepted, 0u);
  EXPECT_EQ(stats.draft_tokens_evaluated, 1u);
  EXPECT_EQ(stats.zero_accept_rounds, 1u);
  EXPECT_EQ(stats.acceptance_length_histogram[0], 1u);
  // The aggregate acceptance-rate denominator (evaluated, not proposed) reflects the same fix.
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 0.0f);
}

TEST_F(StopStringEngineTest, SampledRejectedDraftReplacementStopTelemetryCountsTheEvaluatedRejection) {
  // Sampled-path counterpart: the replacement token flows through StageGeneration() with
  // token_appended == true (the stage loop never calls PromoteFinalStageAsAcceptedDraft for it,
  // since it falls outside confirmed_draft_counts), so the same accepted + 1 rule applies.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto params = MakeGreedyParams(*model_);
  params->search.do_sample = true;
  params->search.top_k = 3;
  params->search.temperature = 0.01f;
  params->search.random_seed = 1234;
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  request->SetDraftTokens(std::array<int32_t, 3>{11, 12, 13});
  engine.executor->SetVerifyRowTokens({7, 12, 13, 14});  // target replaces draft 0 with "STOPX"
  ASSERT_EQ(engine.engine->Run(storage), 1u);

  const auto stats = engine.engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_proposed, 3u);
  EXPECT_EQ(stats.draft_tokens_accepted, 0u);
  EXPECT_EQ(stats.draft_tokens_evaluated, 1u);
  EXPECT_EQ(stats.zero_accept_rounds, 1u);
  EXPECT_EQ(stats.acceptance_length_histogram[0], 1u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 0.0f);
}

// ---------------------------------------------------------------------------------------------
// Rollback and replay of draft verification: the request-local StopStringController checkpoint
// taken at BeginTransaction() covers whatever the draft loop or sampled stage loop observed, so a
// rolled-back step must restore both the draft proposal and the controller for an identical retry.
// ---------------------------------------------------------------------------------------------

TEST_F(StopStringEngineTest, DirectRollbackDuringGreedyDraftVerificationRestoresControllerAndDrafts) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  const auto generated_before = request->TurnGeneratedTokens();

  request->SetDraftTokens(std::array<int32_t, 3>{11, 7, 12});
  engine.executor->SetVerifyRowTokens({11, 7, 12, 13});
  engine.executor->SetNextFailure(ScriptedExecutionFailure::PostProcessing);

  EXPECT_EQ(RunOne(*engine.engine).flags, EngineEventFlagRetryable);
  EXPECT_EQ(request->TurnGeneratedTokens(), generated_before);
  EXPECT_EQ(request->PendingDraftTokenCount(), 3u);
  EXPECT_EQ(request->AcceptedDraftTokenCount(), 0u);
  EXPECT_TRUE(engine.cache->prefix_commits.empty());

  // Retry: re-feeding the exact same draft/verify script must reproduce the exact same match.
  engine.executor->SetVerifyRowTokens({11, 7, 12, 13});
  ASSERT_EQ(engine.engine->Run(storage), 2u);
  EXPECT_EQ(storage[1].token, 7);
  EXPECT_EQ(storage[1].finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(storage[1].matched_stop_string_index, 0);
}

TEST_F(StopStringEngineTest, QueuedRollbackDuringGreedyDraftVerificationRestoresControllerAndDrafts) {
  // Mirrors QueuedRollbackReplaysTheControllerInTheCompletionPhase, but drives the draft loop
  // directly at the Request level so it stays independent of Engine::Run's multi-request queued-
  // restore machinery while still exercising Queue*/Complete* (not Restore*) for the draft path.
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/5 /* "ST" */);
  engine.cache->SetMaxDraftTokensPerStep(3);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 4> storage;
  ASSERT_EQ(engine.engine->Run(storage), 1u);  // commits "ST"

  const auto before = request->Snapshot();
  request->SetDraftTokens(std::array<int32_t, 1>{6});  // "OP": completes "STOP"
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.target_cache_slots = static_cast<size_t>(before.current_sequence_length) + 1;
  plan.draft_token_count = 1;
  PrepareRequestStep(model_, plan);

  request->SaveStateForTransaction();
  request->AppendDraftsForTransaction(1);
  request->CommitAcceptedDraftsForTransaction(1);
  ASSERT_TRUE(request->DraftVerificationCompletedGeneration());
  const auto staged = request->StageDraftCompletionForTransaction();
  ASSERT_EQ(staged.finish_reason, GenerationFinishReason::StopString);

  request->QueueStateRestoreForTransaction();
  request->CompleteStateRestoreForTransaction();
  EXPECT_EQ(request->PendingDraftTokenCount(), 1u);
  EXPECT_EQ(request->AcceptedDraftTokenCount(), 0u);

  // Retry: re-stage and commit the same draft; the match must reappear identically.
  request->SaveStateForTransaction();
  request->AppendDraftsForTransaction(1);
  request->CommitAcceptedDraftsForTransaction(1);
  ASSERT_TRUE(request->DraftVerificationCompletedGeneration());
  const auto retried = request->StageDraftCompletionForTransaction();
  ASSERT_EQ(retried.finish_reason, GenerationFinishReason::StopString);
  ASSERT_EQ(retried.matched_stop_string_index, 0);
  request->CommitStateForTransaction();
  request->CommitStep(plan, retried);

  EXPECT_EQ(request->TurnGeneratedTokens(), 2u);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::StopString);
  EXPECT_EQ(request->MatchedStopStringIndex(), 0);
}

TEST(StopStringMtpSpeculativeTest, AutomaticMtpDraftsAreProposedForAStopEnabledRequestNotYetDone) {
  auto model = LoadSyntheticPagedMtpModel();
  const int32_t eos = EosToken(*model);
  const int32_t filler = eos == 5 ? 6 : 5;  // never EOS, never matches "UNREACHABLE"
  auto engine = MakeMtpDoublesEngine(model, filler);

  auto stop_request = CreateEngineRequest(engine.engine, *model);
  stop_request->BeginTurn(Prompt(), StopOptions({"UNREACHABLE"}));
  auto plain_request = CreateRequestWithPrompt(engine.engine, *model, Prompt());

  std::array<EngineEvent, 8> storage;
  ASSERT_GT(engine.engine->Run(storage), 0u);

  // Both requests receive an MTP shadow now: a stop-enabled turn that has not yet matched is no
  // longer excluded from automatic drafting, and it drafts and verifies request-locally without
  // needing any producer-specific carve-out.
  EXPECT_EQ(engine.mtp_cache->AllocatedCount(), 2u);
}

TEST(StopStringMtpSpeculativeTest, StopEnabledRequestThatJustMatchedIsExcludedFromTheNextMtpDraftBlock) {
  auto model = LoadSyntheticPagedMtpModel();
  auto engine = MakeMtpDoublesEngine(model, /*forced_token=*/5 /* "ST" */);

  auto stop_request = CreateEngineRequest(engine.engine, *model);
  stop_request->BeginTurn(Prompt(), StopOptions({"STOP"}));
  std::array<EngineEvent, 8> storage;
  ASSERT_GT(engine.engine->Run(storage), 0u);
  ASSERT_EQ(engine.mtp_cache->AllocatedCount(), 1u);  // "ST": no match yet, drafts normally

  engine.executor->SetForcedToken(6);  // "OP": completes "STOP"
  const size_t drained = engine.engine->Run(storage);
  ASSERT_GT(drained, 0u);
  bool saw_stop_finish = false;
  for (size_t i = 0; i < drained; ++i) {
    if (storage[i].request == stop_request &&
        (storage[i].flags & EngineEventFlagTurnFinished)) {
      EXPECT_EQ(storage[i].finish_reason, GenerationFinishReason::StopString);
      saw_stop_finish = true;
    }
  }
  EXPECT_TRUE(saw_stop_finish);
  EXPECT_EQ(stop_request->Status(), RequestStatus::TurnComplete);
  // A terminal target result -- this StopString match -- prevents a subsequent draft block: no
  // new MTP shadow is created for it once it is done, exactly like any other finish reason.
  EXPECT_EQ(engine.mtp_cache->AllocatedCount(), 1u);
}

// End-to-end: unlike the test above (where the MTP-proposed draft is rejected and the target's own
// replacement happens to complete the match), this one controls the MTP head's own forced token so
// the *proposed draft itself* is genuinely accepted and completes the match through
// Request::CommitAcceptedDraftsForTransaction()'s greedy loop -- the same generic path a manual
// SetDraftTokens caller uses, exercised here through the fully automatic producer. The doubles'
// CPU device still chains the MTP head to this Engine's full 3-token draft budget (chaining beyond
// one token is not itself CUDA-gated the way the batched device-stage path is), so the proposal is
// "STOPX","STOPX","STOPX"; the match completes on the very first of the three, mid-draft.
TEST(StopStringMtpSpeculativeTest, AutomaticMtpProposedDraftItselfCompletesAMidDraftMatchAndIsNotRedrafted) {
  auto model = LoadSyntheticPagedMtpModel();
  auto engine = MakeMtpDoublesEngine(model, /*forced_token=*/11 /* filler, never matches "STOP" */);
  // Every MTP proposal from here on predicts "STOPX" (token 7).
  engine.mtp_executor->SetForcedToken(7);

  auto stop_request = CreateEngineRequest(engine.engine, *model);
  stop_request->BeginTurn(Prompt(), StopOptions({"STOP"}));

  std::array<EngineEvent, 8> storage;
  ASSERT_GT(engine.engine->Run(storage), 0u);  // prefill + "t11"; MTP proposes 3 "STOPX" drafts
  ASSERT_EQ(engine.mtp_cache->AllocatedCount(), 1u);
  ASSERT_EQ(stop_request->PendingDraftTokenCount(), 3u);
  ASSERT_EQ(stop_request->FinishReason(), GenerationFinishReason::None);

  // The target's own argmax agrees with the proposed draft at position 0 (accepting it, not
  // replacing it), so the draft is genuinely committed through the greedy verification loop; the
  // match completes there and the loop never even looks at positions 1, 2, or the bonus row.
  engine.executor->SetVerifyRowTokens({7, 25, 26, 27});
  const size_t drained = engine.engine->Run(storage);
  ASSERT_GT(drained, 0u);
  bool saw_stop_finish = false;
  for (size_t i = 0; i < drained; ++i) {
    if (storage[i].request == stop_request && (storage[i].flags & EngineEventFlagTurnFinished)) {
      EXPECT_EQ(storage[i].token, 7);
      EXPECT_EQ(storage[i].finish_reason, GenerationFinishReason::StopString);
      EXPECT_EQ(storage[i].matched_stop_string_index, 0);
      saw_stop_finish = true;
    }
  }
  EXPECT_TRUE(saw_stop_finish);
  EXPECT_EQ(stop_request->Status(), RequestStatus::TurnComplete);
  EXPECT_EQ(stop_request->FinishReason(), GenerationFinishReason::StopString);

  // No redraft: the terminal target result this round (result.done == true) makes
  // Engine::PrepareMtpStep skip proposing a new draft block for it entirely.
  EXPECT_EQ(stop_request->PendingDraftTokenCount(), 0u);
  EXPECT_EQ(engine.mtp_cache->AllocatedCount(), 1u);

  // A further Run() neither drafts nor otherwise touches the now-terminal request again.
  ASSERT_EQ(engine.engine->Run(storage), 0u);
  EXPECT_EQ(stop_request->PendingDraftTokenCount(), 0u);
}

// The sibling-keeps-drafting half of the same claim: AutomaticMtpDraftsAreProposedForAStopEnabled-
// RequestNotYetDone above already proves both a stop-enabled and a plain request receive an MTP
// shadow from the same automatic drafter with no producer-specific carve-out; this extends it one
// more round to prove the plain sibling's own drafting keeps working normally, undisturbed by the
// other request's stop configuration. Exact interleaved-row control for a second automatic MTP
// verification round on the *stop-enabled* request's own sibling would require duplicating the
// packed-row bookkeeping above per request; this instead exercises the sibling directly, which is
// the actual property being verified.
TEST(StopStringMtpSpeculativeTest, PlainSiblingContinuesAutomaticMtpDraftingAcrossRounds) {
  auto model = LoadSyntheticPagedMtpModel();
  auto engine = MakeMtpDoublesEngine(model, /*forced_token=*/11);
  engine.mtp_executor->SetForcedToken(12);

  auto stop_request = CreateEngineRequest(engine.engine, *model);
  stop_request->BeginTurn(Prompt(), StopOptions({"UNREACHABLE"}));
  auto plain_request = CreateRequestWithPrompt(engine.engine, *model, Prompt());

  std::array<EngineEvent, 8> storage;
  ASSERT_GT(engine.engine->Run(storage), 0u);
  ASSERT_EQ(engine.mtp_cache->AllocatedCount(), 2u);
  ASSERT_EQ(plain_request->PendingDraftTokenCount(), 3u);
  ASSERT_EQ(stop_request->PendingDraftTokenCount(), 3u);

  // Neither request's proposed draft ("t12") matches "UNREACHABLE"; both continue normally
  // (draft0/1/2 accepted, plus a bonus token that also decodes to unreachable filler text), and
  // the plain sibling keeps receiving fresh automatic drafts round after round.
  engine.executor->SetVerifyRowTokens({12, 12, 12, 13, 12, 12, 12, 14});
  ASSERT_GT(engine.engine->Run(storage), 0u);
  EXPECT_EQ(plain_request->Status(), RequestStatus::Active);
  EXPECT_EQ(stop_request->Status(), RequestStatus::Active);
  EXPECT_EQ(plain_request->PendingDraftTokenCount(), 3u);
  EXPECT_EQ(engine.mtp_cache->AllocatedCount(), 2u);
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
  executing->BeginTurn(Prompt(), StopOptions({"UNREACHABLE"}));
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

TEST_F(StopStringEngineTest, UnserviceableStopEnabledRequestPublishesFailure) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/2);
  auto request = CreateEngineRequest(engine.engine, *model_);
  request->BeginTurn(Prompt(), StopOptions({"UNREACHABLE"}));
  engine.cache->SetUnserviceableRequest(request);

  const auto failure = RunOne(*engine.engine);
  EXPECT_EQ(failure.request, request);
  EXPECT_EQ(failure.flags, EngineEventFlagTurnFinished | EngineEventFlagFailed);
  EXPECT_EQ(failure.error_code, EngineErrorCode::RequestUnserviceable);
  EXPECT_EQ(request->Status(), RequestStatus::TurnComplete);
  EXPECT_EQ(request->FinishReason(), GenerationFinishReason::Failed);
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
