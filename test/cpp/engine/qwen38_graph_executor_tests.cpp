// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Engine-level tests of the static single-slot Qwen3.8-Flash CUDA-graph backend. A scripted device replaces the CUDA
// loop (see qwen38_graph_test_doubles.h); everything else is the real Engine: StaticBatchScheduler, the single-slot
// cache manager, the executor, Request::CommitExternalTokens and the Engine event path. The fixture model is the
// synthetic-paged one because it carries a tiny tokenizer for the stop-string cases (token 5 -> "ST", 6 -> "OP",
// 0/1 special, EOS is 1).

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engine/qwen38_graph_executor.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"
#include "models/qwen38_ort_env.h"
#include "qwen38_graph_test_doubles.h"

namespace Generators::test {
namespace {

constexpr int32_t kEos = 1;

// A reusable barrier for a fixed number of threads. It gives up (instead of hanging the test) when a peer is gone.
class Rendezvous {
 public:
  explicit Rendezvous(size_t parties) : parties_{parties} {}

  // Blocks until every party has arrived; false when the rendezvous was abandoned or the peers never came.
  bool ArriveAndWait() {
    std::unique_lock<std::mutex> lock{mutex_};
    if (abandoned_) return false;
    const size_t generation = generation_;
    if (++arrived_ == parties_) {
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
      return true;
    }
    cv_.wait_for(lock, std::chrono::seconds(30), [&] { return generation_ != generation || abandoned_; });
    return generation_ != generation;
  }

  void Abandon() {
    {
      std::lock_guard<std::mutex> lock{mutex_};
      abandoned_ = true;
    }
    cv_.notify_all();
  }

  size_t Completed() {
    std::lock_guard<std::mutex> lock{mutex_};
    return generation_;
  }

 private:
  const size_t parties_;
  std::mutex mutex_;
  std::condition_variable cv_;
  size_t arrived_{};
  size_t generation_{};
  bool abandoned_{};
};

// Sets a process environment variable for its lifetime and restores the previous state: the stand-in for the variable a
// process-wide publication of the prompt length would write.
class ScopedProcessEnv {
 public:
  ScopedProcessEnv(const char* name, const char* value) : name_{name} {
    if (const char* previous = std::getenv(name)) previous_ = previous;
    Write(value);
  }
  ~ScopedProcessEnv() {
    if (previous_) {
      Write(previous_->c_str());
    } else {
      Unset();
    }
  }
  ScopedProcessEnv(const ScopedProcessEnv&) = delete;
  ScopedProcessEnv& operator=(const ScopedProcessEnv&) = delete;

 private:
  void Write(const char* value) const {
#if defined(_WIN32)
    _putenv_s(name_, value);
#else
    setenv(name_, value, 1);
#endif
  }
  void Unset() const {
#if defined(_WIN32)
    _putenv_s(name_, "");  // an empty value removes the variable
#else
    unsetenv(name_);
#endif
  }

  const char* name_;
  std::optional<std::string> previous_;
};

struct Outcome {
  std::vector<int32_t> tokens;
  GenerationFinishReason finish_reason{GenerationFinishReason::None};
  int32_t matched_stop_string_index{-1};
  TurnUsage usage{};
  size_t runs{};
  size_t finished_events{};
  uint32_t failure_flags{};
};

class Qwen38GraphEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadSyntheticPagedModel();
    model_->config_->engine.dynamic_batching.reset();
    model_->config_->model.qwen38_graph.prefill.chunk = 4;
    device_ = std::make_shared<ScriptedQwen38Device>();
  }

  std::shared_ptr<Engine> MakeEngine() {
    return std::make_shared<Engine>(model_, CreateQwen38GraphDependencies(model_, device_));
  }

  static std::vector<int32_t> Prompt(size_t count, int32_t first = 12) {
    std::vector<int32_t> prompt(count);
    for (size_t i = 0; i < count; ++i) prompt[i] = first + static_cast<int32_t>(i % 20);
    return prompt;
  }

  // Runs the Engine until it has no work and returns what `request` produced.
  static Outcome RunToCompletion(Engine& engine, const std::shared_ptr<Request>& request, size_t max_runs = 10000) {
    Outcome outcome;
    std::array<EngineEvent, 8> buffer;
    while (engine.HasPendingRequests()) {
      if (++outcome.runs > max_runs) {
        ADD_FAILURE() << "the Engine did not finish";
        break;
      }
      const size_t count = engine.Run(buffer);
      for (size_t i = 0; i < count; ++i) {
        const EngineEvent& event = buffer[i];
        if (event.request != request) continue;
        if (event.flags & EngineEventFlagToken) outcome.tokens.push_back(event.token);
        if (event.flags & (EngineEventFlagFailed | EngineEventFlagRetryable | EngineEventFlagCapacityBlocked)) {
          outcome.failure_flags |= event.flags;
        }
        if (event.flags & EngineEventFlagTurnFinished) {
          ++outcome.finished_events;
          outcome.finish_reason = event.finish_reason;
          outcome.matched_stop_string_index = event.matched_stop_string_index;
          outcome.usage = event.usage;
        }
      }
    }
    return outcome;
  }

  std::vector<int32_t> Expected(size_t first_position, size_t count) const {
    std::vector<int32_t> tokens;
    for (size_t i = 0; i < count; ++i) tokens.push_back(device_->TokenAt(first_position + i));
    return tokens;
  }

  // Runs the Engine until an event reports a failure and returns its error code (nullopt: none did).
  static std::optional<EngineErrorCode> RunUntilFailure(Engine& engine, size_t max_runs = 100) {
    std::array<EngineEvent, 8> buffer;
    for (size_t run = 0; run < max_runs && engine.HasPendingRequests(); ++run) {
      const size_t count = engine.Run(buffer);
      for (size_t i = 0; i < count; ++i) {
        if (buffer[i].flags & EngineEventFlagFailed) return buffer[i].error_code;
      }
    }
    return std::nullopt;
  }

  static TurnOptions Limit(size_t max_generated_tokens) {
    TurnOptions options;
    options.max_generated_tokens = max_generated_tokens;
    return options;
  }

  std::shared_ptr<Request> Begin(Engine& engine, std::span<const int32_t> prompt, size_t max_generated_tokens,
                                 std::optional<size_t> max_session_tokens = std::nullopt) {
    RequestOptions request_options;
    request_options.max_session_tokens = max_session_tokens;
    auto request = engine.CreateRequest(request_options);
    auto options = Limit(max_generated_tokens);
    options.request = request;
    request->BeginTurn(prompt, options);
    return request;
  }

  std::shared_ptr<Request> BeginWithStopString(Engine& engine, std::span<const int32_t> prompt, size_t max_generated_tokens) {
    auto request = engine.CreateRequest();
    auto options = Limit(max_generated_tokens);
    options.request = request;
    options.stop_strings = {"STOP"};
    request->BeginTurn(prompt, options);
    return request;
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<ScriptedQwen38Device> device_;
};

// ---- token path ------------------------------------------------------------------------------

TEST_F(Qwen38GraphEngineTest, GeneratesTheScriptedStreamAndEndsAtTheTurnLimit) {
  auto engine = MakeEngine();
  const auto prompt = Prompt(5);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/7);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 7));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::TurnLimit);
  EXPECT_EQ(outcome.finished_events, 1u);
  EXPECT_EQ(outcome.usage.prompt_tokens, 5u);
  EXPECT_EQ(outcome.usage.generated_tokens, 7u);
  EXPECT_EQ(request->TokensCpu().size(), 12u);

  // Accept pattern {accept, reject}: tokens 1 | 2 | 1 | 2 | 1 = 7 over 4 cycles; the last cycle was waited for and
  // no further cycle was enqueued.
  EXPECT_EQ(device_->CallCount("wait"), 4u);
  EXPECT_EQ(device_->cycles_enqueued, 4u);
  EXPECT_FALSE(device_->inflight());
  EXPECT_EQ(device_->calls.back(), "advance:0");
  EXPECT_EQ(device_->ingested, prompt);

  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.rounds, 4u);
  EXPECT_EQ(stats.draft_tokens_proposed, 4u);
  EXPECT_EQ(stats.draft_tokens_accepted, 2u);
  EXPECT_EQ(stats.bonus_tokens, 2u);
  EXPECT_EQ(stats.correction_tokens, 2u);
  EXPECT_EQ(stats.tokens_emitted, 7u);
  EXPECT_EQ(stats.acceptance_length_histogram[0], 2u);
  EXPECT_EQ(stats.acceptance_length_histogram[1], 2u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 0.5f);
  EXPECT_FLOAT_EQ(stats.avg_draft_tokens_per_round, 1.0f);
}

TEST_F(Qwen38GraphEngineTest, TheTurnLimitCutsAnAcceptedCycleAndEndsTheLoop) {
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/2);

  const auto outcome = RunToCompletion(*engine, request);

  // First token + the accepted first cycle would commit 3 tokens; only 2 are allowed.
  EXPECT_EQ(outcome.tokens, Expected(5, 2));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::TurnLimit);
  EXPECT_EQ(device_->calls.back(), "advance:0");
  EXPECT_FALSE(device_->inflight());
  EXPECT_EQ(engine->GetSpeculativeStats().tokens_discarded, 1u);
}

TEST_F(Qwen38GraphEngineTest, ATurnLimitOfOneEndsOnThePromptPredictionWithoutStartingACycle) {
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/1);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 1));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::TurnLimit);
  EXPECT_EQ(outcome.finished_events, 1u);
  EXPECT_EQ(device_->cycles_enqueued, 0u);
  EXPECT_EQ(device_->CallCount("wait"), 0u);
  EXPECT_EQ(engine->GetSpeculativeStats().rounds, 0u);
}

TEST_F(Qwen38GraphEngineTest, AnEndOfSequenceTokenFirstInACycleEndsTheTurnWithoutBeingDelivered) {
  device_->overrides[6] = kEos;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 1));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(outcome.usage.generated_tokens, 1u);
  EXPECT_EQ(device_->calls.back(), "advance:0");
  EXPECT_FALSE(device_->inflight());
  EXPECT_EQ(device_->CallCount("wait"), 1u);
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.tokens_emitted, 1u);
  EXPECT_EQ(stats.tokens_discarded, 2u);
  EXPECT_EQ(stats.draft_tokens_accepted, 0u);
  EXPECT_EQ(stats.bonus_tokens, 0u);
}

TEST_F(Qwen38GraphEngineTest, AnEndOfSequenceTokenSecondInAnAcceptedCycleKeepsTheFirstToken) {
  device_->overrides[7] = kEos;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 2));  // positions 5 and 6; the EOS at 7 is not delivered
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(outcome.usage.generated_tokens, 2u);
  EXPECT_EQ(device_->calls.back(), "advance:0");
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.tokens_emitted, 2u);
  EXPECT_EQ(stats.tokens_discarded, 1u);
  EXPECT_EQ(stats.draft_tokens_accepted, 1u);
  EXPECT_EQ(stats.bonus_tokens, 0u);
}

TEST_F(Qwen38GraphEngineTest, AnEndOfSequenceFirstTokenNeverStartsACycle) {
  device_->overrides[5] = kEos;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_TRUE(outcome.tokens.empty());
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::EosToken);
  EXPECT_EQ(outcome.finished_events, 1u);
  EXPECT_EQ(device_->CallCount("begin:0"), 1u);
  EXPECT_EQ(device_->CallCount("wait"), 0u);
  EXPECT_EQ(device_->cycles_enqueued, 0u);
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.tokens_emitted, 0u);
  EXPECT_EQ(stats.tokens_discarded, 0u);
  EXPECT_EQ(stats.rounds, 0u);
}

TEST_F(Qwen38GraphEngineTest, TheSessionLimitEndsTheTurnWithTheContextLimitReason) {
  auto engine = MakeEngine();
  // 5 prompt tokens, room for 3 more.
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/50, /*max_session_tokens=*/8);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 3));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::ContextLimit);
  EXPECT_EQ(request->CurrentSequenceLength(), 8);
  EXPECT_EQ(device_->calls.back(), "advance:0");
}

TEST_F(Qwen38GraphEngineTest, ASessionThatFillsTheWholeKvCapacityEndsWithTheContextLimitReason) {
  auto engine = MakeEngine();
  const size_t limit = engine->GetCapabilities().max_request_length;
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/limit, /*max_session_tokens=*/limit);

  const auto outcome = RunToCompletion(*engine, request);

  // The device ledger throws when a cycle does not fit, so reaching the limit cleanly proves the reserved tail is honored.
  EXPECT_EQ(outcome.failure_flags, 0u);
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::ContextLimit);
  EXPECT_EQ(outcome.tokens, Expected(5, limit - 5));
  EXPECT_EQ(request->CurrentSequenceLength(), static_cast<int64_t>(limit));
}

TEST_F(Qwen38GraphEngineTest, AStopStringEndsTheTurnAndTheWastedLookAheadCycleIsHarmless) {
  // Tokens 5 and 6 decode to "ST" and "OP": the stop string "STOP" completes on the first token of the first cycle.
  device_->overrides[5] = 5;
  device_->overrides[6] = 6;
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  options.max_generated_tokens = 20;
  options.stop_strings = {"STOP"};
  request->BeginTurn(Prompt(5), options);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(outcome.matched_stop_string_index, 0);
  EXPECT_EQ(outcome.tokens, (std::vector<int32_t>{5, 6}));
  const auto stop_stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stop_stats.tokens_emitted, 2u);
  EXPECT_EQ(stop_stats.tokens_discarded, 1u);
  EXPECT_EQ(stop_stats.draft_tokens_accepted, 1u);
  EXPECT_EQ(stop_stats.bonus_tokens, 0u);
  // The device cannot know about stop strings: the cycle after the one that matched is still in flight...
  EXPECT_TRUE(device_->inflight());
  EXPECT_TRUE(device_->running());

  // ...and a new request waits for it (Reset) before reusing the device.
  const size_t resets = device_->CallCount("reset");
  device_->overrides.clear();
  auto next = Begin(*engine, Prompt(5), /*max_generated_tokens=*/3);
  const auto next_outcome = RunToCompletion(*engine, next);
  EXPECT_EQ(device_->CallCount("reset"), resets + 1);
  EXPECT_EQ(device_->status_checks, 2u);
  EXPECT_EQ(next_outcome.tokens, Expected(5, 3));
}

TEST_F(Qwen38GraphEngineTest, StopStringsSpanningTheTwoTokensOfACycleAreMatched) {
  device_->overrides[6] = 5;
  device_->overrides[7] = 6;
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  options.stop_strings = {"STOP"};
  request->BeginTurn(Prompt(5), options);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(outcome.tokens, (std::vector<int32_t>{device_->TokenAt(5), 5, 6}));
}

TEST_F(Qwen38GraphEngineTest, EveryDraftRejectedCommitsTheGreedyTokenOfEachCycleOnly) {
  device_->accept_pattern = {false};  // the fold.force_reject hook: no draft ever survives its verify
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/6);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 6));  // the same stream as with any other accept pattern
  EXPECT_EQ(device_->CallCount("wait"), 5u);  // the first token comes from the prompt, then one token per cycle
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.rounds, 5u);
  EXPECT_EQ(stats.draft_tokens_accepted, 0u);
  EXPECT_EQ(stats.zero_accept_rounds, 5u);
  EXPECT_EQ(stats.acceptance_length_histogram[0], 5u);
  EXPECT_EQ(device_->ledger_length(), 5u + 5u);
}

TEST_F(Qwen38GraphEngineTest, EveryDraftAcceptedCommitsTwoTokensPerCycle) {
  device_->accept_pattern = {true};
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/9);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 9));
  EXPECT_EQ(device_->CallCount("wait"), 4u);  // 1 + 4 * 2 tokens
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_accepted, 4u);
  EXPECT_EQ(stats.full_accept_rounds, 4u);
  EXPECT_EQ(device_->ledger_length(), 5u + 8u);
  EXPECT_EQ(device_->ledger_bank(), static_cast<int>(device_->ledger_length() % 3));
}

TEST_F(Qwen38GraphEngineTest, WithoutMtpEveryCycleCommitsOneToken) {
  device_ = std::make_shared<ScriptedQwen38Device>(/*capacity=*/68, /*mtp=*/false);
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(4), /*max_generated_tokens=*/5);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(4, 5));
  EXPECT_EQ(device_->CallCount("wait"), 4u);
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_proposed, 0u);
  EXPECT_EQ(stats.target_forward_passes, 4u);
  EXPECT_EQ(stats.tokens_emitted, 5u);
}

// ---- speculative statistics of multi-draft cycles --------------------------------------------
//
// The verify of every cycle proposes 3 drafts and accepts the next entry of the given counts of them. The prompt has 5
// tokens: position 5 is the first token (it comes from the prompt), the first cycle delivers positions 6.. in order
// (accepted drafts, then the correction or, with all three accepted, the bonus). Engine::RecordSpeculativeCommit is the
// reference: a draft counts as evaluated when the turn examined it before its terminal token.

TEST_F(Qwen38GraphEngineTest, ATurnLimitInsideTheAcceptedDraftsOnlyCountsTheDraftItConsumed) {
  device_->SetDrafts(3, {3});
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/2);  // room for one token of the first cycle

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 2));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::TurnLimit);
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.rounds, 1u);
  EXPECT_EQ(stats.draft_tokens_proposed, 3u);
  EXPECT_EQ(stats.draft_tokens_evaluated, 1u);  // not accepted + 1: the second draft was never examined
  EXPECT_EQ(stats.draft_tokens_accepted, 1u);
  EXPECT_EQ(stats.acceptance_length_histogram[1], 1u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 1.0f);
  EXPECT_EQ(stats.tokens_discarded, 3u);
}

TEST_F(Qwen38GraphEngineTest, ATurnLimitAfterSeveralAcceptedDraftsCountsExactlyTheDraftsItConsumed) {
  device_->SetDrafts(3, {3});
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/3);  // room for two tokens of the first cycle

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 3));
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_evaluated, 2u);
  EXPECT_EQ(stats.draft_tokens_accepted, 2u);
  EXPECT_EQ(stats.acceptance_length_histogram[2], 1u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 1.0f);
}

TEST_F(Qwen38GraphEngineTest, ATurnLimitOnTheCorrectionTokenCountsTheRejectedPositionToo) {
  device_->SetDrafts(3, {1});
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/3);  // the accepted draft and the correction

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 3));
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_evaluated, 2u);  // the accepted draft and the position that was rejected
  EXPECT_EQ(stats.draft_tokens_accepted, 1u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 0.5f);
}

TEST_F(Qwen38GraphEngineTest, ATurnLimitRightAfterTheLastAcceptedDraftDoesNotReachTheRejectedPosition) {
  device_->SetDrafts(3, {1});
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/2);  // only the accepted draft fits

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 2));
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_evaluated, 1u);
  EXPECT_EQ(stats.draft_tokens_accepted, 1u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 1.0f);
}

TEST_F(Qwen38GraphEngineTest, TheSessionLimitTruncatesTheEvaluatedDraftsLikeTheTurnLimit) {
  device_->SetDrafts(3, {3});
  auto engine = MakeEngine();
  // 5 prompt tokens, room for 2 more: the first token and one token of the cycle.
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/50, /*max_session_tokens=*/7);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 2));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::ContextLimit);
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_evaluated, 1u);
  EXPECT_EQ(stats.draft_tokens_accepted, 1u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 1.0f);
}

TEST_F(Qwen38GraphEngineTest, AStopStringInsideTheAcceptedDraftsOnlyCountsTheDraftsUpToTheMatch) {
  device_->SetDrafts(3, {3});
  device_->overrides[6] = 5;  // "ST", "OP": the stop string completes on the second accepted draft
  device_->overrides[7] = 6;
  auto engine = MakeEngine();
  auto request = BeginWithStopString(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::StopString);
  EXPECT_EQ(outcome.tokens, (std::vector<int32_t>{device_->TokenAt(5), 5, 6}));
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_evaluated, 2u);
  EXPECT_EQ(stats.draft_tokens_accepted, 2u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 1.0f);
  EXPECT_EQ(stats.tokens_discarded, 2u);
}

TEST_F(Qwen38GraphEngineTest, AStopStringOnTheCorrectionTokenCountsTheRejectedPositionToo) {
  device_->SetDrafts(3, {1});
  device_->overrides[6] = 5;  // the accepted draft
  device_->overrides[7] = 6;  // the correction completes the stop string
  auto engine = MakeEngine();
  auto request = BeginWithStopString(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::StopString);
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_evaluated, 2u);
  EXPECT_EQ(stats.draft_tokens_accepted, 1u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 0.5f);
}

TEST_F(Qwen38GraphEngineTest, AnEndOfSequenceDraftIsExaminedButNotAccepted) {
  device_->SetDrafts(3, {3});
  device_->overrides[7] = kEos;  // the second accepted draft
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 2));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::EosToken);
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_evaluated, 2u);  // like Request::CommitAcceptedDraftsForTransaction: the EOS was examined
  EXPECT_EQ(stats.draft_tokens_accepted, 1u);   // ...but nothing was appended for it
  EXPECT_EQ(stats.acceptance_length_histogram[1], 1u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 0.5f);
}

TEST_F(Qwen38GraphEngineTest, AnEndOfSequenceCorrectionTokenCountsTheRejectedPositionToo) {
  device_->SetDrafts(3, {1});
  device_->overrides[7] = kEos;  // the correction
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 2));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::EosToken);
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_evaluated, 2u);
  EXPECT_EQ(stats.draft_tokens_accepted, 1u);
  EXPECT_EQ(stats.correction_tokens, 0u);  // it was not delivered
}

TEST_F(Qwen38GraphEngineTest, AnEndOfSequenceBonusTokenIsNotAnExaminedDraft) {
  device_->SetDrafts(3, {3});
  device_->overrides[9] = kEos;  // the bonus token after three accepted drafts
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 4));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::EosToken);
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_evaluated, 3u);
  EXPECT_EQ(stats.draft_tokens_accepted, 3u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 1.0f);
}

TEST_F(Qwen38GraphEngineTest, RoundsAreClassifiedByTheNumberOfDraftsThatWereAccepted) {
  device_->SetDrafts(3, {0, 1, 2, 3});
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/11);  // the first token, then 1 + 2 + 3 + 4 tokens

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 11));
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.rounds, 4u);
  EXPECT_EQ(stats.draft_tokens_proposed, 12u);
  EXPECT_EQ(stats.draft_tokens_evaluated, 9u);  // 1 + 2 + 3 + 3: every round but the full one examined its rejected draft
  EXPECT_EQ(stats.draft_tokens_accepted, 6u);
  EXPECT_FLOAT_EQ(stats.acceptance_rate, 6.0f / 9.0f);
  EXPECT_EQ(stats.zero_accept_rounds, 1u);
  EXPECT_EQ(stats.partial_accept_rounds, 2u);
  EXPECT_EQ(stats.full_accept_rounds, 1u);
  for (size_t accepted = 0; accepted <= 3; ++accepted) {
    EXPECT_EQ(stats.acceptance_length_histogram[accepted], 1u) << "accepted=" << accepted;
  }
  EXPECT_EQ(stats.correction_tokens, 3u);
  EXPECT_EQ(stats.bonus_tokens, 1u);
  EXPECT_EQ(stats.tokens_emitted, 11u);
  EXPECT_EQ(stats.tokens_discarded, 0u);
}

TEST_F(Qwen38GraphEngineTest, APartialAcceptRoundIsCountedEvenWhenTheTurnEndsBeforeItsCorrectionToken) {
  device_->SetDrafts(3, {1});
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/2);  // the accepted draft fits, the correction does not

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 2));
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.partial_accept_rounds, 1u);
  EXPECT_EQ(stats.zero_accept_rounds, 0u);
  EXPECT_EQ(stats.full_accept_rounds, 0u);
  EXPECT_EQ(stats.correction_tokens, 0u);  // no correction was delivered, but the round is partial all the same
  EXPECT_EQ(stats.bonus_tokens, 0u);
}

TEST_F(Qwen38GraphEngineTest, ARoundTruncatedInsideItsAcceptedDraftsIsClassifiedByTheDraftsItDelivered) {
  device_->SetDrafts(3, {3});
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/2);  // one of the three accepted drafts fits

  RunToCompletion(*engine, request);

  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_accepted, 1u);
  EXPECT_EQ(stats.partial_accept_rounds, 1u);
  EXPECT_EQ(stats.full_accept_rounds, 0u);
  EXPECT_EQ(stats.bonus_tokens, 0u);
}

TEST_F(Qwen38GraphEngineTest, AnEndOfSequenceFirstDraftIsAZeroAcceptRound) {
  device_->SetDrafts(3, {3});
  device_->overrides[6] = kEos;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 1));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::EosToken);
  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.draft_tokens_evaluated, 1u);
  EXPECT_EQ(stats.draft_tokens_accepted, 0u);
  EXPECT_EQ(stats.zero_accept_rounds, 1u);
  EXPECT_EQ(stats.partial_accept_rounds, 0u);
  EXPECT_EQ(stats.acceptance_length_histogram[0], 1u);
}

TEST_F(Qwen38GraphEngineTest, RoundCountersOfSeveralRequestsAddUp) {
  device_->SetDrafts(3, {1});
  auto engine = MakeEngine();
  for (int i = 0; i < 2; ++i) {
    auto request = Begin(*engine, Prompt(5, 12 + i), /*max_generated_tokens=*/3);  // first token, accepted draft, correction
    ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(5, 3));
  }

  const auto stats = engine->GetSpeculativeStats();
  EXPECT_EQ(stats.rounds, 2u);
  EXPECT_EQ(stats.partial_accept_rounds, 2u);
  EXPECT_EQ(stats.acceptance_length_histogram[1], 2u);
  EXPECT_EQ(stats.correction_tokens, 2u);
  EXPECT_EQ(stats.draft_tokens_evaluated, 4u);
  EXPECT_EQ(stats.draft_tokens_accepted, 2u);
}

// ---- prefill, cancellation, queueing ---------------------------------------------------------

TEST_F(Qwen38GraphEngineTest, ThePromptIsIngestedInChunksOverSeveralSteps) {
  auto engine = MakeEngine();  // prefill.chunk = 4
  const auto prompt = Prompt(11);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/2);

  std::array<EngineEvent, 8> buffer;
  EXPECT_EQ(engine->Run(buffer), 0u);  // chunk 1: no event
  EXPECT_EQ(engine->Run(buffer), 0u);  // chunk 2
  EXPECT_TRUE(engine->HasPendingRequests());
  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(device_->CallCount("prefill:4"), 2u);
  EXPECT_EQ(device_->CallCount("prefill:3"), 1u);
  EXPECT_EQ(device_->ingested, prompt);
  EXPECT_EQ(outcome.tokens, Expected(11, 2));
}

TEST_F(Qwen38GraphEngineTest, AnUnsetChunkUsesTheDefaultOfTheDecodeGraphPath) {
  device_ = std::make_shared<ScriptedQwen38Device>(/*capacity=*/4200);
  model_->config_->model.context_length = 4200;
  model_->config_->model.qwen38_graph.prefill.chunk = 0;
  auto engine = MakeEngine();
  const auto prompt = Prompt(2050);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/2, /*max_session_tokens=*/4100);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(device_->CallCount("prefill:2048"), 1u);  // EffectiveQwen38PrefillChunkTokens of the unset key
  EXPECT_EQ(device_->CallCount("prefill:2"), 1u);
  EXPECT_EQ(outcome.tokens, Expected(2050, 2));
}

TEST_F(Qwen38GraphEngineTest, AnInvalidChunkConfigurationIsRejectedWhenTheBackendIsAssembled) {
  auto& prefill = model_->config_->model.qwen38_graph.prefill;
  for (const int chunk : {1, 5, -2}) {  // an odd chunk would shift the S=2 pairs of everything after it
    prefill.chunk = chunk;
    EXPECT_THROW(CreateQwen38GraphDependencies(model_, device_), std::runtime_error) << "prefill.chunk=" << chunk;
  }
  prefill.chunk = 0;
  EXPECT_NO_THROW(CreateQwen38GraphDependencies(model_, device_));
}

TEST(Qwen38GraphSessionTest, AnOddChunkOfTheOptionsIsRoundedDownToAnEvenOne) {
  auto device = std::make_shared<ScriptedQwen38Device>();
  Qwen38GraphSession::Options options;
  options.prefill_chunk_tokens = 5;  // reachable by direct construction only: the configuration rejects odd chunks
  Qwen38GraphSession session{device, options};
  std::vector<int32_t> prompt(11);
  for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = 12 + static_cast<int32_t>(i);
  Qwen38StepRequest request;
  request.owner = &prompt;
  request.turn_id = 1;
  request.tokens = prompt;
  request.token_budget = 2;
  request.session_budget = 50;

  EXPECT_EQ(session.Step(request).count, 0u);  // chunk 1
  EXPECT_EQ(session.Step(request).count, 0u);  // chunk 2
  EXPECT_EQ(session.Step(request).count, 1u);  // the last chunk returns the first token

  EXPECT_EQ(device->CallCount("prefill:4"), 2u);
  EXPECT_EQ(device->CallCount("prefill:3"), 1u);
}

TEST_F(Qwen38GraphEngineTest, TheDeviceIsToldThePromptLengthAfterTheResetAndBeforeTheFirstChunkOfEveryTurn) {
  auto engine = MakeEngine();  // prefill.chunk = 4
  const auto prompt = Prompt(11);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/3);
  ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(11, 3));

  EXPECT_EQ(device_->CallCount("expect:11"), 1u);
  EXPECT_EQ(device_->prefill_totals, (std::vector<size_t>{11, 11, 11}));  // chunks 4, 4, 3 all see the whole length
  const auto position = [&](const std::string& call) {
    return std::find(device_->calls.begin(), device_->calls.end(), call) - device_->calls.begin();
  };
  EXPECT_LT(position("reset"), position("expect:11"));
  EXPECT_LT(position("expect:11"), position("prefill:4"));

  // The continued turn re-ingests the whole history (11 + 3 + 2 tokens) and announces its new length.
  const std::vector<int32_t> follow_up{30, 31};
  TurnOptions options = Limit(2);
  options.request = request;
  request->BeginTurn(follow_up, options);
  ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(16, 2));
  EXPECT_EQ(device_->CallCount("expect:16"), 1u);
  EXPECT_EQ(device_->prefill_totals, (std::vector<size_t>{11, 11, 11, 16, 16, 16, 16}));
}

// The prompt length of a turn belongs to its device. Two Engines on two owner threads (each with its own model and device)
// are held between the publication of the length and the first chunk of the turn, so the schedule "A publishes, B
// publishes, A ingests" of a process-wide variable happens at every turn start. Each device must see the length of its own
// turn in every chunk, after a Reset (continued turn) and after a cancellation, and nothing may travel through the process
// environment.
TEST_F(Qwen38GraphEngineTest, EnginesOnTwoOwnerThreadsKeepTheirOwnExpectedPrefillTotals) {
  struct Owner {
    size_t prompt;           // the first turn; a continued turn follows it
    size_t canceled_prompt;  // canceled after its first chunk
    size_t last_prompt;
    std::shared_ptr<Model> model;
    std::shared_ptr<ScriptedQwen38Device> device;
    std::vector<std::vector<int32_t>> tokens;
    std::vector<GenerationFinishReason> finish_reasons;
    std::string error;
  };
  const auto make_owner = [](size_t prompt, size_t canceled_prompt, size_t last_prompt) {
    Owner owner{prompt, canceled_prompt, last_prompt, LoadSyntheticPagedModel(), std::make_shared<ScriptedQwen38Device>(/*capacity=*/128), {}, {}, {}};
    owner.model->config_->engine.dynamic_batching.reset();
    owner.model->config_->model.qwen38_graph.prefill.chunk = 4;
    return owner;
  };
  Owner a = make_owner(21, 15, 6);
  Owner b = make_owner(7, 9, 10);

  Rendezvous rendezvous{2};
  for (Owner* owner : {&a, &b}) {
    owner->device->on_expect_prefill = [&rendezvous](size_t) {
      if (!rendezvous.ArriveAndWait()) throw std::runtime_error("the other owner thread never reached its publication");
    };
  }

  const auto run_owner = [&rendezvous](Owner& owner) {
    try {
      auto engine = std::make_shared<Engine>(owner.model, CreateQwen38GraphDependencies(owner.model, owner.device));
      const auto drive = [&](const std::shared_ptr<Request>& request) {
        std::vector<int32_t> tokens;
        auto reason = GenerationFinishReason::None;
        std::array<EngineEvent, 8> buffer;
        for (size_t runs = 0; engine->HasPendingRequests(); ++runs) {
          if (runs > 10000) throw std::runtime_error("the Engine did not finish");
          const size_t count = engine->Run(buffer);
          for (size_t i = 0; i < count; ++i) {
            if (buffer[i].request != request) continue;
            if (buffer[i].flags & EngineEventFlagFailed) throw std::runtime_error("the request failed");
            if (buffer[i].flags & EngineEventFlagToken) tokens.push_back(buffer[i].token);
            if (buffer[i].flags & EngineEventFlagTurnFinished) reason = buffer[i].finish_reason;
          }
        }
        owner.tokens.push_back(std::move(tokens));
        owner.finish_reasons.push_back(reason);
      };

      auto request = engine->CreateRequest();
      TurnOptions first = Limit(3);
      first.request = request;
      request->BeginTurn(Prompt(owner.prompt), first);
      drive(request);

      const std::vector<int32_t> follow_up{30, 31};  // the continued turn re-ingests the whole history after a Reset
      TurnOptions second = Limit(2);
      second.request = request;
      request->BeginTurn(follow_up, second);
      drive(request);

      auto canceled = engine->CreateRequest();
      TurnOptions canceled_options;
      canceled_options.request = canceled;
      const uint64_t turn = canceled->BeginTurn(Prompt(owner.canceled_prompt), canceled_options);
      std::array<EngineEvent, 8> buffer;
      engine->Run(buffer);  // the first chunk of the prompt only
      engine->CancelRequest(canceled, turn);
      drive(canceled);

      auto last = engine->CreateRequest();
      TurnOptions last_options = Limit(2);
      last_options.request = last;
      last->BeginTurn(Prompt(owner.last_prompt), last_options);
      drive(last);
    } catch (const std::exception& error) {
      owner.error = error.what();
      rendezvous.Abandon();
    }
  };

  ScopedProcessEnv sentinel{Qwen38OrtEnv::kPrefillTotalTokens, "424242"};
  std::thread thread_a{[&] { run_owner(a); }};
  std::thread thread_b{[&] { run_owner(b); }};
  thread_a.join();
  thread_b.join();

  EXPECT_EQ(rendezvous.Completed(), 4u);  // one publication per turn on each device
  const auto totals = [](size_t total, size_t chunk) { return std::vector<size_t>((total + chunk - 1) / chunk, total); };
  for (Owner* owner : {&a, &b}) {
    SCOPED_TRACE("prompt of " + std::to_string(owner->prompt) + " tokens");
    EXPECT_EQ(owner->error, "");
    auto expected = totals(owner->prompt, 4);
    for (const auto& part : {totals(owner->prompt + 3 + 2, 4), std::vector<size_t>{owner->canceled_prompt},
                             totals(owner->last_prompt, 4)}) {
      expected.insert(expected.end(), part.begin(), part.end());
    }
    EXPECT_EQ(owner->device->prefill_totals, expected);

    const auto& device = *owner->device;
    ASSERT_EQ(owner->tokens.size(), 4u);
    EXPECT_EQ(owner->tokens[0], (std::vector<int32_t>{device.TokenAt(owner->prompt), device.TokenAt(owner->prompt + 1),
                                                      device.TokenAt(owner->prompt + 2)}));
    EXPECT_EQ(owner->tokens[1],
              (std::vector<int32_t>{device.TokenAt(owner->prompt + 5), device.TokenAt(owner->prompt + 6)}));
    EXPECT_TRUE(owner->tokens[2].empty());
    EXPECT_EQ(owner->tokens[3],
              (std::vector<int32_t>{device.TokenAt(owner->last_prompt), device.TokenAt(owner->last_prompt + 1)}));
    EXPECT_EQ(owner->finish_reasons,
              (std::vector<GenerationFinishReason>{GenerationFinishReason::TurnLimit, GenerationFinishReason::TurnLimit,
                                                   GenerationFinishReason::Canceled, GenerationFinishReason::TurnLimit}));
  }

  const char* published = std::getenv(Qwen38OrtEnv::kPrefillTotalTokens);
  ASSERT_NE(published, nullptr);
  EXPECT_STREQ(published, "424242") << "the executor must not publish the prompt length in the process environment";
}

TEST_F(Qwen38GraphEngineTest, CancelDuringPrefillStopsBeforeDecodingAndTheNextRequestStartsClean) {
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  const uint64_t turn = request->BeginTurn(Prompt(11), options);
  std::array<EngineEvent, 8> buffer;
  ASSERT_EQ(engine->Run(buffer), 0u);  // first chunk only

  EXPECT_TRUE(engine->CancelRequest(request, turn));
  const size_t count = engine->Run(buffer);
  ASSERT_EQ(count, 1u);
  EXPECT_EQ(buffer[0].finish_reason, GenerationFinishReason::Canceled);
  EXPECT_FALSE(engine->HasPendingRequests());
  EXPECT_EQ(device_->CallCount("begin:1") + device_->CallCount("begin:0"), 0u);

  auto next = Begin(*engine, Prompt(6, 20), /*max_generated_tokens=*/3);
  const auto outcome = RunToCompletion(*engine, next);
  EXPECT_EQ(outcome.tokens, Expected(6, 3));
  EXPECT_EQ(device_->ingested, Prompt(6, 20));  // the canceled prompt is gone from the device
}

// A request canceled while it was still ingesting its prompt made no progress, so its next turn restarts it from scratch,
// possibly after another request took the device. The Engine does not enroll such a request in the scheduler again, so the
// scheduler that displaced it must have kept it.
TEST_F(Qwen38GraphEngineTest, ACanceledPrefillRequestRunsAgainAfterAnotherRequestTookTheSlot) {
  auto engine = MakeEngine();
  auto first = engine->CreateRequest();
  TurnOptions first_options;
  first_options.request = first;
  const uint64_t turn = first->BeginTurn(Prompt(11), first_options);
  std::array<EngineEvent, 8> buffer;
  ASSERT_EQ(engine->Run(buffer), 0u);  // first chunk only
  ASSERT_TRUE(engine->CancelRequest(first, turn));
  while (engine->HasPendingRequests()) engine->Run(buffer);

  auto second = Begin(*engine, Prompt(6, 20), /*max_generated_tokens=*/3);
  EXPECT_EQ(RunToCompletion(*engine, second).tokens, Expected(6, 3));

  // The restarted turn replays the canceled prompt ahead of its own input.
  auto restart = Limit(3);
  restart.request = first;
  first->BeginTurn(Prompt(5, 40), restart);
  ASSERT_TRUE(engine->HasPendingRequests()) << "the restarted request is not known to the scheduler";
  const auto outcome = RunToCompletion(*engine, first);
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::TurnLimit);
  EXPECT_EQ(outcome.tokens, Expected(16, 3));
  auto replayed = Prompt(11);
  const auto input = Prompt(5, 40);
  replayed.insert(replayed.end(), input.begin(), input.end());
  EXPECT_EQ(device_->ingested, replayed);
}

// While the slot is still the canceled request's own the next turn continues from what the request kept, which is the
// documented behavior of cancellation: the canceled prompt is followed by the new input.
TEST_F(Qwen38GraphEngineTest, ACanceledPrefillRequestRunsAgainWhileItsSlotIsStillItsOwn) {
  auto engine = MakeEngine();
  auto first = engine->CreateRequest();
  TurnOptions first_options;
  first_options.request = first;
  const uint64_t turn = first->BeginTurn(Prompt(11), first_options);
  std::array<EngineEvent, 8> buffer;
  ASSERT_EQ(engine->Run(buffer), 0u);  // first chunk only
  ASSERT_TRUE(engine->CancelRequest(first, turn));
  while (engine->HasPendingRequests()) engine->Run(buffer);

  auto restart = Limit(3);
  restart.request = first;
  first->BeginTurn(Prompt(5, 40), restart);
  const auto outcome = RunToCompletion(*engine, first);
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::TurnLimit);
  EXPECT_EQ(outcome.tokens, Expected(16, 3));
  auto replayed = Prompt(11);
  const auto input = Prompt(5, 40);
  replayed.insert(replayed.end(), input.begin(), input.end());
  EXPECT_EQ(device_->ingested, replayed);
}

TEST_F(Qwen38GraphEngineTest, CancelWithACycleInFlightIsSafeAndTheNextRequestWaitsForIt) {
  model_->config_->model.qwen38_graph.prefill.chunk = 64;  // the first step ingests the whole prompt
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  const uint64_t turn = request->BeginTurn(Prompt(5), options);
  std::array<EngineEvent, 8> buffer;
  ASSERT_GE(engine->Run(buffer), 1u);  // prefill + first token (the first cycle is now in flight)
  ASSERT_TRUE(device_->inflight());

  EXPECT_TRUE(engine->CancelRequest(request, turn));
  ASSERT_EQ(engine->Run(buffer), 1u);
  EXPECT_EQ(buffer[0].finish_reason, GenerationFinishReason::Canceled);
  EXPECT_TRUE(device_->inflight());  // still running on the device

  auto next = Begin(*engine, Prompt(5, 25), /*max_generated_tokens=*/4);
  const auto outcome = RunToCompletion(*engine, next);
  EXPECT_EQ(outcome.tokens, Expected(5, 4));
  EXPECT_GE(device_->CallCount("reset"), 1u);
}

TEST_F(Qwen38GraphEngineTest, AFurtherRequestQueuesBehindTheResidentOneAndRunsAfterIt) {
  auto engine = MakeEngine();
  auto first = Begin(*engine, Prompt(5), /*max_generated_tokens=*/4);
  auto second = Begin(*engine, Prompt(6, 25), /*max_generated_tokens=*/3);

  std::array<EngineEvent, 8> buffer;
  std::vector<int32_t> first_tokens, second_tokens;
  std::vector<const Request*> order;
  while (engine->HasPendingRequests()) {
    const size_t count = engine->Run(buffer);
    for (size_t i = 0; i < count; ++i) {
      if (!(buffer[i].flags & EngineEventFlagToken)) continue;
      (buffer[i].request == first ? first_tokens : second_tokens).push_back(buffer[i].token);
      order.push_back(buffer[i].request.get());
    }
  }

  EXPECT_EQ(first_tokens, Expected(5, 4));
  EXPECT_EQ(second_tokens, Expected(6, 3));
  // Every event of the first request precedes every event of the second.
  const auto last_first = std::find(order.rbegin(), order.rend(), first.get());
  const auto first_second = std::find(order.begin(), order.end(), second.get());
  EXPECT_LT(std::distance(order.begin(), last_first.base()), std::distance(order.begin(), first_second) + 1);
  EXPECT_GE(device_->CallCount("reset"), 1u);
  EXPECT_EQ(device_->ingested, Prompt(6, 25));
  EXPECT_EQ(engine->GetCapabilities().configured_max_batch_size, 1u);
}

// ---- continuation, rewind --------------------------------------------------------------------

TEST_F(Qwen38GraphEngineTest, AContinuedTurnReIngestsTheWholeHistory) {
  auto engine = MakeEngine();
  const auto prompt = Prompt(5);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/3);
  ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(5, 3));
  const size_t resets = device_->CallCount("reset");

  const std::vector<int32_t> follow_up{30, 31};
  TurnOptions options = Limit(2);
  options.request = request;
  request->BeginTurn(follow_up, options);
  const auto outcome = RunToCompletion(*engine, request);

  std::vector<int32_t> history = prompt;
  for (const int32_t token : Expected(5, 3)) history.push_back(token);
  history.insert(history.end(), follow_up.begin(), follow_up.end());
  EXPECT_EQ(device_->CallCount("reset"), resets + 1);
  EXPECT_EQ(device_->ingested, history);
  EXPECT_EQ(outcome.tokens, Expected(history.size(), 2));
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::TurnLimit);
}

TEST_F(Qwen38GraphEngineTest, RewindToTheStartOfATurnReleasesTheSlotAndReplaysTheRetainedPrefix) {
  auto engine = MakeEngine();
  const auto prompt = Prompt(5);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/3);
  ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(5, 3));
  TurnOptions options = Limit(2);
  options.request = request;
  const uint64_t discarded_turn = request->BeginTurn(std::array<int32_t, 2>{30, 31}, options);
  ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(10, 2));

  request->RewindToStartOfTurn(discarded_turn);
  EXPECT_FALSE(engine->HasPendingRequests());
  EXPECT_EQ(request->CurrentSequenceLength(), 8);

  const std::vector<int32_t> replacement{40};
  options = Limit(2);
  options.request = request;
  request->BeginTurn(replacement, options);
  const auto outcome = RunToCompletion(*engine, request);

  std::vector<int32_t> history = prompt;
  for (const int32_t token : Expected(5, 3)) history.push_back(token);
  history.push_back(40);
  EXPECT_EQ(device_->ingested, history);
  EXPECT_EQ(outcome.tokens, Expected(history.size(), 2));
}

// ---- dense (fast) prefill: chunk streaming and hand-off ---------------------------------------

TEST_F(Qwen38GraphEngineTest, ADeviceWithAPreferredChunkReceivesThePromptInChunksOfThatSizeAndHandsOffOnce) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 8;  // replaces the chunk of the configuration (prefill.chunk = 4)
  auto engine = MakeEngine();
  const auto prompt = Prompt(20);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/5);

  std::array<EngineEvent, 8> buffer;
  EXPECT_EQ(engine->Run(buffer), 0u);  // chunk 1 of 3: no event, no hand-off yet
  EXPECT_EQ(device_->handoffs, 0u);
  EXPECT_EQ(engine->Run(buffer), 0u);  // chunk 2
  EXPECT_EQ(device_->handoffs, 0u);
  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(device_->CallCount("prefill:8"), 2u);
  EXPECT_EQ(device_->CallCount("prefill:4"), 1u);
  EXPECT_EQ(device_->ingested, prompt);
  EXPECT_EQ(device_->handoffs, 1u);
  EXPECT_EQ(device_->handoff_length, 20u);
  EXPECT_EQ(outcome.tokens, Expected(20, 5));
}

TEST_F(Qwen38GraphEngineTest, DenseChunksMayHaveAnOddSizeAndTheLastChunkEndsTheStreamExactly) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 7;
  auto engine = MakeEngine();
  const auto prompt = Prompt(15);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/3);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(device_->CallCount("prefill:7"), 2u);
  EXPECT_EQ(device_->CallCount("prefill:1"), 1u);
  EXPECT_EQ(device_->ingested, prompt);
  EXPECT_EQ(outcome.tokens, Expected(15, 3));
}

TEST_F(Qwen38GraphEngineTest, ADensePromptShorterThanTheChunkIsOneChunk) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 64;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/2);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(device_->CallCount("prefill:5"), 1u);
  EXPECT_EQ(outcome.tokens, Expected(5, 2));
}

TEST_F(Qwen38GraphEngineTest, ADensePromptOfExactlyTheChunkSizeHasNoEmptyTailChunk) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 8;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(16), /*max_generated_tokens=*/2);

  RunToCompletion(*engine, request);

  EXPECT_EQ(device_->CallCount("prefill:8"), 2u);
  EXPECT_EQ(device_->CallCount("prefill:0"), 0u);
}

TEST_F(Qwen38GraphEngineTest, ContinuationAndRewindReIngestTheHistoryThroughTheDenseChunksAndHandOffAgain) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 4;
  auto engine = MakeEngine();
  const auto prompt = Prompt(5);
  auto request = Begin(*engine, prompt, /*max_generated_tokens=*/3);
  ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(5, 3));
  ASSERT_EQ(device_->handoffs, 1u);

  TurnOptions options = Limit(2);
  options.request = request;
  const uint64_t turn = request->BeginTurn(std::array<int32_t, 2>{30, 31}, options);
  ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(10, 2));
  EXPECT_EQ(device_->handoffs, 2u);
  EXPECT_EQ(device_->handoff_length, 10u);  // 5 prompt + 3 generated + 2 follow-up

  request->RewindToStartOfTurn(turn);
  options = Limit(2);
  options.request = request;
  request->BeginTurn(std::array<int32_t, 1>{40}, options);
  const auto outcome = RunToCompletion(*engine, request);
  std::vector<int32_t> history = prompt;
  for (const int32_t token : Expected(5, 3)) history.push_back(token);
  history.push_back(40);
  EXPECT_EQ(device_->ingested, history);
  EXPECT_EQ(device_->handoffs, 3u);
  EXPECT_EQ(device_->handoff_length, history.size());
  EXPECT_EQ(outcome.tokens, Expected(history.size(), 2));
}

TEST_F(Qwen38GraphEngineTest, AFirstTokenThatEndsTheTurnSkipsTheHandOff) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 4;
  device_->overrides[5] = kEos;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/20);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_TRUE(outcome.tokens.empty());
  EXPECT_EQ(device_->handoffs, 0u);
  EXPECT_EQ(device_->CallCount("begin:0"), 1u);
  EXPECT_EQ(device_->status_checks, 0u);
}

TEST_F(Qwen38GraphEngineTest, CancelBetweenDenseChunksNeverHandsOffAndTheNextRequestStartsClean) {
  device_->dense_prefill = true;
  device_->preferred_chunk = 4;
  auto engine = MakeEngine();
  auto first = engine->CreateRequest();
  TurnOptions first_options;
  first_options.request = first;
  const uint64_t turn = first->BeginTurn(Prompt(12), first_options);
  std::array<EngineEvent, 8> buffer;
  ASSERT_EQ(engine->Run(buffer), 0u);  // chunk 1 of 3
  EXPECT_TRUE(engine->CancelRequest(first, turn));
  while (engine->HasPendingRequests()) engine->Run(buffer);
  EXPECT_EQ(device_->handoffs, 0u);

  const auto prompt = Prompt(9, 20);
  auto second = Begin(*engine, prompt, /*max_generated_tokens=*/3);
  const auto outcome = RunToCompletion(*engine, second);
  EXPECT_EQ(device_->ingested, prompt);  // the chunk of the cancelled request is gone
  EXPECT_EQ(device_->handoffs, 1u);
  EXPECT_EQ(outcome.tokens, Expected(9, 3));
}

// ---- end-of-decode device status ---------------------------------------------------------------

TEST_F(Qwen38GraphEngineTest, TheDeviceStatusIsCheckedOnceWhenATurnEndsWithNothingInFlight) {
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/7);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.tokens, Expected(5, 7));
  EXPECT_EQ(device_->status_checks, 1u);
}

TEST_F(Qwen38GraphEngineTest, DeviceErrorBitsAtTheEndOfATurnAreReportedAsAnExecutionFailure) {
  device_->fail_status_check = true;
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/3);

  std::array<EngineEvent, 8> buffer;
  bool saw_failure = false;
  for (int i = 0; i < 100 && engine->HasPendingRequests() && !saw_failure; ++i) {
    const size_t count = engine->Run(buffer);
    for (size_t j = 0; j < count; ++j) {
      if (buffer[j].flags & EngineEventFlagFailed) {
        saw_failure = true;
        EXPECT_EQ(buffer[j].error_code, EngineErrorCode::EngineExecutionFailure);
      }
    }
  }
  EXPECT_TRUE(saw_failure);
}

TEST_F(Qwen38GraphEngineTest, DeviceErrorBitsAfterAStopStringTurnAreSurfacedBeforeReset) {
  device_->overrides[5] = 5;
  device_->overrides[6] = 6;
  device_->fail_status_check = true;
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  options.max_generated_tokens = 20;
  options.stop_strings = {"STOP"};
  const uint64_t turn = request->BeginTurn(Prompt(5), options);

  const auto outcome = RunToCompletion(*engine, request);
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::StopString);

  // Releasing the slot for rewind (or starting a new turn/request) drains the look-ahead cycle and checks decode status.
  EXPECT_THROW(request->RewindToStartOfTurn(turn), std::runtime_error);
  EXPECT_EQ(device_->status_checks, 1u);
}

// The status check of a rewind runs outside a step, where RunStatic's fatal handler does not see its failure. The error bits
// are erased by the next Reset(), so the failure must outlive the exception the caller caught: the context is marked failed
// and the next step fails, instead of the context resuming as if the bits had never been read.
TEST_F(Qwen38GraphEngineTest, ADeviceErrorFoundWhileRewindingStaysFatalAfterTheCallerCaughtIt) {
  device_->overrides[5] = 5;
  device_->overrides[6] = 6;
  device_->fail_status_check = true;
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  options.max_generated_tokens = 20;
  options.stop_strings = {"STOP"};
  const uint64_t turn = request->BeginTurn(Prompt(5), options);
  ASSERT_EQ(RunToCompletion(*engine, request).finish_reason, GenerationFinishReason::StopString);

  ASSERT_THROW(request->RewindToStartOfTurn(turn), std::runtime_error);
  EXPECT_FALSE(device_->context_healthy);

  // What a caller that went on would find: bits that no longer exist and a request that is ready to run.
  device_->fail_status_check = false;
  auto next = Begin(*engine, Prompt(5, 25), /*max_generated_tokens=*/4);
  EXPECT_EQ(RunUntilFailure(*engine), EngineErrorCode::EngineExecutionFailure);
}

// The failure belongs to the device, not to the attempt that found it: a caller that retries the rewind must keep failing
// instead of consuming the error, or a later retry would succeed on a poisoned device and the next step would reset away
// the error bits.
TEST_F(Qwen38GraphEngineTest, RetryingAFailedRewindCannotConsumeTheDeviceFailure) {
  device_->overrides[5] = 5;
  device_->overrides[6] = 6;
  device_->fail_status_check = true;
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  options.max_generated_tokens = 20;
  options.stop_strings = {"STOP"};
  const uint64_t turn = request->BeginTurn(Prompt(5), options);
  ASSERT_EQ(RunToCompletion(*engine, request).finish_reason, GenerationFinishReason::StopString);

  ASSERT_THROW(request->RewindToStartOfTurn(turn), std::runtime_error);
  device_->fail_status_check = false;
  EXPECT_THROW(request->RewindToStartOfTurn(turn), std::runtime_error);
  EXPECT_THROW(request->RewindToStartOfTurn(turn), std::runtime_error);
  EXPECT_FALSE(device_->context_healthy);

  auto next = Begin(*engine, Prompt(5, 25), /*max_generated_tokens=*/4);
  EXPECT_EQ(RunUntilFailure(*engine), EngineErrorCode::EngineExecutionFailure);
}

TEST_F(Qwen38GraphEngineTest, DeviceErrorBitsAfterACanceledTurnAreSurfacedOnTheNextRun) {
  model_->config_->model.qwen38_graph.prefill.chunk = 64;
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  const uint64_t turn = request->BeginTurn(Prompt(5), options);
  std::array<EngineEvent, 8> buffer;
  ASSERT_GE(engine->Run(buffer), 1u);
  ASSERT_TRUE(device_->inflight());

  device_->fail_status_check = true;
  EXPECT_TRUE(engine->CancelRequest(request, turn));
  ASSERT_EQ(engine->Run(buffer), 1u);
  EXPECT_EQ(buffer[0].finish_reason, GenerationFinishReason::Canceled);

  // When the next request claims the slot, Release() drains the canceled cycle and surfaces the device status error.
  auto next = Begin(*engine, Prompt(5, 25), /*max_generated_tokens=*/4);
  bool saw_failure = false;
  for (int i = 0; i < 10 && engine->HasPendingRequests() && !saw_failure; ++i) {
    const size_t count = engine->Run(buffer);
    for (size_t j = 0; j < count; ++j) {
      if (buffer[j].flags & EngineEventFlagFailed) {
        saw_failure = true;
        EXPECT_EQ(buffer[j].error_code, EngineErrorCode::EngineExecutionFailure);
      }
    }
  }
  EXPECT_TRUE(saw_failure);
  EXPECT_EQ(device_->status_checks, 1u);
}

// The wasted look-ahead cycle of a stop-string or canceled turn is never waited for, so its record stays unconsumed: the
// device reports its validation failure only once it was drained, and a Reset() would erase it. The executor must
// therefore drain and check before it lets anything reset the device.
TEST_F(Qwen38GraphEngineTest, AFailureOfTheLookAheadCycleVisibleOnlyAfterDrainSurfacesWhenAStopStringTurnIsRewound) {
  device_->overrides[5] = 5;
  device_->overrides[6] = 6;
  device_->fail_cycle = 2;  // cycle 1 completes the stop string; cycle 2 is the look-ahead nobody waits for
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  options.max_generated_tokens = 20;
  options.stop_strings = {"STOP"};
  const uint64_t turn = request->BeginTurn(Prompt(5), options);

  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::StopString);  // the turn itself is fine
  EXPECT_EQ(outcome.failure_flags, 0u);
  EXPECT_TRUE(device_->inflight());  // the record of the look-ahead cycle is unconsumed...
  EXPECT_TRUE(device_->running());   // ...and nothing has synchronized the stream yet
  EXPECT_EQ(device_->status_checks, 0u);

  EXPECT_THROW(request->RewindToStartOfTurn(turn), std::runtime_error);
  EXPECT_TRUE(device_->inflight());  // Drain() completed the cycle without consuming its record
  EXPECT_FALSE(device_->running());
  EXPECT_EQ(device_->order, (std::vector<std::string>{"reset", "drain", "status"}));  // the only Reset was the first turn's
}

TEST_F(Qwen38GraphEngineTest, AFailureOfTheLookAheadCycleVisibleOnlyAfterDrainFailsTheNextRequestInsteadOfBeingResetAway) {
  device_->overrides[5] = 5;
  device_->overrides[6] = 6;
  device_->fail_cycle = 2;
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  options.max_generated_tokens = 20;
  options.stop_strings = {"STOP"};
  request->BeginTurn(Prompt(5), options);
  ASSERT_EQ(RunToCompletion(*engine, request).finish_reason, GenerationFinishReason::StopString);
  ASSERT_TRUE(device_->inflight());

  // The next request claims the slot: Release() drains the unconsumed cycle and checks the device before any Reset().
  device_->overrides.clear();
  auto next = Begin(*engine, Prompt(5, 25), /*max_generated_tokens=*/4);
  const auto failure = RunUntilFailure(*engine);

  ASSERT_TRUE(failure.has_value()) << "the error bits of the drained cycle were never reported";
  EXPECT_EQ(*failure, EngineErrorCode::EngineExecutionFailure);
  EXPECT_EQ(device_->order, (std::vector<std::string>{"reset", "drain", "status"}));  // no Reset() erased the bits first
  EXPECT_EQ(device_->CallCount("expect:5"), 1u);                                      // only the first request's announcement
}

TEST_F(Qwen38GraphEngineTest, AFailureOfTheFirstCycleVisibleOnlyAfterDrainSurfacesAfterACancellation) {
  model_->config_->model.qwen38_graph.prefill.chunk = 64;
  device_->fail_cycle = 1;  // the cycle BeginDecode enqueued
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  const uint64_t turn = request->BeginTurn(Prompt(5), options);
  std::array<EngineEvent, 8> buffer;
  ASSERT_GE(engine->Run(buffer), 1u);  // prefill and the first token; cycle 1 is running
  ASSERT_TRUE(device_->running());

  EXPECT_TRUE(engine->CancelRequest(request, turn));
  ASSERT_EQ(engine->Run(buffer), 1u);
  EXPECT_EQ(buffer[0].finish_reason, GenerationFinishReason::Canceled);
  EXPECT_EQ(buffer[0].flags & EngineEventFlagFailed, 0u);  // cancellation does not wait for the device
  EXPECT_TRUE(device_->running());

  auto next = Begin(*engine, Prompt(5, 25), /*max_generated_tokens=*/4);
  const auto failure = RunUntilFailure(*engine);

  ASSERT_TRUE(failure.has_value()) << "the error bits of the canceled cycle were never reported";
  EXPECT_EQ(*failure, EngineErrorCode::EngineExecutionFailure);
  EXPECT_EQ(device_->order, (std::vector<std::string>{"reset", "drain", "status"}));
}

TEST_F(Qwen38GraphEngineTest, ADrainedLookAheadCycleWithoutErrorBitsIsCheckedAndTheNextRequestRuns) {
  device_->overrides[5] = 5;
  device_->overrides[6] = 6;
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.request = request;
  options.max_generated_tokens = 20;
  options.stop_strings = {"STOP"};
  request->BeginTurn(Prompt(5), options);
  ASSERT_EQ(RunToCompletion(*engine, request).finish_reason, GenerationFinishReason::StopString);

  device_->overrides.clear();
  auto next = Begin(*engine, Prompt(5, 25), /*max_generated_tokens=*/3);
  const auto outcome = RunToCompletion(*engine, next);

  EXPECT_EQ(outcome.failure_flags, 0u);
  EXPECT_EQ(outcome.tokens, Expected(5, 3));
  EXPECT_EQ(device_->order, (std::vector<std::string>{"reset", "drain", "status", "reset", "status"}));
}

TEST_F(Qwen38GraphEngineTest, CapabilitiesAreHonest) {
  auto engine = MakeEngine();
  const auto capabilities = engine->GetCapabilities();
  EXPECT_EQ(capabilities.configured_max_batch_size, 1u);
  EXPECT_EQ(capabilities.max_request_length, static_cast<uint64_t>(device_->Capacity() - Qwen38DecodeDevice::kReservedTail));
  EXPECT_EQ(engine->MaxDraftTokensPerStep(), 0u);
  EXPECT_FALSE(engine->PrefixCacheStats().has_value());

  RequestOptions too_long;
  too_long.max_session_tokens = capabilities.max_request_length + 1;
  try {
    engine->CreateRequest(too_long);
    FAIL() << "a session longer than the KV capacity must be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("max_request_length"), std::string::npos) << error.what();
  }
  RequestOptions at_limit;
  at_limit.max_session_tokens = capabilities.max_request_length;
  EXPECT_NO_THROW(engine->CreateRequest(at_limit));
}

TEST_F(Qwen38GraphEngineTest, MaxRequestLengthIsTheSmallerOfTheModelContextLengthAndTheKvCapacity) {
  const int kv_limit = device_->Capacity() - Qwen38DecodeDevice::kReservedTail;

  model_->config_->model.context_length = kv_limit - 10;
  EXPECT_EQ(MakeEngine()->GetCapabilities().max_request_length, static_cast<uint64_t>(kv_limit - 10));

  model_->config_->model.context_length = kv_limit + 10;
  EXPECT_EQ(MakeEngine()->GetCapabilities().max_request_length, static_cast<uint64_t>(kv_limit));
}

TEST_F(Qwen38GraphEngineTest, UnsupportedTurnOptionsAreRejectedBeforeTheRequestChanges) {
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  const auto prompt = Prompt(4);
  const auto expect_rejected = [&](const char* label, TurnOptions options, const char* expected_fragment) {
    options.request = request;
    try {
      request->BeginTurn(prompt, options);
      ADD_FAILURE() << label << " must be rejected";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string{error.what()}.find(expected_fragment), std::string::npos) << label << ": " << error.what();
    }
    EXPECT_TRUE(request->IsAwaitingFirstTurn()) << label << " mutated the request";
  };

  TurnOptions sampled;
  sampled.do_sample = true;
  sampled.top_k = 5;
  sampled.temperature = 0.7f;
  expect_rejected("sampling", sampled, "greedily");

  TurnOptions penalty;
  penalty.repetition_penalty = 1.2f;
  expect_rejected("repetition_penalty", penalty, "repetition_penalty");

  TurnOptions ngram;
  ngram.no_repeat_ngram_size = 3;
  expect_rejected("no_repeat_ngram_size", ngram, "no_repeat_ngram_size");

  TurnOptions minimum;
  minimum.min_generated_tokens = 4;
  expect_rejected("min_generated_tokens", minimum, "min_generated_tokens");

  TurnOptions seeded;
  seeded.seed = 7;
  expect_rejected("seed", seeded, "seed");

  TurnOptions guided;
  guided.guidance_type = "lark_grammar";
  guided.guidance_data = "start: \"a\"";
  expect_rejected("guidance", guided, "guidance");

  // The same request still starts a plain greedy turn.
  TurnOptions greedy = Limit(2);
  greedy.request = request;
  EXPECT_NO_THROW(request->BeginTurn(prompt, greedy));
  EXPECT_EQ(RunToCompletion(*engine, request).tokens, Expected(4, 2));
}

TEST_F(Qwen38GraphEngineTest, ADeviceFailureIsFatalAndReportedThroughTheEngineErrorTaxonomy) {
  model_->config_->model.qwen38_graph.prefill.chunk = 64;  // the first step ingests the whole prompt
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/10);
  std::array<EngineEvent, 8> buffer;
  ASSERT_GE(engine->Run(buffer), 1u);  // prefill and the first token succeed
  device_->fail_wait = true;

  const size_t count = engine->Run(buffer);
  ASSERT_GE(count, 1u);
  bool saw_failure = false;
  for (size_t i = 0; i < count; ++i) {
    if (buffer[i].flags & EngineEventFlagFailed) {
      saw_failure = true;
      EXPECT_EQ(buffer[i].error_code, EngineErrorCode::EngineExecutionFailure);
    }
  }
  EXPECT_TRUE(saw_failure);
  EXPECT_THROW(engine->Run(buffer), std::exception);  // the Engine is no longer healthy
}

TEST_F(Qwen38GraphEngineTest, DestroyingTheEngineDrainsTheDevice) {
  model_->config_->model.qwen38_graph.prefill.chunk = 64;  // the first step ingests the whole prompt
  {
    auto engine = MakeEngine();
    auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/10);
    std::array<EngineEvent, 8> buffer;
    ASSERT_GE(engine->Run(buffer), 1u);
    ASSERT_TRUE(device_->running());
  }
  EXPECT_FALSE(device_->running());  // the cycle was waited for (its record stays unconsumed)
  EXPECT_GE(device_->CallCount("drain"), 1u);
}

TEST_F(Qwen38GraphEngineTest, ANewEngineOverAUsedDeviceStartsFromACleanDevice) {
  model_->config_->model.qwen38_graph.prefill.chunk = 64;
  {
    auto engine = MakeEngine();
    auto request = Begin(*engine, Prompt(5), /*max_generated_tokens=*/3);
    ASSERT_EQ(RunToCompletion(*engine, request).tokens, Expected(5, 3));
  }
  const size_t resets = device_->CallCount("reset");

  // The device kept the first Engine's decode state; the next Engine must not prefill on top of it.
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(6, 20), /*max_generated_tokens=*/3);
  const auto outcome = RunToCompletion(*engine, request);

  EXPECT_EQ(device_->CallCount("reset"), resets + 1);
  EXPECT_EQ(device_->ingested, Prompt(6, 20));
  EXPECT_EQ(outcome.tokens, Expected(6, 3));
}

TEST_F(Qwen38GraphEngineTest, MalformedCycleWidthFailsBeforeAnyCandidateCanMutateTheRequest) {
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(3), 10);
  std::array<EngineEvent, 16> events;
  ASSERT_GT(engine->Run(events), 0u);  // prompt prediction
  const auto retained = request->TokensCpu();
  const std::vector<int32_t> before(retained.begin(), retained.end());
  device_->malformed_cycle_count = true;
  engine->Run(events);
  const auto after = request->TokensCpu();
  EXPECT_EQ(std::vector<int32_t>(after.begin(), after.end()), before);
  EXPECT_FALSE(device_->context_healthy);
}

TEST_F(Qwen38GraphEngineTest, FailureHealthPersistsWhenANewEngineTriesToLeaseTheSameContext) {
  auto engine = MakeEngine();
  auto request = Begin(*engine, Prompt(3), 10);
  std::array<EngineEvent, 16> events;
  engine->Run(events);
  device_->fail_wait = true;
  engine->Run(events);
  EXPECT_FALSE(device_->context_healthy);
  engine.reset();
  device_->fail_wait = false;
  EXPECT_THROW(MakeEngine(), std::runtime_error);
}

// ---- cache manager contract ------------------------------------------------------------------

TEST_F(Qwen38GraphEngineTest, TheCacheManagerHoldsExactlyOneRow) {
  auto engine = MakeEngine();
  auto session = std::make_shared<Qwen38GraphSession>(device_, Qwen38GraphSession::Options{});
  Qwen38GraphCacheManager cache{model_, session};
  auto a = engine->CreateRequest();
  auto b = engine->CreateRequest();
  a->BeginTurn(Prompt(3));
  b->BeginTurn(Prompt(3));

  EXPECT_FALSE(cache.CanAllocate({}));
  EXPECT_FALSE(cache.CanAllocate({a, b}));
  ASSERT_TRUE(cache.CanAllocate({a}));
  cache.Allocate({a});
  EXPECT_EQ(cache.ResidentRequestCount(), 1u);
  EXPECT_TRUE(cache.IsResident(a));
  EXPECT_FALSE(cache.IsResident(b));
  EXPECT_FALSE(cache.CanAllocate({b})) << "an active resident row cannot be replaced";
  EXPECT_THROW(cache.Allocate({b}), std::runtime_error);

  std::vector<std::shared_ptr<Request>> wrong{b};
  EXPECT_THROW(cache.Deallocate(wrong), std::runtime_error);
  std::vector<std::shared_ptr<Request>> right{a};
  cache.Deallocate(right);
  EXPECT_EQ(cache.ResidentRequestCount(), 0u);
  EXPECT_TRUE(cache.CanAllocate({b}));
  EXPECT_EQ(cache.MaxBatchSize(), 1u);
  EXPECT_EQ(cache.MaxStaticRequestLength(), static_cast<size_t>(device_->Capacity() - Qwen38DecodeDevice::kReservedTail));
  EXPECT_FALSE(cache.SupportsDynamicBatching());
}

TEST_F(Qwen38GraphEngineTest, DynamicBatchingConfigurationIsRejected) {
  model_->config_->engine.dynamic_batching = Config::Engine::DynamicBatching{};
  EXPECT_THROW(CreateQwen38GraphDependencies(model_, device_), std::runtime_error);
}

TEST_F(Qwen38GraphEngineTest, InvalidPromptTokenIdsAreRejectedAtTurnAdmissionWithoutPoisoningTheDevice) {
  auto engine = MakeEngine();
  auto request = engine->CreateRequest();
  auto options = Limit(3);
  options.request = request;
  const int32_t vocab_size = model_->config_->model.vocab_size;

  for (const int32_t bad_id : {int32_t{-1}, vocab_size}) {
    const std::array<int32_t, 3> bad_prompt = {12, bad_id, 14};
    EXPECT_THROW(request->BeginTurn(bad_prompt, options), std::invalid_argument) << bad_id;
    EXPECT_TRUE(request->IsAwaitingFirstTurn());
    EXPECT_EQ(request->CurrentSequenceLength(), 0);
    EXPECT_TRUE(device_->ContextIsHealthy());
    EXPECT_TRUE(device_->calls.empty());
  }

  // The caller can immediately retry with a valid prompt on the same Request and Engine.
  request->BeginTurn(Prompt(5), options);
  const auto outcome = RunToCompletion(*engine, request);
  EXPECT_EQ(outcome.finish_reason, GenerationFinishReason::TurnLimit);
  EXPECT_EQ(outcome.tokens, Expected(5, 3));
}

// Host cost of one Engine step (Run) with a device that answers instantly: everything the Engine adds around a cycle. Run
// manually: engine_unit_tests --gtest_also_run_disabled_tests --gtest_filter='*RunOverhead*'.
TEST_F(Qwen38GraphEngineTest, DISABLED_RunOverheadPerCycle) {
  device_ = std::make_shared<ScriptedQwen38Device>(/*capacity=*/33028);
  model_->config_->model.context_length = 33028;
  model_->config_->model.qwen38_graph.prefill.chunk = 2048;
  auto engine = MakeEngine();
  const size_t generated = 20000;
  auto request = Begin(*engine, Prompt(16), generated, /*max_session_tokens=*/33024);
  std::array<EngineEvent, 16> buffer;
  size_t runs = 0, tokens = 0;
  const auto start = std::chrono::steady_clock::now();
  while (engine->HasPendingRequests()) {
    const size_t count = engine->Run(buffer);
    ++runs;
    tokens += count;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << "Engine::Run: " << runs << " runs, " << tokens << " events in " << seconds * 1e3 << " ms = "
            << seconds / static_cast<double>(runs) * 1e6 << " us per run" << std::endl;
  EXPECT_EQ(tokens, generated);
}

}  // namespace

// Shared host conformance runs independently of graph names and device bank conventions.
TEST(TokenProducingSessionTest, ProgressSettlesWithoutATokenAndDuplicateAcknowledgementFails) {
  TokenProducingSession host{1};
  int owner;
  const auto plan = host.Plan(0, 0, &owner, 1, 4, false);
  BackendRowResult progress;
  progress.key = plan.key;
  progress.prefill = PrefillProgress{4, false};
  EXPECT_TRUE(host.Collected(progress));
  TokenCommitAck ack;
  ack.key = plan.key;
  EXPECT_NO_THROW(host.Acknowledge(ack));
  EXPECT_THROW(host.Acknowledge(ack), std::logic_error);
}

TEST(TokenProducingSessionTest, RetiringCompletionDiscardsBeforeTheSlotCanBeReused) {
  TokenProducingSession host{1};
  int old_owner, new_owner;
  const auto old = host.Plan(0, 0, &old_owner, 1, 0, false);
  host.Retire(0);
  EXPECT_THROW(host.Plan(0, 0, &new_owner, 1, 0, false), std::logic_error);
  BackendRowResult late;
  late.key = old.key;
  late.selected_tokens.emplace();
  late.selected_tokens->candidate_count = 1;
  late.selected_tokens->candidates[0] = 12;
  EXPECT_FALSE(host.Collected(late));
  const auto current = host.Plan(0, 0, &new_owner, 1, 0, false);
  EXPECT_FALSE(old.key == current.key);
  EXPECT_THROW(host.Collected(late), std::logic_error);
  TokenCommitAck stale;
  stale.key = old.key;
  EXPECT_THROW(host.Acknowledge(stale), std::logic_error);
}

TEST(TokenProducingSessionTest, NewLeasesCannotAliasLocalSlotAndSubmissionCounters) {
  int owner;
  TokenProducingSession first{1};
  TokenProducingSession second{1};
  const auto old = first.Plan(0, 0, &owner, 1, 0, false);
  const auto current = second.Plan(0, 0, &owner, 1, 0, false);
  EXPECT_NE(old.key.context_id, current.key.context_id);
  EXPECT_EQ(old.key.submission_id, current.key.submission_id);
  BackendRowResult stale;
  stale.key = old.key;
  EXPECT_THROW(second.Collected(stale), std::logic_error);
}

TEST(TokenProducingSessionTest, ExactCandidateCountsAndDuplicateCompletionsAreValidated) {
  int owner;
  TokenProducingSession host{1};
  const auto plan = host.Plan(0, 0, &owner, 1, 0, false);
  BackendRowResult result;
  result.key = plan.key;
  result.selected_tokens.emplace();
  result.selected_tokens->candidate_count = 2;
  result.selected_tokens->candidates[0] = 12;
  result.selected_tokens->candidates[1] = 1;
  EXPECT_TRUE(host.Collected(result));
  EXPECT_THROW(host.Collected(result), std::logic_error);
  TokenCommitAck ack;
  ack.key = plan.key;
  ack.candidates_consumed = 3;
  EXPECT_THROW(host.Acknowledge(ack), std::logic_error);
  ack.candidates_consumed = 2;
  ack.tokens_appended = 1;
  ack.tokens_staged_for_events = 1;
  ack.turn_finished = true;
  ack.finish_reason = GenerationFinishReason::EosToken;
  ack.disposition = ContinuationDisposition::Invalidate;
  EXPECT_NO_THROW(host.Acknowledge(ack));
}

}  // namespace Generators::test
