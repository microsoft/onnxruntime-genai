// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Engine contract tests for grammar fast-forward tokens: the forced span a turn's grammar produces
// after a sampled token. These run over the synthetic-paged fixture model, whose checked-in
// tokenizer maps token 5 -> "ST" and token 6 -> "OP" (so the pair completes a "STOP" stop string)
// and tokens 11-63 to unique filler text; the hybrid case uses the synthetic-composite fixture, and
// the USE_GUIDANCE case a real llguidance grammar over tiny-random-gpt2. The recording executor
// never runs the model; it fabricates one-hot logits for the token a test scripts.

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
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

// Long-span tests trigger on token 11 and expect it plus the first 7 forced tokens in one step.
static_assert(kMaxGeneratedTokensPerStep == 8);
std::vector<int32_t> LongForcedSpan() { return {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}; }

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
    if (std::exchange(*fail_next_commit_, false)) {
      throw std::runtime_error("Injected grammar commit failure.");
    }
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
    clone->fail_next_commit_ = fail_next_commit_;
    return clone;
  }

  const std::vector<int32_t>& CommittedTokens() const { return committed_tokens_; }
  // Shared with every clone, so restoring a rolled-back step's checkpoint cannot re-arm it.
  void FailNextCommit() { *fail_next_commit_ = true; }

 private:
  int32_t trigger_;
  std::vector<int32_t> continuation_;
  std::vector<int32_t> committed_tokens_;
  std::vector<int32_t> pending_forced_tokens_;
  std::shared_ptr<bool> fail_next_commit_ = std::make_shared<bool>(false);
};

ForcingGrammar& InstallForcingGrammar(Request& request, int32_t trigger,
                                      std::vector<int32_t> continuation) {
  auto grammar = std::make_unique<ForcingGrammar>(trigger, std::move(continuation));
  auto& installed = *grammar;
  RequestGuidanceTestAccess::Install(request, std::move(grammar));
  return installed;
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

TurnOptions Limit(size_t max_generated_tokens) {
  TurnOptions options;
  options.max_generated_tokens = max_generated_tokens;
  return options;
}

std::vector<int32_t> GeneratedTokens(const Request& request) {
  const auto tokens = request.TokensCpu();
  return {tokens.begin() + static_cast<ptrdiff_t>(Prompt().size()), tokens.end()};
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
// of the span stays private, and the row publishes up to that many more of it, instead of sampling,
// as it reads to its tail.
TEST_F(GuidanceFastForwardTest, LongForcedSpanIsPublishedAtMostEightTokensPerStepBeforeSampling) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto request = CreateEngineRequest(engine.engine);
  request->BeginTurn(Prompt());
  const auto forced = LongForcedSpan();
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

// search.chunk_size bounds prompt chunks only; a row's unread forced tokens are read in one step.
TEST_F(GuidanceFastForwardTest, PrefillChunkSizeDoesNotSplitTheUnreadForcedTokens) {
  model_->config_->search.chunk_size = 2;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto request = CreateEngineRequest(engine.engine);
  request->BeginTurn(Prompt());
  InstallForcingGrammar(*request, /*trigger=*/11, LongForcedSpan());

  std::array<EngineEvent, 16> storage;
  ASSERT_EQ(engine.engine->Run(storage), 0u);  // first 2 prompt tokens
  ASSERT_EQ(engine.engine->Run(storage), 8u);  // last prompt token; samples 11, publishes 7
  engine.executor->SetForcedToken(EosToken(*model_));
  ASSERT_EQ(engine.engine->Run(storage), 4u);
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(engine.executor->decoded_token_counts, (std::vector<size_t>{2, 1, 8, 4}));
}

std::vector<int32_t> LongPrompt(size_t length) {
  std::vector<int32_t> prompt(length);
  std::iota(prompt.begin(), prompt.end(), 11);
  return prompt;
}

// A forcing turn that has sampled 11 and published 7 forced tokens: a catch-up row with 8 unread.
std::shared_ptr<Request> CatchingUpRequest(DoublesEngine& engine) {
  auto forcing = CreateRequestWithPrompt(engine.engine, Prompt());
  InstallForcingGrammar(*forcing, /*trigger=*/11, LongForcedSpan());
  std::array<EngineEvent, 16> storage;
  EXPECT_EQ(engine.engine->Run(storage), kMaxGeneratedTokensPerStep);
  return forcing;
}

// Beside a catch-up row, it and a waiting prompt take turns at spare tokens, whichever waited
// longer since it was last served going first, so neither keeps the other waiting.
TEST_F(GuidanceFastForwardTest, CatchUpRowAndPromptTakeTurnsAtSpareTokens) {
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 6;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto forcing = CatchingUpRequest(engine);
  auto prompt = CreateRequestWithPrompt(engine.engine, LongPrompt(20));

  std::vector<std::vector<size_t>> reads;  // {forcing, prompt} per step
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    std::vector<size_t> step(2);
    for (const auto& entry : context.plan->requests)
      step[entry.request_id == forcing.get() ? 0 : 1] = entry.unprocessed_token_count;
    reads.push_back(step);
  });
  std::array<EngineEvent, 16> storage;
  for (int step = 0; step < 3; ++step)
    engine.engine->Run(storage);
  EXPECT_EQ(reads, (std::vector<std::vector<size_t>>{{5, 1}, {1, 5}, {2, 4}}));
}

// A rolled-back step records no service, so its retry shares spare tokens exactly as it did.
TEST_F(GuidanceFastForwardTest, RolledBackStepDoesNotChangeWhoGetsSpareTokens) {
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 6;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto forcing = CatchingUpRequest(engine);
  auto prompt = CreateRequestWithPrompt(engine.engine, LongPrompt(20));

  std::vector<std::vector<size_t>> reads;  // {forcing, prompt} per attempt
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    std::vector<size_t> attempt(2);
    for (const auto& entry : context.plan->requests)
      attempt[entry.request_id == forcing.get() ? 0 : 1] = entry.unprocessed_token_count;
    reads.push_back(attempt);
  });
  std::array<EngineEvent, 16> storage;
  engine.engine->Run(storage);
  engine.executor->SetNextFailure(ScriptedExecutionFailure::RetryableDuringExecution);
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  ASSERT_EQ(storage[0].flags, EngineEventFlagRetryable);
  engine.engine->Run(storage);
  EXPECT_EQ(reads, (std::vector<std::vector<size_t>>{{5, 1}, {1, 5}, {1, 5}}));
}

// Fast-forward turns arriving every step cannot keep a long prompt from spare tokens: it waits at
// most one turn for each forcing turn with an older stamp, including one admitted in the step the
// prompt was last served.
TEST_F(GuidanceFastForwardTest, SustainedFastForwardArrivalsCannotKeepAPromptWaiting) {
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 10;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto prompt = CreateRequestWithPrompt(engine.engine, LongPrompt(40));

  size_t passed_over = 0;
  size_t max_passed_over = 0;
  size_t catch_up_steps = 0;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    for (const auto& entry : context.plan->requests) {
      catch_up_steps += !entry.is_prefill && entry.ReadsSeveralCommittedTokens();
      if (entry.request_id != prompt.get())
        continue;
      const auto& request = *entry.request;
      const bool waiting = request.CurrentSequenceLength() - request.ProcessedSequenceLength() > 1;
      passed_over = waiting && entry.unprocessed_token_count == 1 ? passed_over + 1 : 0;
      max_passed_over = std::max(max_passed_over, passed_over);
    }
  });
  std::vector<std::shared_ptr<Request>> forcing;
  std::array<EngineEvent, 64> storage;
  for (int step = 0; step < 60 && prompt->IsPrefill(); ++step) {
    if (forcing.size() < 6) {  // one fast-forward arrival per step
      forcing.push_back(CreateRequestWithPrompt(engine.engine, Prompt()));
      InstallForcingGrammar(*forcing.back(), /*trigger=*/11, LongForcedSpan());
    }
    engine.engine->Run(storage);
  }
  ASSERT_FALSE(prompt->IsPrefill());
  ASSERT_GT(catch_up_steps, 0u);
  // Each step's spare tokens fill one catch-up row's share, so the prompt waits at most one turn
  // for each forcing turn.
  EXPECT_LE(max_passed_over, forcing.size());
}

// A forcing turn whose every sampled token forces three more: a catch-up row in every step, so
// every step orders chunk work by waiting time.
std::shared_ptr<Request> AlwaysCatchingUpRequest(DoublesEngine& engine) {
  auto forcing = CreateRequestWithPrompt(engine.engine, Prompt(), Limit(400));
  InstallForcingGrammar(*forcing, /*trigger=*/11, {12, 13, 14});
  std::array<EngineEvent, 16> storage;
  EXPECT_EQ(engine.engine->Run(storage), 4u);
  return forcing;
}

// The prompt among `prompts` that read more than one token in the step, or none.
std::optional<size_t> Leader(const ExecutionContext& context,
                             const std::vector<std::shared_ptr<Request>>& prompts) {
  for (const auto& entry : context.plan->requests) {
    const auto row = std::ranges::find(prompts, entry.request_id, &std::shared_ptr<Request>::get);
    if (row != prompts.end() && entry.unprocessed_token_count > 1)
      return static_cast<size_t>(row - prompts.begin());
  }
  return std::nullopt;
}

// Beside a catch-up row, a newly admitted prompt counts as just served, so it queues behind a
// prompt already waiting: of prompts {first, waiting, arrival}, the arrival leads only after the
// waiting prompt has.
TEST_F(GuidanceFastForwardTest, NewArrivalQueuesBehindAWaitingPrompt) {
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 8;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto forcing = AlwaysCatchingUpRequest(engine);
  std::vector<std::shared_ptr<Request>> prompts{CreateRequestWithPrompt(engine.engine, LongPrompt(20)),
                                                CreateRequestWithPrompt(engine.engine, LongPrompt(20))};
  std::vector<std::optional<size_t>> leaders;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    leaders.push_back(Leader(context, prompts));
  });
  std::array<EngineEvent, 16> storage;
  engine.engine->Run(storage);
  prompts.push_back(CreateRequestWithPrompt(engine.engine, LongPrompt(20)));
  for (int step = 0; step < 4; ++step)
    engine.engine->Run(storage);
  EXPECT_EQ(leaders, (std::vector<std::optional<size_t>>{0, 0, 1, 2, 2}));
}

// Beside a catch-up row, a resident request starting a new turn queues behind a prompt already
// waiting, like an arrival: of {resident, served, waiting}, the new turn leads only after the
// waiting prompt has, and its turn-start stamp keeps it ahead of the prompt served since.
TEST_F(GuidanceFastForwardTest, ContinuationTurnQueuesBehindAWaitingPrompt) {
  model_->config_->engine.dynamic_batching->max_scheduled_tokens = 8;
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto resident = CreateRequestWithPrompt(engine.engine, Prompt(), Limit(1));
  std::array<EngineEvent, 16> storage;
  engine.engine->Run(storage);
  ASSERT_EQ(resident->Status(), RequestStatus::TurnComplete);
  auto forcing = AlwaysCatchingUpRequest(engine);
  std::vector<std::shared_ptr<Request>> prompts{resident, CreateRequestWithPrompt(engine.engine, LongPrompt(20)),
                                                CreateRequestWithPrompt(engine.engine, LongPrompt(20))};
  std::vector<std::optional<size_t>> leaders;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    leaders.push_back(Leader(context, prompts));
  });
  engine.engine->Run(storage);
  resident->BeginTurn(LongPrompt(20));
  for (int step = 0; step < 4; ++step)
    engine.engine->Run(storage);
  EXPECT_EQ(leaders, (std::vector<std::optional<size_t>>{1, 1, 2, 0, 0}));
}

// While a peer verifies drafts, the row reads one unread token per step, so the peer keeps them;
// once nobody drafts, it reads the rest together.
TEST_F(GuidanceFastForwardTest, UnreadForcedTokensAreReadOneAtATimeWhileAPeerVerifiesDrafts) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(1);
  auto drafting = CreateRequestWithPrompt(engine.engine, Prompt());
  auto forcing = CreateRequestWithPrompt(engine.engine, Prompt());
  InstallForcingGrammar(*forcing, /*trigger=*/11, {12, 13});
  std::array<EngineEvent, 8> storage;
  ASSERT_EQ(engine.engine->Run(storage), 4u);  // drafting: 11; forcing: 11, 12, 13

  std::vector<std::vector<std::pair<size_t, size_t>>> forcing_and_drafting_reads;
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    ASSERT_NE(context.plan, nullptr);
    std::pair<size_t, size_t> forcing_read{};
    std::pair<size_t, size_t> drafting_read{};
    for (const auto& entry : context.plan->requests) {
      auto& read = entry.request_id == forcing.get() ? forcing_read : drafting_read;
      read = {entry.unprocessed_token_count, entry.draft_token_count};
    }
    forcing_and_drafting_reads.push_back({forcing_read, drafting_read});
  });
  drafting->SetDraftTokens(std::array<int32_t, 1>{11});
  ASSERT_GT(engine.engine->Run(storage), 0u);
  ASSERT_GT(engine.engine->Run(storage), 0u);

  using Read = std::pair<size_t, size_t>;  // {tokens read, of which drafts}
  EXPECT_EQ(forcing_and_drafting_reads,
            (std::vector<std::vector<Read>>{{Read{1, 0}, Read{2, 1}}, {Read{2, 0}, Read{1, 0}}}));
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
  InstallForcingGrammar(*request, /*trigger=*/11, LongForcedSpan());

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

// Recurrent state cannot checkpoint drafts beside a row reading several tokens, so on a hybrid model
// the row reads one unread token and the peer's draft is still verified.
TEST_F(GuidanceFastForwardTest, HybridPeerKeepsItsDraftWhileTheRowReadsOneUnreadToken) {
  model_ = LoadSyntheticCompositeModel();
  auto engine = MakeCompositeDoublesEngine(model_, /*forced_token=*/11);
  auto drafting = CreateRequestWithPrompt(engine.engine, Prompt());
  auto forcing = CreateRequestWithPrompt(engine.engine, Prompt());
  InstallForcingGrammar(*forcing, /*trigger=*/11, {12, 13});
  std::array<EngineEvent, 8> storage;
  ASSERT_EQ(engine.engine->Run(storage), 4u);  // drafting: 11; forcing: 11, 12, 13
  drafting->SetDraftTokens(std::array<int32_t, 1>{11});

  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    ASSERT_NE(context.plan, nullptr);
    ASSERT_EQ(context.plan->requests.size(), 2u);
    for (const auto& entry : context.plan->requests) {
      const bool is_forcing = entry.request_id == forcing.get();
      EXPECT_EQ(entry.unprocessed_token_count, is_forcing ? 1u : 2u);
      EXPECT_EQ(entry.draft_token_count, is_forcing ? 0u : 1u);
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
    InstallForcingGrammar(*request, /*trigger=*/11, LongForcedSpan());
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
              Concat(Prompt(), Concat({11}, LongForcedSpan())));
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
  InstallForcingGrammar(*request, /*trigger=*/11, LongForcedSpan());

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
  InstallForcingGrammar(*forcing, /*trigger=*/11, LongForcedSpan());
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

// A caller may drain fewer events per Run than a step produces. Tokens still arrive in order,
// exactly once, without another model step until they are drained. The cap is per request, not per
// batch: a peer's first token waits behind at most one capped batch from each forcing request.
TEST_F(GuidanceFastForwardTest, SmallEventBuffersDeliverForcedTokensInOrderWithoutStarvingAPeer) {
  for (const size_t forcing_count : {size_t{1}, size_t{7}}) {
    for (const size_t capacity : {size_t{1}, size_t{3}}) {
      SCOPED_TRACE(::testing::Message() << forcing_count << " forcing, capacity " << capacity);
      auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
      std::vector<std::shared_ptr<Request>> forcing;
      for (size_t i = 0; i < forcing_count; ++i) {
        forcing.push_back(CreateRequestWithPrompt(engine.engine, Prompt(), Limit(12)));
        InstallForcingGrammar(*forcing.back(), /*trigger=*/11, LongForcedSpan());
      }
      auto peer = CreateRequestWithPrompt(engine.engine, Prompt(), Limit(2));

      std::vector<EngineEvent> storage(capacity);
      std::vector<std::vector<int32_t>> forcing_tokens(forcing_count);
      std::vector<int32_t> peer_tokens;
      size_t events_before_peer = 0;
      for (int call = 0; call < 512; ++call) {
        const size_t count = engine.engine->Run(storage);
        const bool all_complete =
            peer->IsTurnComplete() &&
            std::ranges::all_of(forcing, [](const auto& request) { return request->IsTurnComplete(); });
        if (count == 0 && all_complete)
          break;
        for (const auto& event : std::span{storage}.first(count)) {
          if (!(event.flags & EngineEventFlagToken))
            continue;
          if (event.request == peer) {
            // The peer's first token is step 1's last event: no step ran while draining it.
            if (peer_tokens.empty())
              EXPECT_EQ(engine.executor->decode_calls, 1);
            peer_tokens.push_back(event.token);
            continue;
          }
          const auto row = std::ranges::find(forcing, event.request);
          ASSERT_NE(row, forcing.end());
          forcing_tokens[static_cast<size_t>(row - forcing.begin())].push_back(event.token);
          events_before_peer += peer_tokens.empty();
        }
      }

      for (const auto& tokens : forcing_tokens)
        EXPECT_EQ(tokens, Concat({11}, LongForcedSpan()));
      EXPECT_EQ(peer_tokens, (std::vector<int32_t>{11, 11}));
      // All rows run in step 1 and the peer was admitted last, so every forcing row's first capped
      // batch is delivered before the peer's token: exactly the per-request bound times N.
      EXPECT_EQ(events_before_peer, forcing_count * kMaxGeneratedTokensPerStep);
      EXPECT_EQ(engine.executor->decode_calls, 2);
    }
  }
}

// The peer's second draft is rejected in the step where the forcing row reads one unread token; the
// peer keeps its accepted prefix and correction, and the forcing row's tokens are unaffected.
TEST_F(GuidanceFastForwardTest, PartialDraftAcceptanceBesideACatchUpReadKeepsBothRowsExact) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  engine.cache->SetMaxDraftTokensPerStep(2);
  auto peer = CreateRequestWithPrompt(engine.engine, Prompt());
  auto forcing = CreateRequestWithPrompt(engine.engine, Prompt());
  InstallForcingGrammar(*forcing, /*trigger=*/11, LongForcedSpan());
  std::array<EngineEvent, 16> storage;
  ASSERT_EQ(engine.engine->Run(storage), 9u);  // peer: 11; forcing: 11 through 18

  peer->SetDraftTokens(std::array<int32_t, 2>{11, 12});
  // Packed rows: the peer's two draft rows and its final row, then the forcing row.
  engine.executor->SetVerifyRowTokens({11, 20, 30, 40});
  engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
    ASSERT_NE(context.plan, nullptr);
    for (const auto& entry : context.plan->requests) {
      const bool is_forcing = entry.request_id == forcing.get();
      EXPECT_EQ(entry.unprocessed_token_count, is_forcing ? 1u : 3u);
      EXPECT_EQ(entry.draft_token_count, is_forcing ? 0u : 2u);
    }
  });
  ASSERT_EQ(engine.engine->Run(storage), 2u);
  EXPECT_EQ(EventTokens(std::span{storage}.first(2)), (std::vector<int32_t>{11, 20}));

  engine.executor->SetExecutionCallback(nullptr);
  engine.executor->SetVerifyRowTokens({});
  engine.executor->SetForcedToken(EosToken(*model_));
  while (!(forcing->IsTurnComplete() && peer->IsTurnComplete()))
    ASSERT_GT(engine.engine->Run(storage), 0u);
  EXPECT_EQ(GeneratedTokens(*peer), (std::vector<int32_t>{11, 11, 20}));
  EXPECT_EQ(GeneratedTokens(*forcing), Concat({11}, LongForcedSpan()));
  EXPECT_TRUE(ValidateRequestInvariants(peer->Snapshot()).empty());
  EXPECT_TRUE(ValidateRequestInvariants(forcing->Snapshot()).empty());
}

// Turns that start on different steps, one of them while another row is catching up, generate
// exactly what each generates alone.
TEST_F(GuidanceFastForwardTest, StaggeredTurnsGenerateWhatEachGeneratesAlone) {
  struct TurnSpec {
    std::optional<std::vector<int32_t>> forced_after_11;
    size_t limit;
  };
  const std::array<TurnSpec, 3> specs{
      TurnSpec{LongForcedSpan(), 20}, TurnSpec{std::nullopt, 5}, TurnSpec{{{30, 31, 32}}, 9}};
  const auto start_turn = [&](DoublesEngine& engine, const TurnSpec& spec) {
    auto request = CreateRequestWithPrompt(engine.engine, Prompt(), Limit(spec.limit));
    if (spec.forced_after_11)
      InstallForcingGrammar(*request, /*trigger=*/11, *spec.forced_after_11);
    return request;
  };
  const auto run_until_complete = [](DoublesEngine& engine, const auto& requests) {
    std::array<EngineEvent, 16> storage;
    for (int step = 0; step < 64; ++step) {
      engine.engine->Run(storage);
      if (std::all_of(requests.begin(), requests.end(),
                      [](const auto& request) { return request->IsTurnComplete(); }))
        return;
    }
    FAIL() << "Turns did not complete.";
  };

  std::vector<std::vector<int32_t>> alone;
  for (const auto& spec : specs) {
    auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
    const std::array requests{start_turn(engine, spec)};
    run_until_complete(engine, requests);
    alone.push_back(GeneratedTokens(*requests[0]));
  }

  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  std::vector<std::shared_ptr<Request>> requests;
  std::array<EngineEvent, 16> storage;
  for (const auto& spec : specs) {  // a new turn each step
    requests.push_back(start_turn(engine, spec));
    engine.engine->Run(storage);
  }
  run_until_complete(engine, requests);
  for (size_t i = 0; i < requests.size(); ++i) {
    SCOPED_TRACE(::testing::Message() << "turn " << i);
    EXPECT_EQ(GeneratedTokens(*requests[i]), alone[i]);
    EXPECT_TRUE(ValidateRequestInvariants(requests[i]->Snapshot()).empty());
  }
}

// Cache pressure that defers the row holding unpublished forced tokens leaves them private and
// intact; the row publishes them once it runs again.
TEST_F(GuidanceFastForwardTest, CapacityDeferralKeepsTheUnpublishedSpanUntilTheRowRuns) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto forcing = CreateRequestWithPrompt(engine.engine, Prompt(), Limit(12));
  InstallForcingGrammar(*forcing, /*trigger=*/11, LongForcedSpan());
  auto peer = CreateRequestWithPrompt(engine.engine, Prompt(), Limit(3));
  std::array<EngineEvent, 16> storage;
  ASSERT_EQ(engine.engine->Run(storage), 9u);  // forcing: 11 through 18; peer: 11
  const auto deferred = forcing->Snapshot();

  engine.cache->SetCapacityDeferredRequest(forcing);
  ASSERT_EQ(engine.engine->Run(storage), 1u);
  EXPECT_EQ(storage[0].request, peer);
  EXPECT_EQ(forcing->CurrentSequenceLength(), deferred.current_sequence_length);
  EXPECT_EQ(forcing->ProcessedSequenceLength(), deferred.processed_sequence_length);

  engine.cache->SetCapacityDeferredRequest(nullptr);
  ASSERT_EQ(engine.engine->Run(storage), 5u);  // forcing: 19 through 22; peer: 11
  EXPECT_TRUE(forcing->IsTurnComplete());
  EXPECT_EQ(GeneratedTokens(*forcing), Concat({11}, LongForcedSpan()));
  EXPECT_TRUE(ValidateRequestInvariants(forcing->Snapshot()).empty());
}

// A failure after this step staged forced tokens, on the step that samples and forces them or on
// one that publishes held-back ones, rolls them all back; the retry publishes them exactly once.
TEST_F(GuidanceFastForwardTest, FailureAfterForcedTokensAreStagedRollsThemBackForAnExactRetry) {
  for (const bool on_publish_step : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "on publish step " << on_publish_step);
    auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
    auto forcing = CreateRequestWithPrompt(engine.engine, Prompt(), Limit(12));
    InstallForcingGrammar(*forcing, /*trigger=*/11, LongForcedSpan());
    // Staged after the forcing row; its grammar throws while committing its sampled token.
    auto peer = CreateRequestWithPrompt(engine.engine, Prompt(), Limit(3));
    auto& peer_grammar = InstallForcingGrammar(*peer, /*trigger=*/99, {});
    std::array<EngineEvent, 16> storage;
    if (on_publish_step)
      ASSERT_EQ(engine.engine->Run(storage), 9u);
    const auto before = forcing->Snapshot();
    const auto generated_before = GeneratedTokens(*forcing);

    // The forcing row stages its tokens before the peer's commit throws.
    engine.executor->SetExecutionCallback([&](ExecutionContext& context) {
      ASSERT_EQ(context.plan->requests.front().request_id, forcing.get());
    });
    peer_grammar.FailNextCommit();
    ASSERT_EQ(RunOne(*engine.engine).flags, EngineEventFlagRetryable);
    engine.executor->SetExecutionCallback({});
    EXPECT_EQ(forcing->CurrentSequenceLength(), before.current_sequence_length);
    EXPECT_EQ(forcing->ProcessedSequenceLength(), before.processed_sequence_length);
    EXPECT_EQ(GeneratedTokens(*forcing), generated_before);
    // The restored grammar cursor has seen only the tokens committed before the failed step.
    auto* grammar = static_cast<ForcingGrammar*>(RequestGuidanceTestAccess::Get(*forcing));
    ASSERT_NE(grammar, nullptr);
    EXPECT_EQ(grammar->CommittedTokens(),
              on_publish_step ? std::vector<int32_t>{11} : std::vector<int32_t>{});

    const size_t published = on_publish_step ? 4 : 8;
    ASSERT_EQ(engine.engine->Run(storage), published + 1);
    while (!forcing->IsTurnComplete())
      ASSERT_GT(engine.engine->Run(storage), 0u);
    EXPECT_EQ(GeneratedTokens(*forcing), Concat({11}, LongForcedSpan()));
    EXPECT_EQ(forcing->TurnGeneratedTokens(), 12u);
  }
}

// A fatal failure after a publish step staged forced tokens ends the turn with only the output that
// earlier steps published; the tokens that step staged are dropped.
TEST_F(GuidanceFastForwardTest, FatalFailureAfterForcedTokensAreStagedKeepsOnlyPublishedOutput) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, /*forced_token=*/11);
  auto forcing = CreateRequestWithPrompt(engine.engine, Prompt());
  InstallForcingGrammar(*forcing, /*trigger=*/11, LongForcedSpan());
  std::array<EngineEvent, 16> storage;
  ASSERT_EQ(engine.engine->Run(storage), 8u);

  engine.cache->ThrowPrepareFailureOnce();
  const auto failure = RunOne(*engine.engine);
  EXPECT_EQ(failure.request, forcing);
  EXPECT_EQ(failure.flags, EngineEventFlagTurnFinished | EngineEventFlagFailed);
  EXPECT_EQ(forcing->FinishReason(), GenerationFinishReason::Failed);
  EXPECT_EQ(GeneratedTokens(*forcing), (std::vector<int32_t>{11, 12, 13, 14, 15, 16, 17, 18}));
  EXPECT_EQ(forcing->TurnGeneratedTokens(), 8u);
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
  ASSERT_LE(text_tokens.size(), kMaxGeneratedTokensPerStep);  // published in one step

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
