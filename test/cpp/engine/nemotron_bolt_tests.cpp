// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <limits>

#include "dflash2_drafter.h"
#include "engine/engine.h"
#include "engine/nemotron_bolt_executor.h"
#include "engine_test_helpers.h"
#include "models/nemotron_bolt.h"
#include "search.h"

namespace Generators::test {

namespace {

std::vector<int32_t> TargetOnly(Model& model, std::span<const int32_t> prompt, size_t count) {
  std::vector<int32_t> history(prompt.begin(), prompt.end());
  auto params = std::make_shared<GeneratorParams>(model);
  params->search.max_length = static_cast<int>(history.size() + count);
  Generator generator{model, *params};
  for (size_t i = 0; i < count; ++i) {
    // The first generated token is the target's prefill prediction, before speculative verification.
    generator.AppendTokens(history);
    generator.GenerateNextToken();
    history.push_back(generator.GetSequence(0).CopyDeviceToCpu().back());
    generator.RewindToLength(0);
  }
  return {history.begin() + prompt.size(), history.end()};
}

TEST(NemotronBolt, EveryAcceptanceLengthAndBudget) {
  const std::vector<int32_t> drafts{20, 21, 22, 23, 24, 25, 26};
  const std::vector<int> eos{2, 11};
  for (size_t accepted = 0; accepted <= drafts.size(); ++accepted) {
    std::vector<int64_t> predictions(8, 99);
    std::copy_n(drafts.begin(), accepted, predictions.begin());
    EXPECT_EQ(NemotronBoltAcceptedPrefix(drafts, predictions), accepted);
    for (size_t remaining = 0; remaining <= 9; ++remaining) {
      const auto tokens = NemotronBoltRoundTokens(drafts, predictions, remaining, eos);
      ASSERT_EQ(tokens.size(), std::min(accepted + 1, remaining));
      for (size_t i = 0; i < tokens.size(); ++i)
        EXPECT_EQ(tokens[i], i < accepted ? drafts[i] : 99);
    }
  }
}

TEST(NemotronBolt, StopsAtEitherEosIncludingCorrectionAndBonus) {
  for (int eos_token : {2, 11}) {
    for (size_t eos_row = 0; eos_row < 8; ++eos_row) {
      std::vector<int32_t> drafts{20, 21, 22, 23, 24, 25, 26};
      if (eos_row < 7) drafts[eos_row] = eos_token;
      std::vector<int64_t> predictions(drafts.begin(), drafts.end());
      predictions.push_back(eos_row == 7 ? eos_token : 99);
      const auto tokens = NemotronBoltRoundTokens(drafts, predictions, 8, std::vector<int>{2, 11});
      ASSERT_EQ(tokens.size(), eos_row + 1);
      EXPECT_EQ(tokens.back(), eos_token);
    }
    const auto correction = NemotronBoltRoundTokens(
        std::vector<int32_t>(7, 20), std::vector<int64_t>(8, eos_token), 8,
        std::vector<int>{2, 11});
    EXPECT_EQ(correction, std::vector<int32_t>{eos_token});
  }
}

TEST(NemotronBolt, RejectsInvalidVerificationGeometry) {
  EXPECT_THROW(NemotronBoltAcceptedPrefix(std::vector<int32_t>(6), std::vector<int64_t>(8)),
               std::runtime_error);
  EXPECT_THROW(NemotronBoltAcceptedPrefix(std::vector<int32_t>(7), std::vector<int64_t>(7)),
               std::runtime_error);
}

TEST(NemotronBolt, PrefillPlanningPreservesCheckpointAndAuxiliaryBoundariesAcrossTargetWidths) {
  const NemotronBoltPrefillPlan plan{8207, 1025, true};
  EXPECT_EQ(plan.checkpoint, 8200u);
  EXPECT_EQ(plan.draft_start, 7176u);
  EXPECT_EQ(plan.auxiliary_start, 7168u);
  for (size_t rows : {size_t{8}, size_t{1024}, size_t{4095}, size_t{4096}}) {
    for (size_t restored : {size_t{0}, size_t{4096}, size_t{8200}}) {
      SCOPED_TRACE(::testing::Message() << "rows=" << rows << ", restored=" << restored);
      size_t position = restored;
      size_t captures = 0;
      while (position < plan.token_count) {
        const size_t count = plan.NextRows(position, rows);
        EXPECT_GT(count, 0u);
        EXPECT_LE(count, rows);
        if (position < plan.checkpoint) EXPECT_LE(position + count, plan.checkpoint);
        position += count;
        if (position == plan.checkpoint) ++captures;
      }
      EXPECT_EQ(position, plan.token_count);
      EXPECT_EQ(captures, restored == plan.checkpoint ? 0u : 1u);
    }
  }
  EXPECT_EQ(plan.NextRows(8192, 4096), 8u);
  EXPECT_EQ(plan.NextRows(8200, 4096), 7u);
  const NemotronBoltPrefillPlan uncached{8207, 1025, false};
  EXPECT_EQ(uncached.auxiliary_start, uncached.draft_start);
  EXPECT_EQ(uncached.NextRows(8192, 4096), 15u);
}

TEST(NemotronBolt, PrefillPlanningHandlesShortPromptsAndRejectsNonprogressingChunks) {
  for (size_t length : {size_t{1}, size_t{7}, size_t{8}, size_t{9}}) {
    const NemotronBoltPrefillPlan plan{length, 1024, true};
    EXPECT_EQ(plan.draft_start, 0u);
    EXPECT_EQ(plan.auxiliary_start, 0u);
    EXPECT_EQ(plan.NextRows(0, std::numeric_limits<size_t>::max()),
              plan.checkpoint != 0 ? plan.checkpoint : length);
    EXPECT_THROW(plan.NextRows(0, 0), std::runtime_error);
    EXPECT_THROW(plan.NextRows(length, 8), std::runtime_error);
    EXPECT_THROW(plan.NextRows(std::numeric_limits<size_t>::max(), 8), std::runtime_error);
  }
  EXPECT_THROW((NemotronBoltPrefillPlan{0, 1024, true}), std::runtime_error);
}

TEST(NemotronBolt, HonorsGreedyEquivalentSamplingOptions) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.model.vocab_size = 128;
  config.model.context_length = 16;
  config.search.max_length = 16;
  config.speculative.max_draft_tokens = 7;
  struct Case {
    float temperature;
    int top_k;
    bool greedy;
  };
  for (const auto& test : std::array<Case, 8>{{
           {0.0f, 0, true},
           {0.0f, 1, true},
           {0.0f, 50, true},
           {0.7f, 1, true},
           {1.0f, 1, true},
           {0.7f, 50, false},
           {1.0f, 0, false},
           {1.0f, 50, false},
       }}) {
    SCOPED_TRACE(::testing::Message() << "temperature=" << test.temperature << ", top_k=" << test.top_k);
    auto params = std::make_shared<GeneratorParams>(config);
    params->use_graph_capture = true;
    params->search.do_sample = true;
    params->search.temperature = test.temperature;
    params->search.top_k = test.top_k;
    if (test.greedy) {
      EXPECT_NO_THROW(ValidateNemotronBoltSearch(*params));
    } else {
      EXPECT_THROW(ValidateNemotronBoltSearch(*params), std::runtime_error);
    }
  }
}

TEST(NemotronBolt, RejectsUnsupportedSearchInsteadOfIgnoringIt) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.model.dflash2.sliding_window = 1024;
  config.speculative.max_draft_tokens = 7;
  auto params = std::make_shared<GeneratorParams>(config);
  params->use_graph_capture = true;
  EXPECT_NO_THROW(ValidateNemotronBoltSearch(*params));
  const std::vector<std::function<void(GeneratorParams&)>> changes{
      [](auto& p) { p.search.do_sample = true; },
      [](auto& p) { p.search.batch_size = 2; },
      [](auto& p) { p.search.num_beams = 2; },
      [](auto& p) { p.search.num_return_sequences = 2; },
      [](auto& p) { p.search.min_length = 1; },
      [](auto& p) { p.search.repetition_penalty = 1.1f; },
      [](auto& p) { p.search.no_repeat_ngram_size = 2; },
      [](auto& p) { p.search.diversity_penalty = 1; },
      [](auto& p) { p.search.length_penalty = 2; },
      [](auto& p) { p.search.blank_penalty = 1; },
      [](auto& p) { p.guidance_type = "json_schema"; },
      [](auto& p) { p.guidance_data = "{}"; },
      [](auto& p) { p.guidance_ff_tokens_enabled = true; },
      [](auto& p) { p.speculative.ngram_size = 3; },
      [](auto& p) { p.speculative.ngram_chained_lookup = true; },
      [](auto& p) { p.speculative.min_adaptive_k = 2; },
      [](auto& p) { p.speculative.cooldown = true; },
      [](auto& p) { p.speculative.max_draft_tokens = 4; },
      [](auto& p) { p.search.chunk_size = 8; },
      [](auto& p) { p.use_graph_capture = false; }};
  for (const auto& change : changes) {
    auto invalid = std::make_shared<GeneratorParams>(config);
    invalid->use_graph_capture = true;
    change(*invalid);
    EXPECT_THROW(ValidateNemotronBoltSearch(*invalid), std::runtime_error);
  }
  Config sliding = config;
  sliding.model.vocab_size = 128;
  sliding.model.context_length = 16;
  sliding.search.max_length = 16;
  auto valid_sliding_free = std::make_shared<GeneratorParams>(sliding);
  valid_sliding_free->use_graph_capture = true;
  EXPECT_NO_THROW(CreateSearch(*valid_sliding_free));
  sliding.model.decoder.sliding_window = Config::Model::Decoder::SlidingWindow{8};
  auto invalid_sliding = std::make_shared<GeneratorParams>(sliding);
  invalid_sliding->use_graph_capture = true;
  EXPECT_THROW(ValidateNemotronBoltSearch(*invalid_sliding), std::runtime_error);
  EXPECT_THROW(CreateSearch(*invalid_sliding), std::runtime_error);
}

TEST(NemotronBolt, PublicTokenCommitAndResetHaveNoLookaheadLeak) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.model.vocab_size = 128;
  config.model.context_length = 16;
  config.model.eos_token_id = {2, 11};
  config.search.max_length = 11;
  auto params = std::make_shared<GeneratorParams>(config);
  auto search = CreateSearch(*params);
  auto prompt = params->p_device->Allocate<int32_t>(3);
  const std::vector<int32_t> prompt_ids{1, 8, 9};
  prompt.CopyFromCpu(prompt_ids);
  for (int request = 0; request < 2; ++request) {
    search->AppendTokens(prompt);
    const auto queued = NemotronBoltRoundTokens(
        std::vector<int32_t>{20, 21, 22, 23, 24, 25, 26},
        std::vector<int64_t>{20, 21, 22, 23, 24, 25, 26, 27}, 8,
        config.model.eos_token_id);
    EXPECT_EQ(search->GetSequenceLength(), 3);
    for (size_t i = 0; i < queued.size(); ++i) {
      EXPECT_FALSE(search->IsDone());
      search->CommitToken(queued[i]);
      EXPECT_EQ(search->GetSequenceLength(), 4 + i);
      const auto visible = search->GetSequence(0).CopyDeviceToCpu();
      EXPECT_EQ(visible.back(), queued[i]);
    }
    EXPECT_TRUE(search->IsDone());
    search->RewindTo(0);
    EXPECT_EQ(search->GetSequenceLength(), 0);
    EXPECT_FALSE(search->IsDone());
  }
}

TEST(NemotronBolt, TerminalEosIsVisibleAndStopsExactlyOnce) {
  for (int token : {2, 11}) {
    Config config;
    config.model.type = "nemotron_bolt";
    config.model.vocab_size = 128;
    config.model.context_length = 16;
    config.model.eos_token_id = {2, 11};
    config.search.max_length = 16;
    auto params = std::make_shared<GeneratorParams>(config);
    auto search = CreateSearch(*params);
    auto prompt = params->p_device->Allocate<int32_t>(1);
    prompt.CpuSpan()[0] = 1;
    search->AppendTokens(prompt);
    search->CommitToken(token);
    EXPECT_TRUE(search->IsDone());
    EXPECT_EQ(search->GetSequenceLength(), 2);
    EXPECT_EQ(search->GetSequence(0).CopyDeviceToCpu().back(), token);
    EXPECT_THROW(search->CommitToken(20), std::runtime_error);
    search->RewindTo(0);
    EXPECT_FALSE(search->IsDone());
    search->AppendTokens(prompt);
    search->CommitToken(token);
    EXPECT_EQ(search->GetSequenceLength(), 2);
  }
}

TEST(NemotronBolt, PromptAtBudgetIsImmediatelyDone) {
  Config config;
  config.model.type = "nemotron_bolt";
  config.model.vocab_size = 128;
  config.model.context_length = 16;
  config.search.max_length = 1;
  auto params = std::make_shared<GeneratorParams>(config);
  auto search = CreateSearch(*params);
  auto prompt = params->p_device->Allocate<int32_t>(1);
  prompt.CpuSpan()[0] = 20;
  search->AppendTokens(prompt);
  EXPECT_TRUE(search->IsDone());
  EXPECT_EQ(search->GetSequenceLength(), 1);
  EXPECT_THROW(search->CommitToken(21), std::runtime_error);
}

struct ScriptedBoltExecution final : NemotronBoltExecutionState {
  bool PrefixCachingEnabled() const override { return true; }
  size_t Prefill(std::span<const int32_t> tokens, int, PrefixCacheMetrics& events) override {
    ++prefills;
    ++events.lookups;
    events.queried_tokens += tokens.size();
    events.matches += cached != 0;
    events.evictions += cache_events.evictions;
    events.retention_refusals += cache_events.retention_refusals;
    events.publication_refusals += cache_events.publication_refusals;
    return cached;
  }
  int32_t NextToken(size_t) override {
    if (fail_next) throw std::runtime_error("injected post-restore token failure");
    ++stats.rounds;
    return token;
  }
  void Reset() override {
    ++resets;
    if (std::exchange(fail_reset, false)) throw std::runtime_error("injected reset failure");
    stats = {};
  }
  SpeculativeStats Stats() const override { return stats; }
  size_t cached{8};
  int32_t token{7};
  size_t prefills{};
  size_t resets{};
  bool fail_next{};
  bool fail_reset{};
  PrefixCacheMetrics cache_events;
  SpeculativeStats stats;
};

TEST(NemotronBoltEngine, CreditsPrefixAdoptionOnlyAfterAcknowledgementIncludingEos) {
  for (int scenario = 0; scenario < 3; ++scenario) {
    SCOPED_TRACE(scenario);
    auto model = LoadDummyDecoderModel();
    auto execution = std::make_unique<ScriptedBoltExecution>();
    auto* state = execution.get();
    state->fail_next = scenario == 1;
    if (scenario == 2) state->token = EosToken(*model);
    auto engine = std::make_shared<Engine>(
        model, CreateNemotronBoltDependencies(model, std::move(execution)));
    auto request = engine->CreateRequest();
    TurnOptions options;
    options.max_generated_tokens = 1;
    options.do_sample = false;
    const std::vector<int32_t> prompt(8, 3);
    request->BeginTurn(prompt, options);
    EXPECT_EQ(engine->PrefixCacheStats()->hits, 0u);
    std::array<EngineEvent, 8> events;
    engine->Run(events);
    EXPECT_EQ(engine->PrefixCacheStats()->matches, 1u);
    EXPECT_EQ(engine->PrefixCacheStats()->hits, scenario == 1 ? 0u : 1u);
    EXPECT_EQ(engine->PrefixCacheStats()->matched_tokens, scenario == 1 ? 0u : 8u);
    if (scenario == 2) EXPECT_EQ(request->TokensCpu().size(), prompt.size());
  }
}

TEST(NemotronBoltEngine, PreservesCacheEventsWithoutAttributingAnotherEngine) {
  auto model = LoadDummyDecoderModel();
  std::array<std::shared_ptr<Engine>, 2> engines;
  for (size_t i = 0; i < engines.size(); ++i) {
    auto execution = std::make_unique<ScriptedBoltExecution>();
    execution->cache_events.evictions = i + 1;
    execution->cache_events.retention_refusals = i + 2;
    execution->cache_events.publication_refusals = i + 3;
    engines[i] = std::make_shared<Engine>(
        model, CreateNemotronBoltDependencies(model, std::move(execution)));
    auto request = engines[i]->CreateRequest();
    TurnOptions options;
    options.max_generated_tokens = 1;
    options.do_sample = false;
    request->BeginTurn(std::vector<int32_t>(8, 3), options);
    std::array<EngineEvent, 8> events;
    engines[i]->Run(events);
    request->Close();
  }
  for (size_t i = 0; i < engines.size(); ++i) {
    EXPECT_EQ(engines[i]->PrefixCacheStats()->evictions, i + 1);
    EXPECT_EQ(engines[i]->PrefixCacheStats()->retention_refusals, i + 2);
    EXPECT_EQ(engines[i]->PrefixCacheStats()->publication_refusals, i + 3);
  }
}

TEST(NemotronBoltEngine, ResetFailureCannotDisappearOnRetirementRetryOrReplacement) {
  for (bool rewind : {false, true}) {
    SCOPED_TRACE(rewind);
    auto model = LoadDummyDecoderModel();
    auto execution = std::make_unique<ScriptedBoltExecution>();
    auto* state = execution.get();
    auto engine = std::make_shared<Engine>(
        model, CreateNemotronBoltDependencies(model, std::move(execution)));
    auto request = engine->CreateRequest();
    TurnOptions options;
    options.max_generated_tokens = 1;
    options.do_sample = false;
    request->BeginTurn(std::vector<int32_t>(8, 3), options);
    std::array<EngineEvent, 8> events;
    engine->Run(events);
    state->fail_reset = true;
    for (int retry = 0; retry < 2; ++retry) {
      if (rewind)
        EXPECT_THROW(request->RewindToStartOfTurn(request->CurrentTurnId()), std::runtime_error);
      else
        EXPECT_THROW(request->Close(), std::runtime_error);
      EXPECT_EQ(state->resets, 1u);
      EXPECT_EQ(engine->GetSpeculativeStats().rounds, 1u);
    }
    const size_t prefills = state->prefills;
    auto replacement = engine->CreateRequest();
    replacement->BeginTurn(std::vector<int32_t>(8, 4), options);
    engine->Run(events);
    EXPECT_EQ(state->prefills, prefills);
  }
}

TEST(NemotronBoltEngine, ResolvedGreedyOverridesDoNotMutateSampledModelDefaults) {
  auto model = LoadDummyDecoderModel();
  auto& defaults = model->config_->search;
  defaults.do_sample = true;
  defaults.temperature = 1.0f;
  defaults.top_k = 50;
  defaults.repetition_penalty = 1.1f;
  defaults.no_repeat_ngram_size = 2;
  auto execution = std::make_unique<ScriptedBoltExecution>();
  auto* state = execution.get();
  auto engine = std::make_shared<Engine>(
      model, CreateNemotronBoltDependencies(model, std::move(execution)));
  auto request = engine->CreateRequest();
  const std::vector<int32_t> prompt(8, 3);
  EXPECT_THROW(request->BeginTurn(prompt), std::runtime_error);
  EXPECT_TRUE(request->TokensCpu().empty());
  EXPECT_EQ(state->prefills, 0u);
  TurnOptions options;
  options.max_generated_tokens = 1;
  options.do_sample = false;
  options.repetition_penalty = 1.0f;
  options.no_repeat_ngram_size = 0;
  EXPECT_NO_THROW(request->BeginTurn(prompt, options));
  std::array<EngineEvent, 8> events;
  engine->Run(events);
  EXPECT_EQ(state->prefills, 1u);
  EXPECT_TRUE(defaults.do_sample);
  EXPECT_FLOAT_EQ(defaults.repetition_penalty, 1.1f);
  EXPECT_EQ(defaults.no_repeat_ngram_size, 2);
}

// Optional real-graph correctness coverage. No models or GPU are needed for the tests above.
TEST(NemotronBoltGPU, DestructionWaitsForSharedSessionExecution) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  if (!path) GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL to a native Bolt package";
  auto model = std::dynamic_pointer_cast<NemotronBoltModel>(CreateModel(GetOrtEnv(), path));
  ASSERT_NE(model, nullptr);
  auto params = std::make_shared<GeneratorParams>(*model);
  params->search.max_length = 16;
  const std::vector<int32_t> prompt(8, 1010);
  Generator survivor{*model, *params};
  survivor.AppendTokens(prompt);
  auto retiring = std::make_unique<Generator>(*model, *params);
  retiring->AppendTokens(prompt);
  std::promise<void> started;
  auto ready = started.get_future();
  std::unique_lock execution{model->execution_mutex_};
  auto destroyed = std::async(std::launch::async, [generator = std::move(retiring), &started]() mutable {
    started.set_value();
    generator.reset();
  });
  ready.wait();
  EXPECT_EQ(destroyed.wait_for(std::chrono::milliseconds{100}), std::future_status::timeout);
  execution.unlock();
  destroyed.get();
  survivor.GenerateNextToken();
  const auto sequence = survivor.GetSequence(0).CopyDeviceToCpu();
  const std::vector<int32_t> expected(sequence.begin(), sequence.end());
  Generator reference{*model, *params};
  reference.AppendTokens(prompt);
  reference.GenerateNextToken();
  const auto actual = reference.GetSequence(0).CopyDeviceToCpu();
  EXPECT_EQ((std::vector<int32_t>{actual.begin(), actual.end()}), expected);
}

TEST(NemotronBoltGPU, SequentialGeneratorsResetAndPublicBoundaries) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  if (!path) GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL to a native Bolt package";
  auto model = CreateModel(GetOrtEnv(), path);
  // A real, non-thinking chat asking for a palindrome function. Length 35 also exercises a
  // three-row final prefill chunk and its nonzero recurrent/anchor row.
  const std::vector<int32_t> prompt{
      10, 25708, 1010, 11, 1010, 10, 3263, 1010, 18746, 1261, 17616, 2254, 1455,
      29196, 5627, 1261, 3310, 1395, 1261, 5796, 92145, 1044, 1454, 1261, 4958,
      17984, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 13};
  std::vector<int32_t> baseline;
  for (int request = 0; request < 3; ++request) {
    SCOPED_TRACE(request);
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.max_length = static_cast<int>(prompt.size() + 32);
    params->search.do_sample = request != 0;
    params->search.temperature = request == 1 ? 0.0f : 1.0f;
    params->search.top_k = request == 2 ? 1 : 50;
    Generator generator{*model, *params};
    for (int reset = 0; reset < 2; ++reset) {
      if (reset == 1) {
        generator.AppendTokens(prompt);
        generator.GenerateNextToken();
        generator.GenerateNextToken();
        EXPECT_GT(generator.GetSpeculativeStats().tokens_buffered, 0);
        generator.RewindToLength(0);
      }
      generator.AppendTokens(prompt);
      EXPECT_THROW(generator.AppendTokens(prompt), std::runtime_error);
      EXPECT_THROW(generator.GetLogits(), std::runtime_error);
      EXPECT_THROW(generator.RewindToLength(1), std::runtime_error);
      EXPECT_EQ(generator.GetSequence(0).CopyDeviceToCpu().size(), prompt.size());
      while (!generator.IsDone()) {
        const size_t before = generator.TokenCount();
        generator.GenerateNextToken();
        EXPECT_EQ(generator.TokenCount(), before + 1);
      }
      auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
      const std::vector<int32_t> result(sequence.begin(), sequence.end());
      if (baseline.empty()) baseline = result;
      EXPECT_EQ(result, baseline);
      EXPECT_EQ(result.size(), prompt.size() + 32);
      const auto stats = generator.GetSpeculativeStats();
      EXPECT_EQ(stats.tokens_emitted, 32);
      EXPECT_EQ(stats.tokens_queued, 32);
      EXPECT_EQ(stats.tokens_buffered, 0);
      EXPECT_GT(stats.full_accept_rounds, 0);
      EXPECT_GT(stats.partial_accept_rounds, 0);
      generator.RewindToLength(0);
      EXPECT_EQ(generator.TokenCount(), 0);
    }
  }
  {
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.do_sample = true;
    params->search.temperature = 1.0f;
    params->search.top_k = 50;
    EXPECT_THROW(Generator(*model, *params), std::runtime_error);
  }
  for (int budget : {0, 1, 2, 3, 7, 8, 9}) {
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.max_length = static_cast<int>(prompt.size()) + budget;
    Generator generator{*model, *params};
    EXPECT_THROW(generator.AppendTokens(std::vector<int32_t>{-1}), std::runtime_error);
    EXPECT_EQ(generator.TokenCount(), 0);
    generator.AppendTokens(prompt);
    while (!generator.IsDone()) generator.GenerateNextToken();
    const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
    EXPECT_EQ(sequence.size(), prompt.size() + budget);
    EXPECT_TRUE(std::equal(sequence.begin(), sequence.end(), baseline.begin()));
    EXPECT_THROW(generator.GenerateNextToken(), std::runtime_error);
  }
  const auto target_only = TargetOnly(*model, prompt, 32);
  EXPECT_TRUE(std::equal(target_only.begin(), target_only.end(), baseline.begin() + prompt.size()));

  // Cross multiple paged-cache blocks and wrap the drafter's 1024-position sliding-window ring.
  std::vector<int32_t> long_prompt;
  for (int i = 0; i < 32; ++i) long_prompt.insert(long_prompt.end(), prompt.begin(), prompt.end());
  auto long_params = std::make_shared<GeneratorParams>(*model);
  long_params->search.max_length = static_cast<int>(long_prompt.size() + 16);
  std::vector<int32_t> long_result;
  {
    Generator generator{*model, *long_params};
    generator.AppendTokens(long_prompt);
    while (!generator.IsDone()) generator.GenerateNextToken();
    const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
    long_result.assign(sequence.begin() + long_prompt.size(), sequence.end());
  }
  ASSERT_FALSE(long_result.empty());
  {
    const auto expected = TargetOnly(*model, long_prompt, long_result.size());
    EXPECT_EQ(long_result, expected);
  }
}

TEST(NemotronBoltGPU, PrefixCacheAcrossGeneratorsAndEngineRequests) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  if (!path) GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL to a native Bolt package";
  auto config = CreateConfig(GetOrtEnv(), path);
  config->engine.dynamic_batching.emplace();
  config->engine.dynamic_batching->max_batch_size = 1;
  config->engine.dynamic_batching->max_snapshots = 2;
  config->engine.dynamic_batching->snapshot_cache.ram_bytes = size_t{4} * 1024 * 1024 * 1024;
  auto model = std::dynamic_pointer_cast<NemotronBoltModel>(CreateModel(GetOrtEnv(), std::move(config)));
  ASSERT_NE(model, nullptr);
  ASSERT_NE(model->prefix_cache_, nullptr);
  std::vector<int32_t> seed{10, 25708, 1010, 11, 1010, 10, 3263, 1010, 18746, 1261, 17616, 2254,
                            1455, 29196, 5627, 1261, 3310, 1395, 1261, 5796, 92145, 1044, 1454, 1261,
                            4958, 17984, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 13};
  const auto generate = [&](const std::vector<int32_t>& prompt) {
    auto params = std::make_shared<GeneratorParams>(*model);
    params->search.max_length = static_cast<int>(prompt.size() + 32);
    Generator generator{*model, *params};
    generator.AppendTokens(prompt);
    const auto cached = dynamic_cast<NemotronBoltState&>(*generator.state_).CachedPromptTokens();
    const auto forwards = generator.GetSpeculativeStats().target_forward_passes;
    while (!generator.IsDone()) generator.GenerateNextToken();
    const auto sequence = generator.GetSequence(0).CopyDeviceToCpu();
    return std::tuple{std::vector<int32_t>{sequence.begin() + prompt.size(), sequence.end()}, cached, forwards};
  };
  for (size_t length : std::array<size_t, 7>{7, 8, 9, 1031, 1032, 1033, 1040}) {
    SCOPED_TRACE(length);
    std::vector<int32_t> prompt;
    while (prompt.size() < length) prompt.insert(prompt.end(), seed.begin(), seed.end());
    prompt.resize(length);
    auto policy = std::move(model->prefix_cache_);
    const auto [cold, unused, cold_forwards] = generate(prompt);
    model->prefix_cache_ = std::move(policy);
    const auto [published, first_cached, unused_forwards] = generate(prompt);
    const auto [warm, cached, warm_forwards] = generate(prompt);
    EXPECT_EQ(published, cold);
    EXPECT_EQ(warm, cold);
    EXPECT_EQ(cached, length / 8 * 8);
    EXPECT_EQ(warm_forwards + cached / 8, cold_forwards);
    // Divergence after a resident aligned checkpoint must not consume the old speculative tail.
    prompt.push_back(1010);
    auto engine = std::make_shared<Engine>(model);
    std::vector<int32_t> baseline;
    for (int repeat = 0; repeat < 2; ++repeat) {
      auto request = engine->CreateRequest();
      TurnOptions options;
      options.max_generated_tokens = 32;
      options.do_sample = false;
      request->BeginTurn(prompt, options);
      for (size_t runs = 0; !request->IsTurnComplete() && runs < 100; ++runs) {
        std::array<EngineEvent, 16> events;
        const auto count = engine->Run(events);
        for (size_t i = 0; i < count; ++i) EXPECT_EQ(events[i].flags & EngineEventFlagFailed, 0u);
      }
      ASSERT_TRUE(request->IsTurnComplete());
      const auto sequence = request->TokensCpu();
      const std::vector<int32_t> output(sequence.begin() + prompt.size(), sequence.end());
      if (repeat == 0) baseline = output;
      EXPECT_EQ(output, baseline);
      EXPECT_GE(request->TurnCachedPromptTokens(), length / 8 * 8);
      if (length == 1033 && repeat == 0) {
        const size_t old_history = request->TokensCpu().size();
        std::vector<int32_t> history(request->TokensCpu().begin(), request->TokensCpu().end());
        const std::vector<int32_t> appended(9, 1010);
        history.insert(history.end(), appended.begin(), appended.end());
        auto [reference, reused, forwards] = generate(history);
        if (!reference.empty() && std::find(model->config_->model.eos_token_id.begin(),
                                            model->config_->model.eos_token_id.end(), reference.back()) !=
                                      model->config_->model.eos_token_id.end())
          reference.pop_back();
        request->BeginTurn(appended, options);
        for (size_t runs = 0; !request->IsTurnComplete() && runs < 100; ++runs) {
          std::array<EngineEvent, 16> events;
          engine->Run(events);
        }
        ASSERT_TRUE(request->IsTurnComplete());
        EXPECT_EQ(request->TurnCachedPromptTokens(), history.size() / 8 * 8 - old_history);
        const auto continued = request->TokensCpu();
        EXPECT_EQ((std::vector<int32_t>{continued.begin() + history.size(), continued.end()}), reference);
      }
      request->Close();
    }
    const auto [expected, ignored, ignored_forwards] = generate(prompt);
    // Request deliberately excludes terminal EOS, unlike the classic Bolt Generator.
    if (!expected.empty() && std::find(model->config_->model.eos_token_id.begin(),
                                       model->config_->model.eos_token_id.end(), expected.back()) !=
                                 model->config_->model.eos_token_id.end())
      EXPECT_EQ(baseline, (std::vector<int32_t>{expected.begin(), expected.end() - 1}));
    else
      EXPECT_EQ(baseline, expected);
  }

  auto engine = std::make_shared<Engine>(model);
  auto request = engine->CreateRequest();
  TurnOptions options;
  options.max_generated_tokens = 32;
  options.do_sample = false;
  request->BeginTurn(seed, options);
  std::array<EngineEvent, 16> events;
  engine->Run(events);
  engine->Run(events);
  ASSERT_FALSE(request->IsTurnComplete());
  ASSERT_GT(engine->GetSpeculativeStats().tokens_buffered, 0u);
  EXPECT_EQ(engine->GetSpeculativeStats().effective_k, 7u);
  ASSERT_TRUE(engine->CancelRequest(request, request->CurrentTurnId()));
  for (size_t runs = 0; engine->HasPendingRequests() && runs < 10; ++runs) engine->Run(events);
  const size_t old_history = request->TokensCpu().size();
  const std::vector<int32_t> appended(9, 1010);
  std::vector<int32_t> history(request->TokensCpu().begin(), request->TokensCpu().end());
  history.insert(history.end(), appended.begin(), appended.end());
  auto [reference, reused, forwards] = generate(history);
  if (!reference.empty() && std::find(model->config_->model.eos_token_id.begin(),
                                      model->config_->model.eos_token_id.end(), reference.back()) !=
                                model->config_->model.eos_token_id.end())
    reference.pop_back();
  request->BeginTurn(appended, options);
  for (size_t runs = 0; !request->IsTurnComplete() && runs < 100; ++runs) engine->Run(events);
  ASSERT_TRUE(request->IsTurnComplete());
  EXPECT_EQ(request->TurnCachedPromptTokens(), history.size() / 8 * 8 - old_history);
  const auto continued = request->TokensCpu();
  EXPECT_EQ((std::vector<int32_t>{continued.begin() + history.size(), continued.end()}), reference);
  const auto live = engine->GetSpeculativeStats();
  EXPECT_GT(live.tokens_discarded, 0u);
  EXPECT_EQ(live.tokens_emitted, 2u + reference.size());
  EXPECT_EQ(live.effective_k, 7u);
  ASSERT_GT(live.rounds, 0u);
  EXPECT_FLOAT_EQ(live.mean_emitted_tokens_per_round, static_cast<float>(live.tokens_emitted) / live.rounds);
  request->Close();
  const auto completed = engine->GetSpeculativeStats();
  EXPECT_EQ(completed.effective_k, 0u);
  EXPECT_EQ(completed.rounds, live.rounds);
  EXPECT_EQ(completed.tokens_emitted, live.tokens_emitted);
  EXPECT_FLOAT_EQ(completed.mean_emitted_tokens_per_round, live.mean_emitted_tokens_per_round);
}

TEST(NemotronBoltGPU, EngineAcceptsGreedyTurnOverSampledAndPenalizedModelDefaults) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  if (!path) GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL to a native Bolt package";
  auto config = CreateConfig(GetOrtEnv(), path);
  config->search.do_sample = true;
  config->search.temperature = 1.0f;
  config->search.top_k = 50;
  config->search.repetition_penalty = 1.1f;
  config->search.no_repeat_ngram_size = 2;
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  auto engine = std::make_shared<Engine>(model);
  auto request = engine->CreateRequest();
  const std::vector<int32_t> prompt{
      10, 25708, 1010, 11, 1010, 10, 3263, 1010, 18746, 1261, 17616, 2254, 1455,
      29196, 5627, 1261, 3310, 1395, 1261, 5796, 92145, 1044, 1454, 1261, 4958,
      17984, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 13};
  EXPECT_THROW(request->BeginTurn(prompt), std::runtime_error);
  EXPECT_TRUE(request->TokensCpu().empty());
  TurnOptions options;
  options.max_generated_tokens = 1;
  options.do_sample = false;
  options.repetition_penalty = 1.0f;
  options.no_repeat_ngram_size = 0;
  request->BeginTurn(prompt, options);
  std::array<EngineEvent, 8> events;
  engine->Run(events);
  EXPECT_TRUE(request->IsTurnComplete());
  EXPECT_EQ(request->TokensCpu().size(), prompt.size() + 1);
  EXPECT_TRUE(model->config_->search.do_sample);
  EXPECT_FLOAT_EQ(model->config_->search.repetition_penalty, 1.1f);
}

TEST(NemotronBoltGPU, AttributesRealRetentionRefusalsAndEvictionsToInitiatingEngine) {
  const char* path = std::getenv("ORTGENAI_BOLT_TEST_MODEL");
  if (!path) GTEST_SKIP() << "Set ORTGENAI_BOLT_TEST_MODEL to a native Bolt package";
  auto config = CreateConfig(GetOrtEnv(), path);
  config->engine.dynamic_batching.emplace();
  config->engine.dynamic_batching->max_batch_size = 1;
  config->engine.dynamic_batching->snapshot_cache.ram_bytes = 1;
  auto model = std::dynamic_pointer_cast<NemotronBoltModel>(CreateModel(GetOrtEnv(), std::move(config)));
  ASSERT_NE(model, nullptr);
  ASSERT_NE(model->prefix_cache_, nullptr);
  std::array<std::shared_ptr<Engine>, 2> engines{
      std::make_shared<Engine>(model), std::make_shared<Engine>(model)};
  constexpr uint64_t PrefixCacheMetrics::* fields[]{
      &PrefixCacheMetrics::lookups, &PrefixCacheMetrics::queried_tokens,
      &PrefixCacheMetrics::matches, &PrefixCacheMetrics::registered_blocks,
      &PrefixCacheMetrics::evictions, &PrefixCacheMetrics::retention_refusals,
      &PrefixCacheMetrics::publication_refusals};
  const auto run = [&](Engine& engine, const std::vector<int32_t>& prompt) {
    const auto before = *engine.PrefixCacheStats();
    const auto native_before = model->prefix_cache_->Metrics();
    auto request = engine.CreateRequest();
    TurnOptions options;
    options.max_generated_tokens = 1;
    options.do_sample = false;
    request->BeginTurn(prompt, options);
    std::array<EngineEvent, 8> events;
    engine.Run(events);
    EXPECT_TRUE(request->IsTurnComplete());
    request->Close();
    const auto after = *engine.PrefixCacheStats();
    const auto& native_after = model->prefix_cache_->Metrics();
    for (auto field : fields) EXPECT_EQ(after.*field - before.*field, native_after.*field - native_before.*field);
  };
  for (size_t ram_bytes : {size_t{1}, size_t{4} * 1024 * 1024 * 1024}) {
    SCOPED_TRACE(ram_bytes);
    Config::Engine::DynamicBatching settings;
    settings.snapshot_cache.ram_bytes = ram_bytes;
    auto storage = std::make_shared<PrefixSnapshotCache>(
        std::array<size_t, 3>{0, ram_bytes, 0}, 1);
    model->memory_budget_.Configure(settings, storage);
    model->prefix_cache_ = std::make_unique<StateCheckpointCache>(storage, kBoltVerifyRows);
    const auto before = *engines[0]->PrefixCacheStats();
    run(*engines[0], std::vector<int32_t>(8, 1010));
    if (ram_bytes == 1) {
      EXPECT_EQ(engines[0]->PrefixCacheStats()->retention_refusals, before.retention_refusals + 1);
    } else {
      run(*engines[0], std::vector<int32_t>(8, 1011));
      EXPECT_GT(engines[0]->PrefixCacheStats()->evictions, before.evictions);
    }
    const auto first_engine = *engines[0]->PrefixCacheStats();
    {
      auto params = std::make_shared<GeneratorParams>(*model);
      params->search.max_length = 9;
      Generator generator{*model, *params};
      generator.AppendTokens(std::vector<int32_t>(8, 1012));
      generator.GenerateNextToken();
    }
    run(*engines[1], std::vector<int32_t>(8, 1013));
    const auto after = *engines[0]->PrefixCacheStats();
    for (auto field : fields) EXPECT_EQ(after.*field, first_engine.*field);
  }
  constexpr size_t vram_bytes = size_t{4} * 1024 * 1024 * 1024;
  Config::Engine::DynamicBatching settings;
  settings.snapshot_cache.vram_bytes = vram_bytes;
  auto storage = std::make_shared<PrefixSnapshotCache>(
      std::array<size_t, 3>{vram_bytes, 0, 0}, 1);
  model->memory_budget_.Configure(settings, storage);
  model->prefix_cache_ = std::make_unique<StateCheckpointCache>(storage, kBoltVerifyRows);
  auto& allocator = model->p_device_->GetAllocator();
  struct RestoreReserve {
    ~RestoreReserve() { allocator.Reserve = original; }
    Ort::Allocator& allocator;
    decltype(OrtAllocator::Reserve) original;
  } restore{allocator, allocator.Reserve};
  allocator.Reserve = [](OrtAllocator*, size_t) -> void* { throw std::bad_alloc{}; };
  const auto before = *engines[0]->PrefixCacheStats();
  run(*engines[0], std::vector<int32_t>(8, 1014));
  EXPECT_EQ(engines[0]->PrefixCacheStats()->publication_refusals, before.publication_refusals + 1);
  EXPECT_EQ(engines[0]->PrefixCacheStats()->retention_refusals, before.retention_refusals);
}

TEST(NemotronBolt, GgufInitializersConfigParsingAndPathValidation) {
  namespace fs_std = std::filesystem;
  const auto tmp_dir = fs_std::temp_directory_path() / "ortgenai_gguf_init_test";
  std::error_code ec;
  fs_std::remove_all(tmp_dir, ec);
  fs_std::create_directories(tmp_dir);

  auto write_file = [&](const fs_std::path& p, const std::string& content) {
    std::ofstream out(p, std::ios::binary);
    out << content;
  };

  // 1. String form for decoder, inline object form for dflash2
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "nemotron_bolt",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "model.onnx",
        "gguf_initializers": "gguf_initializers.json"
      },
      "dflash2": {
        "filename": "dflash2.onnx",
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "head_size": 64,
        "block_size": 8,
        "num_draft_tokens": 7,
        "selector_top_k": 8,
        "mask_token_id": 100,
        "sliding_window": 1024,
        "gguf_initializers": {
          "files": ["weights.gguf"],
          "share": true,
          "tensors": {
            "embed_w": "token_embd.weight",
            "lm_head": {
              "gguf_name": "output.weight",
              "data_type": 2,
              "shape": [128, 34],
              "share": false
            }
          }
        }
      }
    },
    "search": {}
  })");

  Config cfg(fs::path{tmp_dir.string()}, {});
  EXPECT_EQ(cfg.model.decoder.gguf_initializers.config_file, "gguf_initializers.json");
  ASSERT_EQ(cfg.model.dflash2.gguf_initializers.files.size(), 1u);
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.files[0], "weights.gguf");
  EXPECT_TRUE(cfg.model.dflash2.gguf_initializers.share);
  EXPECT_TRUE(cfg.model.dflash2.gguf_initializers.has_share);
  ASSERT_EQ(cfg.model.dflash2.gguf_initializers.tensors.size(), 2u);
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[0].onnx_name, "embed_w");
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[0].gguf_name, "token_embd.weight");
  EXPECT_FALSE(cfg.model.dflash2.gguf_initializers.tensors[0].has_share);
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[1].onnx_name, "lm_head");
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[1].gguf_name, "output.weight");
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[1].data_type, 2);
  EXPECT_EQ(cfg.model.dflash2.gguf_initializers.tensors[1].shape, (std::vector<int64_t>{128, 34}));
  EXPECT_TRUE(cfg.model.dflash2.gguf_initializers.tensors[1].has_share);
  EXPECT_FALSE(cfg.model.dflash2.gguf_initializers.tensors[1].share);

  auto dflash2_cfg = CreateDflash2Config(cfg);
  EXPECT_EQ(dflash2_cfg->model.decoder.gguf_initializers.files, cfg.model.dflash2.gguf_initializers.files);
  EXPECT_EQ(dflash2_cfg->model.decoder.gguf_initializers.tensors.size(), 2u);

  // Per-tensor file and data type overrides for mtp
  Config mtp_cfg;
  OverlayConfig(mtp_cfg, R"({
    "model": {
      "mtp": {
        "filename": "mtp.onnx",
        "gguf_initializers": {
          "files": ["mtp.gguf"],
          "tensors": {
            "w0": {
              "gguf_name": "blk.0.weight",
              "file": "shard0.gguf",
              "data_type": 1
            }
          }
        }
      }
    }
  })");
  ASSERT_EQ(mtp_cfg.model.mtp.gguf_initializers.files.size(), 1u);
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.files[0], "mtp.gguf");
  ASSERT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors.size(), 1u);
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors[0].onnx_name, "w0");
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors[0].gguf_name, "blk.0.weight");
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors[0].file, "shard0.gguf");
  EXPECT_EQ(mtp_cfg.model.mtp.gguf_initializers.tensors[0].data_type, 1);

  // 2. LoadGgufInitializersConfig from external JSON file and reject path traversal
  write_file(tmp_dir / "gguf_initializers.json", R"({
    "files": ["model.gguf"],
    "tensors": {
      "embed": {
        "gguf_name": "token_embd.weight",
        "share": true
      }
    }
  })");
  Config::Model::GgufInitializers ext_loaded;
  LoadGgufInitializersConfig(fs::path{(tmp_dir / "gguf_initializers.json").string()}, ext_loaded);
  ASSERT_EQ(ext_loaded.files.size(), 1u);
  EXPECT_EQ(ext_loaded.files[0], "model.gguf");
  ASSERT_EQ(ext_loaded.tensors.size(), 1u);
  EXPECT_EQ(ext_loaded.tensors[0].onnx_name, "embed");
  EXPECT_TRUE(ext_loaded.tensors[0].share);

  // Path traversal in external gguf_initializers.json must be rejected
  write_file(tmp_dir / "bad_gguf_initializers.json", R"({
    "files": ["../outside.gguf"],
    "tensors": {}
  })");
  Config::Model::GgufInitializers bad_loaded;
  EXPECT_THROW(LoadGgufInitializersConfig(fs::path{(tmp_dir / "bad_gguf_initializers.json").string()}, bad_loaded),
               std::runtime_error);

  // Path traversal in inline genai_config.json gguf_initializers must be rejected
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "nemotron_bolt",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "model.onnx",
        "gguf_initializers": "../outside.json"
      }
    },
    "search": {}
  })");
  EXPECT_THROW(Config(fs::path{tmp_dir.string()}, {}), std::runtime_error);

  // 3. Synthetic GGUF v3 binary header parsing and AddGgufInitializers validation
  auto write_u32 = [](std::string& buf, uint32_t v) {
    buf.append(reinterpret_cast<const char*>(&v), 4);
  };
  auto write_u64 = [](std::string& buf, uint64_t v) {
    buf.append(reinterpret_cast<const char*>(&v), 8);
  };
  auto write_gguf_str = [&](std::string& buf, std::string_view s) {
    write_u64(buf, s.size());
    buf.append(s.data(), s.size());
  };

  std::string gguf_bytes;
  gguf_bytes.append("GGUF", 4);
  write_u32(gguf_bytes, 3);  // version 3
  write_u64(gguf_bytes, 2);  // tensor_count = 2
  write_u64(gguf_bytes, 1);  // kv_count = 1
  write_gguf_str(gguf_bytes, "general.alignment");
  write_u32(gguf_bytes, 4);  // UINT32
  write_u32(gguf_bytes, 32);

  // Tensor 0: "weight_f32", dims=[4, 2] -> ONNX default shape [2, 4] FLOAT (32 bytes), offset 0
  write_gguf_str(gguf_bytes, "weight_f32");
  write_u32(gguf_bytes, 2);
  write_u64(gguf_bytes, 4);
  write_u64(gguf_bytes, 2);
  write_u32(gguf_bytes, 0);  // F32
  write_u64(gguf_bytes, 0);

  // Tensor 1: "weight_q8", dims=[32, 2] -> ONNX default shape [2, 34] UINT8 (68 bytes), offset 32
  write_gguf_str(gguf_bytes, "weight_q8");
  write_u32(gguf_bytes, 2);
  write_u64(gguf_bytes, 32);
  write_u64(gguf_bytes, 2);
  write_u32(gguf_bytes, 8);  // Q8_0
  write_u64(gguf_bytes, 32);

  while (gguf_bytes.size() % 32 != 0) gguf_bytes.push_back('\0');
  gguf_bytes.append(32 + 68, '\x01');
  write_file(tmp_dir / "synthetic.gguf", gguf_bytes);

  // Missing GGUF tensor name throws from AddGgufInitializers before CreateSession
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "gpt2",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "nonexistent_model.onnx",
        "gguf_initializers": {
          "files": ["synthetic.gguf"],
          "tensors": {
            "w_missing": "does_not_exist"
          }
        }
      }
    },
    "search": {}
  })");
  try {
    CreateModel(GetOrtEnv(), tmp_dir.string().c_str());
    FAIL() << "Expected missing GGUF tensor to throw";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("does_not_exist"), std::string::npos) << e.what();
  }

  // Byte size mismatch throws from AddGgufInitializers
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "gpt2",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "nonexistent_model.onnx",
        "gguf_initializers": {
          "files": ["synthetic.gguf"],
          "tensors": {
            "w0": {
              "gguf_name": "weight_f32",
              "shape": [2, 5]
            }
          }
        }
      }
    },
    "search": {}
  })");
  try {
    CreateModel(GetOrtEnv(), tmp_dir.string().c_str());
    FAIL() << "Expected byte size mismatch to throw";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("Byte size mismatch"), std::string::npos) << e.what();
  }

  // Valid GGUF tensor mappings succeed in AddGgufInitializers and reach CreateSession(nonexistent_model.onnx)
  write_file(tmp_dir / "genai_config.json", R"({
    "model": {
      "type": "gpt2",
      "vocab_size": 128,
      "context_length": 32,
      "decoder": {
        "filename": "nonexistent_model.onnx",
        "gguf_initializers": {
          "files": ["synthetic.gguf"],
          "tensors": {
            "w0": "weight_f32",
            "w1": "weight_q8"
          }
        }
      }
    },
    "search": {}
  })");
  try {
    CreateModel(GetOrtEnv(), tmp_dir.string().c_str());
    FAIL() << "Expected nonexistent_model.onnx to throw after AddGgufInitializers succeeds";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("nonexistent_model.onnx"), std::string::npos) << e.what();
  }

  fs_std::remove_all(tmp_dir, ec);
}

}  // namespace
}  // namespace Generators::test
