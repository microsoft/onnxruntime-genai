// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <barrier>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "constrained_logits_processor.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"
#include "guidance_test_access.h"
#include "models/preprocessing/genai_tokenizer.h"

namespace Generators {
namespace test {

#if USE_GUIDANCE
namespace {

std::shared_ptr<Model> LoadGuidanceModel() {
  return CreateModel(
      GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
}

DeviceSpan<float> LogitsForToken(Model& model, int32_t token) {
  auto logits = model.p_device_inputs_->Allocate<float>(
      static_cast<size_t>(model.config_->model.vocab_size));
  auto cpu_logits = logits.CpuSpan();
  std::fill(cpu_logits.begin(), cpu_logits.end(), 0.0f);
  cpu_logits[token] = 100.0f;
  logits.CopyCpuToDevice();
  return logits;
}

TEST(GuidanceCacheTest, ReusesTokenizerAndCompiledGrammar) {
  auto model = LoadGuidanceModel();
  auto first_params = MakeGreedyParams(*model);
  first_params->SetGuidance("regex", "!", false);
  auto first = CreateGuidanceLogitsProcessor(*model, first_params);

  auto second_params = MakeGreedyParams(*model);
  second_params->SetGuidance("regex", "!", false);
  auto second = CreateGuidanceLogitsProcessor(*model, second_params);

  const auto stats = GetGuidanceCacheStats(*model);
  EXPECT_EQ(stats.tokenizer_initializations, 1u);
  EXPECT_EQ(stats.grammar_misses, 1u);
  EXPECT_EQ(stats.grammar_hits, 1u);
  EXPECT_EQ(stats.cached_grammars, 1u);
  EXPECT_GT(stats.cached_key_bytes, 0u);
}

TEST(GuidanceCacheTest, SingleFlightsConcurrentIdenticalGrammars) {
  auto model = LoadGuidanceModel();
  constexpr size_t kRequestCount = 8;
  std::barrier gate(static_cast<std::ptrdiff_t>(kRequestCount));
  std::vector<std::exception_ptr> errors(kRequestCount);
  std::vector<std::thread> threads;
  threads.reserve(kRequestCount);

  for (size_t i = 0; i < kRequestCount; ++i) {
    threads.emplace_back([&, i] {
      try {
        gate.arrive_and_wait();
        auto params = MakeGreedyParams(*model);
        params->SetGuidance("regex", "!", false);
        auto processor = CreateGuidanceLogitsProcessor(*model, params);
        static_cast<void>(processor);
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    EXPECT_EQ(error, nullptr);
  }

  const auto stats = GetGuidanceCacheStats(*model);
  EXPECT_EQ(stats.tokenizer_initializations, 1u);
  EXPECT_EQ(stats.grammar_misses, 1u);
  EXPECT_EQ(stats.grammar_hits + stats.grammar_waits, kRequestCount - 1);
}

TEST(GuidanceCacheTest, IsolatesConcurrentUniqueGrammars) {
  auto model = LoadGuidanceModel();
  constexpr size_t kRequestCount = 8;
  std::barrier gate(static_cast<std::ptrdiff_t>(kRequestCount));
  std::vector<std::exception_ptr> errors(kRequestCount);
  std::vector<std::thread> threads;
  threads.reserve(kRequestCount);

  for (size_t i = 0; i < kRequestCount; ++i) {
    threads.emplace_back([&, i] {
      try {
        gate.arrive_and_wait();
        auto params = MakeGreedyParams(*model);
        const std::string grammar(1, static_cast<char>('a' + i));
        params->SetGuidance("regex", grammar.c_str(), false);
        auto processor = CreateGuidanceLogitsProcessor(*model, params);
        static_cast<void>(processor);
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    EXPECT_EQ(error, nullptr);
  }

  const auto stats = GetGuidanceCacheStats(*model);
  EXPECT_EQ(stats.tokenizer_initializations, 1u);
  EXPECT_EQ(stats.grammar_misses, kRequestCount);
  EXPECT_EQ(stats.cached_grammars, kRequestCount);
}

TEST(GuidanceCacheTest, PendingMaskOutlivesRemovedProcessor) {
  auto model = LoadGuidanceModel();
  auto tokenizer = model->CreateTokenizer();
  const auto expected_tokens = tokenizer->Encode("!");
  ASSERT_EQ(expected_tokens.size(), 1u);

  auto params = MakeGreedyParams(*model);
  params->SetGuidance("regex", "!!", false);
  auto processor = CreateGuidanceLogitsProcessor(*model, params);
  processor->ProcessLogits(
      LogitsForToken(*model, expected_tokens.front()));
  int32_t token = expected_tokens.front();
  processor->CommitTokens(std::span<int32_t>{&token, 1});
  std::array<ConstrainedLogitsProcessor*, 1> processors{processor.get()};
  ScheduleGuidanceMaskComputation(processors);

  auto observer = processor->Clone();
  processor.reset();

  const auto mask = observer->GetReadyMask();
  ASSERT_FALSE(mask.empty());
  const uint32_t expected = static_cast<uint32_t>(expected_tokens.front());
  EXPECT_NE(mask[expected / 32] & (uint32_t{1} << (expected % 32)), 0u);
}

TEST(GuidanceCacheTest, FailedFutureReschedulesRealProcessor) {
  auto model = LoadGuidanceModel();
  auto params = MakeGreedyParams(*model);
  params->SetGuidance("regex", "!", false);
  auto processor = CreateGuidanceLogitsProcessor(*model, params);
  auto& guidance = dynamic_cast<GuidanceLogitsProcessor&>(*processor);
  GuidanceProcessorTestAccess::InstallFailedMaskFuture(guidance);

  EXPECT_THROW(processor->GetReadyMask(), std::runtime_error);
  EXPECT_TRUE(GuidanceProcessorTestAccess::MaskDirty(guidance));

  const auto retried_mask = processor->GetReadyMask();
  EXPECT_FALSE(retried_mask.empty());
  EXPECT_FALSE(GuidanceProcessorTestAccess::MaskDirty(guidance));
}

TEST(GuidanceCacheTest, FailedFutureRollsBackAndReschedulesEngineRequest) {
  auto model = LoadGuidanceModel();
  auto engine = MakeDoublesEngine(model, /*capacity=*/8, EosToken(*model));
  auto tokenizer = model->CreateTokenizer();
  const auto expected_tokens = tokenizer->Encode("!");
  ASSERT_EQ(expected_tokens.size(), 1u);

  auto params = MakeGreedyParams(*model);
  params->SetGuidance("regex", "!", false);
  auto request = CreateEngineRequest(engine.engine, *params);
  request->BeginTurn(std::array<int32_t, 3>{2, 3, 4},
                     std::optional<size_t>{1});
  auto& guidance = dynamic_cast<GuidanceLogitsProcessor&>(
      *RequestGuidanceTestAccess::Get(*request));
  GuidanceProcessorTestAccess::InstallFailedMaskFuture(guidance);
  engine.executor->SetForcedToken(expected_tokens.front());

  const auto retryable = RunOne(*engine.engine);
  EXPECT_EQ(retryable.flags, EngineEventFlagRetryable);
  EXPECT_EQ(retryable.error_code, EngineErrorCode::RetryableExecution);

  const auto completed = RunOne(*engine.engine);
  EXPECT_EQ(completed.request, request);
  EXPECT_NE(completed.flags & EngineEventFlagTurnFinished, 0u);
}

TEST(GuidanceCacheTest, EvictionKeepsLiveProcessorAssetsValid) {
  auto model = LoadGuidanceModel();
  auto first_params = MakeGreedyParams(*model);
  first_params->SetGuidance("regex", "first", false);
  auto first = CreateGuidanceLogitsProcessor(*model, first_params);

  for (size_t i = 0; i < 65; ++i) {
    auto params = MakeGreedyParams(*model);
    const auto grammar = "value" + std::to_string(i);
    params->SetGuidance("regex", grammar.c_str(), false);
    auto processor = CreateGuidanceLogitsProcessor(*model, params);
    static_cast<void>(processor);
  }

  const auto stats = GetGuidanceCacheStats(*model);
  EXPECT_GT(stats.grammar_evictions, 0u);
  EXPECT_LE(stats.cached_grammars, 64u);
  EXPECT_LE(stats.cached_key_bytes, 16u * 1024u * 1024u);
  const auto first_tokens = model->CreateTokenizer()->Encode("first");
  ASSERT_FALSE(first_tokens.empty());
  int32_t token = first_tokens.front();
  EXPECT_NO_THROW(first->CommitTokens(std::span<int32_t>{&token, 1}));
  EXPECT_NO_THROW(first->GetReadyMask());
}

TEST(GuidanceCacheTest, OversizedGrammarDoesNotEvictCachedEntries) {
  auto model = LoadGuidanceModel();
  auto cached_params = MakeGreedyParams(*model);
  cached_params->SetGuidance("regex", "!", false);
  auto cached = CreateGuidanceLogitsProcessor(*model, cached_params);
  const auto before = GetGuidanceCacheStats(*model);
  ASSERT_EQ(before.cached_grammars, 1u);

  std::string oversized_schema(16u * 1024u * 1024u, ' ');
  oversized_schema.front() = 'x';
  auto oversized_params = MakeGreedyParams(*model);
  oversized_params->SetGuidance(
      "json_schema", oversized_schema.c_str(), false);
  EXPECT_THROW(
      CreateGuidanceLogitsProcessor(*model, oversized_params),
      std::runtime_error);

  const auto after = GetGuidanceCacheStats(*model);
  EXPECT_EQ(after.cached_grammars, before.cached_grammars);
  EXPECT_EQ(after.cached_key_bytes, before.cached_key_bytes);
  EXPECT_EQ(after.grammar_evictions, before.grammar_evictions);
}

TEST(GuidanceCacheTest, DoesNotRetainFailedCompilations) {
  auto model = LoadGuidanceModel();
  for (size_t attempt = 1; attempt <= 2; ++attempt) {
    auto params = MakeGreedyParams(*model);
    params->SetGuidance("regex", "[", false);
    EXPECT_THROW(
        CreateGuidanceLogitsProcessor(*model, params),
        std::runtime_error);

    const auto stats = GetGuidanceCacheStats(*model);
    EXPECT_EQ(stats.grammar_misses, attempt);
    EXPECT_EQ(stats.cached_grammars, 0u);
    EXPECT_EQ(stats.cached_key_bytes, 0u);
  }
}

}  // namespace
#endif

}  // namespace test
}  // namespace Generators
