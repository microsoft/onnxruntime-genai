// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// A deterministic microbenchmark for prefix-aware paged key-value caching.
//
// It drives the production admission, planning, reservation, commit, retention, and eviction path
// -- the real PagedCacheManager, the real dynamic scheduler, and the real Engine transaction --
// over a repeated-prefix workload shaped like multi-turn agent traffic: a long shared prefix
// (system prompt, tool schemas, file context) followed by a short varying suffix.
//
// Model execution is replaced by a double, so the wall-clock figures here are Engine host overhead
// and NOT end-to-end latency: no model runs and no attention is computed. The token counts, block
// residency, and cache-hit figures are exact and hardware-independent, and they are what determines
// the prefill work a real deployment saves.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "engine/engine_invariants.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"

namespace Generators {
namespace test {
namespace {

constexpr size_t kBenchBlockSize = 16;
constexpr size_t kBenchNumBlocks = 512;
constexpr size_t kSharedPrefixTokens = 512;  // System prompt + tool schemas + file context.
constexpr size_t kSuffixTokens = 15;         // The turn's own question.
constexpr int kGeneratedTokens = 8;
constexpr size_t kSequentialTurns = 32;
constexpr size_t kConcurrentRequests = 8;

struct BenchmarkReport {
  double ttft_p50_us{};
  double ttft_p95_us{};
  size_t prefill_tokens{};
  size_t adopted_tokens{};
  size_t decode_tokens{};
  size_t peak_blocks{};
  size_t requests{};
  double concurrent_wall_us{};
  uint64_t lookups{};
  uint64_t hits{};
  uint64_t evictions{};
  uint64_t retention_refusals{};
  uint64_t duplicate_registrations{};
  uint64_t hash_collisions{};
  std::vector<int32_t> first_turn_output;
};

double Percentile(std::vector<double> samples, double fraction) {
  if (samples.empty()) return 0.0;
  std::sort(samples.begin(), samples.end());
  const size_t index = std::min(samples.size() - 1,
                                static_cast<size_t>(fraction * static_cast<double>(samples.size())));
  return samples[index];
}

class PrefixCacheBenchmark : public ::testing::Test {
 protected:
  std::shared_ptr<Model> MakeModel(bool prefix_caching, size_t num_blocks) {
    auto model = LoadDummyDecoderModel();
    Config::Engine::DynamicBatching dynamic_batching;
    dynamic_batching.block_size = kBenchBlockSize;
    dynamic_batching.num_blocks = num_blocks;
    dynamic_batching.max_batch_size = kConcurrentRequests;
    dynamic_batching.max_scheduled_tokens = 4096;
    dynamic_batching.prefix_caching = prefix_caching;
    dynamic_batching.prefix_cache_pool_fraction = 0.5f;
    model->config_->engine.dynamic_batching = dynamic_batching;
    model->config_->model.context_length = 4096;
    return model;
  }

  // Long shared prefix, short varying suffix: the shape multi-turn agent traffic has.
  static std::vector<int32_t> MakeTurnPrompt(size_t turn) {
    std::vector<int32_t> prompt;
    prompt.reserve(kSharedPrefixTokens + kSuffixTokens);
    for (size_t i = 0; i < kSharedPrefixTokens; ++i) {
      prompt.push_back(static_cast<int32_t>(2 + (i % 29)));
    }
    for (size_t i = 0; i < kSuffixTokens; ++i) {
      prompt.push_back(static_cast<int32_t>(2 + ((turn * 7 + i * 3) % 29)));
    }
    return prompt;
  }

  BenchmarkReport Run(bool prefix_caching, size_t num_blocks = kBenchNumBlocks) {
    auto model = MakeModel(prefix_caching, num_blocks);
    auto paged = MakePagedEngine(model);
    BenchmarkReport report;
    std::vector<double> ttft_samples;

    const auto observe_peak = [&]() {
      const auto snapshot = paged.cache->Snapshot();
      report.peak_blocks = std::max(report.peak_blocks,
                                    snapshot.total_blocks - snapshot.free_blocks);
    };

    // Phase 1: a sequential conversation, one turn at a time.
    for (size_t turn = 0; turn < kSequentialTurns; ++turn) {
      const auto prompt = MakeTurnPrompt(turn);
      auto params = MakeGreedyParams(*model);
      params->search.max_length = static_cast<int>(prompt.size()) + kGeneratedTokens;
      auto request = std::make_shared<Request>(params);
      request->AddTokens(prompt);

      const auto admitted_at = std::chrono::steady_clock::now();
      paged.engine->AddRequest(request);
      bool first_token_seen = false;
      std::vector<int32_t> generated;
      while (paged.engine->HasPendingRequests()) {
        auto ready = paged.engine->Step();
        if (!ready) break;
        if (!first_token_seen && ready.get() == request.get() && ready->HasUnseenTokens()) {
          const auto now = std::chrono::steady_clock::now();
          ttft_samples.push_back(
              std::chrono::duration<double, std::micro>(now - admitted_at).count());
          first_token_seen = true;
        }
        while (ready->HasUnseenTokens()) generated.push_back(ready->UnseenToken());
        observe_peak();
      }
      if (turn == 0) report.first_turn_output = generated;
      paged.engine->RemoveRequest(request);
      ++report.requests;
    }

    // Phase 2: the same shared prefix under concurrency.
    std::vector<std::shared_ptr<Request>> concurrent;
    for (size_t i = 0; i < kConcurrentRequests; ++i) {
      const auto prompt = MakeTurnPrompt(kSequentialTurns + i);
      auto params = MakeGreedyParams(*model);
      params->search.max_length = static_cast<int>(prompt.size()) + kGeneratedTokens;
      auto request = std::make_shared<Request>(params);
      request->AddTokens(prompt);
      concurrent.push_back(std::move(request));
    }

    const auto concurrent_start = std::chrono::steady_clock::now();
    for (const auto& request : concurrent) {
      paged.engine->AddRequest(request);
    }
    while (paged.engine->HasPendingRequests()) {
      auto ready = paged.engine->Step();
      if (!ready) break;
      while (ready->HasUnseenTokens()) ready->UnseenToken();
      observe_peak();
    }
    report.concurrent_wall_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - concurrent_start)
            .count();
    for (const auto& request : concurrent) {
      paged.engine->RemoveRequest(request);
    }
    report.requests += concurrent.size();

    report.ttft_p50_us = Percentile(ttft_samples, 0.50);
    report.ttft_p95_us = Percentile(ttft_samples, 0.95);
    report.prefill_tokens = paged.executor->prefill_tokens;
    report.decode_tokens = paged.executor->decode_tokens;
    report.adopted_tokens = paged.executor->adopted_tokens;
    if (const auto* metrics = paged.engine->PrefixCacheStats()) {
      report.lookups = metrics->lookups;
      report.hits = metrics->hits;
      report.evictions = metrics->evictions;
      report.retention_refusals = metrics->retention_refusals;
      report.duplicate_registrations = metrics->duplicate_registrations;
      report.hash_collisions = metrics->hash_collisions;
    }
    EXPECT_TRUE(ValidateCacheInvariants(paged.cache->Snapshot()).empty());
    return report;
  }
};

TEST_F(PrefixCacheBenchmark, RepeatedPrefixWorkloadColdVersusWarm) {
  const auto cold = Run(/*prefix_caching=*/false);
  const auto warm = Run(/*prefix_caching=*/true);

  const double hit_rate =
      warm.lookups == 0 ? 0.0 : 100.0 * static_cast<double>(warm.hits) / static_cast<double>(warm.lookups);
  const size_t offered_prompt_tokens = warm.prefill_tokens + warm.adopted_tokens;
  const double skipped_share =
      offered_prompt_tokens == 0
          ? 0.0
          : 100.0 * static_cast<double>(warm.adopted_tokens) / static_cast<double>(offered_prompt_tokens);
  const size_t block_bytes =
      kBenchBlockSize * 2 /*kv*/ * 2 /*heads*/ * 4 /*head size*/ * 4 /*fp32*/;

  std::printf(
      "\n"
      "prefix cache microbenchmark (engine host path only; model execution is stubbed)\n"
      "  workload            : %zu shared-prefix tokens + %zu varying tokens, %d generated,\n"
      "                        %zu sequential turns then %zu concurrent requests\n"
      "  geometry            : block_size=%zu, num_blocks=%zu, max_batch_size=%zu\n"
      "  ---------------------------------------------  cold  ----  warm  ----\n"
      "  prompt tokens computed                      %8zu      %8zu\n"
      "  prompt tokens skipped (adopted)             %8zu      %8zu\n"
      "  decode tokens                               %8zu      %8zu\n"
      "  peak KV blocks in use                       %8zu      %8zu\n"
      "  peak KV bytes (all layers)                  %8zu      %8zu\n"
      "  admission->first token p50 (us, host only)  %8.1f      %8.1f\n"
      "  admission->first token p95 (us, host only)  %8.1f      %8.1f\n"
      "  concurrent phase wall (us, host only)       %8.1f      %8.1f\n"
      "  prefix lookups / hits                       %8llu/%llu   %8llu/%llu\n"
      "  prefix cache hit rate                                      %7.1f%%\n"
      "  prompt tokens skipped                                      %7.1f%%\n"
      "  evictions / retention refusals                             %llu / %llu\n"
      "  duplicate registrations / hash collisions                  %llu / %llu\n\n",
      kSharedPrefixTokens, kSuffixTokens, kGeneratedTokens, kSequentialTurns, kConcurrentRequests,
      kBenchBlockSize, kBenchNumBlocks, kConcurrentRequests,
      cold.prefill_tokens, warm.prefill_tokens,
      cold.adopted_tokens, warm.adopted_tokens,
      cold.decode_tokens, warm.decode_tokens,
      cold.peak_blocks, warm.peak_blocks,
      cold.peak_blocks * block_bytes, warm.peak_blocks * block_bytes,
      cold.ttft_p50_us, warm.ttft_p50_us,
      cold.ttft_p95_us, warm.ttft_p95_us,
      cold.concurrent_wall_us, warm.concurrent_wall_us,
      static_cast<unsigned long long>(cold.lookups), static_cast<unsigned long long>(cold.hits),
      static_cast<unsigned long long>(warm.lookups), static_cast<unsigned long long>(warm.hits),
      hit_rate, skipped_share,
      static_cast<unsigned long long>(warm.evictions),
      static_cast<unsigned long long>(warm.retention_refusals),
      static_cast<unsigned long long>(warm.duplicate_registrations),
      static_cast<unsigned long long>(warm.hash_collisions));

  // The workload has to actually exercise the feature, and it has to keep the same answers.
  EXPECT_EQ(cold.adopted_tokens, 0u);
  EXPECT_GT(warm.adopted_tokens, 0u);
  EXPECT_LT(warm.prefill_tokens, cold.prefill_tokens);
  EXPECT_EQ(warm.decode_tokens, cold.decode_tokens);
  EXPECT_EQ(warm.first_turn_output, cold.first_turn_output);
  EXPECT_EQ(warm.hash_collisions, 0u);
}

// The same workload on a pool barely large enough to hold one concurrent batch, so retention has to
// give capacity back and the eviction policy is actually exercised.
TEST_F(PrefixCacheBenchmark, RepeatedPrefixWorkloadUnderMemoryPressure) {
  constexpr size_t kTightBlocks = 96;  // Eight concurrent 527-token sequences need 264 uncached.
  const auto warm = Run(/*prefix_caching=*/true, kTightBlocks);

  const double hit_rate =
      warm.lookups == 0 ? 0.0 : 100.0 * static_cast<double>(warm.hits) / static_cast<double>(warm.lookups);
  const size_t offered_prompt_tokens = warm.prefill_tokens + warm.adopted_tokens;
  const double skipped_share =
      offered_prompt_tokens == 0
          ? 0.0
          : 100.0 * static_cast<double>(warm.adopted_tokens) / static_cast<double>(offered_prompt_tokens);

  std::printf(
      "\n"
      "prefix cache microbenchmark under memory pressure (num_blocks=%zu)\n"
      "  prompt tokens computed / skipped            %zu / %zu (%.1f%% skipped)\n"
      "  prefix lookups / hits                       %llu / %llu (%.1f%%)\n"
      "  peak KV blocks in use                       %zu of %zu\n"
      "  evictions / retention refusals              %llu / %llu\n"
      "  admission->first token p50/p95 (us, host)   %.1f / %.1f\n\n",
      kTightBlocks, warm.prefill_tokens, warm.adopted_tokens, skipped_share,
      static_cast<unsigned long long>(warm.lookups), static_cast<unsigned long long>(warm.hits),
      hit_rate, warm.peak_blocks, kTightBlocks,
      static_cast<unsigned long long>(warm.evictions),
      static_cast<unsigned long long>(warm.retention_refusals),
      warm.ttft_p50_us, warm.ttft_p95_us);

  // Every request still completed, so retention gave capacity back rather than starving anyone.
  EXPECT_EQ(warm.requests, kSequentialTurns + kConcurrentRequests);
  EXPECT_LE(warm.peak_blocks, kTightBlocks);
  EXPECT_EQ(warm.hash_collisions, 0u);
}

}  // namespace
}  // namespace test
}  // namespace Generators
