// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "engine/prefix_cache.h"
#include "engine/fixed_state_pool.h"
#include "engine/scheduler.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"

namespace Generators::test {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kBenchmarkBlockSize = 32;
constexpr size_t kBlocksPerPrefix = 128;
constexpr size_t kCachedPrefixCount = 128;
constexpr size_t kCachedBlockCount =
    kBlocksPerPrefix * kCachedPrefixCount;
constexpr size_t kWarmupSamples = 50;
constexpr size_t kMeasuredSamples = 500;
constexpr size_t kOperationsPerSample = 10;

struct LatencySummary {
  double mean_us{};
  double p50_us{};
  double p95_us{};
};

double Percentile(std::vector<double> samples, double percentile) {
  std::sort(samples.begin(), samples.end());
  const double position =
      percentile / 100.0 * static_cast<double>(samples.size() - 1);
  const size_t lower = static_cast<size_t>(position);
  const size_t upper = std::min(lower + 1, samples.size() - 1);
  const double fraction = position - static_cast<double>(lower);
  return samples[lower] +
         (samples[upper] - samples[lower]) * fraction;
}

template <typename Operation>
LatencySummary Measure(Operation&& operation,
                       size_t operations_per_sample = 1,
                       size_t measured_samples = kMeasuredSamples,
                       size_t warmup_samples = kWarmupSamples) {
  for (size_t sample = 0; sample < warmup_samples; ++sample) {
    for (size_t operation_index = 0;
         operation_index < operations_per_sample; ++operation_index) {
      operation();
    }
  }

  std::vector<double> samples;
  samples.reserve(measured_samples);
  for (size_t sample = 0; sample < measured_samples; ++sample) {
    const auto start = Clock::now();
    for (size_t operation_index = 0;
         operation_index < operations_per_sample; ++operation_index) {
      operation();
    }
    const auto stop = Clock::now();
    const double elapsed_us =
        std::chrono::duration<double, std::micro>(stop - start).count();
    samples.push_back(
        elapsed_us / static_cast<double>(operations_per_sample));
  }

  return {
      std::accumulate(samples.begin(), samples.end(), 0.0) /
          static_cast<double>(samples.size()),
      Percentile(samples, 50.0),
      Percentile(samples, 95.0),
  };
}

void PrintResult(std::string_view benchmark,
                 std::string_view scenario,
                 size_t batch_size,
                 size_t matched_blocks,
                 const LatencySummary& latency) {
  std::cout << std::left
            << std::setw(24) << benchmark
            << std::setw(18) << scenario
            << std::right
            << std::setw(8) << batch_size
            << std::setw(16) << matched_blocks
            << std::fixed << std::setprecision(3)
            << std::setw(14) << latency.mean_us
            << std::setw(14) << latency.p50_us
            << std::setw(14) << latency.p95_us
            << '\n';
}

std::vector<int32_t> BuildPrompt(size_t prefix_index) {
  std::vector<int32_t> tokens;
  tokens.reserve(kBlocksPerPrefix * kBenchmarkBlockSize + 1);
  const size_t first_token =
      prefix_index * kBlocksPerPrefix * kBenchmarkBlockSize + 1;
  for (size_t index = 0;
       index < kBlocksPerPrefix * kBenchmarkBlockSize; ++index) {
    tokens.push_back(static_cast<int32_t>(first_token + index));
  }
  tokens.push_back(static_cast<int32_t>(first_token + tokens.size()));
  return tokens;
}

void RegisterPrompt(BlockPool& pool, PrefixCache& cache,
                    std::span<const int32_t> prompt) {
  std::shared_ptr<const BlockIdentity> parent;
  for (size_t offset = 0;
       offset < kBlocksPerPrefix * kBenchmarkBlockSize;
       offset += kBenchmarkBlockSize) {
    auto blocks = pool.AllocateBlocks(kBenchmarkBlockSize);
    if (blocks.size() != 1) {
      throw std::runtime_error(
          "Prefix benchmark could not allocate one block.");
    }
    const auto registration = cache.Register(
        blocks.front(),
        prompt.subspan(offset, kBenchmarkBlockSize),
        parent);
    if (registration.status !=
        PrefixCacheRegistrationStatus::Indexed) {
      throw std::runtime_error(
          "Prefix benchmark produced a duplicate or collision.");
    }
    parent = registration.identity;
    pool.Free(blocks);
  }
}

void PrintHeader() {
  std::cout << '\n'
            << std::left
            << std::setw(24) << "benchmark"
            << std::setw(18) << "scenario"
            << std::right
            << std::setw(8) << "batch"
            << std::setw(16) << "matched_blocks"
            << std::setw(14) << "mean_us"
            << std::setw(14) << "p50_us"
            << std::setw(14) << "p95_us"
            << '\n'
            << std::string(108, '-') << '\n';
}

TEST(PrefixCacheBenchmark, DISABLED_LargePopulatedCacheLookup) {
  BlockPool pool{kBenchmarkBlockSize, kCachedBlockCount};
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = kCachedBlockCount;
  PrefixCache cache{pool, options};

  std::vector<std::vector<int32_t>> cached_prompts;
  cached_prompts.reserve(kCachedPrefixCount);
  for (size_t prefix_index = 0;
       prefix_index < kCachedPrefixCount; ++prefix_index) {
    cached_prompts.push_back(BuildPrompt(prefix_index));
    RegisterPrompt(pool, cache, cached_prompts.back());
  }
  ASSERT_EQ(cache.IndexedBlocks(), kCachedBlockCount);

  const auto& full_hit = cached_prompts[kCachedPrefixCount / 2];
  auto partial_hit = full_hit;
  const size_t partial_blocks = kBlocksPerPrefix / 2;
  partial_hit[partial_blocks * kBenchmarkBlockSize] *= -1;
  const auto cold_miss = BuildPrompt(kCachedPrefixCount + 1);
  size_t observed_tokens = 0;

  PrintHeader();
  const auto run_scenario =
      [&](std::string_view name, std::span<const int32_t> prompt,
          size_t expected_blocks) {
        const auto latency = Measure(
            [&] {
              const auto match =
                  cache.Match(prompt, prompt.size() - 1);
              observed_tokens += match.token_count;
            },
            kOperationsPerSample);
        PrintResult("prefix-cache-match", name, 1,
                    expected_blocks, latency);
      };

  run_scenario("cold-miss", cold_miss, 0);
  run_scenario("partial-hit", partial_hit, partial_blocks);
  run_scenario("full-hit", full_hit, kBlocksPerPrefix);
  EXPECT_GT(observed_tokens, 0u);
}

class DecodePlanningBenchmark {
 public:
  explicit DecodePlanningBenchmark(size_t batch_size)
      : model_{LoadDummyDecoderModel()},
        assign_target_{
            MakeDoublesEngine(model_, 1024, EosToken(*model_)).engine},
        cache_{std::make_shared<RecordingCacheManager>(
            model_, batch_size)},
        scheduler_{model_, cache_} {
    model_->config_->engine.dynamic_batching =
        Config::Engine::DynamicBatching{};
    model_->config_->engine.dynamic_batching->max_batch_size =
        batch_size;
    model_->config_->engine.dynamic_batching->max_scheduled_tokens =
        batch_size;

    for (size_t index = 0; index < batch_size; ++index) {
      MakeDecodeResident(static_cast<int32_t>(index));
    }
  }

  size_t Plan() {
    StepPlan plan;
    const auto result = scheduler_.PlanStep(plan);
    if (!result.executable || plan.requests.empty()) {
      throw std::runtime_error(
          "Decode planning benchmark produced no work.");
    }
    return plan.token_count;
  }

 private:
  void MakeDecodeResident(int32_t seed) {
    const std::array<int32_t, 3> prompt{
        2 + seed % 8, 3 + seed % 8, 4 + seed % 8};
    auto request =
        CreateRequestWithPrompt(assign_target_, prompt);
    scheduler_.AddRequest(request);
    cache_->Allocate({request});
    request->Schedule();

    auto logits = model_->p_device_inputs_->Allocate<float>(
        static_cast<size_t>(model_->config_->model.vocab_size));
    auto cpu_logits = logits.CpuSpan();
    std::fill(cpu_logits.begin(), cpu_logits.end(), 0.0f);
    cpu_logits[5] = 100.0f;
    logits.CopyCpuToDevice();

    const auto before = request->Snapshot();
    RequestStepPlan plan;
    plan.request = request;
    plan.request_id = request.get();
    plan.sequence_length_before = before.current_sequence_length;
    plan.target_cache_slots =
        static_cast<size_t>(before.current_sequence_length);
    PrepareRequestStep(model_, plan);
    request->SaveStateForTransaction();
    const auto result =
        request->ApplyLogitsForTransaction(logits);
    request->CommitStateForTransaction();
    request->CommitStep(plan, result);
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<Engine> assign_target_;
  std::shared_ptr<RecordingCacheManager> cache_;
  DynamicBatchScheduler scheduler_;
};

TEST(SchedulerBenchmark,
     DISABLED_DecodePlanningContributionToInterTokenLatency) {
  PrintHeader();
  size_t planned_tokens = 0;
  for (const size_t batch_size : {1u, 8u, 32u}) {
    DecodePlanningBenchmark benchmark{batch_size};
    const auto latency = Measure([&] {
      planned_tokens += benchmark.Plan();
    });
    PrintResult("decode-step-planning", "steady-state",
                batch_size, 0, latency);
  }
  EXPECT_GT(planned_tokens, 0u);
}

TEST(PrefixCacheBenchmark, DISABLED_HybridCheckpointReclamationScaling) {
  constexpr size_t block_size = 256;
  auto model = LoadSyntheticHybridModel();
  PrintHeader();
  for (const size_t history_blocks : {128u, 512u, 2048u}) {
    for (const size_t histories : {1u, 8u, 32u}) {
      const size_t total_blocks = history_blocks * histories;
      if (total_blocks > 16384) {
        continue;
      }
      BlockPool blocks{block_size, total_blocks};
      PrefixCacheOptions options;
      options.enabled = true;
      options.max_blocks = total_blocks;
      options.requires_checkpoint = true;
      options.max_checkpoints = histories;
      FixedStatePool fixed{model, 1, histories};
      PrefixCache cache{blocks, options};
      std::vector<std::shared_ptr<const BlockIdentity>> endpoints;
      const char request_id{};
      const size_t token_count = history_blocks * block_size;
      for (size_t history = 0; history < histories; ++history) {
        const std::array<FixedStateReservationRequest, 1> requests{
            FixedStateReservationRequest{&request_id, token_count, 0}};
        {
          auto reservation = fixed.Reserve(requests);
          reservation.Commit();
        }
        std::vector<int32_t> tokens(token_count);
        std::iota(tokens.begin(), tokens.end(),
                  static_cast<int32_t>(history * token_count + 1));
        auto owned = blocks.AllocateBlocks(token_count);
        const auto registration = cache.RegisterCheckpointedPrefix(
            owned, tokens, {}, fixed.CapturePrefixCheckpoint(&request_id));
        ASSERT_EQ(registration.status, PrefixCacheRegistrationStatus::Indexed);
        endpoints.push_back(registration.identity);
        blocks.Free(owned);
        fixed.Release(fixed.HandleFor(&request_id));
      }
      ASSERT_EQ(cache.IndexedBlocks(), total_blocks);
      std::vector<size_t> lease_counts{0, (histories + 1) / 2};
      if (histories > 1) {
        lease_counts.push_back(histories);
      }
      for (const size_t leased_histories : lease_counts) {
        std::vector<std::shared_ptr<const FixedStatePrefixCheckpoint>> leases;
        for (size_t history = 0; history < leased_histories; ++history) {
          leases.push_back(cache.DraftBoundary(endpoints[history], token_count));
          ASSERT_NE(leases.back(), nullptr);
        }
        const size_t expected = (histories - leased_histories) * history_blocks;
        ASSERT_EQ(cache.ReclaimableBlocks(), expected);
        size_t observed = 0;
        const auto latency = Measure(
            [&] { observed += cache.ReclaimableBlocks(); }, 1, 5, 1);
        EXPECT_EQ(observed, expected * 6);
        const std::string scenario =
            std::to_string(history_blocks) + "b/" + std::to_string(leased_histories) + "leased";
        PrintResult("hybrid-reclaimable", scenario, histories, total_blocks, latency);
      }
      EXPECT_EQ(cache.ReclaimableBlocks(), total_blocks);
      EXPECT_EQ(cache.Reclaim(total_blocks), total_blocks);
    }
  }
}

}  // namespace
}  // namespace Generators::test
