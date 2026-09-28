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
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "engine/prefix_cache.h"
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
                       size_t operations_per_sample = 1) {
  for (size_t sample = 0; sample < kWarmupSamples; ++sample) {
    for (size_t operation_index = 0;
         operation_index < operations_per_sample; ++operation_index) {
      operation();
    }
  }

  std::vector<double> samples;
  samples.reserve(kMeasuredSamples);
  for (size_t sample = 0; sample < kMeasuredSamples; ++sample) {
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

}  // namespace
}  // namespace Generators::test
