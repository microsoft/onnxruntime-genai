// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_doubles.h"
#include "engine_test_helpers.h"

namespace Generators::test {
namespace {

using Clock = std::chrono::steady_clock;

struct ServingResult {
  std::vector<double> ttft_steps;
  std::vector<double> itl_steps;
  std::vector<double> ttft_ms;
  std::vector<double> itl_ms;
  std::array<size_t, 8> completion_steps{};
  size_t model_steps{};
  size_t query_tokens{};
  size_t generated_tokens{};
  size_t completions{};
  size_t scheduler_wait_steps{};
  uint64_t preemptions{};
  uint64_t recomputed_tokens{};
  double elapsed_ms{};
};

double Percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    throw std::runtime_error("Serving benchmark has no latency samples.");
  }
  std::sort(values.begin(), values.end());
  const double position = fraction * static_cast<double>(values.size() - 1);
  const size_t lower = static_cast<size_t>(position);
  const size_t upper = std::min(lower + 1, values.size() - 1);
  return values[lower] + (values[upper] - values[lower]) * (position - static_cast<double>(lower));
}

ServingResult RunServingWorkload(bool enabled, size_t quantum) {
  auto model = LoadDummyDecoderModelWithChunking(8);
  Config::Engine::DynamicBatching batching;
  batching.block_size = 64;
  batching.num_blocks = 2;
  batching.max_batch_size = 4;
  batching.max_scheduled_tokens = 16;
  batching.enable_recompute_preemption = enabled;
  batching.min_decode_steps_before_preemption = quantum;
  model->config_->engine.dynamic_batching = batching;
  auto test = MakeCompositeDoublesEngine(model, 5);
  std::array<std::shared_ptr<Request>, 8> requests;
  std::array<bool, 8> arrived{};
  std::array<bool, 8> completed{};
  std::array<size_t, 8> arrival_steps{};
  std::array<size_t, 8> last_token_steps{};
  std::array<size_t, 8> token_counts{};
  std::array<double, 8> arrival_ms{};
  std::array<double, 8> last_token_ms{};
  for (auto& request : requests) {
    request = test.engine->CreateRequest();
  }
  for (size_t index = 2; index < requests.size(); ++index) {
    arrival_steps[index] = 4 + (index - 2) * 3;
  }
  ServingResult result;
  const auto start = Clock::now();
  const auto elapsed = [&] {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  };
  std::array<EngineEvent, 8> events;
  for (size_t calls = 0; result.completions < requests.size() && calls < 10000; ++calls) {
    const size_t step = static_cast<size_t>(test.executor->decode_calls);
    for (size_t index = 0; index < requests.size(); ++index) {
      if (!arrived[index] && step >= arrival_steps[index]) {
        const bool long_request = index < 2;
        const std::vector<int32_t> prompt(long_request ? 12 : 4, 2 + static_cast<int32_t>(index));
        arrival_ms[index] = elapsed();
        requests[index]->BeginTurn(prompt, long_request ? 44 : 8);
        arrived[index] = true;
      }
    }
    const size_t count = test.engine->Run(events);
    const size_t committed_step = static_cast<size_t>(test.executor->decode_calls);
    if (committed_step != step) {
      for (size_t index = 0; index < requests.size(); ++index) {
        if (arrived[index] && !completed[index] &&
            !test.cache->IsResident(requests[index])) {
          ++result.scheduler_wait_steps;
        }
      }
    }
    for (const auto& event : std::span{events}.first(count)) {
      if ((event.flags & (EngineEventFlagFailed | EngineEventFlagCapacityBlocked |
                          EngineEventFlagRetryable)) != 0) {
        throw std::runtime_error("Serving benchmark encountered an unsuccessful Engine event.");
      }
      const auto request = std::find(requests.begin(), requests.end(), event.request);
      if (request == requests.end()) {
        throw std::runtime_error("Serving benchmark received an unknown Request.");
      }
      const size_t index = static_cast<size_t>(request - requests.begin());
      const double now = elapsed();
      if ((event.flags & EngineEventFlagToken) != 0) {
        ++result.generated_tokens;
        if (token_counts[index]++ == 0) {
          // TTFT population: the six short arrivals behind the two long residents.
          if (index >= 2) {
            result.ttft_steps.push_back(static_cast<double>(committed_step - arrival_steps[index]));
            result.ttft_ms.push_back(now - arrival_ms[index]);
          }
        } else {
          result.itl_steps.push_back(static_cast<double>(committed_step - last_token_steps[index]));
          result.itl_ms.push_back(now - last_token_ms[index]);
        }
        last_token_steps[index] = committed_step;
        last_token_ms[index] = now;
      }
      if ((event.flags & EngineEventFlagTurnFinished) != 0) {
        if (completed[index]) {
          throw std::runtime_error("Serving benchmark received duplicate completion.");
        }
        completed[index] = true;
        result.completion_steps[index] = committed_step;
        ++result.completions;
        result.preemptions += requests[index]->PreemptionCount();
        result.recomputed_tokens += requests[index]->RecomputedTokenCount();
        requests[index]->Close();
      }
    }
  }
  result.elapsed_ms = elapsed();
  result.model_steps = static_cast<size_t>(test.executor->decode_calls);
  result.query_tokens = std::accumulate(
      test.executor->decoded_token_counts.begin(), test.executor->decoded_token_counts.end(), size_t{0});
  if (result.completions != requests.size() || result.generated_tokens != 136) {
    throw std::runtime_error("Serving benchmark did not complete the fixed workload.");
  }
  return result;
}

void PrintServingResult(const char* policy, const ServingResult& result, const ServingResult& baseline) {
  std::cout << std::fixed << std::setprecision(3)
            << policy
            << " steps=" << result.model_steps
            << " query_tokens=" << result.query_tokens
            << " completions=" << result.completions
            << " generated=" << result.generated_tokens
            << " short_ttft_steps_p50=" << Percentile(result.ttft_steps, 0.50)
            << " short_ttft_steps_p95=" << Percentile(result.ttft_steps, 0.95)
            << " itl_steps_p50=" << Percentile(result.itl_steps, 0.50)
            << " itl_steps_p95=" << Percentile(result.itl_steps, 0.95)
            << " short_ttft_ms_p50=" << Percentile(result.ttft_ms, 0.50)
            << " short_ttft_ms_p95=" << Percentile(result.ttft_ms, 0.95)
            << " itl_ms_p50=" << Percentile(result.itl_ms, 0.50)
            << " itl_ms_p95=" << Percentile(result.itl_ms, 0.95)
            << " host_tokens_per_second=" << 1000.0 * static_cast<double>(result.generated_tokens) / result.elapsed_ms
            << " wait_request_steps=" << result.scheduler_wait_steps
            << " preemptions=" << result.preemptions
            << " recomputed_tokens=" << result.recomputed_tokens;
  for (size_t index = 0; index < 2; ++index) {
    std::cout << " long_" << index << "_slowdown="
              << static_cast<double>(result.completion_steps[index]) /
                     static_cast<double>(baseline.completion_steps[index]);
  }
  std::cout << '\n';
}

TEST(RecomputePreemptionBenchmark, DISABLED_MixedServingUnderFixedKVPressure) {
  // Production scheduler, cache, Search and event API; scripted logits, not LLM execution.
  // Step latencies count model invocations, never API calls used only to drain ready events.
  RunServingWorkload(false, 8);
  RunServingWorkload(true, 8);
  std::cout << "\nCPU host-policy microbenchmark; two 64-slot KV blocks, two long and six short Requests."
            << "\nWall-clock timings and tokens/s exclude model kernels; they are not GPU serving measurements.\n";
  for (size_t repetition = 0; repetition < 5; ++repetition) {
    const auto baseline = RunServingWorkload(false, 8);
    const auto eager = RunServingWorkload(true, 1);
    const auto bounded = RunServingWorkload(true, 8);
    PrintServingResult("wait-only", baseline, baseline);
    PrintServingResult("preempt-q1", eager, baseline);
    PrintServingResult("preempt-q8", bounded, baseline);
    EXPECT_EQ(baseline.preemptions, 0u);
    EXPECT_GT(bounded.preemptions, 0u);
    EXPECT_LE(bounded.recomputed_tokens, eager.recomputed_tokens);
  }
}

}  // namespace
}  // namespace Generators::test
