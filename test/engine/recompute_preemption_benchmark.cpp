// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// A deterministic microbenchmark for recompute preemption on the continuous-batching dynamic path.
//
// It drives the production DynamicBatchScheduler, PagedCacheManager, PagedKeyValueCache, and
// Request lifecycle. Only model execution is replaced, so scheduling, admission, block accounting,
// reservation, commit, and preemption are the real code paths. Because there is no model run, the
// unit of time is one committed model step rather than a wall-clock interval: the numbers below
// describe scheduling behavior (how long a request waits, how evenly it is served, how much work is
// recomputed), not kernel or provider performance.
//
// What this measures:
//   * time to first token, in committed steps, per request;
//   * inter-token latency, in committed steps, per request;
//   * aggregate tokens per committed step, completions, and total steps;
//   * scheduler wait, preemption count, blocks reclaimed, tokens recomputed;
//   * victim slowdown, comparing a preempted request's end-to-end steps with the same request's
//     end-to-end steps in the wait-only baseline.
//
// What it does not measure: kernel time, memory bandwidth, provider behavior, or anything that
// depends on real attention. A real-model measurement is still required before claiming a
// throughput or latency result for a deployed configuration.
//
// The benchmark runs as an ordinary test so it stays compiled and correct, and prints its table to
// stdout. It asserts only the properties the policy is supposed to guarantee, never a timing value.

#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engine/engine.h"
#include "engine/scheduler.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"

namespace Generators {
namespace test {
namespace {

struct WorkloadRequest {
  size_t prompt_length{};
  int max_length{};
  size_t arrival_step{};  // Committed step at which the request is handed to the engine.
};

struct RequestOutcome {
  size_t prompt_length{};
  size_t arrival_step{};
  size_t first_token_step{};
  size_t completion_step{};
  size_t tokens{};
  size_t preemptions{};
  size_t recomputed_tokens{};
  std::vector<size_t> inter_token_gaps;

  size_t TimeToFirstToken() const { return first_token_step - arrival_step; }
  size_t EndToEnd() const { return completion_step - arrival_step; }
};

struct RunResult {
  std::vector<RequestOutcome> requests;
  size_t committed_steps{};       // Engine ticks; the unit of time for every latency below.
  size_t model_runs{};            // Ticks that actually ran the model rather than draining results.
  size_t generated_tokens{};
  size_t completions{};
  size_t scheduler_wait_steps{};  // Request-ticks spent waiting instead of resident.
  SchedulerPreemptionMetrics preemption;
  size_t capacity_deferred_steps{};
};

double Percentile(std::vector<size_t> values, double percentile) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  const double rank = percentile * static_cast<double>(values.size() - 1);
  const size_t lower = static_cast<size_t>(rank);
  const size_t upper = std::min(lower + 1, values.size() - 1);
  const double fraction = rank - static_cast<double>(lower);
  return static_cast<double>(values[lower]) +
         fraction * static_cast<double>(values[upper] - values[lower]);
}

std::vector<size_t> Collect(const std::vector<RequestOutcome>& outcomes,
                            size_t (RequestOutcome::*member)() const) {
  std::vector<size_t> values;
  values.reserve(outcomes.size());
  for (const auto& outcome : outcomes)
    values.push_back((outcome.*member)());
  return values;
}

std::vector<size_t> CollectGaps(const std::vector<RequestOutcome>& outcomes) {
  std::vector<size_t> values;
  for (const auto& outcome : outcomes)
    values.insert(values.end(), outcome.inter_token_gaps.begin(),
                  outcome.inter_token_gaps.end());
  return values;
}

// A mixed workload: two long resident requests arrive first and occupy the key-value budget, then
// short requests arrive behind them. Without preemption the short requests wait for the long ones
// to finish; with preemption they are admitted by reclaiming a long request's blocks.
std::vector<WorkloadRequest> MixedWorkload() {
  std::vector<WorkloadRequest> workload;
  workload.push_back(WorkloadRequest{12, 56, 0});
  workload.push_back(WorkloadRequest{12, 56, 0});
  for (size_t i = 0; i < 6; ++i)
    workload.push_back(WorkloadRequest{4, 12, 3 + i * 2});
  return workload;
}

std::vector<int32_t> Prompt(size_t length, int32_t seed, int32_t vocab_size) {
  std::vector<int32_t> prompt;
  prompt.reserve(length);
  for (size_t i = 0; i < length; ++i)
    prompt.push_back(2 + (seed + static_cast<int32_t>(i)) % (vocab_size - 3));
  return prompt;
}

RunResult RunWorkload(const std::vector<WorkloadRequest>& workload,
                      size_t num_blocks, size_t block_size,
                      bool enable_preemption, size_t min_decode_steps) {
  auto model = LoadDummyDecoderModel();
  Config::Engine::DynamicBatching dynamic_batching;
  dynamic_batching.block_size = block_size;
  dynamic_batching.num_blocks = num_blocks;
  dynamic_batching.max_batch_size = 8;
  dynamic_batching.max_scheduled_tokens = 256;
  dynamic_batching.enable_recompute_preemption = enable_preemption;
  dynamic_batching.max_preemptions_per_step = 1;
  dynamic_batching.min_decode_steps_before_preemption = min_decode_steps;
  model->config_->engine.dynamic_batching = dynamic_batching;

  auto paged = MakePagedEngine(model, /*forced_token=*/5);
  const auto vocab_size = static_cast<int32_t>(model->config_->model.vocab_size);

  std::vector<std::shared_ptr<Request>> requests;
  requests.reserve(workload.size());
  for (size_t i = 0; i < workload.size(); ++i) {
    auto params = MakeGreedyParams(*model);
    params->search.max_length = workload[i].max_length;
    auto request = std::make_shared<Request>(params);
    const auto prompt = Prompt(workload[i].prompt_length,
                               static_cast<int32_t>(i * 5), vocab_size);
    request->AddTokens(prompt);
    requests.push_back(std::move(request));
  }

  RunResult result;
  result.requests.resize(workload.size());
  std::vector<size_t> last_token_step(workload.size(), 0);
  std::vector<bool> added(workload.size(), false);
  std::vector<bool> completed(workload.size(), false);
  for (size_t i = 0; i < workload.size(); ++i) {
    result.requests[i].prompt_length = workload[i].prompt_length;
    result.requests[i].arrival_step = workload[i].arrival_step;
  }

  // One iteration is one engine tick, which is one Engine::Step() call. Arrival times, time to
  // first token, and inter-token latency are all expressed in those ticks.
  size_t step = 0;
  constexpr size_t kStepBudget = 20000;
  while (step < kStepBudget) {
    for (size_t i = 0; i < workload.size(); ++i) {
      if (!added[i] && workload[i].arrival_step <= step) {
        paged.engine->AddRequest(requests[i]);
        added[i] = true;
        last_token_step[i] = step;
      }
    }
    const bool all_arrived =
        std::all_of(added.begin(), added.end(), [](bool value) { return value; });
    if (all_arrived && !paged.engine->HasPendingRequests())
      break;

    if (paged.engine->HasPendingRequests()) {
      try {
        paged.engine->Step();
      } catch (const EngineStepError& error) {
        // Capacity exhaustion is a legitimate outcome for the wait-only policy under this budget;
        // the tick still passes, and later arrivals or completions can change the picture.
        if (error.Outcome().kind != StepOutcomeKind::CapacityDeferred)
          throw;
        ++result.capacity_deferred_steps;
      }
    }
    ++step;
    ++result.committed_steps;

    for (size_t i = 0; i < workload.size(); ++i) {
      if (!added[i] || completed[i])
        continue;
      auto& outcome = result.requests[i];
      while (requests[i]->HasUnseenTokens()) {
        requests[i]->UnseenToken();
        ++outcome.tokens;
        if (outcome.tokens == 1) {
          outcome.first_token_step = step;
        } else {
          outcome.inter_token_gaps.push_back(step - last_token_step[i]);
        }
        last_token_step[i] = step;
      }
      if (requests[i]->status_ == RequestStatus::Completed) {
        completed[i] = true;
        outcome.completion_step = step;
        ++result.completions;
      } else if (requests[i]->status_ == RequestStatus::Assigned ||
                 requests[i]->status_ == RequestStatus::Suspended) {
        ++result.scheduler_wait_steps;
      }
    }
  }

  result.model_runs = static_cast<size_t>(paged.executor->decode_calls);
  for (size_t i = 0; i < workload.size(); ++i) {
    result.requests[i].preemptions = requests[i]->PreemptionCount();
    result.requests[i].recomputed_tokens = requests[i]->RecomputedTokenCount();
    result.generated_tokens += result.requests[i].tokens;
    if (!completed[i])
      result.requests[i].completion_step = step;
  }
  result.preemption = paged.scheduler->PreemptionMetrics();
  return result;
}

void PrintRun(const char* label, const RunResult& run) {
  const auto ttft = Collect(run.requests, &RequestOutcome::TimeToFirstToken);
  const auto gaps = CollectGaps(run.requests);
  std::printf(
      "%-18s ticks=%5zu model_runs=%5zu tokens=%5zu completions=%2zu "
      "ttft_p50=%6.1f ttft_p95=%6.1f itl_p50=%5.2f itl_p95=%5.2f "
      "tok/run=%5.2f wait=%5zu deferred=%3zu preempt=%3llu recomputed=%5llu reclaimed=%4llu\n",
      label, run.committed_steps, run.model_runs, run.generated_tokens, run.completions,
      Percentile(ttft, 0.50), Percentile(ttft, 0.95), Percentile(gaps, 0.50),
      Percentile(gaps, 0.95),
      run.model_runs == 0
          ? 0.0
          : static_cast<double>(run.generated_tokens) /
                static_cast<double>(run.model_runs),
      run.scheduler_wait_steps, run.capacity_deferred_steps,
      static_cast<unsigned long long>(run.preemption.preemptions),
      static_cast<unsigned long long>(run.preemption.recomputed_tokens),
      static_cast<unsigned long long>(run.preemption.reclaimed_blocks));
}

// Short requests are the ones that arrive behind the long resident requests, so their admission
// latency is what preemption is supposed to improve.
std::vector<RequestOutcome> ShortRequests(const RunResult& run) {
  std::vector<RequestOutcome> shorts;
  for (const auto& outcome : run.requests) {
    if (outcome.arrival_step > 0)
      shorts.push_back(outcome);
  }
  return shorts;
}

TEST(RecomputePreemptionBenchmark, WaitOnlyVersusRecomputePreemptionUnderAFixedKeyValueBudget) {
  const auto workload = MixedWorkload();
  // A budget that holds the two long requests and nothing else, so the short requests behind them
  // are admitted only when capacity is reclaimed.
  constexpr size_t kNumBlocks = 2;
  constexpr size_t kBlockSize = 64;

  const auto baseline =
      RunWorkload(workload, kNumBlocks, kBlockSize, /*enable_preemption=*/false,
                  /*min_decode_steps=*/1);
  const auto eager =
      RunWorkload(workload, kNumBlocks, kBlockSize, /*enable_preemption=*/true,
                  /*min_decode_steps=*/1);
  const auto shipped_default =
      RunWorkload(workload, kNumBlocks, kBlockSize, /*enable_preemption=*/true,
                  /*min_decode_steps=*/8);

  std::printf(
      "\n[recompute preemption] unit of time is one engine tick; no model is executed.\n");
  PrintRun("wait-only", baseline);
  PrintRun("preempt q=1", eager);
  PrintRun("preempt q=8 (dflt)", shipped_default);

  const auto baseline_short = ShortRequests(baseline);
  const auto eager_short = ShortRequests(eager);
  const auto default_short = ShortRequests(shipped_default);
  std::printf(
      "short-request admission: wait-only ttft_p50=%.1f p95=%.1f | q=1 "
      "ttft_p50=%.1f p95=%.1f | q=8 ttft_p50=%.1f p95=%.1f\n",
      Percentile(Collect(baseline_short, &RequestOutcome::TimeToFirstToken), 0.50),
      Percentile(Collect(baseline_short, &RequestOutcome::TimeToFirstToken), 0.95),
      Percentile(Collect(eager_short, &RequestOutcome::TimeToFirstToken), 0.50),
      Percentile(Collect(eager_short, &RequestOutcome::TimeToFirstToken), 0.95),
      Percentile(Collect(default_short, &RequestOutcome::TimeToFirstToken), 0.50),
      Percentile(Collect(default_short, &RequestOutcome::TimeToFirstToken), 0.95));

  // Victim slowdown: the long requests pay for the short requests' admission.
  for (size_t i = 0; i < baseline.requests.size(); ++i) {
    if (shipped_default.requests[i].preemptions == 0)
      continue;
    const double slowdown =
        baseline.requests[i].EndToEnd() == 0
            ? 0.0
            : static_cast<double>(shipped_default.requests[i].EndToEnd()) /
                  static_cast<double>(baseline.requests[i].EndToEnd());
    std::printf(
        "victim %zu: preemptions=%zu recomputed_tokens=%zu end_to_end %zu -> %zu (slowdown %.2fx)\n",
        i, shipped_default.requests[i].preemptions,
        shipped_default.requests[i].recomputed_tokens, baseline.requests[i].EndToEnd(),
        shipped_default.requests[i].EndToEnd(), slowdown);
  }

  // Every request must still finish, whichever policy ran.
  EXPECT_EQ(baseline.completions, workload.size());
  EXPECT_EQ(eager.completions, workload.size());
  EXPECT_EQ(shipped_default.completions, workload.size());
  // Preemption has to actually engage for the comparison to mean anything.
  EXPECT_GT(eager.preemption.preemptions, 0u);
  EXPECT_GT(shipped_default.preemption.preemptions, 0u);
  EXPECT_EQ(baseline.preemption.preemptions, 0u);
  // Recomputation is bounded by what was reclaimed, never invented.
  EXPECT_EQ(eager.preemption.recomputed_tokens > 0,
            eager.preemption.preemptions > 0);
  // The service quantum is what bounds churn: the default must recompute less than the eager one.
  EXPECT_LT(shipped_default.preemption.recomputed_tokens,
            eager.preemption.recomputed_tokens);
  // Preemption exists to admit the requests waiting behind the long ones sooner.
  EXPECT_LT(Percentile(Collect(default_short, &RequestOutcome::TimeToFirstToken), 0.50),
            Percentile(Collect(baseline_short, &RequestOutcome::TimeToFirstToken), 0.50));
  EXPECT_LE(Percentile(Collect(default_short, &RequestOutcome::TimeToFirstToken), 0.95),
            Percentile(Collect(baseline_short, &RequestOutcome::TimeToFirstToken), 0.95));
  // Aggregate scheduler wait must not get worse for the workload as a whole.
  EXPECT_LE(shipped_default.scheduler_wait_steps, baseline.scheduler_wait_steps);
}

}  // namespace
}  // namespace test
}  // namespace Generators
