// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "engine/execution_context.h"
#include "engine/step_transaction.h"

namespace Generators {
namespace {

constexpr size_t kWarmupIterations = 1000;
constexpr size_t kMeasuredIterations = 10000;
constexpr size_t kRepetitions = 5;

struct OrchestrationScratch {
  explicit OrchestrationScratch(size_t batch_size) {
    plan.requests.resize(batch_size);
    results.reserve(batch_size);
    staged_ready.reserve(batch_size);
    published_ready.reserve(batch_size);
  }

  StepPlan plan;
  std::vector<uint8_t> results;
  std::vector<const void*> staged_ready;
  std::vector<const void*> published_ready;
  uintptr_t sink{};
};

struct NoopCacheReservation {
  void Commit() { committed = true; }
  bool committed{};
};

struct NoopExecutor {
  void Execute(ExecutionContext& context) {
    context.execution_started = true;
    context.execution_completed = true;
  }
};

double RunOrchestration(OrchestrationScratch& scratch,
                        size_t iterations,
                        bool transactional) {
  const auto start = std::chrono::steady_clock::now();
  for (size_t iteration = 0; iteration < iterations; ++iteration) {
    scratch.results.clear();
    scratch.staged_ready.clear();

    if (transactional) {
      NoopCacheReservation reservation;
      NoopExecutor executor;
      ExecutionContext context{iteration + 1, &scratch.plan};
      StepTransaction transaction{scratch.plan};
      transaction.MarkReserved();
      transaction.MarkExecuting();
      executor.Execute(context);
      transaction.MarkExecuted();
      for (const auto& entry : scratch.plan.requests) {
        scratch.results.push_back(1);
        scratch.staged_ready.push_back(entry.request_id);
      }
      reservation.Commit();
      scratch.published_ready.swap(scratch.staged_ready);
      transaction.Commit();
      scratch.sink += reservation.committed +
                      context.execution_completed;
    } else {
      for (const auto& entry : scratch.plan.requests) {
        scratch.results.push_back(1);
        scratch.staged_ready.push_back(entry.request_id);
      }
      scratch.published_ready.swap(scratch.staged_ready);
    }

    scratch.sink += scratch.results.size() +
                    scratch.published_ready.size();
    scratch.published_ready.clear();
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  return std::chrono::duration<double, std::micro>(elapsed).count() /
         static_cast<double>(iterations);
}

double MedianAddedOverhead(size_t batch_size) {
  OrchestrationScratch scratch{batch_size};
  const auto result_capacity = scratch.results.capacity();
  const auto staged_capacity = scratch.staged_ready.capacity();
  const auto published_capacity = scratch.published_ready.capacity();

  RunOrchestration(scratch, kWarmupIterations, false);
  RunOrchestration(scratch, kWarmupIterations, true);

  std::array<double, kRepetitions> added_overhead;
  for (size_t repetition = 0; repetition < kRepetitions; ++repetition) {
    const double baseline =
        RunOrchestration(scratch, kMeasuredIterations, false);
    const double transactional =
        RunOrchestration(scratch, kMeasuredIterations, true);
    added_overhead[repetition] =
        std::max(0.0, transactional - baseline);
  }
  std::sort(added_overhead.begin(), added_overhead.end());

  EXPECT_EQ(scratch.results.capacity(), result_capacity);
  EXPECT_EQ(scratch.staged_ready.capacity(), staged_capacity);
  EXPECT_EQ(scratch.published_ready.capacity(), published_capacity);
  EXPECT_NE(scratch.sink, 0u);
  return added_overhead[kRepetitions / 2];
}

TEST(TransactionOverheadTest, AddedOrchestrationStaysWithinBudget) {
  const double batch_1 = MedianAddedOverhead(1);
  const double batch_4 = MedianAddedOverhead(4);
  const double batch_8 = MedianAddedOverhead(8);

  std::cout << "Transaction added median orchestration (us): batch1="
            << batch_1 << ", batch4=" << batch_4
            << ", batch8=" << batch_8 << '\n';
  EXPECT_LT(batch_8, 25.0);
}

}  // namespace
}  // namespace Generators
