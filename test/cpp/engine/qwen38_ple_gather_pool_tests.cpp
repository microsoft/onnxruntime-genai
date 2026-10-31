// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// GPU-free tests of the worker pool behind the host-side PLE gather of libqwen38_plane
// (src/qwen38_plane/qwen38_parallel_pool.h): joinable workers, Stop() that joins them, a start that fails cleanly.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "process_thread_count.h"
#include "qwen38_plane/qwen38_parallel_pool.h"

namespace Generators::test {

namespace {

using qwen38_plane::ParallelPool;

const std::error_code kNoError{};
const std::error_code kThreadLimit = std::make_error_code(std::errc::resource_unavailable_try_again);

// Counts the threads of a pool: the ones started, and the ones whose entry function has not returned yet. A thread decrements
// `running` as the last thing it does, so once the pool has joined it `running` no longer counts it. Optionally fails the
// creation of one thread, as std::thread does when the system is out of threads.
struct ThreadLedger {
  enum class Failure { kSystemError,
                       kBadAlloc,
                       kOther };

  ParallelPool::ThreadStarter Starter() {
    return [this](std::function<void()> entry) {
      if (attempts++ == fail_at) {
        switch (failure) {
          case Failure::kSystemError:
            throw std::system_error(kThreadLimit);
          case Failure::kBadAlloc:
            throw std::bad_alloc();
          default:
            throw std::runtime_error("cannot start a thread");
        }
      }
      ++started;
      ++running;
      return std::thread([this, entry = std::move(entry)] {
        entry();
        --running;
      });
    };
  }

  int attempts = 0;  // creations tried; the pool creates its threads one at a time from the thread that calls Run
  int fail_at = -1;  // the 0-based creation that fails, -1 for none
  Failure failure = Failure::kSystemError;
  std::atomic<int> started{0};
  std::atomic<int> running{0};
};

// Polls `condition` for up to 30 seconds.
bool WaitUntil(const std::function<bool()>& condition) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!condition()) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

}  // namespace

TEST(Qwen38ParallelPoolTest, DefaultWorkerCountIsOneBelowTheProcessorsBetweenThreeAndTwentyThree) {
  using qwen38_plane::DefaultParallelWorkerCount;
  // An unknown count (hardware_concurrency() returns 0) and a single processor must not wrap around to billions of threads.
  EXPECT_EQ(DefaultParallelWorkerCount(0), 3);
  EXPECT_EQ(DefaultParallelWorkerCount(1), 3);
  EXPECT_EQ(DefaultParallelWorkerCount(2), 3);
  EXPECT_EQ(DefaultParallelWorkerCount(4), 3);
  EXPECT_EQ(DefaultParallelWorkerCount(5), 4);
  EXPECT_EQ(DefaultParallelWorkerCount(16), 15);
  EXPECT_EQ(DefaultParallelWorkerCount(24), 23);
  EXPECT_EQ(DefaultParallelWorkerCount(25), 23);
  EXPECT_EQ(DefaultParallelWorkerCount(512), 23);
}

TEST(Qwen38ParallelPoolTest, RunsTheBodyOnEveryWorkerAndOnTheCaller) {
  constexpr int kWorkers = 4;
  ParallelPool pool{kWorkers};
  EXPECT_EQ(pool.ThreadCount(), 0) << "the workers start with the first Run";
  for (int run = 0; run < 3; ++run) {
    std::mutex mutex;
    std::set<std::thread::id> threads;
    std::atomic<int> calls{0};
    EXPECT_EQ(pool.Run([&] {
      ++calls;
      std::lock_guard<std::mutex> lock(mutex);
      threads.insert(std::this_thread::get_id());
    }),
              kNoError);
    EXPECT_EQ(calls.load(), kWorkers + 1);
    EXPECT_EQ(threads.size(), static_cast<size_t>(kWorkers + 1));
    EXPECT_EQ(threads.count(std::this_thread::get_id()), 1u);
    EXPECT_EQ(pool.ThreadCount(), kWorkers);
  }
}

TEST(Qwen38ParallelPoolTest, ARunFinishesTheWorkOfEveryParticipantBeforeItReturns) {
  constexpr int kRows = 50000;
  constexpr int kSpan = 32;
  ParallelPool pool{3};
  for (int run = 0; run < 5; ++run) {
    std::vector<std::atomic<int>> visits(kRows);
    for (auto& visit : visits) visit.store(0);
    std::atomic<int> next{0};
    ASSERT_EQ(pool.Run([&] {
      for (;;) {
        const int begin = next.fetch_add(kSpan);
        if (begin >= kRows) break;
        for (int row = begin; row < std::min(begin + kSpan, kRows); ++row) ++visits[static_cast<size_t>(row)];
      }
    }),
              kNoError);
    // Every row was claimed exactly once, and all of it is done now: nothing is left running in a worker.
    for (int row = 0; row < kRows; ++row) {
      ASSERT_EQ(visits[static_cast<size_t>(row)].load(), 1) << "row " << row << " of run " << run;
    }
  }
}

TEST(Qwen38ParallelPoolTest, StopJoinsEveryWorkerIsIdempotentAndARunRestartsThem) {
  constexpr int kWorkers = 4;
  const int baseline = ProcessThreadCount();
  ThreadLedger ledger;
  ParallelPool pool{kWorkers, ledger.Starter()};

  pool.Stop();  // before the first Run there is nothing to stop
  EXPECT_EQ(pool.ThreadCount(), 0);

  std::atomic<int> calls{0};
  ASSERT_EQ(pool.Run([&] { ++calls; }), kNoError);
  EXPECT_EQ(pool.ThreadCount(), kWorkers);
  EXPECT_EQ(ledger.running.load(), kWorkers);

  pool.Stop();
  EXPECT_EQ(pool.ThreadCount(), 0);
  EXPECT_EQ(ledger.running.load(), 0) << "Stop returned before a worker had returned";
  EXPECT_TRUE(ProcessThreadCountSettlesAtOrBelow(baseline)) << "a worker thread is still alive";

  pool.Stop();
  pool.Stop();
  EXPECT_EQ(pool.ThreadCount(), 0);

  ASSERT_EQ(pool.Run([&] { ++calls; }), kNoError);
  EXPECT_EQ(calls.load(), 2 * (kWorkers + 1));
  EXPECT_EQ(pool.ThreadCount(), kWorkers);
  EXPECT_EQ(ledger.started.load(), 2 * kWorkers);
  pool.Stop();
  EXPECT_EQ(ledger.running.load(), 0);
  EXPECT_TRUE(ProcessThreadCountSettlesAtOrBelow(baseline));
}

TEST(Qwen38ParallelPoolTest, TheDestructorJoinsTheWorkers) {
  const int baseline = ProcessThreadCount();
  ThreadLedger ledger;
  {
    ParallelPool pool{5, ledger.Starter()};
    ASSERT_EQ(pool.Run([] {}), kNoError);
    EXPECT_EQ(ledger.running.load(), 5);
  }
  EXPECT_EQ(ledger.running.load(), 0);
  EXPECT_TRUE(ProcessThreadCountSettlesAtOrBelow(baseline));
}

TEST(Qwen38ParallelPoolTest, AFailedThreadStartJoinsTheWorkersThatStartedAndReportsTheError) {
  constexpr int kWorkers = 5;
  // The creation of the first, of a middle and of the last worker fails.
  for (const int fail_at : {0, 2, kWorkers - 1}) {
    SCOPED_TRACE(fail_at);
    const int baseline = ProcessThreadCount();
    ThreadLedger ledger;
    ledger.fail_at = fail_at;
    ParallelPool pool{kWorkers, ledger.Starter()};

    std::atomic<int> calls{0};
    EXPECT_EQ(pool.Run([&] { ++calls; }), kThreadLimit);
    EXPECT_EQ(calls.load(), 0) << "the body ran although the pool could not start";
    EXPECT_EQ(ledger.started.load(), fail_at);
    EXPECT_EQ(ledger.running.load(), 0) << "a worker that had started was not joined";
    EXPECT_EQ(pool.ThreadCount(), 0);
    EXPECT_TRUE(ProcessThreadCountSettlesAtOrBelow(baseline));

    // Nothing is left over from the failed start: the next Run starts all the workers.
    EXPECT_EQ(pool.Run([&] { ++calls; }), kNoError);
    EXPECT_EQ(calls.load(), kWorkers + 1);
    EXPECT_EQ(pool.ThreadCount(), kWorkers);
    EXPECT_EQ(ledger.started.load(), fail_at + kWorkers);
    pool.Stop();
    EXPECT_EQ(ledger.running.load(), 0);
  }
}

TEST(Qwen38ParallelPoolTest, AStartThatRunsOutOfMemoryReportsItAndJoinsTheStartedWorkers) {
  ThreadLedger ledger;
  ledger.fail_at = 1;
  ledger.failure = ThreadLedger::Failure::kBadAlloc;
  ParallelPool pool{4, ledger.Starter()};
  EXPECT_EQ(pool.Run([] {}), std::make_error_code(std::errc::not_enough_memory));
  EXPECT_EQ(ledger.started.load(), 1);
  EXPECT_EQ(ledger.running.load(), 0);
  EXPECT_EQ(pool.ThreadCount(), 0);
}

TEST(Qwen38ParallelPoolTest, AStartThatThrowsSomethingElseStillJoinsTheStartedWorkersBeforeItPropagates) {
  ThreadLedger ledger;
  ledger.fail_at = 2;
  ledger.failure = ThreadLedger::Failure::kOther;
  ParallelPool pool{4, ledger.Starter()};
  EXPECT_THROW(pool.Run([] {}), std::runtime_error);
  EXPECT_EQ(ledger.started.load(), 2);
  EXPECT_EQ(ledger.running.load(), 0);
  EXPECT_EQ(pool.ThreadCount(), 0);
  EXPECT_EQ(pool.Run([] {}), kNoError);  // the failure was transient and the pool is usable
  EXPECT_EQ(pool.ThreadCount(), 4);
}

TEST(Qwen38ParallelPoolTest, StopWaitsForARunInFlightSoNoWorkIsLost) {
  constexpr int kWorkers = 3;
  ParallelPool pool{kWorkers};
  std::atomic<int> entered{0};
  std::atomic<int> finished{0};
  std::promise<void> gate;
  const std::shared_future<void> open = gate.get_future().share();

  std::error_code run_error = std::make_error_code(std::errc::interrupted);
  std::thread runner{[&] {
    run_error = pool.Run([&] {
      ++entered;
      open.wait();
      ++finished;
    });
  }};
  ASSERT_TRUE(WaitUntil([&] { return entered.load() == kWorkers + 1; })) << "not every participant entered the body";

  std::future<void> stopped = std::async(std::launch::async, [&] { pool.Stop(); });
  EXPECT_EQ(stopped.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout)
      << "Stop returned while a Run was still in flight";
  EXPECT_EQ(finished.load(), 0);

  gate.set_value();
  runner.join();
  stopped.get();
  EXPECT_EQ(run_error, kNoError);
  EXPECT_EQ(finished.load(), kWorkers + 1) << "a copy of the body was lost";
  EXPECT_EQ(pool.ThreadCount(), 0);
}

TEST(Qwen38ParallelPoolTest, RunsDoNotOverlap) {
  constexpr int kWorkers = 3;
  constexpr int kCallers = 4;
  constexpr int kRunsPerCaller = 15;
  ParallelPool pool{kWorkers};
  std::atomic<int> active{0};
  std::atomic<int> most_active{0};
  std::atomic<int> failures{0};
  const auto body = [&] {
    const int now = ++active;
    int seen = most_active.load();
    while (seen < now && !most_active.compare_exchange_weak(seen, now)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    --active;
  };
  std::vector<std::thread> callers;
  for (int c = 0; c < kCallers; ++c) {
    callers.emplace_back([&] {
      for (int i = 0; i < kRunsPerCaller; ++i) {
        if (pool.Run(body)) ++failures;
      }
    });
  }
  for (std::thread& caller : callers) caller.join();
  EXPECT_EQ(failures.load(), 0);
  EXPECT_LE(most_active.load(), kWorkers + 1) << "two Runs executed their bodies at the same time";
}

TEST(Qwen38ParallelPoolTest, StopsThatRaceRunsNeverSplitOrDropARun) {
  constexpr int kWorkers = 3;
  constexpr int kCallers = 3;
  constexpr int kRunsPerCaller = 150;
  const int baseline = ProcessThreadCount();
  ThreadLedger ledger;
  {
    // The pool creates its threads under its own lock, one caller at a time, which is what the ledger's counter relies on.
    ParallelPool pool{kWorkers, ledger.Starter()};
    std::atomic<int> calls{0};
    std::atomic<int> failures{0};
    std::atomic<bool> done{false};
    std::vector<std::thread> callers;
    for (int c = 0; c < kCallers; ++c) {
      callers.emplace_back([&] {
        for (int i = 0; i < kRunsPerCaller; ++i) {
          if (pool.Run([&] { ++calls; })) ++failures;
        }
      });
    }
    std::thread stopper{[&] {
      while (!done.load()) {
        pool.Stop();
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }};
    for (std::thread& caller : callers) caller.join();
    done.store(true);
    stopper.join();
    EXPECT_EQ(failures.load(), 0);
    EXPECT_EQ(calls.load(), kCallers * kRunsPerCaller * (kWorkers + 1)) << "a Run lost or repeated a copy of its body";
  }
  EXPECT_EQ(ledger.running.load(), 0);
  EXPECT_TRUE(ProcessThreadCountSettlesAtOrBelow(baseline));
}

}  // namespace Generators::test
