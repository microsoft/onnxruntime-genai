// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// The worker threads of the host-side PLE gather of libqwen38_plane. Header-only and free of CUDA so the unit tests can build
// it without a GPU (test/cpp/engine/qwen38_ple_gather_pool_tests.cpp).
//
// The workers run code of the plane library, so their lifetime is bounded by the library's: they are joinable, Stop() joins
// them, and the plane stops the pool before the last loader detaches (see qwen38_plane_detach), i.e. before the host can
// FreeLibrary / dlclose the module. Nothing here runs at static-destruction or DllMain time.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace qwen38_plane {

// Worker threads for a machine with `hardware_threads` logical processors (0 when the count is unknown): one fewer than the
// processors, because the calling thread takes part too, but between 3 and 23.
inline int DefaultParallelWorkerCount(unsigned int hardware_threads) {
  constexpr unsigned int kMinWorkers = 3;
  constexpr unsigned int kMaxWorkers = 23;
  const unsigned int workers = hardware_threads > 1 ? hardware_threads - 1 : 0;
  return static_cast<int>(workers < kMinWorkers ? kMinWorkers : (workers > kMaxWorkers ? kMaxWorkers : workers));
}

// A fixed set of worker threads that run one body together with the calling thread.
//
// Run() executes `body` once on every worker and once on the caller, concurrently, and returns when every copy has returned.
// A body shares its work out by itself (for example by claiming ranges from an atomic counter) and must not throw. Run()
// calls are serialised. The workers are started by the first Run() and kept until Stop().
class ParallelPool {
 public:
  // Starts one thread that runs `entry`. Throws std::system_error when the thread cannot be created, like std::thread.
  using ThreadStarter = std::function<std::thread(std::function<void()> entry)>;

  // `workers` is the number of worker threads (at least 1). A null `starter` creates plain std::thread objects.
  explicit ParallelPool(int workers, ThreadStarter starter = nullptr)
      : workers_(workers > 0 ? workers : 1),
        starter_(starter ? std::move(starter) : ThreadStarter{[](std::function<void()> entry) {
          return std::thread(std::move(entry));
        }}) {}

  ~ParallelPool() { Stop(); }

  ParallelPool(const ParallelPool&) = delete;
  ParallelPool& operator=(const ParallelPool&) = delete;

  // Runs `body` on every worker and on the calling thread. Returns an empty error code once all of them have returned. When
  // the workers cannot be started (a thread cannot be created, or memory ran out) it returns the reason after it has stopped
  // and joined the workers that did start; `body` has not run at all then, and the next Run() starts the workers again.
  std::error_code Run(const std::function<void()>& body) {
    std::lock_guard<std::mutex> call_lock(call_mu_);
    if (const std::error_code error = StartLocked()) return error;
    {
      std::lock_guard<std::mutex> lock(mu_);
      body_ = &body;
      pending_ = static_cast<int>(threads_.size());
      ++epoch_;
    }
    wake_.notify_all();
    try {
      body();
    } catch (...) {
      WaitForWorkers();  // the workers still read `body`
      throw;
    }
    WaitForWorkers();
    return {};
  }

  // Makes every worker return and joins it. A Run() in flight completes first, so no work is lost. Idempotent; a later Run()
  // starts new workers.
  void Stop() {
    std::lock_guard<std::mutex> call_lock(call_mu_);
    StopLocked();
  }

  // The number of worker threads that are alive (0 before the first Run() and after Stop()).
  int ThreadCount() const { return thread_count_.load(std::memory_order_acquire); }

 private:
  // Starts the workers unless they are running. All or nothing: a failure joins the ones that started. Caller holds call_mu_.
  std::error_code StartLocked() {
    if (!threads_.empty()) return {};
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = false;
      epoch_ = 0;
      body_ = nullptr;
      pending_ = 0;
    }
    try {
      threads_.reserve(static_cast<size_t>(workers_));
      for (int i = 0; i < workers_; ++i) {
        threads_.push_back(starter_([this] { WorkerMain(); }));
      }
    } catch (const std::system_error& error) {
      StopLocked();
      return error.code();
    } catch (const std::bad_alloc&) {
      StopLocked();
      return std::make_error_code(std::errc::not_enough_memory);
    } catch (...) {
      StopLocked();
      throw;
    }
    thread_count_.store(static_cast<int>(threads_.size()), std::memory_order_release);
    return {};
  }

  // Caller holds call_mu_, so no Run() is in flight.
  void StopLocked() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& thread : threads_) {
      if (thread.joinable()) thread.join();
    }
    threads_.clear();
    thread_count_.store(0, std::memory_order_release);
  }

  void WaitForWorkers() {
    std::unique_lock<std::mutex> lock(mu_);
    done_.wait(lock, [this] { return pending_ == 0; });
    body_ = nullptr;
  }

  // Each worker runs the body once per Run() (epoch), so that Run() returns only after all of them have finished.
  void WorkerMain() {
    std::uint64_t seen = 0;
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
      wake_.wait(lock, [&] { return stop_ || epoch_ != seen; });
      if (stop_) return;
      seen = epoch_;
      const std::function<void()>* body = body_;
      lock.unlock();
      (*body)();
      lock.lock();
      if (--pending_ == 0) done_.notify_all();
    }
  }

  const int workers_;
  const ThreadStarter starter_;
  std::mutex call_mu_;                // one Run() or Stop() at a time
  std::vector<std::thread> threads_;  // guarded by call_mu_
  std::atomic<int> thread_count_{0};
  std::mutex mu_;  // the state below
  std::condition_variable wake_;
  std::condition_variable done_;
  const std::function<void()>* body_ = nullptr;
  int pending_ = 0;
  std::uint64_t epoch_ = 0;
  bool stop_ = false;
};

}  // namespace qwen38_plane
