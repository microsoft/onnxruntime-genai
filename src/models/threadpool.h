// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace Generators {

class ThreadPool {
 public:
  using RangeFunction = std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>;

  explicit ThreadPool(size_t num_threads);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  static void TryParallelFor(ThreadPool* thread_pool,
                             std::ptrdiff_t total,
                             double cost_per_unit,
                             const RangeFunction& function);

  void Compute(const std::function<void(size_t)>& func);

 private:
  void WorkerLoop();
  void ExecuteJob();
  void Run(std::ptrdiff_t total, double cost_per_unit, const RangeFunction& function);

  size_t num_threads_;
  std::vector<std::thread> threads_;

  std::mutex submission_mutex_;
  std::mutex state_mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  bool shutdown_{};
  size_t generation_{};
  size_t workers_remaining_{};

  const RangeFunction* function_{};
  std::ptrdiff_t total_{};
  std::ptrdiff_t chunk_size_{};
  std::atomic<std::ptrdiff_t> next_{};
  std::atomic<bool> cancelled_{};
  std::mutex exception_mutex_;
  std::exception_ptr exception_;
};

}  // namespace Generators
