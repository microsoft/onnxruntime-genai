// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "threadpool.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Generators {
namespace {

constexpr double kMinimumParallelCost = 16384.0;
thread_local bool is_thread_pool_callback = false;

struct CallbackScope {
  CallbackScope() : previous_{is_thread_pool_callback} {
    is_thread_pool_callback = true;
  }

  ~CallbackScope() {
    is_thread_pool_callback = previous_;
  }

 private:
  bool previous_;
};

}  // namespace

ThreadPool::ThreadPool(size_t num_threads) : num_threads_{num_threads} {
  threads_.reserve(num_threads_);
  for (size_t i = 0; i < num_threads_; ++i) {
    threads_.emplace_back([this] { WorkerLoop(); });
  }
}

ThreadPool::~ThreadPool() {
  std::lock_guard submission_lock{submission_mutex_};
  {
    std::lock_guard state_lock{state_mutex_};
    shutdown_ = true;
    ++generation_;
  }
  work_ready_.notify_all();
  for (auto& thread : threads_) {
    thread.join();
  }
}

void ThreadPool::TryParallelFor(ThreadPool* thread_pool,
                                std::ptrdiff_t total,
                                double cost_per_unit,
                                const RangeFunction& function) {
  if (total < 0) {
    throw std::invalid_argument("ThreadPool::TryParallelFor total must not be negative");
  }
  if (total == 0) {
    return;
  }
  if (!thread_pool || is_thread_pool_callback) {
    CallbackScope callback_scope;
    function(0, total);
    return;
  }
  thread_pool->Run(total, cost_per_unit, function);
}

void ThreadPool::Run(std::ptrdiff_t total, double cost_per_unit, const RangeFunction& function) {
  const double total_cost = static_cast<double>(total) * std::max(0.0, cost_per_unit);
  if (threads_.empty() || total == 1 || !std::isfinite(total_cost) ||
      total_cost < kMinimumParallelCost) {
    CallbackScope callback_scope;
    function(0, total);
    return;
  }

  // Concurrent top-level submissions are intentionally serialized. This keeps
  // one reusable worker set while nested submissions remain synchronous.
  std::unique_lock submission_lock{submission_mutex_};
  const auto desired_chunks = static_cast<std::ptrdiff_t>((threads_.size() + 1) * 4);
  const auto chunk_size = std::max<std::ptrdiff_t>(1, (total + desired_chunks - 1) / desired_chunks);

  {
    std::lock_guard state_lock{state_mutex_};
    function_ = &function;
    total_ = total;
    chunk_size_ = chunk_size;
    next_.store(0, std::memory_order_relaxed);
    cancelled_.store(false, std::memory_order_relaxed);
    exception_ = nullptr;
    workers_remaining_ = threads_.size();
    ++generation_;
  }
  work_ready_.notify_all();

  ExecuteJob();

  {
    std::unique_lock state_lock{state_mutex_};
    work_done_.wait(state_lock, [this] { return workers_remaining_ == 0; });
    function_ = nullptr;
  }

  if (exception_) {
    std::rethrow_exception(exception_);
  }
}

void ThreadPool::ExecuteJob() {
  CallbackScope callback_scope;
  while (!cancelled_.load(std::memory_order_acquire)) {
    const auto first = next_.fetch_add(chunk_size_, std::memory_order_relaxed);
    if (first >= total_ || cancelled_.load(std::memory_order_acquire)) {
      break;
    }
    const auto last = std::min(total_, first + chunk_size_);
    try {
      (*function_)(first, last);
    } catch (...) {
      {
        std::lock_guard exception_lock{exception_mutex_};
        if (!exception_) {
          exception_ = std::current_exception();
        }
      }
      cancelled_.store(true, std::memory_order_release);
      break;
    }
  }
}

void ThreadPool::WorkerLoop() {
  size_t observed_generation = 0;
  for (;;) {
    {
      std::unique_lock state_lock{state_mutex_};
      work_ready_.wait(state_lock, [this, observed_generation] {
        return shutdown_ || generation_ != observed_generation;
      });
      if (shutdown_) {
        return;
      }
      observed_generation = generation_;
    }

    ExecuteJob();

    {
      std::lock_guard state_lock{state_mutex_};
      if (--workers_remaining_ == 0) {
        work_done_.notify_one();
      }
    }
  }
}

void ThreadPool::Compute(const std::function<void(size_t)>& func) {
  TryParallelFor(this, static_cast<std::ptrdiff_t>(num_threads_),
                 kMinimumParallelCost, [&](std::ptrdiff_t first, std::ptrdiff_t last) {
                   for (auto i = first; i < last; ++i) {
                     func(static_cast<size_t>(i));
                   }
                 });
}

}  // namespace Generators
