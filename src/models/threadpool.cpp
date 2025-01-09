// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "threadpool.h"

namespace Generators {

ThreadPool::ThreadPool(size_t num_threads) : num_threads_{num_threads} {}

void ThreadPool::Compute(const std::function<void(size_t)>& func) {
  for (size_t i = 0; i < num_threads_; ++i) {
    threads_.emplace_back([&, i] { func(i); });
  }

  for (auto& thread : threads_) {
    thread.join();
  }

  threads_.clear();
}

}  // namespace Generators