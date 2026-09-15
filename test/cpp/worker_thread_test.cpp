// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "worker_thread.h"

#include <atomic>
#include <vector>

#include <gtest/gtest.h>

namespace Generators::test {

TEST(WorkerThreadTest, EnqueueOneThenWaitForOne) {
  constexpr size_t num_work_items = 64;

  std::atomic<size_t> work_counter = 0;
  auto do_work = [&work_counter]() { ++work_counter; };

  WorkerThread worker{};

  for (size_t i = 0; i < num_work_items; ++i) {
    auto work_item_future = worker.Enqueue(do_work);
    ASSERT_TRUE(work_item_future.valid());
    work_item_future.wait();
  }

  EXPECT_EQ(work_counter, num_work_items);
}

TEST(WorkerThreadTest, EnqueueMultipleThenWaitForMultiple) {
  constexpr size_t num_work_items = 64;

  std::atomic<size_t> work_counter = 0;
  auto do_work = [&work_counter]() { ++work_counter; };

  WorkerThread worker{};
  std::vector<std::future<void>> work_item_futures;

  for (size_t i = 0; i < num_work_items; ++i) {
    auto work_item_future = worker.Enqueue(do_work);
    work_item_futures.emplace_back(std::move(work_item_future));
  }

  for (auto& work_item_future : work_item_futures) {
    ASSERT_TRUE(work_item_future.valid());
    work_item_future.get();
  }

  EXPECT_EQ(work_counter, num_work_items);
}

}  // namespace Generators::test
