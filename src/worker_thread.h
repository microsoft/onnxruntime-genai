// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <future>
#include <mutex>
#include <queue>
#include <thread>

namespace Generators {

// A worker thread that performs the work items submitted to it.
// A work item is something callable with signature `void WorkItem()`.
class WorkerThread {
 public:
  using Task = std::packaged_task<void()>;

 public:
  ~WorkerThread() {
    Stop();
  }

  // Stops the worker thread.
  // If there are remaining work items in the queue, they will not be completed.
  void Stop() {
    if (!thread_.joinable()) {
      return;
    }

    {
      std::scoped_lock l{sync_state_.m};
      sync_state_.stop_requested = true;
    }

    sync_state_.notify_worker_cv.notify_all();

    thread_.join();
  }

  // Enqueues Task `task` as a work item.
  // The work item completion can be monitored with `task`'s associated std::future.
  void EnqueueTask(Task&& task) {
    bool work_queue_was_empty{};
    {
      std::scoped_lock l{sync_state_.m};
      work_queue_was_empty = sync_state_.work_queue.empty();
      sync_state_.work_queue.push(std::move(task));
    }

    if (work_queue_was_empty) {
      sync_state_.notify_worker_cv.notify_one();
    }
  }

  // Enqueues `fn` as a work item.
  // Returns the std::future associated with a Task that wraps `fn`.
  // The work item completion can be monitored with the returned std::future.
  template <typename Fn>
  std::future<void> Enqueue(Fn&& fn) {
    Task task{std::forward<Fn>(fn)};
    auto future = task.get_future();
    EnqueueTask(std::move(task));
    return future;
  }

 private:
  struct SynchronizedState {
    std::mutex m{};
    std::condition_variable notify_worker_cv{};

    // Note: The SynchronizedState data members declared below should be accessed while SynchronizedState::m is locked.

    bool stop_requested{false};
    std::queue<Task> work_queue{};
  };

 private:
  static void WorkerLoop(SynchronizedState& sync_state) {
    while (true) {
      std::unique_lock l{sync_state.m};
      sync_state.notify_worker_cv.wait(l, [&sync_state]() {
        return sync_state.stop_requested || !sync_state.work_queue.empty();
      });

      // stop?
      if (sync_state.stop_requested) {
        break;
      }

      // get work item
      Task work_item = std::move(sync_state.work_queue.front());
      sync_state.work_queue.pop();

      l.unlock();

      // do work item
      work_item();
    }
  }

 private:
  SynchronizedState sync_state_{};

  std::thread thread_{&WorkerThread::WorkerLoop, std::ref(sync_state_)};
};

}  // namespace Generators
