// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kv_cache.h"

#include <future>
#include <thread>

namespace Generators {

class WorkerThread {
 public:
  using Task = std::packaged_task<void()>;

 public:
  ~WorkerThread() {
    Stop();
  }

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
    bool stop_requested{false};
    std::queue<Task> work_queue{};
  };

 private:
  static void WorkerLoop(SynchronizedState& sync_state) {
    while (true) {
      std::unique_lock l{sync_state.m};
      sync_state.notify_worker_cv.wait(l, [&sync_state]() { return sync_state.stop_requested || !sync_state.work_queue.empty(); });

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

struct WindowedKeyValueCache : KeyValueCache {
  WindowedKeyValueCache(State& state);

  void Add() override;
  void AddEncoder() override {
    throw std::runtime_error("WindowedKeyValueCache does not support AddEncoder.");
  };

  std::future<void> EnqueueSlideTask(std::span<size_t> layer_indices);

  void Update(DeviceSpan<int32_t> beam_indices, int current_length) override;

  void RewindTo(size_t index) override {
    throw std::runtime_error("WindowedKeyValueCache does not support RewindTo.");
  }

 private:
  void SlideForLayer(size_t layer_idx);
  void SlideForAllLayers();

  State& state_;
  const Model& model_{state_.model_};
  int layer_count_{};
  int window_size_{};
  size_t num_windows_{};
  size_t window_index_{};
  size_t input_index_{~0U}, output_index_{~0U};

  std::array<int64_t, 4> key_cache_shape_in_, key_cache_shape_out_;
  std::array<int64_t, 4> value_cache_shape_in_, value_cache_shape_out_;
  ONNXTensorElementDataType type_;

  std::vector<std::unique_ptr<OrtValue>> key_caches_in_, value_caches_in_;
  std::vector<std::unique_ptr<OrtValue>> key_caches_out_, value_caches_out_;
  std::vector<std::string> input_name_strings_, output_name_strings_;

  bool is_first_update_{true};

  WorkerThread worker_thread_{};
};

}  // namespace Generators
