// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "generators.h"
#include "engine/cache_manager.h"
#include "engine/model_executor.h"
#include "engine/scheduler.h"
#include "engine/engine.h"
#include "engine/decoders/decoder.h"

namespace Generators {
namespace test {

// A single ordered log the recording doubles append to, so a test can assert the sequence of
// collaborator calls the engine and scheduler make within a step (for example that a batch is
// allocated before it is decoded). Shared by the cache manager and executor doubles.
struct CallTrace {
  std::vector<std::string> entries;

  void Record(std::string entry) { entries.push_back(std::move(entry)); }
};

// An in-memory CacheManager double. It keeps deterministic allocation accounting (so a real
// scheduler drives it correctly) while recording every call the scheduler makes. Capacity and the
// CanAllocate verdict are scriptable so admission and backpressure can be exercised without a real
// paged cache or GPU.
struct RecordingCacheManager : CacheManager {
  RecordingCacheManager(std::shared_ptr<Model> model, size_t capacity,
                        std::shared_ptr<CallTrace> trace = nullptr)
      : CacheManager(std::move(model)), capacity_{capacity}, trace_{std::move(trace)} {}

  bool CanAllocate(const std::vector<std::shared_ptr<Request>>& requests) const override {
    can_allocate_calls++;
    if (trace_) trace_->Record("CanAllocate");
    if (!can_allocate_verdict_) return false;
    return allocated_.size() + requests.size() <= capacity_;
  }

  void Allocate(const std::vector<std::shared_ptr<Request>>& requests) override {
    allocate_calls++;
    if (trace_) trace_->Record("Allocate");
    for (const auto& request : requests) {
      if (std::find(allocated_.begin(), allocated_.end(), request) == allocated_.end())
        allocated_.push_back(request);
    }
  }

  void Step() override {
    step_calls++;
    if (trace_) trace_->Record("Step");
  }

  void Deallocate(std::vector<std::shared_ptr<Request>>& requests) override {
    deallocate_calls++;
    if (trace_) trace_->Record("Deallocate");
    for (const auto& request : requests) {
      allocated_.erase(std::remove(allocated_.begin(), allocated_.end(), request), allocated_.end());
    }
  }

  bool SupportsDynamicBatching() const override { return true; }

  std::vector<std::shared_ptr<Request>> AllocatedRequests() const override { return allocated_; }

  // Scriptable knobs.
  void SetCanAllocate(bool verdict) { can_allocate_verdict_ = verdict; }
  size_t AllocatedCount() const { return allocated_.size(); }

  // Recorded call counts.
  mutable int can_allocate_calls{0};
  int allocate_calls{0};
  int deallocate_calls{0};
  int step_calls{0};

 private:
  size_t capacity_;
  std::shared_ptr<CallTrace> trace_;
  bool can_allocate_verdict_{true};
  std::vector<std::shared_ptr<Request>> allocated_;
};

// A DecoderIO that fabricates logits instead of running the model: for each scheduled request it
// returns a vocab-sized logits row whose maximum is at `forced_token`, so the request's real greedy
// search deterministically selects that token. Using the model's end-of-stream token drives requests
// to completion in one step.
struct ScriptedDecoderIO : DecoderIO {
  ScriptedDecoderIO(std::shared_ptr<Model> model, ScheduledRequests& scheduled_requests,
                    std::shared_ptr<CacheManager> cache_manager, int32_t forced_token)
      : DecoderIO(model, scheduled_requests, cache_manager),
        vocab_size_{static_cast<int64_t>(model->config_->model.vocab_size)} {
    if (forced_token < 0 || forced_token >= vocab_size_) {
      throw std::runtime_error("ScriptedDecoderIO: forced_token out of vocabulary range");
    }
    const int64_t batch_size = static_cast<int64_t>(scheduled_requests.size());
    logits_ = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<float>);
    const std::array<int64_t, 2> shape{batch_size, vocab_size_};
    logits_->CreateTensor(shape);
    auto device_span = logits_->GetDeviceSpan<float>();
    auto cpu_span = device_span.CpuSpan();
    std::fill(cpu_span.begin(), cpu_span.end(), 0.0f);
    for (int64_t row = 0; row < batch_size; ++row) {
      cpu_span[row * vocab_size_ + forced_token] = 100.0f;
    }
    device_span.CopyCpuToDevice();
  }

  std::vector<DeviceSpan<float>> ProcessLogits() override {
    std::vector<DeviceSpan<float>> rows;
    auto all = logits_->GetDeviceSpan<float>();
    for (size_t i = 0; i < scheduled_requests_.size(); ++i) {
      rows.push_back(all.subspan(i * vocab_size_, vocab_size_));
    }
    return rows;
  }

 private:
  int64_t vocab_size_;
};

// A ModelExecutor double that records how many batches it is asked to decode (and how large each
// was) without running the model, then attaches scripted logits so the batch can sample its next
// tokens. This isolates the engine's schedule/decode call sequence from real inference.
struct RecordingModelExecutor : ModelExecutor {
  RecordingModelExecutor(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager,
                         int32_t forced_token, std::shared_ptr<CallTrace> trace = nullptr)
      : model_{std::move(model)},
        cache_manager_{std::move(cache_manager)},
        forced_token_{forced_token},
        trace_{std::move(trace)} {}

  void Decode(ScheduledRequests& scheduled_requests) override {
    decode_calls++;
    decoded_batch_sizes.push_back(scheduled_requests.size());
    if (trace_) trace_->Record("Decode");
    scheduled_requests.AddDecoderState(
        std::make_unique<ScriptedDecoderIO>(model_, scheduled_requests, cache_manager_, forced_token_));
  }

  int decode_calls{0};
  std::vector<size_t> decoded_batch_sizes;

 private:
  std::shared_ptr<Model> model_;
  std::shared_ptr<CacheManager> cache_manager_;
  int32_t forced_token_;
  std::shared_ptr<CallTrace> trace_;
};

// An Engine wired with the recording doubles above, together with non-owning observers of those
// doubles and the shared call trace so a test can drive Engine::Step and then inspect how the engine
// scheduled and decoded. The engine owns the doubles; the raw pointers stay valid for its lifetime.
struct DoublesEngine {
  std::shared_ptr<Engine> engine;
  RecordingCacheManager* cache;
  RecordingModelExecutor* executor;
  std::shared_ptr<CallTrace> trace;
};

inline DoublesEngine MakeDoublesEngine(std::shared_ptr<Model> model, size_t capacity, int32_t forced_token) {
  auto trace = std::make_shared<CallTrace>();
  auto cache = std::make_shared<RecordingCacheManager>(model, capacity, trace);
  auto scheduler = Scheduler::Create(model, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(model, cache, forced_token, trace);

  RecordingCacheManager* cache_observer = cache.get();
  RecordingModelExecutor* executor_observer = executor.get();

  EngineDependencies dependencies{std::move(cache), std::move(scheduler), std::move(executor)};
  auto engine = std::make_shared<Engine>(std::move(model), std::move(dependencies));

  return DoublesEngine{std::move(engine), cache_observer, executor_observer, std::move(trace)};
}

}  // namespace test
}  // namespace Generators
