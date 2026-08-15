// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <utility>
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
                        std::shared_ptr<CallTrace> trace = nullptr,
                        bool supports_dynamic_batching = true)
      : CacheManager(std::move(model)),
        capacity_{capacity},
        trace_{std::move(trace)},
        supports_dynamic_batching_{supports_dynamic_batching} {}

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

  bool SupportsDynamicBatching() const override {
    return supports_dynamic_batching_;
  }

  bool SupportsRecomputePreemption() const override {
    return supports_recompute_preemption_;
  }

  // Mirrors the paged manager: the request's blocks go back to the pool and it stops being
  // resident. Each allocated request is modelled as owning `blocks_per_request_` blocks.
  ReclaimedCacheOwnership ReclaimRequestCache(
      const std::shared_ptr<Request>& request) override {
    reclaim_calls++;
    if (trace_) trace_->Record("ReclaimRequestCache");
    const auto it = std::find(allocated_.begin(), allocated_.end(), request);
    if (it == allocated_.end())
      throw std::runtime_error("Recording cache cannot reclaim an unallocated request.");
    allocated_.erase(it);
    return ReclaimedCacheOwnership{
        blocks_per_request_,
        static_cast<size_t>(request->ProcessedSequenceLength()),
    };
  }

  PagedCacheSnapshot Snapshot() const override {
    PagedCacheSnapshot snapshot;
    snapshot.block_size = 1;
    snapshot.total_blocks = capacity_ * blocks_per_request_;
    size_t next_block_id = 0;
    for (const auto& request : allocated_) {
      RequestBlockSnapshot request_snapshot;
      request_snapshot.request_id = request.get();
      for (size_t i = 0; i < blocks_per_request_; ++i)
        request_snapshot.block_ids.push_back(next_block_id++);
      request_snapshot.used_slots = blocks_per_request_;
      snapshot.requests.push_back(std::move(request_snapshot));
    }
    snapshot.free_blocks = snapshot.total_blocks - next_block_id;
    return snapshot;
  }

  size_t MaxBatchSize() const override { return capacity_; }

  std::vector<std::shared_ptr<Request>> AllocatedRequests() const override { return allocated_; }

  StepPlanningResult PlanStepResources(StepPlan& plan) const override {
    const size_t request_limit =
        plan.scheduled_request_limit == 0 ? capacity_ : plan.scheduled_request_limit;
    const size_t new_admission_limit =
        plan.new_admission_limit == 0 ? capacity_ : plan.new_admission_limit;
    size_t selected_requests = 0;
    size_t selected_new_requests = 0;
    bool capacity_deferred = false;
    const void* unserviceable_request_id = nullptr;
    BlockCapacityShortfall blocked_admission;
    std::vector<const void*> request_ids;
    request_ids.reserve(plan.requests.size());
    for (size_t i = 0; i < plan.requests.size(); ++i) {
      const auto& entry = plan.requests[i];
      if (std::find(request_ids.begin(), request_ids.end(), entry.request_id) != request_ids.end()) {
        throw std::runtime_error("Recording cache received a duplicate request.");
      }
      request_ids.push_back(entry.request_id);

      const bool resident =
          std::find(allocated_.begin(), allocated_.end(), entry.request) != allocated_.end();
      if (entry.newly_admitted == resident) {
        throw std::runtime_error("Recording cache received invalid request membership.");
      }
      if (entry.request_id == unserviceable_request_id_) {
        unserviceable_request_id = unserviceable_request_id_;
        continue;
      }
      // Stands in for a waiting request that only the block pool is holding up, which is the one
      // deferral reason reclaiming a resident's blocks can relieve.
      if (entry.newly_admitted && entry.request_id == admission_blocked_request_id_) {
        capacity_deferred = true;
        if (!blocked_admission.Any()) {
          blocked_admission =
              BlockCapacityShortfall{admission_blocked_request_id_, admission_block_shortfall_};
        }
        continue;
      }
      if (entry.request_id == capacity_deferred_request_id_ ||
          selected_requests >= request_limit ||
          (entry.newly_admitted &&
           (!can_allocate_verdict_ ||
            allocated_.size() + selected_new_requests >= capacity_ ||
            selected_new_requests >= new_admission_limit))) {
        capacity_deferred = true;
        continue;
      }
      const bool newly_admitted = entry.newly_admitted;
      if (selected_requests != i) {
        plan.requests[selected_requests] = std::move(plan.requests[i]);
      }
      ++selected_requests;
      if (newly_admitted) {
        ++selected_new_requests;
      }
    }
    plan.requests.resize(selected_requests);

    if (!plan.requests.empty()) {
      return StepPlanningResult{
          true,
          capacity_deferred,
          unserviceable_request_id,
          blocked_admission,
          {},
          {StepOutcomeKind::Committed, plan.transaction_id, nullptr},
      };
    }
    if (unserviceable_request_id) {
      return StepPlanningResult{
          false,
          capacity_deferred,
          unserviceable_request_id,
          blocked_admission,
          {},
          {StepOutcomeKind::UnserviceableRequest,
           plan.transaction_id,
           unserviceable_request_id},
      };
    }
    return StepPlanningResult{
        false,
        capacity_deferred,
        nullptr,
        blocked_admission,
        {},
        {capacity_deferred ? StepOutcomeKind::CapacityDeferred : StepOutcomeKind::NoWork,
         plan.transaction_id,
         nullptr},
    };
  }

  std::unique_ptr<CacheStepReservation> ReserveStep(const StepPlan& plan) override {
    struct Reservation final : CacheStepReservation {
      Reservation(RecordingCacheManager& cache, const StepPlan& plan)
          : cache_{cache} {
        for (const auto& entry : plan.requests) {
          if (entry.newly_admitted) {
            newly_admitted_.push_back(entry.request);
          }
        }
      }

      void Commit() override {
        if (committed_) {
          throw std::logic_error("Recording cache reservation can only commit once.");
        }
        cache_.Allocate(newly_admitted_);
        committed_ = true;
      }

      void Release() override {
        if (committed_) {
          throw std::logic_error("Cannot release a committed recording cache reservation.");
        }
        released_ = true;
      }

      RecordingCacheManager& cache_;
      std::vector<std::shared_ptr<Request>> newly_admitted_;
      bool committed_{};
      bool released_{};
    };

    return std::make_unique<Reservation>(*this, plan);
  }

  // Scriptable knobs.
  void SetCanAllocate(bool verdict) { can_allocate_verdict_ = verdict; }
  void SetUnserviceableRequest(const std::shared_ptr<Request>& request) {
    unserviceable_request_id_ = request.get();
  }
  void SetCapacityDeferredRequest(const std::shared_ptr<Request>& request) {
    capacity_deferred_request_id_ = request.get();
  }
  // Reports `request` as a waiting admission that only free blocks are holding up, needing
  // `shortfall` more of them.
  void SetAdmissionBlockedRequest(const std::shared_ptr<Request>& request, size_t shortfall) {
    admission_blocked_request_id_ = request.get();
    admission_block_shortfall_ = shortfall;
  }
  void ClearAdmissionBlockedRequest() {
    admission_blocked_request_id_ = nullptr;
    admission_block_shortfall_ = 0;
  }
  void SetSupportsRecomputePreemption(bool supported) {
    supports_recompute_preemption_ = supported;
  }
  void SetBlocksPerRequest(size_t blocks) { blocks_per_request_ = blocks; }
  size_t AllocatedCount() const { return allocated_.size(); }

  // Recorded call counts.
  mutable int can_allocate_calls{0};
  int allocate_calls{0};
  int deallocate_calls{0};
  int step_calls{0};
  int reclaim_calls{0};

 private:
  size_t capacity_;
  std::shared_ptr<CallTrace> trace_;
  bool can_allocate_verdict_{true};
  const void* unserviceable_request_id_{};
  const void* capacity_deferred_request_id_{};
  const void* admission_blocked_request_id_{};
  size_t admission_block_shortfall_{};
  bool supports_dynamic_batching_{true};
  bool supports_recompute_preemption_{true};
  size_t blocks_per_request_{1};
  std::vector<std::shared_ptr<Request>> allocated_;
};

// A DecoderIO that fabricates logits instead of running the model: for each scheduled request it
// returns a vocab-sized logits row whose maximum is at `forced_token`, so the request's real greedy
// search deterministically selects that token. Using the model's end-of-stream token drives requests
// to completion in one step.
struct ScriptedDecoderIO : DecoderIO {
  ScriptedDecoderIO(std::shared_ptr<Model> model, ScheduledRequests& scheduled_requests,
                    std::shared_ptr<CacheManager> cache_manager, int32_t forced_token,
                    bool fail_process_logits = false)
      : ScriptedDecoderIO(model, scheduled_requests, cache_manager,
                          std::vector<int32_t>(scheduled_requests.size(), forced_token),
                          fail_process_logits) {}

  // Per-request variant: row i's logits peak at forced_tokens[i], so a test can make the "model"
  // output depend on what each request actually fed in this step.
  ScriptedDecoderIO(std::shared_ptr<Model> model, ScheduledRequests& scheduled_requests,
                    std::shared_ptr<CacheManager> cache_manager,
                    const std::vector<int32_t>& forced_tokens,
                    bool fail_process_logits = false)
      : DecoderIO(model, scheduled_requests, cache_manager),
        vocab_size_{static_cast<int64_t>(model->config_->model.vocab_size)},
        fail_process_logits_{fail_process_logits} {
    if (forced_tokens.size() != scheduled_requests.size()) {
      throw std::runtime_error("ScriptedDecoderIO: one forced token per scheduled request is required");
    }
    for (const int32_t forced_token : forced_tokens) {
      if (forced_token < 0 || forced_token >= vocab_size_) {
        throw std::runtime_error("ScriptedDecoderIO: forced_token out of vocabulary range");
      }
    }
    const int64_t batch_size = static_cast<int64_t>(scheduled_requests.size());
    logits_ = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<float>);
    const std::array<int64_t, 2> shape{batch_size, vocab_size_};
    logits_->CreateTensor(shape);
    auto device_span = logits_->GetDeviceSpan<float>();
    auto cpu_span = device_span.CpuSpan();
    std::fill(cpu_span.begin(), cpu_span.end(), 0.0f);
    for (int64_t row = 0; row < batch_size; ++row) {
      cpu_span[row * vocab_size_ + forced_tokens[static_cast<size_t>(row)]] = 100.0f;
    }
    device_span.CopyCpuToDevice();
  }

  std::vector<DeviceSpan<float>> ProcessLogits() override {
    if (fail_process_logits_) {
      throw std::runtime_error("Injected post-processing failure.");
    }
    std::vector<DeviceSpan<float>> rows;
    auto all = logits_->GetDeviceSpan<float>();
    for (size_t i = 0; i < scheduled_requests_.size(); ++i) {
      rows.push_back(all.subspan(i * vocab_size_, vocab_size_));
    }
    return rows;
  }

 private:
  int64_t vocab_size_;
  bool fail_process_logits_{};
};

enum class ScriptedExecutionFailure {
  None,
  RetryableBeforeExecution,
  RetryableDuringExecution,
  CapacityExceeded,
  Fatal,
  PostProcessing,
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

  void Decode(ScheduledRequests& scheduled_requests,
              ExecutionContext& context) override {
    decode_calls++;
    decoded_batch_sizes.push_back(scheduled_requests.size());
    if (context.plan) {
      decoded_token_counts.push_back(context.plan->token_count);
      std::vector<const void*> request_ids;
      request_ids.reserve(context.plan->requests.size());
      for (const auto& entry : context.plan->requests)
        request_ids.push_back(entry.request_id);
      decoded_request_ids.push_back(std::move(request_ids));
      decoded_transaction_ids.push_back(context.plan->transaction_id);
    }
    if (trace_) trace_->Record("Decode");
    ScriptedExecutionFailure failure = ScriptedExecutionFailure::None;
    if (!failures_.empty()) {
      failure = failures_.front();
      failures_.pop_front();
    }
    if (failure == ScriptedExecutionFailure::RetryableBeforeExecution) {
      throw ModelExecutionError{ExecutionFailureKind::RetryableAbort,
                                "Injected retryable execution failure."};
    }
    if (failure == ScriptedExecutionFailure::RetryableDuringExecution) {
      throw ModelExecutionError{ExecutionFailureKind::RetryableAbort,
                                "Injected retryable in-execution failure."};
    }
    if (failure == ScriptedExecutionFailure::CapacityExceeded) {
      throw ModelExecutionError{ExecutionFailureKind::CapacityExceeded,
                                "Injected execution capacity failure."};
    }
    if (failure == ScriptedExecutionFailure::Fatal) {
      throw std::runtime_error("Injected fatal execution failure.");
    }
    std::vector<int32_t> forced_tokens;
    forced_tokens.reserve(scheduled_requests.size());
    for (const auto& request : scheduled_requests) {
      forced_tokens.push_back(next_token_selector_ ? next_token_selector_(*request)
                                                   : forced_token_);
    }
    scheduled_requests.AddDecoderState(
        std::make_unique<ScriptedDecoderIO>(
            model_, scheduled_requests, cache_manager_, forced_tokens,
            failure == ScriptedExecutionFailure::PostProcessing));
    static_cast<void>(context);
  }

  // Replaces the single forced token with a per-request choice, so a test can make the fabricated
  // "model" output depend on the tokens a request actually pushed through this step.
  using NextTokenSelector = std::function<int32_t(Request&)>;
  void SetNextTokenSelector(NextTokenSelector selector) {
    next_token_selector_ = std::move(selector);
  }

  void SetNextFailure(ScriptedExecutionFailure failure) {
    failures_ = {failure};
  }

  void SetFailures(
      std::initializer_list<ScriptedExecutionFailure> failures) {
    failures_ = failures;
  }

  int decode_calls{0};
  std::vector<size_t> decoded_batch_sizes;
  std::vector<size_t> decoded_token_counts;
  std::vector<std::vector<const void*>> decoded_request_ids;
  std::vector<StepTransactionId> decoded_transaction_ids;

 private:
  std::shared_ptr<Model> model_;
  std::shared_ptr<CacheManager> cache_manager_;
  int32_t forced_token_;
  std::shared_ptr<CallTrace> trace_;
  std::deque<ScriptedExecutionFailure> failures_;
  NextTokenSelector next_token_selector_;
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
  if (!model->config_->engine.dynamic_batching)
    model->config_->engine.dynamic_batching = Config::Engine::DynamicBatching{};
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

// An Engine wired with the production paged cache manager and dynamic scheduler, with only model
// execution replaced. Admission, block accounting, reservation, commit, and preemption all run
// through the real code, so capacity pressure in a test is real block pressure.
struct PagedDoublesEngine {
  std::shared_ptr<Engine> engine;
  std::shared_ptr<PagedCacheManager> cache;
  DynamicBatchScheduler* scheduler;
  RecordingModelExecutor* executor;
};

inline PagedDoublesEngine MakePagedEngine(std::shared_ptr<Model> model, int32_t forced_token) {
  if (!model->config_->engine.dynamic_batching)
    throw std::runtime_error("MakePagedEngine requires engine.dynamic_batching to be configured.");

  auto cache = std::make_shared<PagedCacheManager>(model);
  auto scheduler = std::make_unique<DynamicBatchScheduler>(model, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(model, cache, forced_token);

  DynamicBatchScheduler* scheduler_observer = scheduler.get();
  RecordingModelExecutor* executor_observer = executor.get();
  std::shared_ptr<PagedCacheManager> cache_observer = cache;

  EngineDependencies dependencies{std::move(cache), std::move(scheduler), std::move(executor)};
  auto engine = std::make_shared<Engine>(std::move(model), std::move(dependencies));

  return PagedDoublesEngine{std::move(engine), std::move(cache_observer), scheduler_observer,
                            executor_observer};
}

}  // namespace test
}  // namespace Generators
