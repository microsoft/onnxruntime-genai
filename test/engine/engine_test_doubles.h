// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "generators.h"
#include "search.h"
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

  size_t MaxBatchSize() const override { return capacity_; }

  size_t MaxQueryTokensPerRequest() const override { return max_query_tokens_per_request_; }

  size_t MaxDraftTokensPerStep() const override { return max_draft_tokens_per_step_; }

  std::vector<std::shared_ptr<Request>> AllocatedRequests() const override { return allocated_; }

  bool IsResident(const std::shared_ptr<Request>& request) const override {
    return std::find(allocated_.begin(), allocated_.end(), request) != allocated_.end();
  }

  size_t ResidentRequestCount() const override { return allocated_.size(); }

  StepPlanningResult PlanStepResources(StepPlan& plan) const override {
    const size_t request_limit =
        plan.scheduled_request_limit == 0 ? capacity_ : plan.scheduled_request_limit;
    size_t selected_requests = 0;
    size_t selected_new_requests = 0;
    bool capacity_deferred = false;
    const void* unserviceable_request_id = nullptr;
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
      if (entry.request_id == capacity_deferred_request_id_ ||
          selected_requests >= request_limit ||
          (entry.newly_admitted &&
           (!can_allocate_verdict_ ||
            allocated_.size() + selected_new_requests >= capacity_))) {
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
      if (scripted_fixed_plan_) {
        plan.fixed_state = *scripted_fixed_plan_;
      }
      return StepPlanningResult{
          true,
          capacity_deferred,
          unserviceable_request_id,
          {StepOutcomeKind::Committed, plan.transaction_id, nullptr},
      };
    }
    if (unserviceable_request_id) {
      return StepPlanningResult{
          false,
          capacity_deferred,
          unserviceable_request_id,
          {StepOutcomeKind::UnserviceableRequest,
           plan.transaction_id,
           unserviceable_request_id},
      };
    }
    return StepPlanningResult{
        false,
        capacity_deferred,
        nullptr,
        {capacity_deferred ? StepOutcomeKind::CapacityDeferred : StepOutcomeKind::NoWork,
         plan.transaction_id,
         nullptr},
    };
  }

  std::unique_ptr<CacheStepReservation> ReserveStep(const StepPlan& plan) override {
    reserve_calls++;
    struct Reservation final : CacheStepReservation {
      Reservation(RecordingCacheManager& cache, const StepPlan& plan)
          : cache_{cache} {
        for (const auto& entry : plan.requests) {
          if (entry.newly_admitted) {
            newly_admitted_.push_back(entry.request);
          }
        }
        // Replay any scripted fixed-state slots so a test can force the Engine's plan/reservation
        // consistency guard to fire without a real fixed-state pool.
        fixed_state_slots_ = cache.scripted_fixed_slots_;
        fixed_state_staging_bytes_ = cache.scripted_fixed_staging_bytes_;
      }

      std::span<const FixedStateSlotHandle> FixedStateSlots() const override {
        return fixed_state_slots_;
      }

      size_t FixedStateStagingBytes() const override {
        return fixed_state_staging_bytes_;
      }

      void CommitPrefix(size_t row, const void* request_id,
                        size_t step_tokens, size_t kept_tokens) override {
        if (committed_) {
          throw std::logic_error("Recording cache reservation is already committed.");
        }
        cache_.prefix_commits.push_back({row, request_id, step_tokens, kept_tokens});
      }

      void Commit() override {
        if (committed_) {
          throw std::logic_error("Recording cache reservation can only commit once.");
        }
        cache_.Allocate(newly_admitted_);
        cache_.reservation_commit_calls++;
        committed_ = true;
      }

      void Release() override {
        if (committed_) {
          throw std::logic_error("Cannot release a committed recording cache reservation.");
        }
        cache_.reservation_release_calls++;
        released_ = true;
      }

      RecordingCacheManager& cache_;
      std::vector<std::shared_ptr<Request>> newly_admitted_;
      std::vector<FixedStateSlotHandle> fixed_state_slots_;
      size_t fixed_state_staging_bytes_{};
      bool committed_{};
      bool released_{};
    };

    size_t newly_admitted = 0;
    for (const auto& entry : plan.requests) {
      newly_admitted += entry.newly_admitted ? 1 : 0;
    }
    reserved_new_request_counts.push_back(newly_admitted);
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
  void SetMaxQueryTokensPerRequest(size_t token_count) {
    max_query_tokens_per_request_ = token_count;
  }
  void SetMaxDraftTokensPerStep(size_t token_count) {
    max_draft_tokens_per_step_ = token_count;
  }
  // Forces the composite plan/reservation consistency guard in Engine::StepDynamic. PlanStepResources
  // publishes `plan`, and every reservation reports `slots`/`staging_bytes`, so a test can make the
  // planned fixed-state resources disagree with the reservation the Engine actually receives.
  void ScriptFixedStateMismatch(FixedStateResourcePlan plan,
                                std::vector<FixedStateSlotHandle> slots,
                                size_t staging_bytes) {
    scripted_fixed_plan_ = plan;
    scripted_fixed_slots_ = std::move(slots);
    scripted_fixed_staging_bytes_ = staging_bytes;
  }
  size_t AllocatedCount() const { return allocated_.size(); }

  // Every accepted-prefix narrowing the Engine asked a reservation to perform, in call order.
  struct PrefixCommit {
    size_t row{};
    const void* request_id{};
    size_t step_tokens{};
    size_t kept_tokens{};
  };
  std::vector<PrefixCommit> prefix_commits;

  // Recorded call counts.
  mutable int can_allocate_calls{0};
  int allocate_calls{0};
  int deallocate_calls{0};
  int step_calls{0};
  int reserve_calls{0};
  int reservation_commit_calls{0};
  int reservation_release_calls{0};
  std::vector<size_t> reserved_new_request_counts;

 private:
  size_t capacity_;
  size_t max_query_tokens_per_request_{};
  size_t max_draft_tokens_per_step_{};
  std::shared_ptr<CallTrace> trace_;
  bool can_allocate_verdict_{true};
  const void* unserviceable_request_id_{};
  const void* capacity_deferred_request_id_{};
  bool supports_dynamic_batching_{true};
  std::vector<std::shared_ptr<Request>> allocated_;
  std::optional<FixedStateResourcePlan> scripted_fixed_plan_;
  std::vector<FixedStateSlotHandle> scripted_fixed_slots_;
  size_t scripted_fixed_staging_bytes_{};
};

// A DecoderIO that fabricates logits instead of running the model: for each scheduled request it
// returns a vocab-sized logits row whose maximum is at `forced_token`, so the request's real greedy
// search deterministically selects that token. Using the model's end-of-stream token drives requests
// to completion in one step.
//
// A speculative step needs one row per draft on top of that. When the plan carries draft tokens,
// `row_tokens` scripts the argmax of every row in packed order, which is what decides how much of
// each proposal the Engine accepts.
struct ScriptedDecoderIO : DecoderIO {
  ScriptedDecoderIO(std::shared_ptr<Model> model, ScheduledRequests& scheduled_requests,
                    std::shared_ptr<CacheManager> cache_manager, int32_t forced_token,
                    bool fail_process_logits = false,
                    const StepPlan* plan = nullptr,
                    std::span<const int32_t> row_tokens = {},
                    int64_t hidden_size = 0,
                    ONNXTensorElementDataType hidden_type = Ort::TypeToTensorType<float>)
      : DecoderIO(model, scheduled_requests, cache_manager),
        vocab_size_{static_cast<int64_t>(model->config_->model.vocab_size)},
        fail_process_logits_{fail_process_logits} {
    if (forced_token < 0 || forced_token >= vocab_size_) {
      throw std::runtime_error("ScriptedDecoderIO: forced_token out of vocabulary range");
    }
    size_t rows = scheduled_requests.size();
    if (plan) {
      for (const auto& entry : plan->requests) {
        rows += entry.draft_token_count;
      }
    }
    if (!row_tokens.empty() && row_tokens.size() != rows) {
      throw std::runtime_error("ScriptedDecoderIO: row token script does not cover every row.");
    }
    row_count_ = rows;
    logits_ = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<float>);
    const std::array<int64_t, 2> shape{static_cast<int64_t>(rows), vocab_size_};
    logits_->CreateTensor(shape);
    auto device_span = logits_->GetDeviceSpan<float>();
    auto cpu_span = device_span.CpuSpan();
    std::fill(cpu_span.begin(), cpu_span.end(), 0.0f);
    for (size_t row = 0; row < rows; ++row) {
      const int32_t token = row_tokens.empty() ? forced_token : row_tokens[row];
      if (token < 0 || token >= vocab_size_) {
        throw std::runtime_error("ScriptedDecoderIO: scripted row token out of vocabulary range");
      }
      cpu_span[static_cast<int64_t>(row) * vocab_size_ + token] = 100.0f;
    }
    device_span.CopyCpuToDevice();

    if (hidden_size > 0) {
      hidden_states_ = std::make_unique<Tensor>(model->p_device_inputs_, hidden_type);
      const size_t hidden_rows = plan ? plan->token_count : scheduled_requests.size();
      const std::array<int64_t, 2> hidden_shape{
          static_cast<int64_t>(hidden_rows), hidden_size};
      hidden_states_->CreateTensor(hidden_shape);
      hidden_states_->GetByteSpan().Zero();
    }
  }

  std::vector<DeviceSpan<float>> ProcessLogits() override {
    if (fail_process_logits_) {
      throw std::runtime_error("Injected post-processing failure.");
    }
    std::vector<DeviceSpan<float>> rows;
    auto all = logits_->GetDeviceSpan<float>();
    for (size_t i = 0; i < row_count_; ++i) {
      rows.push_back(all.subspan(i * vocab_size_, vocab_size_));
    }
    return rows;
  }

  Tensor* HiddenStates() const override { return hidden_states_.get(); }

 private:
  int64_t vocab_size_;
  size_t row_count_{};
  bool fail_process_logits_{};
  std::unique_ptr<Tensor> hidden_states_;
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
      std::vector<int64_t> sequence_lengths;
      sequence_lengths.reserve(context.plan->requests.size());
      for (const auto& entry : context.plan->requests)
        sequence_lengths.push_back(entry.sequence_length_before);
      decoded_sequence_lengths_before.push_back(std::move(sequence_lengths));
    }
    used_device_input_ids.push_back(!context.input_ids.empty());
    if (trace_) trace_->Record("Decode");
    const auto failure = std::exchange(next_failure_, ScriptedExecutionFailure::None);
    if (failure == ScriptedExecutionFailure::RetryableBeforeExecution) {
      throw ModelExecutionError{ExecutionFailureKind::RetryableAbort,
                                "Injected retryable execution failure."};
    }
    // Executes after the pre-execution failure hook but before any in-execution or post-processing
    // failure, so a test can inspect the reserved fixed-state bindings/slots and write staged
    // outputs for the row order the Engine actually scheduled.
    if (on_execute_) {
      on_execute_(context);
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
    std::vector<int32_t> row_tokens;
    if (!queued_row_tokens_.empty()) {
      row_tokens = std::move(queued_row_tokens_.front());
      queued_row_tokens_.pop_front();
    }
    const auto& active_row_tokens = row_tokens.empty() ? verify_row_tokens_ : row_tokens;
    scheduled_requests.AddDecoderState(
        std::make_unique<ScriptedDecoderIO>(
            model_, scheduled_requests, cache_manager_, forced_token_,
            failure == ScriptedExecutionFailure::PostProcessing,
            context.plan, active_row_tokens, hidden_size_, hidden_type_));
    static_cast<void>(context);
  }

  void SetNextFailure(ScriptedExecutionFailure failure) { next_failure_ = failure; }
  void SetForcedToken(int32_t token) { forced_token_ = token; }
  // Scripts the argmax of every packed logits row of the next step, in plan order. Empty restores
  // the single forced token for all rows.
  void SetVerifyRowTokens(std::vector<int32_t> tokens) {
    verify_row_tokens_ = std::move(tokens);
  }
  void QueueRowTokens(std::vector<int32_t> tokens) {
    queued_row_tokens_.push_back(std::move(tokens));
  }
  void EnableHiddenStatesOutput(
      int64_t hidden_size,
      ONNXTensorElementDataType hidden_type = Ort::TypeToTensorType<float>) {
    hidden_size_ = hidden_size;
    hidden_type_ = hidden_type;
  }
  void SetExecutionCallback(std::function<void(ExecutionContext&)> callback) {
    on_execute_ = std::move(callback);
  }

  int decode_calls{0};
  std::vector<size_t> decoded_batch_sizes;
  std::vector<size_t> decoded_token_counts;
  std::vector<std::vector<const void*>> decoded_request_ids;
  std::vector<std::vector<int64_t>> decoded_sequence_lengths_before;
  std::vector<bool> used_device_input_ids;

 private:
  std::shared_ptr<Model> model_;
  std::shared_ptr<CacheManager> cache_manager_;
  int32_t forced_token_;
  std::shared_ptr<CallTrace> trace_;
  ScriptedExecutionFailure next_failure_{ScriptedExecutionFailure::None};
  std::vector<int32_t> verify_row_tokens_;
  std::deque<std::vector<int32_t>> queued_row_tokens_;
  std::function<void(ExecutionContext&)> on_execute_;
  int64_t hidden_size_{};
  ONNXTensorElementDataType hidden_type_{Ort::TypeToTensorType<float>};
};

struct CountingCudaDeviceState {
  size_t device_to_host_copies{};
  size_t synchronize_calls{};
  std::vector<int> argmax_rows;
};

struct CountingCudaMemory final : DeviceBuffer {
  CountingCudaMemory(size_t size, std::shared_ptr<CountingCudaDeviceState> state)
      : storage_{std::make_unique<uint8_t[]>(size)}, state_{std::move(state)} {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = storage_.get();
  }

  CountingCudaMemory(void* memory, size_t size,
                     std::shared_ptr<CountingCudaDeviceState> state)
      : state_{std::move(state)} {
    size_in_bytes_ = size;
    p_cpu_ = p_device_ = static_cast<uint8_t*>(memory);
  }

  const char* GetType() const override { return "test_cuda"; }
  void AllocateCpu() override {}
  void CopyDeviceToCpu() override { ++state_->device_to_host_copies; }
  void CopyCpuToDevice() override {}
  void CopyFrom(size_t begin_dest, DeviceBuffer& source,
                size_t begin_source, size_t size_in_bytes) override {
    std::memmove(p_device_ + begin_dest, source.p_device_ + begin_source, size_in_bytes);
  }
  void Zero() override { std::memset(p_device_, 0, size_in_bytes_); }

 private:
  std::unique_ptr<uint8_t[]> storage_;
  std::shared_ptr<CountingCudaDeviceState> state_;
};

struct CountingCudaDevice final : DeviceInterface {
  CountingCudaDevice()
      : state{std::make_shared<CountingCudaDeviceState>()} {}

  DeviceType GetType() const override { return DeviceType::CUDA; }
  void InitOrt(const OrtApi&, Ort::Allocator&) override {}
  Ort::Allocator& GetAllocator() override {
    return GetDeviceInterface(DeviceType::CPU)->GetAllocator();
  }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override { return {}; }
  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<CountingCudaMemory>(size, state);
  }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* memory, size_t size) override {
    return std::make_shared<CountingCudaMemory>(memory, size, state);
  }
  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    return std::make_unique<GreedySearch_Cpu>(params);
  }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return std::make_unique<BeamSearch_Cpu>(params);
  }
  void Synchronize() override { ++state->synchronize_calls; }

  bool ArgMaxDevice(const void* logits, ONNXTensorElementDataType logits_type,
                    int num_rows, int vocab_size,
                    DeviceSpan<int32_t> out_tokens) override {
    if (logits_type != Ort::TypeToTensorType<float> ||
        out_tokens.size() < static_cast<size_t>(num_rows)) {
      return false;
    }
    state->argmax_rows.push_back(num_rows);
    const auto* values = static_cast<const float*>(logits);
    auto output = out_tokens.Span();
    for (int row = 0; row < num_rows; ++row) {
      const auto* begin = values + static_cast<size_t>(row) * vocab_size;
      output[row] = static_cast<int32_t>(
          std::max_element(begin, begin + vocab_size) - begin);
    }
    return true;
  }

  std::shared_ptr<CountingCudaDeviceState> state;
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

struct MtpDoublesEngine {
  std::unique_ptr<CountingCudaDevice> device;
  std::shared_ptr<Engine> engine;
  RecordingCacheManager* cache;
  RecordingModelExecutor* executor;
  RecordingCacheManager* mtp_cache;
  RecordingModelExecutor* mtp_executor;
  std::shared_ptr<CountingCudaDeviceState> device_state;
};

// A composite Engine wired over the *real* PagedCacheManager (so a real PagedKeyValueCache and, when
// the model declares fixed groups, a real FixedStatePool are exercised) but driven by the recording
// model-executor double. The executor fabricates logits and never runs the ONNX graph, so the tests
// exercise the composite reserve/validate/prepare/publish transaction against real state pools while
// controlling execution and injecting failures. The engine owns the doubles; the raw pointers stay
// valid for its lifetime.
struct CompositeDoublesEngine {
  std::shared_ptr<Engine> engine;
  PagedCacheManager* cache;
  RecordingModelExecutor* executor;
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

inline CompositeDoublesEngine MakeCompositeDoublesEngine(std::shared_ptr<Model> model,
                                                         int32_t forced_token) {
  // Bypass the public compatibility gate so this PR can exercise the complete composite resource
  // manager before the following packed-IO PR enables fixed-state model execution.
  auto cache = std::make_shared<PagedCacheManager>(model);
  auto* cache_observer = dynamic_cast<PagedCacheManager*>(cache.get());
  if (!cache_observer) {
    throw std::logic_error("Composite Engine tests require a paged cache manager.");
  }
  auto scheduler = Scheduler::Create(model, cache);
  auto executor = std::make_unique<RecordingModelExecutor>(model, cache, forced_token);
  auto* executor_observer = executor.get();

  EngineDependencies dependencies{std::move(cache), std::move(scheduler), std::move(executor)};
  auto engine = std::make_shared<Engine>(std::move(model), std::move(dependencies));
  return CompositeDoublesEngine{std::move(engine), cache_observer, executor_observer};
}

}  // namespace test
}  // namespace Generators
