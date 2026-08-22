// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"
#include "../models/io/kv_cache.h"
#include "fixed_state_pool.h"
#include "paged_key_value_cache.h"
#include "execution_context.h"
#include "step_plan.h"

namespace Generators {

struct CacheStepReservation {
  virtual PagedCacheReservation* PagedReservation() { return nullptr; }
  // The fixed reservation itself, so a speculative step can roll a row back to the state after its
  // accepted prefix before the commit. Null for a paged-only reservation.
  virtual FixedStateReservation* FixedReservation() { return nullptr; }
  // Fixed decoder-state resources this reservation owns, in scheduled request row order. Empty for
  // a paged-only reservation.
  virtual std::span<const FixedStateSlotHandle> FixedStateSlots() const { return {}; }
  virtual std::span<const FixedStateBinding> FixedStateBindings() const { return {}; }
  virtual size_t FixedStateStagingBytes() const { return 0; }
  virtual void ValidateCommit() const {}
  virtual void PrepareCommit() { ValidateCommit(); }
  virtual void Commit() = 0;
  virtual void Release() = 0;
  virtual ~CacheStepReservation() = default;
};

struct KeyValueCacheState : State {
  KeyValueCacheState(const GeneratorParams& params, const Model& model)
      : State(params, model) {}

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override {
    return {};
  }
};

struct CacheManager {
  CacheManager(std::shared_ptr<Model> model) : model_{model} {}

  static std::unique_ptr<CacheManager> Create(std::shared_ptr<Model> model);

  virtual bool CanAllocate(const std::vector<std::shared_ptr<Request>>& requests) const = 0;

  virtual void Allocate(const std::vector<std::shared_ptr<Request>>& requests) = 0;

  virtual void Step() = 0;

  virtual void PrepareStep(const std::vector<std::shared_ptr<Request>>&,
                           ExecutionContext&) {
    Step();
  }

  KeyValueCacheState* Cache() { return key_value_cache_state_.get(); };

  virtual void Deallocate(std::vector<std::shared_ptr<Request>>& requests) = 0;

  virtual bool SupportsDynamicBatching() const = 0;

  virtual size_t MaxBatchSize() const { return 4; }

  virtual std::vector<std::shared_ptr<Request>> AllocatedRequests() const = 0;

  virtual bool IsResident(const std::shared_ptr<Request>& request) const = 0;

  virtual size_t ResidentRequestCount() const = 0;

  // Columns in the block table the model will see this step, or 0 when the cache does not use one.
  // The decode path multiplies it by the block size to get the KV length bound it reports through
  // `attention_metadata`.
  virtual size_t BlockTableColumns() const { return 0; }

  // Maximum query tokens one request can contribute to a step, or 0 when the cache imposes no
  // per-request limit. Sliding-window rings use this to prevent a step from overwriting live KV.
  virtual size_t MaxQueryTokensPerRequest() const { return 0; }

  // Immutable snapshot of the cache's block accounting for invariant validation and state
  // inspection. Caches that do not use paged blocks return an empty snapshot.
  virtual PagedCacheSnapshot Snapshot() const { return {}; }

  virtual StepPlanningResult PlanStepResources(StepPlan&) const {
    throw std::logic_error("Cache manager does not support transactional step planning.");
  }

  virtual std::unique_ptr<CacheStepReservation> ReserveStep(const StepPlan&) {
    throw std::logic_error("Cache manager does not support transactional reservation.");
  }

  virtual ~CacheManager() = default;

 protected:
  std::shared_ptr<Model> model_;
  std::unique_ptr<KeyValueCacheState> key_value_cache_state_;
};

struct StaticCacheManager : CacheManager {
  StaticCacheManager(std::shared_ptr<Model> model);

  bool CanAllocate(const std::vector<std::shared_ptr<Request>>& requests) const override;

  void Allocate(const std::vector<std::shared_ptr<Request>>& requests) override;

  void Step() override;

  void Deallocate(std::vector<std::shared_ptr<Request>>& requests) override;

  bool SupportsDynamicBatching() const override;

  std::vector<std::shared_ptr<Request>> AllocatedRequests() const override;

  bool IsResident(const std::shared_ptr<Request>& request) const override;

  size_t ResidentRequestCount() const override { return cache_allocated_requests_.size(); }

 private:
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<KeyValueCache> key_value_cache_;
  std::vector<std::shared_ptr<Request>> cache_allocated_requests_;
};

struct PagedCacheManager : CacheManager {
  PagedCacheManager(std::shared_ptr<Model> model);

  bool CanAllocate(const std::vector<std::shared_ptr<Request>>& requests) const override;

  void Allocate(const std::vector<std::shared_ptr<Request>>& requests) override;

  void Step() override;

  void PrepareStep(const std::vector<std::shared_ptr<Request>>& requests,
                   ExecutionContext& context) override;

  void Deallocate(std::vector<std::shared_ptr<Request>>& requests) override;

  bool SupportsDynamicBatching() const override;

  size_t MaxBatchSize() const override {
    return model_->config_->engine.dynamic_batching->max_batch_size;
  }

  std::vector<std::shared_ptr<Request>> AllocatedRequests() const override;

  bool IsResident(const std::shared_ptr<Request>& request) const override;

  size_t ResidentRequestCount() const override { return cache_allocated_requests_.size(); }

  size_t BlockTableColumns() const override { return key_value_cache_->BlockTableColumns(); }

  size_t MaxQueryTokensPerRequest() const override {
    return key_value_cache_->MaxQueryTokensPerRequest();
  }

  PagedCacheSnapshot Snapshot() const override { return key_value_cache_->Snapshot(); }

  StepPlanningResult PlanStepResources(StepPlan& plan) const override;

  std::unique_ptr<CacheStepReservation> ReserveStep(const StepPlan& plan) override;

 private:
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<PagedKeyValueCache> key_value_cache_;
  std::unique_ptr<FixedStatePool> fixed_state_pool_;
  std::vector<std::shared_ptr<Request>> cache_allocated_requests_;
};

}  // namespace Generators
