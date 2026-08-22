// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cache_manager.h"

#include <algorithm>
#include <optional>

#include "../models/model_state_manifest.h"

namespace Generators {

namespace {

// One transaction reservation over both the paged KV cache and, when the model declares fixed
// decoder-state groups, the fixed-state pool. It plans and reserves both up front, validates both
// before publishing either, stages all fallible fixed device work into inactive banks during
// PrepareCommit, and then publishes both at an infallible boundary.
//
// A composite reservation wraps exactly one PagedCacheReservation. The Engine holds at most one
// composite reservation at a time and the fixed pool permits a single live reservation, so no two
// paged reservations ever share the cache's committed_tables_ vector concurrently. The paged
// reservation's constructor therefore reserves its own committed_tables_ headroom, and its
// CommitValidated() is allocation-free even though it publishes every newly admitted table in one
// call. This is why no cross-reservation aggregate capacity preflight is needed here.
class CompositeCacheStepReservation final : public CacheStepReservation {
 public:
  CompositeCacheStepReservation(PagedKeyValueCache& cache,
                                FixedStatePool* fixed_state_pool,
                                std::vector<std::shared_ptr<Request>>& allocated_requests,
                                const StepPlan& plan)
      : allocated_requests_{allocated_requests} {
    std::vector<PagedCacheReservationRequest> paged_requests;
    paged_requests.reserve(plan.requests.size());
    std::vector<FixedStateReservationRequest> fixed_requests;
    if (fixed_state_pool) {
      fixed_requests.reserve(plan.requests.size());
    }
    newly_admitted_.reserve(plan.requests.size());
    for (const auto& entry : plan.requests) {
      paged_requests.push_back(PagedCacheReservationRequest{
          entry.request_id,
          entry.target_cache_slots,
          entry.newly_admitted,
          entry.whole_sequence_cache_slots,
      });
      if (fixed_state_pool) {
        // Fixed and paged track the same per-request cache-slot boundary: the fixed target_tokens
        // mirror the paged target so both commit at one token boundary. The pool infers resident
        // vs. new ownership itself, so no newly_admitted flag is passed.
        fixed_requests.push_back(FixedStateReservationRequest{
            entry.request_id,
            entry.target_cache_slots,
        });
      }
      if (entry.newly_admitted) {
        newly_admitted_.push_back(entry.request);
      }
    }

    allocated_requests_.reserve(allocated_requests_.size() + newly_admitted_.size());
    // Reserve paged blocks first, then fixed slots. If the fixed reservation throws, the already
    // constructed paged reservation member is destroyed as the constructor unwinds, which releases
    // its blocks, so no half-built reservation escapes.
    paged_reservation_.emplace(cache.Reserve(paged_requests));
    if (fixed_state_pool) {
      fixed_reservation_.emplace(
          fixed_state_pool->Reserve(fixed_requests, plan.fixed_state.capture_checkpoints));
    }
  }

  PagedCacheReservation* PagedReservation() override {
    return &*paged_reservation_;
  }

  FixedStateReservation* FixedReservation() override {
    return fixed_reservation_ ? &*fixed_reservation_ : nullptr;
  }

  std::span<const FixedStateSlotHandle> FixedStateSlots() const override {
    return fixed_reservation_ ? fixed_reservation_->Handles()
                              : std::span<const FixedStateSlotHandle>{};
  }

  std::span<const FixedStateBinding> FixedStateBindings() const override {
    return fixed_reservation_ ? fixed_reservation_->Bindings()
                              : std::span<const FixedStateBinding>{};
  }

  size_t FixedStateStagingBytes() const override {
    return fixed_reservation_ ? fixed_reservation_->PlannedStagingBytes() : 0;
  }

  void ValidateCommit() const override {
    if (committed_) {
      throw std::logic_error(
          "Composite cache step reservation can only be committed once.");
    }
    // Validate both sub-reservations up front so neither is published if either cannot commit.
    if (fixed_reservation_) {
      fixed_reservation_->ValidateCommit();
    }
    paged_reservation_->ValidateCommit();
  }

  void PrepareCommit() override {
    if (prepared_) {
      throw std::logic_error(
          "Composite cache step reservation can only be prepared once.");
    }
    ValidateCommit();
    // All fallible device work runs here, staging fixed outputs into the pool's inactive banks. The
    // visible fixed state and committed paged tables are untouched, so a failure leaves committed
    // state intact even though the Engine treats it as fatal (inactive banks may be partly written).
    if (fixed_reservation_) {
      fixed_reservation_->PrepareCommit();
    }
    prepared_ = true;
  }

  void Commit() override {
    if (committed_) {
      throw std::logic_error(
          "Composite cache step reservation can only be committed once.");
    }
    if (!prepared_) {
      PrepareCommit();
    }
    // Publication crosses the transaction boundary and is never retried. Publish paged occupancy
    // first: for this single validated reservation CommitValidated() performs no fallible allocation
    // or device work, but it is not noexcept, so the Engine treats any throw as fatal. The fixed
    // bank flip is noexcept and cannot leave the two states at different boundaries once reached.
    paged_reservation_->CommitValidated();
    if (fixed_reservation_) {
      fixed_reservation_->PublishCommit();
    }
    allocated_requests_.insert(allocated_requests_.end(),
                               newly_admitted_.begin(),
                               newly_admitted_.end());
    committed_ = true;
  }

  void Release() override {
    if (committed_) {
      throw std::logic_error(
          "Cannot release a committed composite cache step reservation.");
    }
    // Discard the fixed provisional slots and release the paged blocks. Neither was published, so
    // both are safe to reclaim. Aggregate errors so one failure does not skip the other cleanup.
    std::exception_ptr release_error;
    if (fixed_reservation_) {
      try {
        fixed_reservation_->Discard();
      } catch (...) {
        release_error = std::current_exception();
      }
    }
    try {
      paged_reservation_->Release();
    } catch (...) {
      if (!release_error) {
        release_error = std::current_exception();
      }
    }
    if (release_error) {
      std::rethrow_exception(release_error);
    }
  }

 private:
  std::vector<std::shared_ptr<Request>>& allocated_requests_;
  std::vector<std::shared_ptr<Request>> newly_admitted_;
  // Destroyed in reverse declaration order: the fixed reservation unwinds before the paged one,
  // matching the discard order in Release().
  std::optional<PagedCacheReservation> paged_reservation_;
  std::optional<FixedStateReservation> fixed_reservation_;
  bool prepared_{};
  bool committed_{};
};

}  // namespace

std::unique_ptr<CacheManager> CacheManager::Create(std::shared_ptr<Model> model) {
  const ModelStateManifest manifest{model->config_->model.decoder};
  if (model->config_->engine.dynamic_batching) {
    ModelStateManifest::ValidateDynamicEngineCompatibility(model->config_->model.decoder);
    return std::make_unique<PagedCacheManager>(model);
  }
  if (manifest.HasFixedStateGroups()) {
    throw std::runtime_error(
        "Fixed decoder state groups require engine.dynamic_batching");
  }

  return std::make_unique<StaticCacheManager>(model);
}

StaticCacheManager::StaticCacheManager(std::shared_ptr<Model> model)
    : CacheManager(model) {}

bool StaticCacheManager::CanAllocate(const std::vector<std::shared_ptr<Request>>& requests) const {
  if (cache_allocated_requests_.empty()) {
    return true;
  }

  if (std::all_of(cache_allocated_requests_.begin(), cache_allocated_requests_.end(),
                  [](const std::shared_ptr<Request>& request) {
                    return IsTurnComplete(request->status_) ||
                           IsClosed(request->status_);
                  })) {
    return true;
  }

  return false;
}

void StaticCacheManager::Allocate(const std::vector<std::shared_ptr<Request>>& requests) {
  assert(CanAllocate(requests));

  if (!cache_allocated_requests_.empty() &&
      std::all_of(cache_allocated_requests_.begin(), cache_allocated_requests_.end(),
                  [](const std::shared_ptr<Request>& request) {
                    return IsTurnComplete(request->status_) ||
                           IsClosed(request->status_);
                  })) {
    // If every request is TurnComplete or Closed, recycle the static batch before allocating new requests.
    Deallocate(cache_allocated_requests_);
  }

  for (const auto& request : requests) {
    cache_allocated_requests_.push_back(request);
  }

  if (!key_value_cache_) {
    auto request_with_max_max_sequence_length =
        std::max_element(
            requests.begin(), requests.end(),
            [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
              return a->Params()->search.max_length < b->Params()->search.max_length;
            });

    params_ = std::make_shared<GeneratorParams>(*model_);
    params_->search.max_length = (*request_with_max_max_sequence_length)->Params()->search.max_length;
    params_->search.batch_size = static_cast<int>(cache_allocated_requests_.size());

    key_value_cache_state_ = std::make_unique<KeyValueCacheState>(*params_, *model_);
    key_value_cache_ = std::make_unique<DefaultKeyValueCache>(*key_value_cache_state_);

    key_value_cache_->Add();
  }
}

bool StaticCacheManager::SupportsDynamicBatching() const { return false; }

void StaticCacheManager::Step() {
  auto request_with_max_sequence_length =
      std::max_element(
          cache_allocated_requests_.begin(), cache_allocated_requests_.end(),
          [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
            return a->CurrentSequenceLength() < b->CurrentSequenceLength();
          });

  const int64_t max_sequence_length = (*request_with_max_sequence_length)->CurrentSequenceLength();

  key_value_cache_->Update({}, static_cast<int>(max_sequence_length));
}

void StaticCacheManager::Deallocate(std::vector<std::shared_ptr<Request>>& requests) {
  if (std::set<std::shared_ptr<Request>>{requests.begin(), requests.end()} !=
      std::set<std::shared_ptr<Request>>{cache_allocated_requests_.begin(), cache_allocated_requests_.end()}) {
    throw std::runtime_error("Cannot dynamically deallocate statically batched requests.");
  }

  key_value_cache_.reset();
  key_value_cache_state_.reset();
  params_.reset();
  cache_allocated_requests_.clear();
}

std::vector<std::shared_ptr<Request>> StaticCacheManager::AllocatedRequests() const {
  return cache_allocated_requests_;
}

bool StaticCacheManager::IsResident(const std::shared_ptr<Request>& request) const {
  return std::find(cache_allocated_requests_.begin(), cache_allocated_requests_.end(), request) !=
         cache_allocated_requests_.end();
}

PagedCacheManager::PagedCacheManager(std::shared_ptr<Model> model)
    : CacheManager(model),
      params_(std::make_shared<GeneratorParams>(*model_)) {
  // The paged cache resolves its own paged_kv group. The fixed pool is created only when the
  // decoder manifest declares generic fixed state groups, so a dense paged model owns no pool and
  // the composite path degrades to paged-only. Its capacity matches the paged batch limit so paged
  // admission (bounded by max_batch_size) can never outrun fixed slots.
  ModelStateManifest manifest{model->config_->model.decoder};
  if (manifest.HasFixedStateGroups()) {
    auto fixed_state_pool = std::make_unique<FixedStatePool>(
        model, manifest,
        model_->config_->engine.dynamic_batching->max_batch_size);
    // Continuous batching schedules a variable number of rows per step, so a fixed group whose
    // tensors declare a static (non-symbolic) batch axis can never be served: every reservation
    // would have to have exactly that many rows. Reject it here, at load, with a clear message
    // rather than constructing the pool and failing fatally on the first step whose row count
    // differs from the static extent (FixedStatePool::Reserve enforces the exact match). The public
    // compatibility gate remains closed in this PR; this additional check protects direct manager
    // construction in component tests and remains necessary when the packed-IO follow-up opens it.
    if (fixed_state_pool->SessionBatchSize() != 0) {
      throw std::runtime_error(
          "Dynamic batching requires fixed decoder state groups with a dynamic batch axis, but "
          "this model declares a fixed (static) batch dimension of " +
          std::to_string(fixed_state_pool->SessionBatchSize()) + ".");
    }
    fixed_state_pool_ = std::move(fixed_state_pool);
  }
  // Allocate fixed banks before auto-sizing paged KV. ComputeNumBlocks queries current device free
  // memory, so the fixed pool's complete persistent footprint is already excluded without
  // duplicating its geometry/accounting in the paged cache.
  key_value_cache_ = std::make_unique<PagedKeyValueCache>(model);
  key_value_cache_state_ = std::make_unique<KeyValueCacheState>(*params_, *model_);
}

bool PagedCacheManager::CanAllocate(const std::vector<std::shared_ptr<Request>>& requests) const {
  if (cache_allocated_requests_.size() + requests.size() > model_->config_->engine.dynamic_batching->max_batch_size) {
    return false;
  }

  for (auto& request : requests) {
    if (!key_value_cache_->CanAdd(request)) {
      return false;
    }
  }

  return true;
}

void PagedCacheManager::Allocate(const std::vector<std::shared_ptr<Request>>& requests) {
  for (auto& request : requests) {
    cache_allocated_requests_.push_back(request);
    key_value_cache_->Add(request);
  }
}

void PagedCacheManager::Step() {
  for (auto& request : cache_allocated_requests_) {
    if (IsTurnComplete(request->status_)) {
      continue;
    }

    if (!key_value_cache_->CanAppendTokens(request)) {
      throw std::runtime_error("Cannot append tokens to request that is not ready.");
    }

    key_value_cache_->AppendTokens(request);
  }

  key_value_cache_->UpdateState(*key_value_cache_state_, cache_allocated_requests_);
}

void PagedCacheManager::PrepareStep(
    const std::vector<std::shared_ptr<Request>>& requests,
    ExecutionContext& context) {
  if (!context.cache_reservation) {
    Step();
    return;
  }
  if (!context.plan) {
    throw std::logic_error("Transactional cache preparation requires a step plan.");
  }

  key_value_cache_->UpdateState(*key_value_cache_state_,
                                requests,
                                *context.cache_reservation,
                                context.plan->proposed_block_table_columns);
}

void PagedCacheManager::Deallocate(std::vector<std::shared_ptr<Request>>& requests) {
  // Removal and reaping release paged and fixed ownership together, mirroring how the transaction
  // committed them together. Only requests that actually own committed state are touched; a request
  // that was queued but never committed owns neither and is skipped so HandleFor cannot throw.
  std::vector<std::shared_ptr<Request>> allocated_to_remove;
  allocated_to_remove.reserve(requests.size());
  std::vector<FixedStateSlotHandle> fixed_handles;
  if (fixed_state_pool_) {
    fixed_handles.reserve(requests.size());
  }
  for (const auto& request : requests) {
    if (std::find(cache_allocated_requests_.begin(),
                  cache_allocated_requests_.end(),
                  request) == cache_allocated_requests_.end()) {
      continue;
    }
    allocated_to_remove.push_back(request);
    if (fixed_state_pool_) {
      // A committed allocation always owns a committed fixed slot (they are published together), so
      // this handle resolves; collect them all before releasing so a stale handle cannot leave the
      // two states partially released.
      fixed_handles.push_back(fixed_state_pool_->HandleFor(request.get()));
    }
  }

  // Drop the torn-down requests from this manager's allocated list first. The erase is allocation
  // free and prevents a later teardown retry from processing and releasing the same requests twice.
  // The fixed handles were all validated above; releasing those handles and the corresponding paged
  // tables is therefore effectively infallible under the manager's serialized ownership contract.
  cache_allocated_requests_.erase(
      std::remove_if(cache_allocated_requests_.begin(), cache_allocated_requests_.end(),
                     [&allocated_to_remove](const std::shared_ptr<Request>& request) {
                       return std::find(allocated_to_remove.begin(),
                                        allocated_to_remove.end(),
                                        request) != allocated_to_remove.end();
                     }),
      cache_allocated_requests_.end());

  // Release fixed slots (host-only ownership reset; safe even if a fatal device error left the pool
  // unhealthy, so teardown is never blocked) and the paged blocks. Both are effectively infallible
  // for handles/tables this manager owns.
  for (const auto& handle : fixed_handles) {
    fixed_state_pool_->Release(handle);
  }
  for (auto& request : allocated_to_remove) {
    key_value_cache_->Remove(request);
  }
}

bool PagedCacheManager::SupportsDynamicBatching() const { return true; }

std::vector<std::shared_ptr<Request>> PagedCacheManager::AllocatedRequests() const {
  return cache_allocated_requests_;
}

bool PagedCacheManager::IsResident(const std::shared_ptr<Request>& request) const {
  return std::find(cache_allocated_requests_.begin(), cache_allocated_requests_.end(), request) !=
         cache_allocated_requests_.end();
}

StepPlanningResult PagedCacheManager::PlanStepResources(StepPlan& plan) const {
  plan.fixed_state = {};
  auto result = key_value_cache_->PlanStepResources(plan);
  if (!fixed_state_pool_ || !result.executable) {
    return result;
  }

  // Paged and fixed ownership advance together at commit, so their committed counts must agree
  // between steps. A divergence means a prior commit or removal left them out of step.
  // Uses cheap pool accessors (no per-step snapshot allocation) since planning runs every step.
  if (fixed_state_pool_->CommittedSlotCount() != cache_allocated_requests_.size()) {
    throw std::logic_error(
        "Paged and fixed state committed ownership counts differ.");
  }

  size_t new_slot_count = 0;
  for (const auto& entry : plan.requests) {
    // Cross-check per-request ownership against the fixed pool: a resident must own a committed
    // fixed slot and a new admission must not. This catches drift between the paged planner's
    // membership decision and fixed ownership before the reservation is built.
    const bool owns_fixed_slot =
        fixed_state_pool_->OwnsCommittedSlot(entry.request_id);
    if (entry.newly_admitted == owns_fixed_slot) {
      throw std::logic_error(
          "Fixed state ownership does not match the paged step plan membership.");
    }
    if (entry.newly_admitted) {
      ++new_slot_count;
    }
  }
  // Paged admission is bounded by max_batch_size, which equals fixed capacity, so this always holds;
  // a violation would mean paged and fixed capacity accounting diverged and is a logic error.
  if (new_slot_count > fixed_state_pool_->AvailableSlots()) {
    throw std::logic_error(
        "Paged planning selected more admissions than fixed state can reserve.");
  }

  // Rolling a rejected draft back needs the operators' per-token state series, which costs a
  // window's worth of extra staging, so only a step that actually verifies drafts asks for it.
  const bool capture_checkpoints = std::any_of(
      plan.requests.begin(), plan.requests.end(),
      [](const RequestStepPlan& entry) { return entry.draft_token_count != 0; });
  if (capture_checkpoints && !fixed_state_pool_->SupportsCheckpoints()) {
    throw std::logic_error(
        "Planned draft verification on a model whose fixed state declares no checkpoints.");
  }

  plan.fixed_state = FixedStateResourcePlan{
      true,
      plan.requests.size(),
      new_slot_count,
      fixed_state_pool_->PlannedStagingBytes(plan.requests.size(), capture_checkpoints),
      capture_checkpoints,
  };
  return result;
}

std::unique_ptr<CacheStepReservation> PagedCacheManager::ReserveStep(const StepPlan& plan) {
  return std::make_unique<CompositeCacheStepReservation>(
      *key_value_cache_, fixed_state_pool_.get(), cache_allocated_requests_, plan);
}

}  // namespace Generators
