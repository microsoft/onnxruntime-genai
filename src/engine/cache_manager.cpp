// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cache_manager.h"

#include <optional>

#include "../models/model_state_manifest.h"

namespace Generators {

namespace {

class CompositeCacheStepReservation final : public CacheStepReservation {
 public:
  CompositeCacheStepReservation(PagedKeyValueCache& cache,
                                FixedStatePool* fixed_state_pool,
                                std::vector<std::shared_ptr<Request>>& allocated_requests,
                                const StepPlan& plan)
      : allocated_requests_{allocated_requests} {
    std::vector<PagedCacheReservationRequest> requests;
    requests.reserve(plan.requests.size());
    newly_admitted_.reserve(plan.requests.size());
    for (const auto& entry : plan.requests) {
      requests.push_back(PagedCacheReservationRequest{
          entry.request_id,
          entry.target_cache_slots,
          entry.newly_admitted,
          entry.whole_sequence_cache_slots,
      });
      if (entry.newly_admitted) {
        newly_admitted_.push_back(entry.request);
      }
    }

    allocated_requests_.reserve(allocated_requests_.size() + newly_admitted_.size());
    paged_reservation_.emplace(cache.Reserve(requests));
    if (fixed_state_pool) {
      std::vector<FixedStateReservationRequest> fixed_requests;
      fixed_requests.reserve(plan.requests.size());
      for (const auto& entry : plan.requests) {
        fixed_requests.push_back(FixedStateReservationRequest{
            entry.request_id, entry.target_cache_slots, 0});
      }
      fixed_reservation_.emplace(fixed_state_pool->Reserve(fixed_requests));
    }
  }

  PagedCacheReservation* PagedReservation() override {
    return &*paged_reservation_;
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
      throw std::logic_error("Composite cache step reservation can only be committed once.");
    }
    if (fixed_reservation_) {
      fixed_reservation_->ValidateCommit();
    }
    paged_reservation_->ValidateCommit();
  }

  void PrepareCommit() override {
    if (prepared_) {
      throw std::logic_error("Composite cache step reservation can only be prepared once.");
    }
    ValidateCommit();
    if (fixed_reservation_) {
      fixed_reservation_->PrepareCommit();
    }
    prepared_ = true;
  }

  void Commit() override {
    if (committed_) {
      throw std::logic_error("Composite cache step reservation can only be committed once.");
    }
    if (!prepared_) {
      PrepareCommit();
    }
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
      throw std::logic_error("Cannot release a committed paged cache step reservation.");
    }
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
    key_value_cache_ = model_->p_device_kvcache_->CreateKeyValueCache(*key_value_cache_state_);
    if (!key_value_cache_)
      throw std::runtime_error("The selected execution provider did not create a KV cache for the static Engine.");
    if (key_value_cache_->IsModelManaged()) {
      throw std::runtime_error(
          "Static Engine does not support model-managed KV caches. Use the Generator API for stateful "
          "execution-provider models, or use a model with exposed past/present KV cache tensors.");
    }

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
  const ModelStateManifest manifest{model->config_->model.decoder};
  if (manifest.HasFixedStateGroups()) {
    fixed_state_pool_ = std::make_unique<FixedStatePool>(
        model, model_->config_->engine.dynamic_batching->max_batch_size);
  }
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
  std::vector<FixedStateSlotHandle> fixed_handles;
  if (fixed_state_pool_) {
    fixed_handles.reserve(requests.size());
    for (const auto& request : requests) {
      if (IsResident(request)) {
        fixed_handles.push_back(fixed_state_pool_->HandleFor(request.get()));
      }
    }
  }
  for (const auto& handle : fixed_handles) {
    fixed_state_pool_->Release(handle);
  }
  for (auto& request : requests) {
    key_value_cache_->Remove(request);
  }

  cache_allocated_requests_.erase(
      std::remove_if(cache_allocated_requests_.begin(), cache_allocated_requests_.end(),
                     [&requests](const std::shared_ptr<Request>& request) {
                       return std::find(requests.begin(), requests.end(), request) != requests.end();
                     }),
      cache_allocated_requests_.end());
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
  auto result = key_value_cache_->PlanStepResources(plan);
  if (!fixed_state_pool_ || !result.executable) {
    return result;
  }
  size_t new_slot_count = 0;
  for (const auto& entry : plan.requests) {
    if (entry.newly_admitted) {
      ++new_slot_count;
    }
  }
  if (new_slot_count > fixed_state_pool_->AvailableSlots()) {
    throw std::logic_error(
        "Paged planning selected more admissions than fixed state can reserve.");
  }
  return result;
}

std::unique_ptr<CacheStepReservation> PagedCacheManager::ReserveStep(const StepPlan& plan) {
  return std::make_unique<CompositeCacheStepReservation>(
      *key_value_cache_, fixed_state_pool_.get(), cache_allocated_requests_, plan);
}

}  // namespace Generators
