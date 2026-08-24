// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cache_manager.h"

#include <optional>

#include "../models/model_state_manifest.h"

namespace Generators {

namespace {

class PagedCacheStepReservation final : public CacheStepReservation {
 public:
  PagedCacheStepReservation(PagedKeyValueCache& cache,
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
    reservation_.emplace(cache.Reserve(requests));
  }

  PagedCacheReservation* PagedReservation() override {
    return &*reservation_;
  }

  void Commit() override {
    if (committed_) {
      throw std::logic_error("Paged cache step reservation can only be committed once.");
    }
    reservation_->Commit();
    allocated_requests_.insert(allocated_requests_.end(),
                               newly_admitted_.begin(),
                               newly_admitted_.end());
    committed_ = true;
  }

  void Release() override {
    if (committed_) {
      throw std::logic_error("Cannot release a committed paged cache step reservation.");
    }
    reservation_->Release();
  }

 private:
  std::vector<std::shared_ptr<Request>>& allocated_requests_;
  std::vector<std::shared_ptr<Request>> newly_admitted_;
  std::optional<PagedCacheReservation> reservation_;
  bool committed_{};
};

}  // namespace

std::unique_ptr<CacheManager> CacheManager::Create(std::shared_ptr<Model> model) {
  if (model->config_->engine.dynamic_batching) {
    ModelStateManifest::ValidateDynamicEngineCompatibility(model->config_->model.decoder);
    return std::make_unique<PagedCacheManager>(model);
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
      params_(std::make_shared<GeneratorParams>(*model_)),
      key_value_cache_(std::make_unique<PagedKeyValueCache>(model)) {
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

std::unique_ptr<CacheStepReservation> PagedCacheManager::ReserveStep(const StepPlan& plan) {
  return std::make_unique<PagedCacheStepReservation>(
      *key_value_cache_, cache_allocated_requests_, plan);
}

}  // namespace Generators
