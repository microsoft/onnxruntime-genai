// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cache_manager.h"

namespace Generators {

std::unique_ptr<CacheManager> CreateCacheManager(std::shared_ptr<Model> model) {
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
                    return request->status_ == RequestStatus::Completed;
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
                    return request->status_ == RequestStatus::Completed;
                  })) {
    // If all requests are completed, we can deallocate them before allocating the new requests.
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

}  // namespace Generators
