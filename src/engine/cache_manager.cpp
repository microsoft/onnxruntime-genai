// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cache_manager.h"

namespace Generators {

std::unique_ptr<CacheManager> CreateCacheManager(std::shared_ptr<Model> model) {
  return std::make_unique<StaticCacheManager>(model);
}

bool StaticCacheManager::CanAllocate(const std::vector<std::shared_ptr<Request>>& requests) const {
  if (cache_allocated_requests_.empty()) {
    return true;
  }

  return false;
}

void StaticCacheManager::Allocate(const std::vector<std::shared_ptr<Request>>& requests) {
  for (const auto& request : requests) {
    cache_allocated_requests_.push_back(request);
  }

  if (!key_value_cache_) {
    key_value_cache_state_ = std::make_unique<KeyValueCacheState>(model_->Params(), *model_);
    key_value_cache_ = std::make_unique<DefaultKeyValueCache>(model_);
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

  key_value_cache_->Update({}, max_sequence_length);
}

}  // namespace Generators
