// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"
#include "../models/kv_cache.h"

namespace Generators {

struct KeyValueCacheState : State {
  KeyValueCacheState(const GeneratorParams& params, const Model& model)
      : State(params, model) {}
};

struct CacheManager {
  CacheManager(std::shared_ptr<Model> model) : model_{model} {}

  bool CanAllocate(const std::vector<std::shared_ptr<Request>>& requests) const { return true; };

  void Allocate(const std::vector<std::shared_ptr<Request>>& requests){};

  void Step(){};

  KeyValueCacheState* Cache() { return nullptr; };

  bool SupportsContinuousBatching() const { return false; };

 private:
  std::shared_ptr<Model> model_;
  std::vector<std::shared_ptr<Request>> cache_allocated_requests_;
  std::unique_ptr<KeyValueCacheState> key_value_cache_state_;
  std::unique_ptr<KeyValueCache> key_value_cache_;
};

}  // namespace Generators
