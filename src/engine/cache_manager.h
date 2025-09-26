// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "request.h"
#include "../models/kv_cache.h"
#include "paged_key_value_cache.h"

namespace Generators {

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

  KeyValueCacheState* Cache() { return key_value_cache_state_.get(); };

  virtual void Deallocate(std::vector<std::shared_ptr<Request>>& requests) = 0;

  virtual bool SupportsDynamicBatching() const = 0;

  virtual std::vector<std::shared_ptr<Request>> AllocatedRequests() const = 0;

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

  void Deallocate(std::vector<std::shared_ptr<Request>>& requests) override;

  bool SupportsDynamicBatching() const override;

  std::vector<std::shared_ptr<Request>> AllocatedRequests() const override;

 private:
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<PagedKeyValueCache> key_value_cache_;
  std::vector<std::shared_ptr<Request>> cache_allocated_requests_;
};

}  // namespace Generators
