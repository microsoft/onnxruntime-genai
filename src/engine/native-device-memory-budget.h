// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string_view>

#include "../models/model.h"

namespace Generators {

class PrefixSnapshotCache;

std::optional<size_t> NativeSnapshotDeviceMemoryLimit(
    const Config::Engine::DynamicBatching& settings, std::optional<size_t> initial_free_bytes,
    size_t free_bytes, size_t snapshot_bytes);

class NativeDeviceMemoryBudget {
 public:
  explicit NativeDeviceMemoryBudget(const Model& model) : model_{model} {}
  std::shared_ptr<PrefixSnapshotCache> ConfigurePrefixCache(
      const Config& config, const char* disabled_reason = nullptr, std::string_view warning_component = {});
  void Configure(const std::optional<Config::Engine::DynamicBatching>& settings,
                 const std::shared_ptr<PrefixSnapshotCache>& cache);
  void Refresh() const;

 private:
  const Model& model_;
  Config::Engine::DynamicBatching settings_;
  std::weak_ptr<PrefixSnapshotCache> snapshots_;
};

}  // namespace Generators
