// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "native-device-memory-budget.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "paged_key_value_cache.h"
#include "prefix_snapshot_cache.h"

namespace Generators {

std::optional<size_t> NativeSnapshotDeviceMemoryLimit(
    const Config::Engine::DynamicBatching& settings, std::optional<size_t> initial_free_bytes,
    size_t free_bytes, size_t snapshot_bytes) {
  if (!settings.device_memory_budget_bytes && !settings.gpu_utilization_factor &&
      !settings.device_memory_headroom_bytes)
    return std::nullopt;
  if (settings.gpu_utilization_factor &&
      (!std::isfinite(*settings.gpu_utilization_factor) || *settings.gpu_utilization_factor <= 0 ||
       *settings.gpu_utilization_factor > 1))
    throw std::runtime_error("gpu_utilization_factor must be > 0 and <= 1.");
  if (snapshot_bytes > std::numeric_limits<size_t>::max() - free_bytes)
    throw std::overflow_error("Native snapshot memory accounting overflows size_t.");
  // Snapshots are reclaimable; only the remaining allocations belong to fixed native state.
  const size_t without_snapshots = free_bytes + snapshot_bytes;
  if (settings.device_memory_budget_bytes) {
    if (!settings.device_memory_headroom_bytes)
      throw std::runtime_error("device_memory_budget_bytes requires an explicit device_memory_headroom_bytes allowance.");
    if (!initial_free_bytes)
      throw std::runtime_error("device_memory_budget_bytes must be configured before loading the model.");
    const size_t fixed_bytes = *initial_free_bytes > without_snapshots ? *initial_free_bytes - without_snapshots : 0;
    if (fixed_bytes > *settings.device_memory_budget_bytes)
      throw std::runtime_error("device_memory_budget_bytes is too small for the model and fixed native state.");
    const auto remaining = std::min<uint64_t>(*settings.device_memory_budget_bytes - fixed_bytes, without_snapshots);
    if (*settings.device_memory_headroom_bytes > remaining)
      throw std::runtime_error("device_memory_budget_bytes is too small after reserving device_memory_headroom_bytes.");
    return static_cast<size_t>(remaining - *settings.device_memory_headroom_bytes);
  }
  const auto headroom = settings.device_memory_headroom_bytes.value_or(0);
  if (headroom > without_snapshots)
    throw std::runtime_error("Native device memory is too small after reserving device_memory_headroom_bytes.");
  const size_t available = without_snapshots - static_cast<size_t>(headroom);
  return settings.gpu_utilization_factor
             ? PagedCacheMemoryBudget(available, *settings.gpu_utilization_factor)
             : available;
}

void NativeDeviceMemoryBudget::Configure(
    const std::optional<Config::Engine::DynamicBatching>& settings,
    const std::shared_ptr<PrefixSnapshotCache>& cache) {
  settings_ = settings.value_or(Config::Engine::DynamicBatching{});
  snapshots_ = cache;
  if (!settings_.device_memory_budget_bytes && !settings_.gpu_utilization_factor &&
      !settings_.device_memory_headroom_bytes)
    return;
  if (model_.p_device_->GetType() != DeviceType::CUDA)
    throw std::runtime_error("Native device memory controls require the CUDA execution provider.");
  if (settings_.device_memory_budget_bytes && !settings_.device_memory_headroom_bytes)
    throw std::runtime_error("device_memory_budget_bytes requires an explicit device_memory_headroom_bytes allowance.");
}

void NativeDeviceMemoryBudget::Refresh() const {
  if (!settings_.device_memory_budget_bytes && !settings_.gpu_utilization_factor &&
      !settings_.device_memory_headroom_bytes)
    return;
  model_.p_device_->Synchronize();
  size_t free_bytes, total_bytes;
  model_.p_device_->GetAvailableMemory(free_bytes, total_bytes);
  const auto snapshots = snapshots_.lock();
  const auto limit = NativeSnapshotDeviceMemoryLimit(
      settings_, model_.device_memory_budget_initial_free_bytes_, free_bytes,
      snapshots ? snapshots->Used()[0] : 0);
  if (snapshots && !snapshots->SetDeviceMemoryLimit(*limit, model_.p_device_))
    throw std::runtime_error("Native device memory budget cannot reclaim leased prefix checkpoints.");
}

}  // namespace Generators
