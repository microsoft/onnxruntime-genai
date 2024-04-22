// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "cache_manager.h"

#include <iostream>

namespace Generators {

namespace {

constexpr char KeyCacheNamePrefix[] = "key_cache.";
constexpr char ValueCacheNamePrefix[] = "value_cache.";
constexpr char BlockTablesName[] = "block_tables";
constexpr char SlotMappingName[] = "slot_mapping";
constexpr int32_t BlockTablePadValue = -1;
constexpr int32_t DefaultBlockSize = 16;
constexpr int32_t BlockAvailable = 0;
constexpr float DefaultCacheGPUUtilizationFactor = 0.3f;

std::pair<std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>> AllocateLayerCache(
    const CacheOptions& options, Ort::Allocator& gpu_allocator) {
  std::vector<int64_t> shape = {options.num_blocks_, options.block_size_ * options.num_kv_heads_ * options.head_size_};
  std::unique_ptr<OrtValue> key_cache = OrtValue::CreateTensor(gpu_allocator, shape, options.dtype_);
  std::unique_ptr<OrtValue> value_cache = OrtValue::CreateTensor(gpu_allocator, shape, options.dtype_);

  return {std::move(key_cache), std::move(value_cache)};
}

void ComputeNumBlocks(CacheOptions& options) {
  if (options.num_blocks_ > 0) {
    return;
  }

  size_t dtype_size = 0;
  switch (options.dtype_) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      dtype_size = 4;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      dtype_size = 2;
      break;
    default:
      throw std::runtime_error("Invalid cache dtype: " + std::to_string(options.dtype_));
  }

  size_t free_bytes, total_bytes;
  CudaCheck() == cudaMemGetInfo(&free_bytes, &total_bytes);

  constexpr float memory_fragmentation_factor = 0.9f;
  constexpr size_t num_caches_per_layer = 2;  // K and V caches

  // Use the free memory to compute the number of blocks needed to achieve the given gpu_utilization_factor.
  options.num_blocks_ =
      (free_bytes *
       memory_fragmentation_factor *
       options.gpu_utilization_factor_) /
      (options.block_size_ *
       options.num_kv_heads_ *
       options.head_size_ *
       options.num_layers_ *
       dtype_size *
       num_caches_per_layer);
}

}  // namespace

CacheOptions::CacheOptions(const int32_t num_layers, const std::optional<int32_t>& block_size,
                           const int32_t num_kv_heads, const int32_t head_size,
                           const ONNXTensorElementDataType dtype,
                           const std::optional<int32_t>& num_blocks,
                           const std::optional<float>& gpu_utilization_factor)
    : num_layers_(num_layers),
      block_size_(block_size.value_or(DefaultBlockSize)),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      dtype_(dtype) {
  if (num_blocks.has_value() && gpu_utilization_factor.has_value()) {
    throw std::runtime_error("Both num_blocks and gpu_utilization_factor cannot be set at the same time.");
  } else if (num_blocks.has_value()) {
    num_blocks_ = *num_blocks;
  } else {
    gpu_utilization_factor_ = gpu_utilization_factor.value_or(DefaultCacheGPUUtilizationFactor);
  }
}

PagedCacheManager::PagedCacheManager(const CacheOptions& cache_options,
                                     Ort::Allocator* cpu_allocator,
                                     Ort::Allocator* gpu_allocator)
    : options_(cache_options),
      cpu_allocator_(cpu_allocator),
      gpu_allocator_(gpu_allocator) {
  ComputeNumBlocks(options_);
  block_refs_.resize(options_.num_blocks_, BlockAvailable);
  for (int64_t i = 0; i < options_.num_layers_; ++i) {
    cache_.emplace_back(AllocateLayerCache(options_, *gpu_allocator_));
  }
}

std::vector<size_t> PagedCacheManager::FindAvailableBlocks(size_t num_blocks_needed) {
  std::vector<size_t> free_blocks;
  for (size_t i = 0; i < block_refs_.size(); ++i) {
    if (block_refs_[i] == BlockAvailable) {
      free_blocks.push_back(i);
      if (free_blocks.size() == num_blocks_needed) {
        break;
      }
    }
  }

  if (free_blocks.size() != num_blocks_needed) {
    throw std::runtime_error("Not enough free blocks available to serve the request.");
  }

  return free_blocks;
}

void PagedCacheManager::ReserveBlocks(const std::vector<size_t>& block_ids) {
  for (size_t block_id : block_ids) {
    block_refs_[block_id]++;
  }
}

void PagedCacheManager::ReleaseBlocks(const std::vector<size_t>& block_ids) {
  for (size_t block_id : block_ids) {
    block_refs_[block_id]--;
    if (block_refs_[block_id] < 0) {
      // This should never happen.
      throw std::runtime_error("Block reference count is negative.");
    }
  }
}

void PagedCacheManager::AddToken(size_t sequence_id) {
  auto found_block_info = block_tables_.find(sequence_id);
  if (found_block_info == block_tables_.end()) {
    return;
  }

  auto& block_info = *(found_block_info->second);
  block_info.is_prompt = false;  // AddToken is only called during decoding stage.
  block_info.context_length++;   // Increment the context length for the sequence.
  if ((block_info.slot_ids.back() + 1) % options_.block_size_ == 0) {
    // The current block is full. Allocate a new block.
    auto block_ids = FindAvailableBlocks(1U);
    ReserveBlocks(block_ids);
    auto block_id = block_ids.front();

    block_info.block_ids.push_back(block_id);
    block_info.slot_ids = {block_id * options_.block_size_};
  } else {
    block_info.slot_ids = {block_info.slot_ids.back() + 1};
  }
}

void PagedCacheManager::Remove(size_t sequence_id) {
  auto found_block_info = block_tables_.find(sequence_id);
  if (found_block_info == block_tables_.end()) {
    return;
  }

  ReleaseBlocks(found_block_info->second->block_ids);
  block_infos_.erase(found_block_info->second);
  block_tables_.erase(found_block_info);
}

void PagedCacheManager::Add(size_t sequence_id, size_t prompt_token_size) {
  if (block_tables_.count(sequence_id)) {
    // The blocks for the given sequence_id is already allocated.
    return;
  }

  const size_t num_slots_needed = prompt_token_size;

  size_t num_blocks_needed = (num_slots_needed + options_.block_size_) / options_.block_size_;
  auto block_ids = FindAvailableBlocks(num_blocks_needed);
  ReserveBlocks(block_ids);
  std::vector<size_t> slot_ids(num_slots_needed);
  size_t token_id = 0;
  for (const auto& block_id : block_ids) {
    for (size_t slot_id = block_id * options_.block_size_;
         slot_id < (block_id + 1) * options_.block_size_ && token_id < num_slots_needed;
         ++slot_id) {
      slot_ids[token_id++] = slot_id;
    }
  }

  block_infos_.emplace_back(sequence_id, true /* is_prompt */, block_ids, slot_ids, prompt_token_size);
  block_tables_.emplace(sequence_id, --block_infos_.end());
}

std::pair<OrtValue*, OrtValue*> PagedCacheManager::Cache(size_t layer_id) {
  return {cache_[layer_id].first.get(), cache_[layer_id].second.get()};
}

OrtValue* PagedCacheManager::BlockTables() {
  size_t max_blocks = 0;
  for (const auto& block_info : block_infos_) {
    max_blocks = std::max(max_blocks, block_info.block_ids.size());
  }

  std::vector<int64_t> shape = {static_cast<int64_t>(block_infos_.size()), static_cast<int64_t>(max_blocks)};
  block_tables_value_ = OrtValue::CreateTensor(*cpu_allocator_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto* block_tables_data = block_tables_value_->GetTensorMutableData<int32_t>();
  size_t block_info_idx = 0;
  for (const auto& block_info : block_infos_) {
    for (size_t i = 0; i < block_info.block_ids.size(); ++i) {
      block_tables_data[block_info_idx * max_blocks + i] = block_info.block_ids[i];
    }
    for (size_t i = block_info.block_ids.size(); i < max_blocks; ++i) {
      block_tables_data[block_info_idx * max_blocks + i] = BlockTablePadValue;
    }
    ++block_info_idx;
  }

  return block_tables_value_.get();
}

OrtValue* PagedCacheManager::SlotMapping() {
  size_t num_tokens = std::accumulate(block_infos_.begin(), block_infos_.end(), num_tokens,
                                      [](size_t acc, const auto& block_info) { return acc + block_info.slot_ids.size(); });
  std::vector<int64_t> shape = {static_cast<int64_t>(num_tokens)};
  slot_mapping_value_ = OrtValue::CreateTensor(*cpu_allocator_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto* slot_mapping_data = slot_mapping_value_->GetTensorMutableData<int32_t>();
  size_t block_info_idx = 0;
  for (const auto& block_info : block_infos_) {
    for (size_t i = 0; i < block_info.slot_ids.size(); ++i) {
      slot_mapping_data[block_info_idx++] = block_info.slot_ids[i];
    }
  }
  return slot_mapping_value_.get();
}

// std::vector<size_t> PagedCacheManager::Order() const {
//   std::vector<size_t> order;
//   for (const auto& block_info : block_infos_) {
//     order.push_back(block_info.sequence_id);
//   }
//   return order;
// }

void PagedCacheManager::ReorderCache(const std::unordered_map<size_t, size_t>& sequence_id_mapping) {
  // After reordering the cache, we might have shared resources between sequences.
  // We need to update the block_refs_ accordingly.
  // IMP: Remember that the cache may contain sequences that the user may not be interested in.
  // so, offer a way for the user to specify the mapping from the sequence id to the location in the cache

  // 1. Update the block_refs_ for the new order.
  // [0, 1, 2, 3]
  // [2, 1, 1, 3]
  for (auto& [pointer, pointee] : sequence_id_mapping) {
    if (pointer == pointee) {
      continue;
    }
    block_refs_[pointee] = block_refs_[pointer];

  }
  // std::unordered_set<size_t> retained_sequence_ids(sequence_index_permutation.begin(), sequence_index_permutation.end());
  // for (auto& [sequence_id, block_info] : block_tables_) {
  //   if (retained_sequence_ids.count(sequence_id)) {
      
  //   }

  //   // The sequence is missing in the new order. Release the blocks.
  //   ReleaseBlocks(block_info->block_ids);
  //   missing_sequence_ids.insert(sequence_id);
  // }
  // std::unordered_set<size_t> missing_sequence_ids;

  // 2. Update the block tables so sequences point to the correct blocks in memory based on the new shape.

  // 3. Reorder the cache based on the new order.
  
}

}  // namespace Generators