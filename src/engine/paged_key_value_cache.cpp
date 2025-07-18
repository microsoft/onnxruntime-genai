// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cache_manager.h"

#include <iostream>

namespace Generators {

namespace {

size_t ComputeNumBlocks(std::shared_ptr<Model> model) {
  if (model->config_->engine->dynamic_batching->num_blocks.has_value()) {
    return *model->config_->engine->dynamic_batching->num_blocks;
  }

  const auto dtype_size = Ort::SizeOf(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  size_t free_bytes, total_bytes;
  model->p_device_kvcache_->GetAvailableMemory(free_bytes, total_bytes);

  constexpr float memory_fragmentation_factor = 0.9f;
  constexpr size_t num_caches_per_layer = 2;  // 2 for key and value caches

  // Use the free memory to compute the number of blocks needed to achieve the given gpu_utilization_factor.
  return (free_bytes *
          memory_fragmentation_factor *
          *model->config_->engine->dynamic_batching->gpu_utilization_factor) /
         (model->config_->engine->dynamic_batching->block_size *
          model->config_->model.decoder.num_key_value_heads *
          model->config_->model.decoder.head_size *
          model->config_->model.decoder.num_hidden_layers *
          dtype_size *
          num_caches_per_layer);
}

}  // namespace

PagedKeyValueCache::PagedKeyValueCache(std::shared_ptr<Model> model)
    : model_(model) {
  const auto num_blocks = ComputeNumBlocks(model_);
  const std::vector<int64_t> cache_shape_per_layer{static_cast<int64_t>(num_blocks),
                                                   static_cast<int64_t>(model->config_->engine->dynamic_batching->block_size) *
                                                       model->config_->model.decoder.num_key_value_heads *
                                                       model->config_->model.decoder.head_size};
  const auto dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  for (size_t i = 0; i < model->config_->model.decoder.num_hidden_layers; ++i) {
    cache_.emplace_back(OrtValue::CreateTensor(model->p_device_kvcache_->GetAllocator(), cache_shape_per_layer, dtype),   // Key cache
                        OrtValue::CreateTensor(model->p_device_kvcache_->GetAllocator(), cache_shape_per_layer, dtype));  // Value cache
  }
  block_pool_ = std::make_unique<BlockPool>(model->config_->engine->dynamic_batching->block_size, num_blocks);
}

bool PagedKeyValueCache::CanAdd(std::shared_ptr<Request> request) const {
  return true;
  // return block_allocator_->NumFreeBlocks() > block_allocator_->NumBlocksNeeded(num_tokens);
}

void PagedKeyValueCache::Add(std::shared_ptr<Request> request) {
  // if (!CanAdd(num_tokens)) {
  //   throw std::runtime_error("Not enough free blocks available to serve the request.");
  // }

  // // Allocate blocks for the first sequence id in sequence_ids.
  // Add(sequence_ids.front(), num_tokens);

  // // Fork the blocks for the remaining sequence ids.
  // for (size_t i = 1; i < sequence_ids.size(); ++i) {
  //   Fork(sequence_ids.front(), sequence_ids[i]);
}

// void PagedKeyValueCache::Add(size_t sequence_id, size_t num_tokens) {
// if (!CanAdd(num_tokens)) {
//   throw std::runtime_error("Not enough free blocks available to serve the request.");
// }

// // Create an empty block table for the sequence id.
// block_table_pool_.emplace_back(BlockTable{sequence_id, {}});
// auto block_table = --block_table_pool_.end();
// block_tables_[sequence_id] = block_table;

// // Allocate blocks for the sequence id.
// block_table->blocks = block_allocator_->AllocateBlocks(num_tokens);
// }

bool PagedKeyValueCache::CanAppendTokens(std::shared_ptr<Request> request, size_t num_tokens) const {
  return true;
  //   const auto block_table = block_tables_.find(sequence_id);
  //   if (block_table == block_tables_.end()) {
  //     return false;
  //   }

  //   const size_t num_slots_available = block_table->second->blocks.back()->NumEmptySlots() +
  //                                      block_allocator_->NumFreeBlocks() * options_.block_size_;

  //   // TODO: Implement copy on write functionality

  //   return num_slots_available >= num_tokens;
  // }

  // void PagedKeyValueCache::AppendTokens(size_t sequence_id, size_t num_tokens) {
  //   if (!CanAppendTokens(sequence_id, num_tokens)) {
  //     throw std::runtime_error("Not enough free slots available to serve the request.");
  //   }

  //   auto block_table = block_tables_.find(sequence_id);
  //   if (block_table == block_tables_.end()) {
  //     throw std::runtime_error("Given sequence id " + std::to_string(sequence_id) + " is not found in the cache.");
  //   }

  //   if (block_table->second->blocks.back()->RefCount() > 1) {
  //     // The last block has multiple references. Copy-on-write is needed.
  //     auto src_block = block_table->second->blocks.back();
  //     auto dst_block = block_allocator_->AllocateBlock(src_block->NumOccupiedSlots(), src_block->PreviosBlock());
  //     src_block->DecrementRefCount();
  //     block_table->second->blocks.back() = dst_block;
  //     copy_on_writes_.emplace(dst_block->Id(), src_block->Id());
  //   }

  //   if (!block_table->second->blocks.back()->IsFull()) {
  //     for (size_t i = 0; i < std::min(num_tokens, block_table->second->blocks.back()->NumEmptySlots()); ++i) {
  //       block_table->second->blocks.back()->AddSlot();
  //       --num_tokens;
  //     }
  //   }

  //   std::shared_ptr<Block> previous_block = block_table->second->blocks.back();
  //   for (int32_t remaining_tokens = static_cast<int32_t>(num_tokens);
  //        remaining_tokens > 0;
  //        remaining_tokens -= options_.block_size_) {
  //     previous_block = block_allocator_->AllocateBlock(std::min(remaining_tokens, options_.block_size_), previous_block);
  //     block_table->second->blocks.push_back(previous_block);
  //   }
}

void PagedKeyValueCache::Remove(std::shared_ptr<Request> request) {
  // auto block_table = block_tables_.find(sequence_id);
  // if (block_table == block_tables_.end()) {
  //   return;
  // }

  // block_allocator_->Free(block_table->second->blocks);

  // block_table_pool_.erase(block_table->second);
  // block_tables_.erase(block_table);
}

std::pair<OrtValue*, OrtValue*> PagedKeyValueCache::Cache(size_t layer_id) {
  return {cache_[layer_id].first.get(), cache_[layer_id].second.get()};
}

std::unique_ptr<OrtValue> PagedKeyValueCache::BlockTables(const std::vector<size_t>& sequence_ids) const {
  return nullptr;  // TODO: Implement this function.
  // size_t max_blocks = 0;
  // for (const auto& sequence_id : sequence_ids) {
  //   auto block_table = block_tables_.find(sequence_id);
  //   if (block_table == block_tables_.end()) {
  //     throw std::runtime_error("Given sequence id " + std::to_string(sequence_id) + " is not found in the cache.");
  //   }
  //   max_blocks = std::max(max_blocks, block_table->second->blocks.size());
  // }

  // std::vector<int64_t> shape = {static_cast<int64_t>(sequence_ids.size()), static_cast<int64_t>(max_blocks)};
  // auto block_tables_value = OrtValue::CreateTensor(*cpu_allocator_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  // auto* block_tables_data = block_tables_value->GetTensorMutableData<int32_t>();

  // constexpr int32_t block_tables_pad_value = -1;

  // for (size_t i = 0; i < sequence_ids.size(); ++i) {
  //   auto block_table = block_tables_.find(sequence_ids[i]);
  //   for (size_t j = 0; j < block_table->second->blocks.size(); ++j) {
  //     block_tables_data[i * max_blocks + j] = block_table->second->blocks[j]->Id();
  //   }
  //   for (size_t j = block_table->second->blocks.size(); j < max_blocks; ++j) {
  //     block_tables_data[i * max_blocks + j] = block_tables_pad_value;
  //   }
  // }

  // return block_tables_value;
}

std::unique_ptr<OrtValue> PagedKeyValueCache::SlotMapping(const std::vector<size_t>& sequence_ids) const {
  return nullptr;
  // size_t num_tokens = 0U;
  // for (const auto& sequence_id : sequence_ids) {
  //   auto block_table = block_tables_.find(sequence_id);
  //   if (block_table == block_tables_.end()) {
  //     throw std::runtime_error("Given sequence id " + std::to_string(sequence_id) + " is not found in the cache.");
  //   }
  //   for (const auto& block : block_table->second->blocks) {
  //     num_tokens += block->NumOccupiedSlots();
  //   }
  // }
  // std::vector<int64_t> shape = {static_cast<int64_t>(num_tokens)};
  // auto slot_mapping_value = OrtValue::CreateTensor(*cpu_allocator_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  // auto* slot_mapping_data = slot_mapping_value->GetTensorMutableData<int32_t>();
  // for (size_t i = 0; i < sequence_ids.size(); ++i) {
  //   auto block_table = block_tables_.find(sequence_ids[i]);
  //   for (size_t i = 0, block_id = 0; block_id < block_table->second->blocks.size(); ++block_id) {
  //     const auto slot_ids = block_table->second->blocks[block_id]->SlotIds();
  //     for (size_t j = 0; j < slot_ids.size(); ++j, ++i) {
  //       slot_mapping_data[i] = slot_ids[j];
  //     }
  //   }
  // }
  // return slot_mapping_value;
}

}  // namespace Generators
