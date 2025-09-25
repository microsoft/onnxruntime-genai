// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cache_manager.h"

namespace Generators {

namespace {

size_t ComputeNumBlocks(std::shared_ptr<Model> model) {
  if (model->config_->engine.dynamic_batching->num_blocks.has_value()) {
    return *model->config_->engine.dynamic_batching->num_blocks;
  }

  const auto dtype_size = Ort::SizeOf(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  size_t free_bytes, total_bytes;
  model->p_device_kvcache_->GetAvailableMemory(free_bytes, total_bytes);

  constexpr float memory_fragmentation_factor = 0.9f;
  constexpr size_t num_caches_per_layer = 2;  // 2 for key and value caches

  // Use the free memory to compute the number of blocks needed to achieve the given gpu_utilization_factor.
  return static_cast<size_t>(free_bytes *
                             memory_fragmentation_factor *
                             *model->config_->engine.dynamic_batching->gpu_utilization_factor) /
         (model->config_->engine.dynamic_batching->block_size *
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
                                                   static_cast<int64_t>(model->config_->engine.dynamic_batching->block_size),
                                                   static_cast<int64_t>(model->config_->model.decoder.num_key_value_heads),
                                                   static_cast<int64_t>(model->config_->model.decoder.head_size)};
  const auto dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  for (size_t i = 0; i < model->config_->model.decoder.num_hidden_layers; ++i) {
    cache_.push_back(LayerCache{
        OrtValue::CreateTensor(model->p_device_kvcache_->GetAllocator(), cache_shape_per_layer, dtype),      // Key cache
        OrtValue::CreateTensor(model->p_device_kvcache_->GetAllocator(), cache_shape_per_layer, dtype),      // Value cache
        ComposeKeyValueName(model->config_->model.decoder.inputs.past_key_names, static_cast<int>(i)),       // Key cache name
        ComposeKeyValueName(model->config_->model.decoder.inputs.past_value_names, static_cast<int>(i)),     // Value cache name
        ComposeKeyValueName(model->config_->model.decoder.outputs.present_key_names, static_cast<int>(i)),   // Key cache output name
        ComposeKeyValueName(model->config_->model.decoder.outputs.present_value_names, static_cast<int>(i))  // Value cache output name
    });
  }
  block_pool_ = std::make_unique<BlockPool>(model->config_->engine.dynamic_batching->block_size, num_blocks);
}

bool PagedKeyValueCache::CanAdd(std::shared_ptr<Request> request) const {
  return block_pool_->AvailableBlocks() > block_pool_->BlocksNeeded(request->UnprocessedTokens().size());
}

void PagedKeyValueCache::Add(std::shared_ptr<Request> request) {
  if (!CanAdd(request)) {
    throw std::runtime_error("Not enough free blocks available to serve the request.");
  }

  auto allocated_blocks = block_pool_->AllocateBlocks(request->UnprocessedTokens().size());
  block_tables_.emplace_back(BlockTable{request, std::move(allocated_blocks)});
}

bool PagedKeyValueCache::CanAppendTokens(std::shared_ptr<Request> request) const {
  const auto block_table_it = std::find_if(block_tables_.begin(), block_tables_.end(),
                                           [&request](const BlockTable& block_table) {
                                             return block_table.request == request;
                                           });
  if (block_table_it == block_tables_.end()) {
    throw std::runtime_error("Given request is not found in the cache.");
  }

  const size_t num_required_slots = request->UnprocessedTokens().size();
  const size_t num_slots_available = block_table_it->blocks.back()->EmptySlots() +
                                     block_pool_->AvailableBlocks() * block_table_it->blocks.back()->Capacity();

  return num_slots_available >= num_required_slots;
}

void PagedKeyValueCache::AppendTokens(std::shared_ptr<Request> request) {
  if (!CanAppendTokens(request)) {
    throw std::runtime_error("Not enough free slots available to append tokens to the request.");
  }

  const auto block_table_it = std::find_if(block_tables_.begin(), block_tables_.end(),
                                           [&request](const BlockTable& block_table) {
                                             return block_table.request == request;
                                           });
  assert(block_table_it != block_tables_.end());

  size_t num_slots = request->UnprocessedTokens().size();
  if (!block_table_it->blocks.back()->IsFull()) {
    for (size_t i = 0; i < std::min(num_slots, block_table_it->blocks.back()->EmptySlots()); ++i) {
      block_table_it->blocks.back()->AddSlot();
      --num_slots;
    }
  }

  auto allocated_blocks = block_pool_->AllocateBlocks(num_slots);
  std::move(allocated_blocks.begin(), allocated_blocks.end(),
            std::back_inserter(block_table_it->blocks));
}

void PagedKeyValueCache::Remove(std::shared_ptr<Request> request) {
  for (auto request_it = block_tables_.begin(); request_it != block_tables_.end(); ++request_it) {
    if (request_it->request == request) {
      block_pool_->Free(request_it->blocks);
      block_tables_.erase(request_it);
      return;
    }
  }
}

std::vector<std::pair<OrtValue*, OrtValue*>> PagedKeyValueCache::Cache() {
  std::vector<std::pair<OrtValue*, OrtValue*>> cache;
  for (auto& layer_cache : cache_) {
    cache.emplace_back(layer_cache.key_cache.get(), layer_cache.value_cache.get());
  }
  return cache;
}

std::vector<std::pair<const char*, const char*>> PagedKeyValueCache::Names() {
  std::vector<std::pair<const char*, const char*>> names;
  for (const auto& layer_cache : cache_) {
    names.emplace_back(layer_cache.key_cache_name.c_str(), layer_cache.value_cache_name.c_str());
  }
  return names;
}

std::vector<std::pair<const char*, const char*>> PagedKeyValueCache::OutputNames() {
  std::vector<std::pair<const char*, const char*>> output_names;
  for (const auto& layer_cache : cache_) {
    output_names.emplace_back(layer_cache.key_cache_output_name.c_str(), layer_cache.value_cache_output_name.c_str());
  }
  return output_names;
}

std::pair<OrtValue*, const char*> PagedKeyValueCache::BlockTables(const std::vector<std::shared_ptr<Request>>& requests) {
  size_t max_blocks = 0;
  for (auto& block_table : block_tables_) {
    if (std::find(requests.begin(), requests.end(), block_table.request) != requests.end()) {
      max_blocks = std::max(max_blocks, block_table.blocks.size());
    } else {
      throw std::runtime_error("Given request is not found in the cache. Please add it before requesting block tables.");
    }
  }

  std::vector<int64_t> shape = {static_cast<int64_t>(requests.size()), static_cast<int64_t>(max_blocks)};
  block_tables_value_ = OrtValue::CreateTensor(model_->allocator_cpu_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto* block_table_data = block_tables_value_->GetTensorMutableData<int32_t>();

  constexpr int32_t block_tables_pad_value = -1;

  for (auto& block_table : block_tables_) {
    auto it = std::find(requests.begin(), requests.end(), block_table.request);
    if (it == requests.end()) {
      throw std::runtime_error("Given request is not found in the cache. Please add it before requesting block tables.");
    }
    size_t index = std::distance(requests.begin(), it);
    for (size_t j = 0; j < block_table.blocks.size(); ++j) {
      block_table_data[index * max_blocks + j] = static_cast<int32_t>(block_table.blocks[j]->Id());
    }
    for (size_t j = block_table.blocks.size(); j < max_blocks; ++j) {
      block_table_data[index * max_blocks + j] = block_tables_pad_value;
    }
  }

  return {block_tables_value_.get(), model_->config_->model.decoder.inputs.block_table.c_str()};
}

void PagedKeyValueCache::UpdateState(State& state, const std::vector<std::shared_ptr<Request>>& requests) {
  auto cache = Cache();
  auto cache_names = Names();
  auto cache_output_names = OutputNames();

  if (state.inputs_.empty()) {
    // Number of layers * 2 for key and value caches + 1 for block tables
    state.inputs_.resize(cache.size() * 2 + 1);
    state.input_names_.resize(cache.size() * 2 + 1);
    state.outputs_.resize(cache.size() * 2);
    state.output_names_.resize(cache.size() * 2);
  }

  for (size_t layer_idx = 0; layer_idx < cache.size(); ++layer_idx) {
    // Key cache
    state.inputs_[layer_idx * 2] = cache[layer_idx].first;
    state.outputs_[layer_idx * 2] = cache[layer_idx].first;

    // Key cache name
    state.input_names_[layer_idx * 2] = cache_names[layer_idx].first;
    state.output_names_[layer_idx * 2] = cache_output_names[layer_idx].first;

    // Value cache
    state.inputs_[layer_idx * 2 + 1] = cache[layer_idx].second;
    state.outputs_[layer_idx * 2 + 1] = cache[layer_idx].second;

    // Value cache name
    state.input_names_[layer_idx * 2 + 1] = cache_names[layer_idx].second;
    state.output_names_[layer_idx * 2 + 1] = cache_output_names[layer_idx].second;
  }

  auto block_tables = BlockTables(requests);
  state.inputs_.back() = block_tables.first;
  state.input_names_.back() = block_tables.second;
}

}  // namespace Generators
