// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cache_manager.h"

#include <numeric>

namespace Generators {

namespace {

ONNXTensorElementDataType KeyValueCacheType(std::shared_ptr<Model> model) {
  const auto key_name = ComposeKeyValueName(model->config_->model.decoder.inputs.past_key_names, 0);
  return model->session_info_.GetInputDataType(key_name);
}

size_t ComputeNumBlocks(std::shared_ptr<Model> model) {
  if (model->config_->engine.dynamic_batching->num_blocks.has_value()) {
    return *model->config_->engine.dynamic_batching->num_blocks;
  }

  const auto dtype_size = Ort::SizeOf(KeyValueCacheType(model));

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

// Slots reserved for a request but not yet written to.
size_t EmptySlots(const std::vector<std::shared_ptr<Block>>& blocks) {
  return std::accumulate(blocks.begin(), blocks.end(), size_t{0},
                         [](size_t sum, const std::shared_ptr<Block>& block) {
                           return sum + block->EmptySlots();
                         });
}

size_t UsedSlots(const std::vector<std::shared_ptr<Block>>& blocks) {
  return std::accumulate(blocks.begin(), blocks.end(), size_t{0},
                         [](size_t sum, const std::shared_ptr<Block>& block) {
                           return sum + block->Size();
                         });
}

// Number of KV slots the model will have addressed once the pending step has run, i.e. one per
// token whose key and value live in the cache afterwards.
//
// This has to match how VarlenDecoderIO fills `past_sequence_lengths`: the decoder writes the
// unprocessed tokens at absolute positions [past, past + unprocessed), so the cache must own
// `past + unprocessed` slots. CurrentSequenceLength() already counts the unprocessed tokens for
// a prefill request and excludes them for a generation request, hence the branch.
//
// Counting appended tokens instead leaves the cache exactly one slot short on every decode step.
// That stays invisible while the last block still has room, and turns into an out-of-bounds
// block-table entry the moment a sequence length lands on a block boundary.
size_t RequiredSlots(const std::shared_ptr<Request>& request) {
  const size_t sequence_length = request->CurrentSequenceLength();
  return request->IsPrefill() ? sequence_length : sequence_length + request->UnprocessedTokens().size();
}

}  // namespace

PagedKeyValueCache::PagedKeyValueCache(std::shared_ptr<Model> model)
    : model_(model) {
  const auto num_blocks = ComputeNumBlocks(model_);
  const std::vector<int64_t> cache_shape_per_layer{static_cast<int64_t>(num_blocks),
                                                   static_cast<int64_t>(model->config_->engine.dynamic_batching->block_size),
                                                   static_cast<int64_t>(model->config_->model.decoder.num_key_value_heads),
                                                   static_cast<int64_t>(model->config_->model.decoder.head_size)};
  const auto dtype = KeyValueCacheType(model);
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

  graph_capture_ = IsGraphCaptureEnabled(model->config_->model.decoder.session_options);
  if (graph_capture_) {
    const size_t block_size = model->config_->engine.dynamic_batching->block_size;
    max_block_table_rows_ = model->config_->engine.dynamic_batching->max_batch_size;
    // A sequence can never need more blocks than the model's context window, and it can never use
    // more than the pool holds. The buffer is allocated once at that ceiling; individual steps view
    // a smaller [rows, columns] window of it.
    max_block_table_columns_ = std::min<size_t>(
        (static_cast<size_t>(model->config_->model.context_length) + block_size - 1) / block_size,
        num_blocks);
    max_block_table_columns_ = std::max<size_t>(max_block_table_columns_, 1);
    block_tables_tensor_ = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int32_t>);
    block_tables_tensor_->CreateTensor(
        std::vector<int64_t>{static_cast<int64_t>(max_block_table_rows_),
                             static_cast<int64_t>(max_block_table_columns_)},
        /*make_static=*/true);
  }
}

bool PagedKeyValueCache::CanAdd(std::shared_ptr<Request> request) const {
  return block_pool_->AvailableBlocks() >= block_pool_->BlocksNeeded(RequiredSlots(request));
}

void PagedKeyValueCache::Add(std::shared_ptr<Request> request) {
  if (!CanAdd(request)) {
    throw std::runtime_error("Not enough free blocks available to serve the request.");
  }

  // Reserve the blocks the prompt will need, but leave their slots empty. The slots are marked
  // used in AppendTokens() once the tokens are actually written to the cache. Marking them here
  // too would count the prompt twice and force the pool to be sized at roughly twice the
  // capacity it can actually use.
  auto reserved_blocks = block_pool_->ReserveBlocks(RequiredSlots(request));
  block_tables_.emplace_back(BlockTable{request, std::move(reserved_blocks)});
}

bool PagedKeyValueCache::CanAppendTokens(std::shared_ptr<Request> request) const {
  const auto block_table_it = std::find_if(block_tables_.begin(), block_tables_.end(),
                                           [&request](const BlockTable& block_table) {
                                             return block_table.request == request;
                                           });
  if (block_table_it == block_tables_.end()) {
    throw std::runtime_error("Given request is not found in the cache.");
  }

  const size_t required_slots = RequiredSlots(request);
  const size_t used_slots = UsedSlots(block_table_it->blocks);
  if (required_slots <= used_slots) {
    return true;
  }

  const size_t num_slots_available = EmptySlots(block_table_it->blocks) +
                                     block_pool_->AvailableBlocks() * block_pool_->BlockSize();

  return num_slots_available >= required_slots - used_slots;
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

  const size_t required_slots = RequiredSlots(request);
  const size_t used_slots = UsedSlots(block_table_it->blocks);
  if (required_slots <= used_slots) {
    return;
  }
  size_t num_slots = required_slots - used_slots;

  // Consume the slots already reserved for this request before asking the pool for more.
  for (auto& block : block_table_it->blocks) {
    while (num_slots > 0 && !block->IsFull()) {
      block->AddSlot();
      --num_slots;
    }
    if (num_slots == 0) {
      break;
    }
  }

  if (num_slots > 0) {
    auto allocated_blocks = block_pool_->AllocateBlocks(num_slots);
    std::move(allocated_blocks.begin(), allocated_blocks.end(),
              std::back_inserter(block_table_it->blocks));
  }
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

PagedCacheSnapshot PagedKeyValueCache::Snapshot() const {
  PagedCacheSnapshot snapshot;
  snapshot.block_size = block_pool_->BlockSize();
  snapshot.total_blocks = block_pool_->Capacity();
  snapshot.free_blocks = block_pool_->AvailableBlocks();
  snapshot.block_table_columns = block_table_columns_;
  snapshot.requests.reserve(block_tables_.size());
  for (const auto& block_table : block_tables_) {
    RequestBlockSnapshot request_snapshot;
    request_snapshot.request_id = block_table.request.get();
    request_snapshot.block_ids.reserve(block_table.blocks.size());
    for (const auto& block : block_table.blocks) {
      request_snapshot.block_ids.push_back(block->Id());
      request_snapshot.used_slots += block->Size();
      request_snapshot.empty_slots += block->EmptySlots();
    }
    snapshot.requests.push_back(std::move(request_snapshot));
  }
  return snapshot;
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
  int32_t* block_table_data = nullptr;
  DeviceSpan<int32_t> device_span;
  if (graph_capture_) {
    // Round the column count up to a power of two so that a run producing steadily longer sequences
    // settles on a handful of distinct shapes instead of a new one every time a block is appended.
    // Each distinct shape is captured under its own annotation id.
    size_t columns = 8;
    while (columns < max_blocks) {
      columns *= 2;
    }
    block_table_columns_ = std::min(columns, max_block_table_columns_);
    if (max_blocks > block_table_columns_ || requests.size() > max_block_table_rows_) {
      throw std::runtime_error("Block table exceeds the capacity reserved for graph capture.");
    }
    max_blocks = block_table_columns_;
    shape[1] = static_cast<int64_t>(max_blocks);
    // Re-creating the view keeps the same underlying buffer, so the address baked into a captured
    // graph stays valid while the shape tracks the current bucket.
    block_tables_tensor_->CreateTensor(shape, /*make_static=*/true);
    device_span = block_tables_tensor_->GetDeviceSpan<int32_t>();
    block_table_data = device_span.CpuSpan().data();
  } else {
    block_table_columns_ = max_blocks;
    block_tables_value_ = OrtValue::CreateTensor(model_->allocator_cpu_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    block_table_data = block_tables_value_->GetTensorMutableData<int32_t>();
  }

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

  if (graph_capture_) {
    device_span.CopyCpuToDevice();
    return {block_tables_tensor_->GetOrtTensor(), model_->config_->model.decoder.inputs.block_table.c_str()};
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
