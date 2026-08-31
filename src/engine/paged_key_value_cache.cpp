// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cache_manager.h"

#include <numeric>
#include <set>

#include "sequence_positions.h"
#include "window_ring.h"

namespace Generators {

namespace {

using StateGroup = Config::Model::Decoder::StateGroup;
using StateGroupKind = Config::Model::Decoder::StateGroupKind;

StateGroup ResolvePagedKeyValueGroup(const Config::Model::Decoder& decoder) {
  if (!decoder.state_groups) {
    StateGroup group;
    group.kind = StateGroupKind::PagedKeyValue;
    group.layer_ids.reserve(decoder.num_hidden_layers);
    for (int layer_id = 0; layer_id < decoder.num_hidden_layers; ++layer_id) {
      group.layer_ids.push_back(layer_id);
    }
    if (group.layer_ids.empty()) {
      throw std::runtime_error(
          "Dynamic batching requires at least one paged_kv decoder layer");
    }
    return group;
  }

  const StateGroup* paged_group = nullptr;
  for (const auto& group : *decoder.state_groups) {
    if (group.kind != StateGroupKind::PagedKeyValue) {
      continue;
    }
    if (paged_group) {
      throw std::runtime_error(
          "Dynamic batching supports only one paged_kv decoder state group");
    }
    paged_group = &group;
  }
  if (!paged_group) {
    throw std::runtime_error(
        "Dynamic batching requires one paged_kv decoder state group");
  }
  if (paged_group->layer_ids.empty()) {
    throw std::runtime_error(
        "Dynamic batching requires a non-empty paged_kv decoder state group");
  }
  return *paged_group;
}

ONNXTensorElementDataType KeyValueCacheType(const std::shared_ptr<Model>& model,
                                            const StateGroup& paged_group) {
  const auto key_name = ComposeKeyValueName(
      model->config_->model.decoder.inputs.past_key_names,
      paged_group.layer_ids.front());
  return model->session_info_.GetInputDataType(key_name);
}

// Layers whose KV cache is a ring sized to the sliding window instead of to the context length.
// A model qualifies only if it was built with a second block table for them; without it every
// layer reads the same table and the ring would be indexed as if it were a linear cache.
std::set<int> WindowedLayers(const std::shared_ptr<Model>& model,
                             const StateGroup& paged_group) {
  const auto& decoder = model->config_->model.decoder;
  if (!decoder.sliding_window.has_value() ||
      decoder.inputs.block_table_windowed.empty() ||
      !model->session_info_.HasInput(decoder.inputs.block_table_windowed)) {
    return {};
  }

  const std::set<int> paged_layers{
      paged_group.layer_ids.begin(), paged_group.layer_ids.end()};
  std::set<int> layers;
  for (int layer : decoder.sliding_window->layers) {
    if (layer < 0 || static_cast<size_t>(layer) >= decoder.num_hidden_layers) {
      throw std::runtime_error("Sliding-window layer index is outside the decoder layer range.");
    }
    if (paged_layers.count(layer) == 0) {
      throw std::runtime_error(
          "Every sliding-window layer must belong to the paged_kv decoder state group.");
    }
    layers.insert(layer);
  }
  return layers;
}

// Bytes one block of one layer occupies across the key and the value cache.
size_t BytesPerBlock(const std::shared_ptr<Model>& model,
                     ONNXTensorElementDataType dtype) {
  constexpr size_t num_caches_per_layer = 2;  // key and value
  return model->config_->engine.dynamic_batching->block_size *
         model->config_->model.decoder.num_key_value_heads *
         model->config_->model.decoder.head_size *
         Ort::SizeOf(dtype) *
         num_caches_per_layer;
}

// Blocks per layer for the layers that keep the whole sequence. The windowed layers are budgeted
// separately and their (small, fixed) cost is taken off the top, so freeing them up is what lets
// the full-attention layers hold more of the sequence in the same memory.
size_t ComputeNumBlocks(std::shared_ptr<Model> model,
                        size_t full_layer_count,
                        size_t windowed_bytes,
                        ONNXTensorElementDataType dtype) {
  if (model->config_->engine.dynamic_batching->num_blocks.has_value()) {
    return *model->config_->engine.dynamic_batching->num_blocks;
  }

  size_t free_bytes, total_bytes;
  model->p_device_kvcache_->GetAvailableMemory(free_bytes, total_bytes);

  return ComputePagedBlockCapacity(
      free_bytes,
      *model->config_->engine.dynamic_batching->gpu_utilization_factor,
      windowed_bytes,
      model->config_->engine.dynamic_batching->block_size,
      model->config_->model.decoder.num_key_value_heads,
      model->config_->model.decoder.head_size,
      full_layer_count,
      Ort::SizeOf(dtype));
}

size_t UsedSlots(const std::vector<std::shared_ptr<Block>>& blocks) {
  return std::accumulate(blocks.begin(), blocks.end(), size_t{0},
                         [](size_t sum, const std::shared_ptr<Block>& block) {
                           return sum + block->Size();
                         });
}

// Once the pending step completes, every token currently in the sequence has a KV slot.
size_t RequiredSlots(const std::shared_ptr<Request>& request) {
  return static_cast<size_t>(request->CurrentSequenceLength());
}

}  // namespace

size_t ComputePagedBlockCapacity(size_t available_memory_bytes,
                                 float gpu_utilization_factor,
                                 size_t reserved_memory_bytes,
                                 size_t block_size,
                                 size_t num_key_value_heads,
                                 size_t head_size,
                                 size_t full_layer_count,
                                 size_t element_size) {
  if (block_size == 0 || num_key_value_heads == 0 || head_size == 0 ||
      full_layer_count == 0 || element_size == 0) {
    throw std::invalid_argument(
        "Paged cache capacity dimensions must be greater than zero");
  }
  constexpr float memory_fragmentation_factor = 0.9f;
  const auto budget = static_cast<size_t>(
      available_memory_bytes * memory_fragmentation_factor * gpu_utilization_factor);
  if (budget <= reserved_memory_bytes) {
    throw std::runtime_error(
        "The key-value cache budget is too small to hold the reserved decoder state.");
  }

  constexpr size_t num_caches_per_layer = 2;
  return (budget - reserved_memory_bytes) /
         (block_size *
          num_key_value_heads *
          head_size *
          full_layer_count *
          element_size *
          num_caches_per_layer);
}

PagedKeyValueCache::PagedKeyValueCache(std::shared_ptr<Model> model)
    : model_(model) {
  const auto& decoder = model->config_->model.decoder;
  const size_t block_size = model->config_->engine.dynamic_batching->block_size;
  const size_t max_batch_size = model->config_->engine.dynamic_batching->max_batch_size;
  const auto paged_group = ResolvePagedKeyValueGroup(decoder);
  const auto dtype = KeyValueCacheType(model_, paged_group);

  const auto windowed = WindowedLayers(model, paged_group);
  size_t num_window_blocks = 0;
  if (!windowed.empty()) {
    if (decoder.sliding_window->window_size <= 0) {
      throw std::runtime_error("Sliding-window size must be greater than zero.");
    }
    window_size_ = static_cast<size_t>(decoder.sliding_window->window_size);

    // A step covers positions [past, past + scheduled) and each of those queries reaches back
    // window_size positions, so the ring has to hold scheduled + window_size - 1 positions at
    // once. `scheduled` is bounded by the prefill chunk size, which therefore has to be set: an
    // unchunked prefill would need a ring as long as the prompt and there would be nothing to save.
    const auto& chunk_size = model->config_->search.chunk_size;
    if (!chunk_size.has_value() || *chunk_size == 0) {
      throw std::runtime_error(
          "This model holds its sliding-window layers in a ring of blocks, which only works if the "
          "prefill is chunked. Set search.chunk_size in genai_config.json.");
    }

    window_live_span_ = WindowLiveSpan(*chunk_size, window_size_);
    window_ring_blocks_ = WindowRingBlocks(*chunk_size, window_size_, block_size);
    num_window_blocks = window_ring_blocks_ * max_batch_size;
  }

  const size_t num_full_layers = paged_group.layer_ids.size() - windowed.size();
  if (num_full_layers == 0) {
    throw std::runtime_error("A paged model needs at least one layer that keeps the whole sequence.");
  }
  const auto num_blocks = ComputeNumBlocks(model_, num_full_layers,
                                           num_window_blocks * windowed.size() *
                                               BytesPerBlock(model, dtype),
                                           dtype);

  for (const int layer_id : paged_group.layer_ids) {
    const auto blocks = windowed.count(layer_id) != 0 ? num_window_blocks : num_blocks;
    const std::vector<int64_t> cache_shape_per_layer{static_cast<int64_t>(blocks),
                                                     static_cast<int64_t>(block_size),
                                                     static_cast<int64_t>(decoder.num_key_value_heads),
                                                     static_cast<int64_t>(decoder.head_size)};
    cache_.push_back(LayerCache{
        OrtValue::CreateTensor(model->p_device_kvcache_->GetAllocator(), cache_shape_per_layer, dtype),  // Key cache
        OrtValue::CreateTensor(model->p_device_kvcache_->GetAllocator(), cache_shape_per_layer, dtype),  // Value cache
        ComposeKeyValueName(decoder.inputs.past_key_names, layer_id),
        ComposeKeyValueName(decoder.inputs.past_value_names, layer_id),
        ComposeKeyValueName(decoder.outputs.present_key_names, layer_id),
        ComposeKeyValueName(decoder.outputs.present_value_names, layer_id)});
  }
  block_pool_ = std::make_unique<BlockPool>(block_size, num_blocks);
  if (Windowed()) {
    window_block_pool_ = std::make_unique<BlockPool>(block_size, num_window_blocks);
  }

  max_batch_size_ = max_batch_size;
  block_table_index_ = std::make_unique<RequestIndex>(max_batch_size_);
  graph_capture_ = IsGraphCaptureEnabled(decoder.session_options);
  if (graph_capture_) {
    max_block_table_rows_ = max_batch_size_;
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
    if (Windowed()) {
      // Same shape as the full table: the operator indexes both with the token's true position,
      // only the ids repeat, so the two tables also share a capture bucket.
      window_block_tables_tensor_ = std::make_unique<Tensor>(model->p_device_inputs_, Ort::TypeToTensorType<int32_t>);
      window_block_tables_tensor_->CreateTensor(
          std::vector<int64_t>{static_cast<int64_t>(max_block_table_rows_),
                               static_cast<int64_t>(max_block_table_columns_)},
          /*make_static=*/true);
    }
  }
}

bool PagedKeyValueCache::CanAdd(std::shared_ptr<Request> request) const {
  if (block_pool_->AvailableBlocks() < block_pool_->BlocksNeeded(RequiredSlots(request))) {
    return false;
  }
  // The ring is a fixed cost per request, so admission is the only place it can fail.
  return !Windowed() || window_block_pool_->AvailableBlocks() >= window_ring_blocks_;
}

void PagedKeyValueCache::Add(std::shared_ptr<Request> request) {
  if (!CanAdd(request)) {
    throw std::runtime_error("Not enough free blocks available to serve the request.");
  }
  if (block_table_index_->Find(request.get()) ||
      block_table_index_->Size() >= block_table_index_->Capacity()) {
    throw std::logic_error(
        "Paged cache request index cannot admit this request.");
  }

  // Reserve the blocks the prompt will need, but leave their slots empty. The slots are marked
  // used in AppendTokens() once the tokens are actually written to the cache. Marking them here
  // too would count the prompt twice and force the pool to be sized at roughly twice the
  // capacity it can actually use.
  auto reserved_blocks = block_pool_->ReserveBlocks(RequiredSlots(request));

  // The ring, in contrast, is claimed in full up front. Its slots are written over and over as the
  // window slides, so there is nothing to grow later and no meaningful notion of a used slot.
  std::vector<std::shared_ptr<Block>> window_blocks;
  if (Windowed()) {
    window_blocks = window_block_pool_->AllocateBlocks(window_ring_blocks_ * window_block_pool_->BlockSize());
  }

  block_tables_.emplace_back(
      PagedCacheBlockTable{request.get(), 0, std::move(reserved_blocks), std::move(window_blocks)});
  if (!block_table_index_->Insert(
          request.get(), block_tables_.size() - 1)) {
    std::terminate();
  }
}

bool PagedKeyValueCache::CanAppendTokens(std::shared_ptr<Request> request) const {
  const auto block_table_it = std::find_if(block_tables_.begin(), block_tables_.end(),
                                           [&request](const PagedCacheBlockTable& block_table) {
                                             return block_table.request_id_ == request.get();
                                           });
  if (block_table_it == block_tables_.end()) {
    throw std::runtime_error("Given request is not found in the cache.");
  }

  const size_t required_slots = RequiredSlots(request);
  const size_t used_slots = block_table_it->committed_slots_;
  if (required_slots <= used_slots) {
    return true;
  }

  const size_t num_slots_available =
      block_table_it->blocks_.size() * block_pool_->BlockSize() - used_slots +
      block_pool_->AvailableBlocks() * block_pool_->BlockSize();

  return num_slots_available >= required_slots - used_slots;
}

void PagedKeyValueCache::AppendTokens(std::shared_ptr<Request> request) {
  if (!CanAppendTokens(request)) {
    throw std::runtime_error("Not enough free slots available to append tokens to the request.");
  }

  if (Windowed() && WindowLiveSpan(request->UnprocessedTokens().size(), window_size_) > window_live_span_) {
    // More tokens in one step than the ring was sized for, so the oldest positions this step still
    // has to attend to would be overwritten by the newest ones before they are read.
    throw std::runtime_error(
        "A step carries more tokens than the sliding-window ring can hold together with the window. "
        "Lower search.chunk_size, or rebuild the model with a larger one.");
  }

  const auto block_table_it = std::find_if(block_tables_.begin(), block_tables_.end(),
                                           [&request](const PagedCacheBlockTable& block_table) {
                                             return block_table.request_id_ == request.get();
                                           });
  assert(block_table_it != block_tables_.end());

  const size_t required_slots = RequiredSlots(request);
  const size_t used_slots = block_table_it->committed_slots_;
  assert(UsedSlots(block_table_it->blocks_) == used_slots);
  assert(used_slots == static_cast<size_t>(request->ProcessedSequenceLength()));
  if (required_slots <= used_slots) {
    return;
  }
  const size_t growth = required_slots - used_slots;
  const size_t committed_capacity =
      block_table_it->blocks_.size() * block_pool_->BlockSize();
  const size_t tail_capacity = committed_capacity - used_slots;
  const size_t new_slots = growth > tail_capacity ? growth - tail_capacity : 0;
  const size_t new_block_count = block_pool_->BlocksNeeded(new_slots);

  // Complete every fallible allocation before changing existing block occupancy.
  block_table_it->blocks_.reserve(
      block_table_it->blocks_.size() + new_block_count);
  auto allocated_blocks = block_pool_->AllocateBlocks(new_slots);

  size_t num_slots = growth - new_slots;
  size_t block_index = used_slots / block_pool_->BlockSize();
  while (num_slots > 0 && block_index < block_table_it->blocks_.size()) {
    auto& block = block_table_it->blocks_[block_index++];
    const size_t slots = std::min(num_slots, block->EmptySlots());
    block->AddSlots(slots);
    num_slots -= slots;
  }
  std::move(allocated_blocks.begin(), allocated_blocks.end(),
            std::back_inserter(block_table_it->blocks_));
  block_table_it->committed_slots_ = required_slots;
  ++block_table_it->mutation_generation_;
  block_pool_->RecordOccupancyMutation();
}

void PagedKeyValueCache::Remove(std::shared_ptr<Request> request) {
  RemovePagedCacheBlockTable(*block_pool_, window_block_pool_.get(),
                             block_tables_, request.get());
  RebuildBlockTableIndex();
}

void PagedKeyValueCache::ValidateRemove(const void* request_id) const {
  ValidateRemovePagedCacheBlockTable(
      *block_pool_, window_block_pool_.get(), block_tables_, request_id);
}

void PagedKeyValueCache::RemoveValidated(const void* request_id) noexcept {
  RemoveValidatedPagedCacheBlockTable(
      *block_pool_, window_block_pool_.get(), block_tables_, request_id);
  RebuildBlockTableIndex();
}

bool PagedKeyValueCache::OwnsRequest(
    const void* request_id) const noexcept {
  const auto index = block_table_index_->Find(request_id);
  return index && *index < block_tables_.size() &&
         block_tables_[*index].request_id_ == request_id;
}

size_t PagedKeyValueCache::CommittedSlots(
    const void* request_id) const {
  const auto index = block_table_index_->Find(request_id);
  if (!index || *index >= block_tables_.size() ||
      block_tables_[*index].request_id_ != request_id) {
    throw StepPlanningConsistencyError(
        "A cache resident has no committed paged cache table.");
  }
  return block_tables_[*index].committed_slots_;
}

PagedCacheReservation PagedKeyValueCache::Reserve(std::span<const PagedCacheReservationRequest> requests) {
  return PagedCacheReservation{*block_pool_, block_tables_, requests,
                               window_block_pool_.get(), window_ring_blocks_,
                               block_table_index_.get()};
}

void PagedKeyValueCache::RebuildBlockTableIndex() noexcept {
  block_table_index_->Clear();
  for (size_t index = 0; index < block_tables_.size(); ++index) {
    if (!block_table_index_->Insert(
            block_tables_[index].request_id_, index)) {
      std::terminate();
    }
  }
}

StepPlanningResult PagedKeyValueCache::PlanStepResources(StepPlan& plan) const {
  const size_t committed_request_count = block_tables_.size();
  if (committed_request_count > max_batch_size_) {
    throw StepPlanningConsistencyError(
        "Committed paged cache requests exceed the configured batch size.");
  }
  const size_t scheduled_request_limit =
      plan.scheduled_request_limit == 0
          ? max_batch_size_
          : plan.scheduled_request_limit;
  if (scheduled_request_limit > max_batch_size_) {
    throw StepPlanningConsistencyError(
        "Step plan request limit exceeds the configured batch size.");
  }

  const size_t available_blocks = block_pool_->AvailableBlocks();
  const size_t available_window_blocks =
      Windowed() ? window_block_pool_->AvailableBlocks() : 0;
  size_t planned_blocks = 0;
  size_t selected_requests = 0;
  size_t selected_new_requests = 0;
  size_t max_blocks_per_request = 0;
  bool capacity_deferred = false;
  const void* unserviceable_request_id = nullptr;

  struct CacheGrowth {
    size_t proposed_blocks{};
    size_t new_blocks{};
  };
  // Blocks the request has to own for this step. A chunked prefill is planned one chunk at a time,
  // but the blocks are taken for the whole sequence: admitting a prompt on the strength of its
  // first chunk and then losing the rest of the pool to another request would stall it part way
  // through, holding the blocks it already took. PagedCacheReservation reserves the same blocks.
  const auto calculate_growth = [&](const RequestStepPlan& entry,
                                    const PagedCacheBlockTable* table) {
    const size_t committed_slots = table ? table->committed_slots_ : 0;
    if (entry.target_cache_slots < committed_slots) {
      throw StepPlanningConsistencyError(
          "Step plan target precedes the committed cache boundary.");
    }

    const size_t reserved_slots =
        std::max(entry.whole_sequence_cache_slots, entry.target_cache_slots);
    const size_t committed_blocks = table ? table->blocks_.size() : 0;
    const size_t committed_capacity = committed_blocks * block_pool_->BlockSize();
    const size_t additional_slots =
        reserved_slots > committed_capacity ? reserved_slots - committed_capacity : 0;
    const size_t new_blocks = block_pool_->BlocksNeeded(additional_slots);
    return CacheGrowth{committed_blocks + new_blocks, new_blocks};
  };
  const auto permanently_unserviceable = [&](const CacheGrowth& growth) {
    return growth.proposed_blocks > block_pool_->Capacity() ||
           (graph_capture_ &&
            growth.proposed_blocks > max_block_table_columns_);
  };
  const auto select = [&](size_t request_index,
                          const CacheGrowth& growth) {
    // Compact selected entries in place. Requests skipped for temporary capacity pressure remain
    // pending with their committed block tables untouched and can be reconsidered next Step().
    planned_blocks += growth.new_blocks;
    max_blocks_per_request =
        std::max(max_blocks_per_request, growth.proposed_blocks);
    if (selected_requests != request_index) {
      plan.requests[selected_requests] =
          std::move(plan.requests[request_index]);
    }
    ++selected_requests;
  };
  const auto find_table = [this](const void* request_id) {
    const auto index = block_table_index_->Find(request_id);
    return index ? &block_tables_[*index] : nullptr;
  };
  RequestIndex request_ids{plan.requests.size()};
  for (size_t i = 0; i < plan.requests.size(); ++i) {
    auto& candidate = plan.requests[i];
    if (!request_ids.Insert(candidate.request_id, i)) {
      throw StepPlanningConsistencyError(
          "Step plan contains a duplicate request.");
    }

    const auto* table = find_table(candidate.request_id);
    if (candidate.newly_admitted && table) {
      throw StepPlanningConsistencyError(
          "New step plan request already belongs to the paged cache.");
    }
    if (!candidate.newly_admitted && !table) {
      throw StepPlanningConsistencyError(
          "Step plan resident membership does not match the committed cache.");
    }

    auto growth = calculate_growth(candidate, table);
    // Drafts are optional acceleration work. Reduce them until the request fits rather than
    // deferring or rejecting a base decode that can make progress without the full proposal.
    while (candidate.draft_token_count > 0 &&
           (permanently_unserviceable(growth) ||
            planned_blocks + growth.new_blocks > available_blocks)) {
      --candidate.draft_token_count;
      --candidate.unprocessed_token_count;
      --candidate.target_cache_slots;
      growth = calculate_growth(candidate, table);
    }
    if (permanently_unserviceable(growth)) {
      if (!unserviceable_request_id) {
        unserviceable_request_id = candidate.request_id;
      }
      continue;
    }

    if (selected_requests >= scheduled_request_limit ||
        (candidate.newly_admitted &&
         (committed_request_count + selected_new_requests >= max_batch_size_ ||
          (Windowed() &&
           (selected_new_requests + 1) * window_ring_blocks_ > available_window_blocks))) ||
        planned_blocks + growth.new_blocks > available_blocks) {
      capacity_deferred = true;
      continue;
    }

    const bool newly_admitted = candidate.newly_admitted;
    select(i, growth);
    if (newly_admitted) {
      ++selected_new_requests;
    }
  }
  plan.requests.resize(selected_requests);

  size_t block_table_columns = max_blocks_per_request;
  if (graph_capture_) {
    block_table_columns =
        GetGraphBlockTableColumns(max_blocks_per_request, max_block_table_columns_);
  }

  plan.proposed_block_table_columns = block_table_columns;

  if (!plan.requests.empty()) {
    return StepPlanningResult{
        true,
        capacity_deferred,
        unserviceable_request_id,
        {StepOutcomeKind::Committed, plan.transaction_id, nullptr},
    };
  }
  if (unserviceable_request_id) {
    return StepPlanningResult{
        false,
        capacity_deferred,
        unserviceable_request_id,
        {StepOutcomeKind::UnserviceableRequest, plan.transaction_id, unserviceable_request_id},
    };
  }
  if (capacity_deferred) {
    return StepPlanningResult{
        false,
        true,
        nullptr,
        {StepOutcomeKind::CapacityDeferred, plan.transaction_id, nullptr},
    };
  }
  return StepPlanningResult{
      false,
      false,
      nullptr,
      {StepOutcomeKind::NoWork, plan.transaction_id, nullptr},
  };
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
    request_snapshot.request_id = block_table.request_id_;
    request_snapshot.block_ids.reserve(block_table.blocks_.size());
    for (const auto& block : block_table.blocks_) {
      request_snapshot.block_ids.push_back(block->Id());
      request_snapshot.used_slots += block->Size();
      request_snapshot.empty_slots += block->EmptySlots();
    }
    snapshot.requests.push_back(std::move(request_snapshot));
  }
  if (Windowed()) {
    snapshot.window_blocks.total_blocks = window_block_pool_->Capacity();
    snapshot.window_blocks.free_blocks = window_block_pool_->AvailableBlocks();
    snapshot.window_blocks.blocks_per_request = window_ring_blocks_;
    snapshot.window_blocks.requests.reserve(block_tables_.size());
    for (const auto& block_table : block_tables_) {
      RequestBlockSnapshot request_snapshot;
      request_snapshot.request_id = block_table.request_id_;
      request_snapshot.block_ids.reserve(block_table.window_blocks_.size());
      for (const auto& block : block_table.window_blocks_) {
        request_snapshot.block_ids.push_back(block->Id());
      }
      snapshot.window_blocks.requests.push_back(std::move(request_snapshot));
    }
  }
  return snapshot;
}

PagedCacheSnapshot PagedKeyValueCache::Snapshot(
    const PagedCacheReservation& reservation) const {
  auto snapshot = Snapshot();
  snapshot.transaction_reserved_block_ids.reserve(
      reservation.ReservedBlocks().size());
  for (const auto& block : reservation.ReservedBlocks()) {
    snapshot.transaction_reserved_block_ids.push_back(block->Id());
  }
  snapshot.window_blocks.transaction_reserved_block_ids.reserve(
      reservation.ReservedWindowBlocks().size());
  for (const auto& block : reservation.ReservedWindowBlocks()) {
    snapshot.window_blocks.transaction_reserved_block_ids.push_back(block->Id());
  }
  snapshot.reservations.reserve(reservation.Deltas().size());
  for (const auto& delta : reservation.Deltas()) {
    RequestReservationSnapshot request_reservation;
    request_reservation.request_id = delta.request_id;
    request_reservation.committed_slots = delta.committed_slots;
    request_reservation.target_slots = delta.target_slots;
    request_reservation.reserved_block_ids.reserve(
        delta.reserved_block_count);
    for (size_t i = 0; i < delta.reserved_block_count; ++i) {
      request_reservation.reserved_block_ids.push_back(
          reservation.ReservedBlocks()[delta.reserved_block_offset + i]->Id());
    }
    snapshot.reservations.push_back(std::move(request_reservation));
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

void PagedKeyValueCache::FillBlockTables(const std::vector<std::shared_ptr<Request>>& requests,
                                         bool windowed, int32_t* data, size_t columns) {
  constexpr int32_t block_tables_pad_value = -1;

  for (auto& block_table : block_tables_) {
    auto it = std::find_if(requests.begin(), requests.end(),
                           [&block_table](const std::shared_ptr<Request>& request) {
                             return request.get() == block_table.request_id_;
                           });
    if (it == requests.end()) {
      throw std::runtime_error("Given request is not found in the cache. Please add it before requesting block tables.");
    }
    size_t index = std::distance(requests.begin(), it);

    if (windowed) {
      // Repeat the ring across every column. There is nothing to pad: whichever column the
      // operator reaches for, the ring block it names is the one holding that position.
      for (size_t j = 0; j < columns; ++j) {
        const auto& block = block_table.window_blocks_[WindowRingColumn(j, window_ring_blocks_)];
        data[index * columns + j] = static_cast<int32_t>(block->Id());
      }
      continue;
    }

    for (size_t j = 0; j < block_table.blocks_.size(); ++j) {
      data[index * columns + j] = static_cast<int32_t>(block_table.blocks_[j]->Id());
    }
    for (size_t j = block_table.blocks_.size(); j < columns; ++j) {
      data[index * columns + j] = block_tables_pad_value;
    }
  }
}

std::pair<OrtValue*, const char*> PagedKeyValueCache::BlockTables(const std::vector<std::shared_ptr<Request>>& requests) {
  size_t max_blocks = 0;
  for (auto& block_table : block_tables_) {
    if (std::find_if(requests.begin(), requests.end(),
                     [&block_table](const std::shared_ptr<Request>& request) {
                       return request.get() == block_table.request_id_;
                     }) != requests.end()) {
      max_blocks = std::max(max_blocks, block_table.blocks_.size());
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
    block_table_columns_ =
        GetGraphBlockTableColumns(max_blocks, max_block_table_columns_);
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

  FillBlockTables(requests, /*windowed=*/false, block_table_data, max_blocks);

  if (graph_capture_) {
    device_span.CopyCpuToDevice();
    return {block_tables_tensor_->GetOrtTensor(), model_->config_->model.decoder.inputs.block_table.c_str()};
  }

  return {block_tables_value_.get(), model_->config_->model.decoder.inputs.block_table.c_str()};
}

std::pair<OrtValue*, const char*> PagedKeyValueCache::BlockTables(
    const std::vector<std::shared_ptr<Request>>& requests,
    const PagedCacheReservation& reservation,
    size_t columns) {
  if (columns < reservation.RequiredBlockTableColumns()) {
    throw std::runtime_error("Proposed block table is narrower than the cache reservation.");
  }

  std::vector<int64_t> shape{
      static_cast<int64_t>(requests.size()),
      static_cast<int64_t>(columns),
  };
  int32_t* block_table_data = nullptr;
  DeviceSpan<int32_t> device_span;
  if (graph_capture_) {
    if (columns > max_block_table_columns_ ||
        requests.size() > max_block_table_rows_) {
      throw std::runtime_error("Proposed block table exceeds the graph capture buffer.");
    }
    block_tables_tensor_->CreateTensor(shape, /*make_static=*/true);
    device_span = block_tables_tensor_->GetDeviceSpan<int32_t>();
    block_table_data = device_span.CpuSpan().data();
  } else {
    block_tables_value_ =
        OrtValue::CreateTensor(model_->allocator_cpu_, shape,
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    block_table_data = block_tables_value_->GetTensorMutableData<int32_t>();
  }

  std::vector<const void*> request_ids;
  request_ids.reserve(requests.size());
  for (const auto& request : requests) {
    request_ids.push_back(request.get());
  }
  reservation.FillBlockTable(
      request_ids,
      columns,
      std::span<int32_t>{block_table_data, requests.size() * columns});
  block_table_columns_ = columns;

  if (graph_capture_) {
    device_span.CopyCpuToDevice();
    return {block_tables_tensor_->GetOrtTensor(),
            model_->config_->model.decoder.inputs.block_table.c_str()};
  }
  return {block_tables_value_.get(),
          model_->config_->model.decoder.inputs.block_table.c_str()};
}

std::pair<OrtValue*, const char*> PagedKeyValueCache::WindowBlockTables(const std::vector<std::shared_ptr<Request>>& requests) {
  // Must run after BlockTables(), which is what settles the column count for this step. Both
  // tables carry the same columns so that a position resolves through the same index in each.
  const size_t columns = block_table_columns_;
  const std::vector<int64_t> shape = {static_cast<int64_t>(requests.size()), static_cast<int64_t>(columns)};

  int32_t* data = nullptr;
  DeviceSpan<int32_t> device_span;
  if (graph_capture_) {
    window_block_tables_tensor_->CreateTensor(shape, /*make_static=*/true);
    device_span = window_block_tables_tensor_->GetDeviceSpan<int32_t>();
    data = device_span.CpuSpan().data();
  } else {
    window_block_tables_value_ = OrtValue::CreateTensor(model_->allocator_cpu_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    data = window_block_tables_value_->GetTensorMutableData<int32_t>();
  }

  FillBlockTables(requests, /*windowed=*/true, data, columns);

  if (graph_capture_) {
    device_span.CopyCpuToDevice();
    return {window_block_tables_tensor_->GetOrtTensor(), model_->config_->model.decoder.inputs.block_table_windowed.c_str()};
  }

  return {window_block_tables_value_.get(), model_->config_->model.decoder.inputs.block_table_windowed.c_str()};
}

std::pair<OrtValue*, const char*> PagedKeyValueCache::WindowBlockTables(
    const std::vector<std::shared_ptr<Request>>& requests,
    const PagedCacheReservation& reservation,
    size_t columns) {
  const std::vector<int64_t> shape = {static_cast<int64_t>(requests.size()),
                                      static_cast<int64_t>(columns)};
  int32_t* data = nullptr;
  DeviceSpan<int32_t> device_span;
  if (graph_capture_) {
    window_block_tables_tensor_->CreateTensor(shape, /*make_static=*/true);
    device_span = window_block_tables_tensor_->GetDeviceSpan<int32_t>();
    data = device_span.CpuSpan().data();
  } else {
    window_block_tables_value_ = OrtValue::CreateTensor(
        model_->allocator_cpu_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    data = window_block_tables_value_->GetTensorMutableData<int32_t>();
  }

  std::vector<const void*> request_ids;
  request_ids.reserve(requests.size());
  for (const auto& request : requests) {
    request_ids.push_back(request.get());
  }
  reservation.FillWindowBlockTable(
      request_ids, columns,
      std::span<int32_t>{data, requests.size() * columns});

  if (graph_capture_) {
    device_span.CopyCpuToDevice();
    return {window_block_tables_tensor_->GetOrtTensor(),
            model_->config_->model.decoder.inputs.block_table_windowed.c_str()};
  }
  return {window_block_tables_value_.get(),
          model_->config_->model.decoder.inputs.block_table_windowed.c_str()};
}

void PagedKeyValueCache::BindCache(State& state) {
  auto cache = Cache();
  auto cache_names = Names();
  auto cache_output_names = OutputNames();

  const size_t num_block_tables = Windowed() ? 2 : 1;
  if (state.inputs_.empty()) {
    // Number of layers * 2 for key and value caches + one entry per block table
    state.inputs_.resize(cache.size() * 2 + num_block_tables);
    state.input_names_.resize(cache.size() * 2 + num_block_tables);
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
}

void PagedKeyValueCache::UpdateState(State& state, const std::vector<std::shared_ptr<Request>>& requests) {
  BindCache(state);
  auto block_tables = BlockTables(requests);
  state.inputs_[cache_.size() * 2] = block_tables.first;
  state.input_names_[cache_.size() * 2] = block_tables.second;

  if (Windowed()) {
    // Depends on the column count BlockTables() just settled, so it has to follow it.
    auto window_block_tables = WindowBlockTables(requests);
    state.inputs_[cache_.size() * 2 + 1] = window_block_tables.first;
    state.input_names_[cache_.size() * 2 + 1] = window_block_tables.second;
  }
}

void PagedKeyValueCache::UpdateState(
    State& state,
    const std::vector<std::shared_ptr<Request>>& requests,
    const PagedCacheReservation& reservation,
    size_t columns) {
  BindCache(state);
  auto block_tables = BlockTables(requests, reservation, columns);
  const size_t block_table_index = cache_.size() * 2;
  state.inputs_[block_table_index] = block_tables.first;
  state.input_names_[block_table_index] = block_tables.second;
  if (Windowed()) {
    auto window_block_tables = WindowBlockTables(requests, reservation, columns);
    state.inputs_[block_table_index + 1] = window_block_tables.first;
    state.input_names_[block_table_index + 1] = window_block_tables.second;
  }
}

}  // namespace Generators
