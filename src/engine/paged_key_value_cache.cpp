// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cache_manager.h"

#include <numeric>

#include "sequence_positions.h"

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

PrefixCacheOptions ComposePrefixCacheOptions(const Config::Engine::DynamicBatching& config,
                                             size_t num_blocks) {
  PrefixCacheOptions options;
  options.enabled = config.prefix_caching;
  options.min_match_blocks = std::max<size_t>(config.prefix_cache_min_blocks, 1);
  if (config.prefix_cache_max_blocks.has_value()) {
    options.max_blocks = std::min(*config.prefix_cache_max_blocks, num_blocks);
  } else {
    const float fraction = std::clamp(config.prefix_cache_pool_fraction, 0.0f, 1.0f);
    options.max_blocks = static_cast<size_t>(static_cast<float>(num_blocks) * fraction);
  }
  return options;
}

}  // namespace

bool MakeTailBlockExclusive(PagedCacheBlockTable& table,
                            size_t target_slots,
                            BlockPool& pool,
                            BlockCopier& copier) {
  if (target_slots <= table.committed_slots || pool.BlockSize() == 0) {
    return false;
  }

  const size_t block_index = table.committed_slots / pool.BlockSize();
  if (block_index >= table.blocks.size()) {
    return false;
  }

  auto& block = table.blocks[block_index];
  if (!block->IsShared()) {
    return false;
  }

  auto replacement = pool.ReserveBlocks(pool.BlockSize());
  if (replacement.size() != 1) {
    throw std::runtime_error("Copy-on-write needs exactly one replacement block.");
  }
  copier.CopyBlock(block->Id(), replacement.front()->Id());
  replacement.front()->AddSlots(block->Size());

  auto shared = block;
  table.blocks[block_index] = replacement.front();
  if (table.sealed_blocks > block_index) {
    throw std::logic_error(
        "Copy-on-write reached a block the prefix cache already sealed, which is never written.");
  }
  pool.Free({shared});
  return true;
}

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
  prefix_cache_ = std::make_unique<PrefixCache>(
      *block_pool_,
      ComposePrefixCacheOptions(*model->config_->engine.dynamic_batching, num_blocks));
  block_bytes_per_layer_ = model->config_->engine.dynamic_batching->block_size *
                           model->config_->model.decoder.num_key_value_heads *
                           model->config_->model.decoder.head_size *
                           Ort::SizeOf(dtype);

  max_batch_size_ = model->config_->engine.dynamic_batching->max_batch_size;
  graph_capture_ = IsGraphCaptureEnabled(model->config_->model.decoder.session_options);
  if (graph_capture_) {
    const size_t block_size = model->config_->engine.dynamic_batching->block_size;
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
  block_tables_.emplace_back(PagedCacheBlockTable{request.get(), 0, std::move(reserved_blocks)});
}

bool PagedKeyValueCache::CanAppendTokens(std::shared_ptr<Request> request) const {
  const auto block_table_it = std::find_if(block_tables_.begin(), block_tables_.end(),
                                           [&request](const PagedCacheBlockTable& block_table) {
                                             return block_table.request_id == request.get();
                                           });
  if (block_table_it == block_tables_.end()) {
    throw std::runtime_error("Given request is not found in the cache.");
  }

  const size_t required_slots = RequiredSlots(request);
  const size_t used_slots = block_table_it->committed_slots;
  if (required_slots <= used_slots) {
    return true;
  }

  const size_t num_slots_available =
      block_table_it->blocks.size() * block_pool_->BlockSize() - used_slots +
      block_pool_->AvailableBlocks() * block_pool_->BlockSize();

  return num_slots_available >= required_slots - used_slots;
}

void PagedKeyValueCache::AppendTokens(std::shared_ptr<Request> request) {
  if (!CanAppendTokens(request)) {
    throw std::runtime_error("Not enough free slots available to append tokens to the request.");
  }

  const auto block_table_it = std::find_if(block_tables_.begin(), block_tables_.end(),
                                           [&request](const PagedCacheBlockTable& block_table) {
                                             return block_table.request_id == request.get();
                                           });
  assert(block_table_it != block_tables_.end());

  const size_t required_slots = RequiredSlots(request);
  const size_t used_slots = block_table_it->committed_slots;
  assert(UsedSlots(block_table_it->blocks) == used_slots);
  assert(used_slots == static_cast<size_t>(request->ProcessedSequenceLength()));
  if (required_slots <= used_slots) {
    return;
  }
  size_t num_slots = required_slots - used_slots;

  size_t block_index = used_slots / block_pool_->BlockSize();
  while (num_slots > 0 && block_index < block_table_it->blocks.size()) {
    auto& block = block_table_it->blocks[block_index++];
    const size_t slots = std::min(num_slots, block->EmptySlots());
    block->AddSlots(slots);
    num_slots -= slots;
  }

  if (num_slots > 0) {
    auto allocated_blocks = block_pool_->AllocateBlocks(num_slots);
    std::move(allocated_blocks.begin(), allocated_blocks.end(),
              std::back_inserter(block_table_it->blocks));
  }
  block_table_it->committed_slots = required_slots;
}

void PagedKeyValueCache::Remove(std::shared_ptr<Request> request) {
  for (auto request_it = block_tables_.begin(); request_it != block_tables_.end(); ++request_it) {
    if (request_it->request_id == request.get()) {
      // Blocks the prefix cache indexed keep its reference and stay resident, so the next turn of
      // this conversation can adopt them. Everything else, including the partially filled tail
      // block, goes straight back to the pool.
      block_pool_->Free(request_it->blocks);
      block_tables_.erase(request_it);
      return;
    }
  }
}

PagedKeyValueCache::~PagedKeyValueCache() {
  // Release the block tables before the prefix cache and the pool unwind, so every reference is
  // accounted for regardless of whether the engine drained its requests first. The single-block
  // release allocates nothing, so teardown cannot fail on a failed allocation.
  for (auto& table : block_tables_) {
    for (const auto& block : table.blocks) {
      block_pool_->Release(block);
    }
  }
  block_tables_.clear();
}

PrefixCacheMatch PagedKeyValueCache::MatchPrefix(std::span<const int32_t> tokens,
                                                 size_t max_adoptable_tokens) {
  return prefix_cache_->Match(tokens, max_adoptable_tokens);
}

bool PagedKeyValueCache::PrefixCachingEnabled() const {
  return prefix_cache_->Enabled();
}

const PrefixCacheMetrics& PagedKeyValueCache::PrefixMetrics() const {
  return prefix_cache_->Metrics();
}

void PagedKeyValueCache::SealCommittedBlocks(const void* request_id,
                                             std::span<const int32_t> tokens) {
  if (!prefix_cache_->Enabled()) {
    return;
  }

  const auto table_it = std::find_if(block_tables_.begin(), block_tables_.end(),
                                     [request_id](const PagedCacheBlockTable& table) {
                                       return table.request_id == request_id;
                                     });
  if (table_it == block_tables_.end()) {
    return;
  }

  const size_t block_size = block_pool_->BlockSize();
  const size_t full_blocks = table_it->committed_slots / block_size;
  if (full_blocks <= table_it->sealed_blocks) {
    return;
  }
  if (tokens.size() < full_blocks * block_size) {
    throw std::runtime_error(
        "The request's token mirror is shorter than the slots the paged cache committed for it.");
  }

  auto parent = table_it->sealed_identity;
  for (size_t index = table_it->sealed_blocks; index < full_blocks; ++index) {
    auto chained = prefix_cache_->Register(table_it->blocks[index],
                                           tokens.subspan(index * block_size, block_size),
                                           parent);
    if (!chained) {
      // Nothing after an unreachable identity can be reached either, so the chain stops here and
      // the remaining blocks stay private to this request.
      break;
    }
    parent = std::move(chained);
    table_it->sealed_blocks = index + 1;
    table_it->sealed_identity = parent;
  }
}

PagedCacheReservation PagedKeyValueCache::Reserve(std::span<const PagedCacheReservationRequest> requests) {
  // A step only ever writes into a request's tail block, and sharing is restricted to full blocks
  // that are never written again, so this normally finds nothing to do. When a tail block is shared
  // anyway the writer takes a private copy of it here, before any reservation state exists, rather
  // than diverging into the other owner's key-value data.
  if (prefix_cache_->Enabled()) {
    LayerBlockCopier copier{*this};
    for (const auto& request : requests) {
      if (request.newly_admitted) {
        continue;
      }
      const auto table_it = std::find_if(block_tables_.begin(), block_tables_.end(),
                                         [&request](const PagedCacheBlockTable& value) {
                                           return value.request_id == request.request_id;
                                         });
      if (table_it == block_tables_.end()) {
        continue;
      }
      MakeTailBlockExclusive(*table_it, request.target_slots, *block_pool_, copier);
    }
  }

  RetainedBlockReclaimer reclaimer{*prefix_cache_};
  return PagedCacheReservation{*block_pool_, block_tables_, requests, &reclaimer};
}

void PagedKeyValueCache::LayerBlockCopier::CopyBlock(size_t source_block_id,
                                                     size_t destination_block_id) {
  const size_t block_bytes = cache_.block_bytes_per_layer_;
  if (block_bytes == 0) {
    throw std::runtime_error("The paged cache cannot copy a block of unknown size.");
  }

  const auto copy = [&](OrtValue& tensor) {
    auto* data = tensor.GetTensorMutableRawData();
    const size_t total_bytes = cache_.block_pool_->Capacity() * block_bytes;
    auto buffer = cache_.model_->p_device_kvcache_->WrapMemoryBase(data, total_bytes);
    buffer->CopyFrom(destination_block_id * block_bytes, *buffer, source_block_id * block_bytes,
                     block_bytes);
  };

  for (auto& layer : cache_.cache_) {
    copy(*layer.key_cache);
    copy(*layer.value_cache);
  }
}

StepPlanningResult PagedKeyValueCache::PlanStepResources(StepPlan& plan) const {
  const size_t committed_request_count = block_tables_.size();
  if (committed_request_count > max_batch_size_) {
    throw std::runtime_error("Committed paged cache requests exceed the configured batch size.");
  }
  const size_t scheduled_request_limit =
      plan.scheduled_request_limit == 0
          ? max_batch_size_
          : plan.scheduled_request_limit;
  if (scheduled_request_limit > max_batch_size_) {
    throw std::runtime_error("Step plan request limit exceeds the configured batch size.");
  }

  // Retained prefix blocks are always reclaimable, so they count as capacity a step can take. The
  // reservation reclaims exactly what it needs, in least-recently-used order.
  const size_t available_blocks =
      block_pool_->AvailableBlocks() + prefix_cache_->ReclaimableBlocks();
  size_t planned_blocks = 0;
  size_t selected_requests = 0;
  size_t selected_new_requests = 0;
  size_t max_blocks_per_request = 0;
  bool capacity_deferred = false;
  const void* unserviceable_request_id = nullptr;

  struct CacheGrowth {
    size_t proposed_blocks{};
    size_t new_blocks{};
    // Retained prefix blocks this admission would adopt that no other selected request has already
    // been charged for. They are counted as available above, but adopting them takes them out of
    // the reclaimable pool, so they have to be charged to this step as well or the reservation
    // could be planned to reclaim capacity it has just claimed.
    std::vector<size_t> claimed_retained_blocks;
  };
  // Retained blocks already charged to this step, so several requests adopting the same prefix pay
  // for it once instead of being deferred for capacity that is not actually needed twice.
  std::vector<size_t> charged_retained_blocks;
  // Blocks the request has to own for this step. A chunked prefill is planned one chunk at a time,
  // but the blocks are taken for the whole sequence: admitting a prompt on the strength of its
  // first chunk and then losing the rest of the pool to another request would stall it part way
  // through, holding the blocks it already took. PagedCacheReservation reserves the same blocks.
  // Blocks adopted from the prefix cache already exist, so they widen the request's block table
  // without drawing on the free pool.
  const auto calculate_growth = [&](const RequestStepPlan& entry,
                                    const PagedCacheBlockTable* table) {
    const size_t adopted_blocks =
        entry.prefix_match ? entry.prefix_match->blocks.size() : 0;
    const size_t committed_slots =
        table ? table->committed_slots : (entry.prefix_match ? entry.prefix_match->token_count : 0);
    if (entry.target_cache_slots < committed_slots) {
      throw std::runtime_error("Step plan target precedes the committed cache boundary.");
    }

    std::vector<size_t> claimed_retained_blocks;
    if (entry.prefix_match) {
      for (const auto& block : entry.prefix_match->blocks) {
        if (block->RefCount() == 1 &&
            std::find(charged_retained_blocks.begin(), charged_retained_blocks.end(),
                      block->Id()) == charged_retained_blocks.end()) {
          claimed_retained_blocks.push_back(block->Id());
        }
      }
    }

    const size_t reserved_slots =
        std::max(entry.whole_sequence_cache_slots, entry.target_cache_slots);
    const size_t committed_blocks = table ? table->blocks.size() : adopted_blocks;
    const size_t committed_capacity = committed_blocks * block_pool_->BlockSize();
    const size_t additional_slots =
        reserved_slots > committed_capacity ? reserved_slots - committed_capacity : 0;
    const size_t new_blocks = block_pool_->BlocksNeeded(additional_slots);
    return CacheGrowth{committed_blocks + new_blocks, new_blocks,
                       std::move(claimed_retained_blocks)};
  };
  const auto permanently_unserviceable = [&](const CacheGrowth& growth) {
    return growth.proposed_blocks > block_pool_->Capacity() ||
           (graph_capture_ &&
            growth.proposed_blocks > max_block_table_columns_);
  };
  const auto select = [&](size_t request_index,
                          CacheGrowth& growth) {
    // Compact selected entries in place. Requests skipped for temporary capacity pressure remain
    // pending with their committed block tables untouched and can be reconsidered next Step().
    planned_blocks += growth.new_blocks + growth.claimed_retained_blocks.size();
    charged_retained_blocks.insert(charged_retained_blocks.end(),
                                   growth.claimed_retained_blocks.begin(),
                                   growth.claimed_retained_blocks.end());
    max_blocks_per_request =
        std::max(max_blocks_per_request, growth.proposed_blocks);
    if (selected_requests != request_index) {
      plan.requests[selected_requests] =
          std::move(plan.requests[request_index]);
    }
    ++selected_requests;
  };
  const auto find_table = [this](const void* request_id) {
    const auto it = std::find_if(block_tables_.begin(), block_tables_.end(),
                                 [request_id](const PagedCacheBlockTable& table) {
                                   return table.request_id == request_id;
                                 });
    return it == block_tables_.end() ? nullptr : &*it;
  };
  std::vector<const void*> request_ids;
  request_ids.reserve(plan.requests.size());
  for (size_t i = 0; i < plan.requests.size(); ++i) {
    const auto& candidate = plan.requests[i];
    if (std::find(request_ids.begin(), request_ids.end(),
                  candidate.request_id) != request_ids.end()) {
      throw std::runtime_error("Step plan contains a duplicate request.");
    }
    request_ids.push_back(candidate.request_id);

    const auto* table = find_table(candidate.request_id);
    if (candidate.newly_admitted && table) {
      throw std::runtime_error("New step plan request already belongs to the paged cache.");
    }
    if (!candidate.newly_admitted && !table) {
      throw std::runtime_error("Step plan resident membership does not match the committed cache.");
    }

    auto growth = calculate_growth(candidate, table);
    if (permanently_unserviceable(growth)) {
      if (!unserviceable_request_id) {
        unserviceable_request_id = candidate.request_id;
      }
      continue;
    }

    // Charging the retained blocks an adoption claims alongside the blocks it still has to take
    // makes adoption capacity-neutral: every adopted block removes one from `new_blocks` and adds
    // one back here, so a step is never admitted on capacity the reservation then cannot reclaim,
    // and a prefix hit never makes a request harder to admit than recomputing it would.
    if (selected_requests >= scheduled_request_limit ||
        (candidate.newly_admitted &&
         committed_request_count + selected_new_requests >= max_batch_size_) ||
        planned_blocks + growth.new_blocks + growth.claimed_retained_blocks.size() >
            available_blocks) {
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
    block_table_columns = 8;
    while (block_table_columns < max_blocks_per_request) {
      block_table_columns *= 2;
    }
    block_table_columns = std::min(block_table_columns, max_block_table_columns_);
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
  for (const auto& block : block_pool_->OwnedBlocks()) {
    snapshot.blocks.push_back(CachedBlockSnapshot{
        block->Id(),
        block->RefCount(),
        block->Size(),
        block->IsFull(),
        block->HasIdentity(),
    });
  }
  snapshot.requests.reserve(block_tables_.size());
  for (const auto& block_table : block_tables_) {
    RequestBlockSnapshot request_snapshot;
    request_snapshot.request_id = block_table.request_id;
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

PagedCacheSnapshot PagedKeyValueCache::Snapshot(
    const PagedCacheReservation& reservation) const {
  auto snapshot = Snapshot();
  snapshot.transaction_reserved_block_ids.reserve(
      reservation.ReservedBlocks().size());
  for (const auto& block : reservation.ReservedBlocks()) {
    snapshot.transaction_reserved_block_ids.push_back(block->Id());
  }
  snapshot.transaction_adopted_block_ids.reserve(
      reservation.AdoptedBlocks().size());
  for (const auto& block : reservation.AdoptedBlocks()) {
    snapshot.transaction_adopted_block_ids.push_back(block->Id());
  }
  snapshot.reservations.reserve(reservation.Deltas().size());
  for (const auto& delta : reservation.Deltas()) {
    RequestReservationSnapshot request_reservation;
    request_reservation.request_id = delta.request_id;
    request_reservation.committed_slots = delta.committed_slots;
    request_reservation.target_slots = delta.target_slots;
    request_reservation.tail_slots_to_consume = delta.tail_slots_to_consume;
    request_reservation.reserved_block_ids.reserve(
        delta.reserved_block_count);
    for (size_t i = 0; i < delta.reserved_block_count; ++i) {
      request_reservation.reserved_block_ids.push_back(
          reservation.ReservedBlocks()[delta.reserved_block_offset + i]->Id());
    }
    request_reservation.adopted_block_ids.reserve(delta.adopted_block_count);
    for (size_t i = 0; i < delta.adopted_block_count; ++i) {
      request_reservation.adopted_block_ids.push_back(
          reservation.AdoptedBlocks()[delta.adopted_block_offset + i]->Id());
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

std::pair<OrtValue*, const char*> PagedKeyValueCache::BlockTables(const std::vector<std::shared_ptr<Request>>& requests) {
  size_t max_blocks = 0;
  for (auto& block_table : block_tables_) {
    if (std::find_if(requests.begin(), requests.end(),
                     [&block_table](const std::shared_ptr<Request>& request) {
                       return request.get() == block_table.request_id;
                     }) != requests.end()) {
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
    auto it = std::find_if(requests.begin(), requests.end(),
                           [&block_table](const std::shared_ptr<Request>& request) {
                             return request.get() == block_table.request_id;
                           });
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

void PagedKeyValueCache::BindCache(State& state) {
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
}

void PagedKeyValueCache::UpdateState(State& state, const std::vector<std::shared_ptr<Request>>& requests) {
  BindCache(state);
  auto block_tables = BlockTables(requests);
  state.inputs_.back() = block_tables.first;
  state.input_names_.back() = block_tables.second;
}

void PagedKeyValueCache::UpdateState(
    State& state,
    const std::vector<std::shared_ptr<Request>>& requests,
    const PagedCacheReservation& reservation,
    size_t columns) {
  BindCache(state);
  auto block_tables = BlockTables(requests, reservation, columns);
  state.inputs_.back() = block_tables.first;
  state.input_names_.back() = block_tables.second;
}

}  // namespace Generators
