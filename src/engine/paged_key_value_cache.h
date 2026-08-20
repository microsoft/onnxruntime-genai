// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <stdint.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include <list>

#include "block.h"
#include "engine_invariants.h"
#include "paged_cache_reservation.h"
#include "request.h"
#include "step_plan.h"

namespace Generators {

inline constexpr size_t kMinGraphBlockTableColumns = 8;

inline size_t GetGraphBlockTableColumns(size_t max_blocks, size_t max_columns) {
  if (max_columns == 0) {
    throw std::runtime_error("Graph block-table capacity must be non-zero.");
  }

  size_t columns = std::min(kMinGraphBlockTableColumns, max_columns);
  while (columns < max_blocks) {
    if (columns > max_columns / 2) {
      return max_columns;
    }
    columns *= 2;
  }
  return columns;
}

/*
 * PagedKeyValueCache manages a paged key-value cache for models that use the PagedAttention operator.
 * The cache is divided into blocks, each containing a fixed number of slots. Each slot holds
 * the key and value vectors for a single token across all attention heads.
 * The cache is allocated on a device (e.g., GPU) and is shared across multiple requests.
 * Requests can be added to the cache, and blocks are allocated as needed. The cache
 * supports appending tokens to existing requests and removing requests from the cache.
 * The cache also provides methods to retrieve the current key-value cache and block tables
 * for all requests.
 */
struct PagedKeyValueCache {
 public:
  PagedKeyValueCache(std::shared_ptr<Model> model);

  bool CanAdd(std::shared_ptr<Request> request) const;

  void Add(std::shared_ptr<Request> request);

  bool CanAppendTokens(std::shared_ptr<Request> request) const;

  void AppendTokens(std::shared_ptr<Request> request);

  void Remove(std::shared_ptr<Request> request);

  PagedCacheReservation Reserve(std::span<const PagedCacheReservationRequest> requests);

  // Selects the active and pending requests whose immediate cache growth fits this step.
  StepPlanningResult PlanStepResources(StepPlan& plan) const;

  // Returns the K, V cache.
  std::vector<std::pair<OrtValue*, OrtValue*>> Cache();

  std::vector<std::pair<const char*, const char*>> Names();

  std::vector<std::pair<const char*, const char*>> OutputNames();

  // Shape: [batch_size, max_num_blocks_per_sequence]
  // Assume that the block tables are requested for 3 sequences
  // Assume the block tables for given sequence / requests are:
  // {
  //   [0, 1, 2],
  //   [3, 7, 9],
  //   [4, 5, 6, 8]
  // }
  // Invoking this function will return the block tables as:
  // [ [0, 1, 2, -1],
  //   [3, 7, 9, -1],
  //   [4, 5, 6, 8] ]
  //
  // This implies that the sequence in the first request has its kv cache stored in blocks with ids [0, 1, 2],
  // the sequence in the second request has its kv cache stored in blocks with ids [3, 7, 9], and
  // the sequence in the third request has its kv cache stored in blocks with ids [4, 5, 6, 8].
  // -1 is used to pad the block tables to the max blocks per sequence from the given sequences.
  // The order of the block tables is based on the order the provided requests.
  std::pair<OrtValue*, const char*> BlockTables(const std::vector<std::shared_ptr<Request>>& requests);
  std::pair<OrtValue*, const char*> BlockTables(
      const std::vector<std::shared_ptr<Request>>& requests,
      const PagedCacheReservation& reservation,
      size_t columns);

  // Block table for the sliding-window layers, same shape as BlockTables() but filled by repeating
  // each request's ring of blocks: column j holds ring[j % ring_blocks]. The operator indexes it
  // with the token's true position, so position p resolves to slot p % (ring_blocks * block_size)
  // and positions that have fallen out of the window are overwritten in place.
  //
  //   ring = [5, 9], block_size = 4, so the table is
  //   [5, 9, 5, 9, 5, 9, ...]
  //
  // Only valid when WindowedLayers() is non-empty.
  std::pair<OrtValue*, const char*> WindowBlockTables(const std::vector<std::shared_ptr<Request>>& requests);
  std::pair<OrtValue*, const char*> WindowBlockTables(
      const std::vector<std::shared_ptr<Request>>& requests,
      const PagedCacheReservation& reservation,
      size_t columns);

  // Number of columns in the block table handed to the model on the most recent BlockTables() call.
  // With graph capture this is a bucketed capacity rather than the exact longest table, so the shape
  // is stable across steps; it also bounds the KV length any sequence in the batch can reach, which
  // is what `attention_metadata` has to report for a captured step.
  size_t BlockTableColumns() const { return block_table_columns_; }
  size_t MaxBlockTableColumns() const { return max_block_table_columns_; }
  size_t MaxBlockTableRows() const { return max_block_table_rows_; }
  bool GraphCaptureEnabled() const { return graph_capture_; }
  size_t MaxQueryTokensPerRequest() const {
    return Windowed() ? window_live_span_ - window_size_ + 1 : 0;
  }

  void UpdateState(State& state, const std::vector<std::shared_ptr<Request>>& requests);
  void UpdateState(State& state,
                   const std::vector<std::shared_ptr<Request>>& requests,
                   const PagedCacheReservation& reservation,
                   size_t columns);

  // Captures an immutable snapshot of the cache's block accounting (free/allocated blocks, and per
  // request block ids and used/empty slots) for invariant validation and state inspection. The
  // snapshot copies out the state and holds no reference into the cache.
  PagedCacheSnapshot Snapshot() const;
  PagedCacheSnapshot Snapshot(const PagedCacheReservation& reservation) const;

 private:
  struct LayerCache {
    std::unique_ptr<OrtValue> key_cache;    // Shape: [num_blocks, block_size, num_kv_heads, head_size]
    std::unique_ptr<OrtValue> value_cache;  // Shape: [num_blocks, block_size, num_kv_heads, head_size]
    std::string key_cache_name;
    std::string value_cache_name;
    std::string key_cache_output_name;
    std::string value_cache_output_name;
  };

  void BindCache(State& state);

  //   The key and the value cache is represented as an array of blocks. Each block contains
  //   a number of slots equal to the block size. Each slot contains num_kv_heads * head_size
  //   elements. Here the slot represents data generated by the model for a single token.
  //   This key-value cache is allocated for each layer in the model.
  //   Although the cache is preallocated, the actual memory is alloted to a request only as needed.
  //   View of the cache for each layer (LayerCache):
  //         -->|size of each block = block_size(M) * size of each slot|<--
  //            |______________________________________________________|
  //            |       -->|          |<-- size of each slot = num_kv_heads * head_size
  //            |          |          |                                |
  //            |__________|__________|________________________________|
  //   block 0  |  slot 0  |  slot 1  |  slot 2  |     .    |  slot M  |
  //   block 1  |          |          |          |          |          |
  //   block 2  |          |          |          |          |          |
  //   block 3  |          |          |          |          |          |
  //      .     |          |          |          |          |          |
  //      .     |          |          |          |          |          |
  //      .     |          |          |          |          |          |
  //            |          |          |          |          |          |
  //   block N  |__________|__________|__________|__________|__________|
  //   N = num_blocks per layer
  //   M = block_size per block

  // Fills `data` with `columns` block ids per request, in the order the requests were given.
  void FillBlockTables(const std::vector<std::shared_ptr<Request>>& requests, bool windowed,
                       int32_t* data, size_t columns);
  std::shared_ptr<Model> model_;
  std::vector<LayerCache> cache_;                   // Pair of key and value caches for all layers
  std::unique_ptr<BlockPool> block_pool_;           // Allocator for blocks
  std::vector<PagedCacheBlockTable> block_tables_;  // Block table for all requests in the cache
  std::unique_ptr<OrtValue> block_tables_value_;    // Block tables for all requests in the cache

  // Sliding-window layers hold their KV in a ring of `window_ring_blocks_` blocks rather than one
  // block per position, so they get their own much smaller pool and their own block table. The
  // ring has to span the whole live set of a step: the queries reach back `window_size` positions
  // and a chunked prefill advances by up to `chunk_size` at once, so it must cover
  // `chunk_size + window_size - 1` positions. Zero when the model has no windowed layers.
  size_t window_size_{};
  size_t window_ring_blocks_{};
  size_t window_live_span_{};  // chunk_size + window_size - 1, checked against each step
  std::unique_ptr<BlockPool> window_block_pool_;
  std::unique_ptr<OrtValue> window_block_tables_value_;
  std::unique_ptr<Tensor> window_block_tables_tensor_;

  bool Windowed() const { return window_ring_blocks_ > 0; }

  // Graph capture needs the block table at a device address that never moves and at a shape that
  // repeats across steps, so it gets a dedicated persistent tensor instead of the per-step CPU one.
  bool graph_capture_{};
  size_t max_batch_size_{};
  size_t block_table_columns_{};
  size_t max_block_table_columns_{};
  size_t max_block_table_rows_{};
  std::unique_ptr<Tensor> block_tables_tensor_;
};

}  // namespace Generators