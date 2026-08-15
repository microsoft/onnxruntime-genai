// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <stdint.h>
#include <memory>
#include <vector>
#include <list>

#include "block.h"
#include "engine_invariants.h"
#include "paged_cache_reservation.h"
#include "prefix_cache.h"
#include "request.h"
#include "step_plan.h"

namespace Generators {

// Copies the key and value data of one physical block onto another. Implemented by the paged cache
// over its per-layer tensors; separated so the divergence policy below can be exercised without a
// device.
struct BlockCopier {
  virtual void CopyBlock(size_t source_block_id, size_t destination_block_id) = 0;
  virtual ~BlockCopier() = default;
};

/**
 * @brief Gives `table` a private copy of the block the next step is about to write into, when that
 *        block is still referenced by another owner.
 * @param table The request's committed block table.
 * @param target_slots Slots the table will hold once the pending step commits.
 * @param pool The pool the private replacement block is taken from.
 * @param copier Moves the shared block's key-value data into the replacement.
 * @return True when a private copy was made.
 *
 * Only full blocks are ever shared, and a full block is never written again, so under the prefix
 * cache's own policy this finds nothing to do. It exists so that a partially filled block that is
 * shared anyway diverges by copy instead of corrupting the other owner's key-value data, which is
 * what makes "shared blocks are immutable" an enforced property rather than an assumption.
 */
bool MakeTailBlockExclusive(PagedCacheBlockTable& table,
                            size_t target_slots,
                            BlockPool& pool,
                            BlockCopier& copier);

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
  ~PagedKeyValueCache();

  bool CanAdd(std::shared_ptr<Request> request) const;

  void Add(std::shared_ptr<Request> request);

  bool CanAppendTokens(std::shared_ptr<Request> request) const;

  void AppendTokens(std::shared_ptr<Request> request);

  void Remove(std::shared_ptr<Request> request);

  PagedCacheReservation Reserve(std::span<const PagedCacheReservationRequest> requests);

  // Selects the active and pending requests whose immediate cache growth fits this step.
  StepPlanningResult PlanStepResources(StepPlan& plan) const;

  /**
   * @brief Longest run of leading prompt tokens already resident in the cache, or an empty match.
   * @param tokens The request's whole sequence.
   * @param max_adoptable_tokens Tokens the caller may skip; always at least one less than the
   *        sequence length so the request still computes a token and produces logits.
   *
   * The returned blocks are the physical blocks the request will point at. Nothing is committed
   * here: the reservation takes the references, so a step that never runs adopts nothing.
   */
  PrefixCacheMatch MatchPrefix(std::span<const int32_t> tokens, size_t max_adoptable_tokens);

  /**
   * @brief Gives every block that filled up during the committed step a content identity, so later
   *        prompts sharing the same prefix can adopt it.
   *
   * Called once the step's reservation is committed, at which point the host token mirror of each
   * request covers exactly the slots the cache holds for it.
   */
  void SealCommittedBlocks(const void* request_id, std::span<const int32_t> tokens);

  bool PrefixCachingEnabled() const;
  const PrefixCacheMetrics& PrefixMetrics() const;

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

  // Number of columns in the block table handed to the model on the most recent BlockTables() call.
  // With graph capture this is a bucketed capacity rather than the exact longest table, so the shape
  // is stable across steps; it also bounds the KV length any sequence in the batch can reach, which
  // is what `attention_metadata` has to report for a captured step.
  size_t BlockTableColumns() const { return block_table_columns_; }
  size_t MaxBlockTableColumns() const { return max_block_table_columns_; }
  size_t MaxBlockTableRows() const { return max_block_table_rows_; }
  bool GraphCaptureEnabled() const { return graph_capture_; }

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

  // Copies a block's key and value data across every layer, used only by MakeTailBlockExclusive.
  class LayerBlockCopier final : public BlockCopier {
   public:
    explicit LayerBlockCopier(PagedKeyValueCache& cache) : cache_{cache} {}
    void CopyBlock(size_t source_block_id, size_t destination_block_id) override;

   private:
    PagedKeyValueCache& cache_;
  };

  void BindCache(State& state);

  // Reclaims retained prefix blocks so the pool can satisfy a reservation.
  class RetainedBlockReclaimer final : public BlockReclaimer {
   public:
    explicit RetainedBlockReclaimer(PrefixCache& prefix_cache) : prefix_cache_{prefix_cache} {}
    size_t Reclaim(size_t blocks_needed) override { return prefix_cache_.Reclaim(blocks_needed); }

   private:
    PrefixCache& prefix_cache_;
  };

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

  std::shared_ptr<Model> model_;
  std::vector<LayerCache> cache_;                   // Pair of key and value caches for all layers
  std::unique_ptr<BlockPool> block_pool_;           // Allocator for blocks
  // Declared after the pool so it releases its retained references before the pool goes away.
  std::unique_ptr<PrefixCache> prefix_cache_;       // Content-addressed index over filled blocks
  std::vector<PagedCacheBlockTable> block_tables_;  // Block table for all requests in the cache
  std::unique_ptr<OrtValue> block_tables_value_;    // Block tables for all requests in the cache
  size_t block_bytes_per_layer_{};                  // Bytes one block occupies in one layer's cache

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