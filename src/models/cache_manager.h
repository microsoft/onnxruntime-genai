// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <stdint.h>
#include <cstdint>
#include <memory>
#include <tuple>
#include <optional>
#include <vector>
#include <list>
#include "block.h"
#include "onnxruntime_api.h"
#include "../server/engine_utils.h"


namespace Generators {

struct CacheOptions {
  CacheOptions(const int32_t num_layers,
               const std::optional<int32_t>& block_size,
               const int32_t num_kv_heads, const int32_t head_size,
               const std::optional<ONNXTensorElementDataType> dtype,
               const std::optional<int32_t>& num_blocks,
               const std::optional<float>& gpu_utilization_factor);

  int32_t num_layers_{};
  int32_t block_size_{16};
  int32_t num_kv_heads_{};
  int32_t head_size_{};
  ONNXTensorElementDataType dtype_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
  int32_t num_blocks_{};
  float gpu_utilization_factor_{0.3f};
  float watermark{0.01f};
};

struct BlockTable {
  size_t sequence_id;
  std::vector<std::shared_ptr<Block>>
      blocks;  // List of blocks alloted to the sequence_id
};

struct CacheManager {
 public:
  CacheManager(const CacheOptions& cache_options, Ort::Allocator* cpu_allocator,
               Ort::Allocator* gpu_allocator);

  CacheManager(const CacheManager&) = delete;

  CacheManager& operator=(const CacheManager&) = delete;

  bool CanAdd(size_t num_tokens) const;
  AllocateStatus CanAllocate(const SequenceGroup& seq_group) const;

  // Allocates blocks needed to serve the given sequence_id for the given prompt
  // token size. Cache additions happen one sequence at a time.
  void Add(const std::vector<size_t>& sequence_ids, size_t num_tokens);
  void Allocate(const SequenceGroup& seq_group);
  void Free(const Sequence& seq);

  bool CanAppendTokens(size_t sequence_id, size_t num_tokens) const;
  bool CanAppendSlots(const SequenceGroup& seq_group,
                      int num_lookahead_slots = 0) const;

  // Before running a decoding step, the cache needs to allot a new slot for the
  // given sequence_id. If the block has been completely filled up, a new block
  // will be allocated as well. This function should be called before each
  // decoding step.
  void AppendTokens(size_t sequence_id, size_t num_tokens);
  std::vector<std::tuple<int, int>> AppendSlots(const Sequence& seq,
                                                int num_lookahead_slots = 0);

  bool CanSwapOut(const SequenceGroup& seq_group);
  std::vector<std::tuple<int, int>> SwapOut(const SequenceGroup& seq_group);

  AllocateStatus CanSwapIn(const SequenceGroup& seq_group);
  std::vector<std::tuple<int, int>> SwapIn(const SequenceGroup& seq_group);

  // Removes the allocated blocks for the given sequence_id and makes it
  // available for other sequences.
  void Remove(size_t sequence_id);

  void Fork(size_t parent_seq_id, size_t child_seq_id);
  void ForkSeq(const Sequence& parent_seq, const Sequence& child_seq);

  // Returns the K, V cache for the given layer_id.
  std::pair<OrtValue*, OrtValue*> Cache(size_t layer_id);

  // Shape: [batch_size, max_num_blocks_per_sequence]
  // Assume that the block tables are requested for sequences with ids [2, 5, 7]
  // Assume the block tables for given sequence ids are:
  // {
  //   2: [0, 1, 2],
  //   5: [3, 7, 9],
  //   7: [4, 5, 6, 8]
  // }
  // Invoking this function will return the block tables as:
  // [ [0, 1, 2, -1],
  //   [3, 7, 9, -1],
  //   [4, 5, 6, 8] ]
  //
  // This implies that the sequence with sequence id 2 has its kv cache stored
  // in blocks with ids [0, 1, 2], the sequence with sequence id 5 has its kv
  // cache stored in blocks with ids [3, 7, 9], and and the sequencewith
  // sequence id 7 has its kv cache stored in blocks with ids [4, 5, 6, 8]. -1
  // is used to pad the block tables to the max blocks per sequence from the
  // given sequences. The order of the block tables is based on the order the
  // provided sequence_ids.
  std::unique_ptr<OrtValue> BlockTables(
      const std::vector<size_t>& sequence_ids) const;
  std::vector<int> GetBlockTable(const Sequence& seq) const;

  // Shape: [num_tokens]
  // Prompt stage:
  // Assume the cache contains the blocks for sequences with ids [2, 5, 7]
  // Assume that the slot mapping for the given sequence ids are:
  // {
  //   2: 32, 33, 34, 35
  //   5: 0, 1, 2, 3, 4
  //   7: 16, 17, 18
  // }
  // And assume that the block size is 16.
  // The slot mapping tells us that the sequence with id 2 should fill its
  // prompt KV cache tokens at slots [0, 1, 2, 3] (slot_id % 16) in block 2
  // (slot_id / 16), the sequence with id 5 should fill its prompt KV cache
  // tokens at slots [0, 1, 2, 3, 4] in block 0, and the sequence with id 7
  // should fill its prompt KV cache tokens at slots [0, 1, 2] in block 1.
  // Invoking this function will return the slot mapping as:
  // [ | 32, 33, 34, 35, | 0, 1, 2, 3, 4, | 16, 17, 18 | ]
  // Decoding stage:
  // The same principle applies for the decoding stage, but the slot mapping
  // will only contain the slot ids for the new token generated by the model.
  // For example, assume that the cache contains the blocks for sequences with
  // ids [2, 5, 7] Assume that the slot mapping for the given sequence ids are:
  // {
  //   2: 43,
  //   5: 29,
  //   7: 12
  // }
  // And assume that the block size is 16.
  // The slot mapping tells us that the sequence with id 2 should fill its KV
  // cache token at slot 11 (43 % 16) in block 2 (43 / 16), the sequence with id
  // 5 should fill its KV cache token at slot 13 (29 % 16) in block 1, and the
  // sequence with id 7 should fill its KV cache token at slot 12 (12 % 16) in
  // block 0. The order of the clot mapping is based on the order the provided
  // sequence_ids.
  std::unique_ptr<OrtValue> SlotMapping(
      const std::vector<size_t>& sequence_ids) const;

 private:
  using LayerCache =
      std::unique_ptr<OrtValue>;  // Shape: [num_blocks, block_size *
                                  // num_kv_heads * head_size]
  /*
  The K and the V Cache is represented as an array of blocks. Each block
  contains a number of slots equal to the block size. Each slot contains
  num_kv_heads * head_size elements. Here the slot represents data generated by
  the model for a single token. This KV cache is allocated for each layer in the
  model. Although the cache is preallocated, the actual memory is alloted to a
  sequence_id only as needed.

  View of the cache for each layer (LayerCache):

        -->|size of each block = block_size(M) * size of each slot|<--
           |______________________________________________________|
           |       -->|          |<-- size of each slot = num_kv_heads *
  head_size |          |          |                                |
           |__________|__________|________________________________|
  block 0  |  slot 0  |  slot 1  |  slot 2  |     .    |  slot M  |
  block 1  |          |          |          |          |          |
  block 2  |          |          |          |          |          |
  block 3  |          |          |          |          |          |
     .     |          |          |          |          |          |
     .     |          |          |          |          |          |
     .     |          |          |          |          |          |
           |          |          |          |          |          |
  block N  |__________|__________|__________|__________|__________|

  N = num_blocks per layer
  M = block_size per block

  */

  void Add(size_t sequence_id, size_t num_tokens);

  CacheOptions options_;
  Ort::Allocator* cpu_allocator_;
  Ort::Allocator* gpu_allocator_;
  std::vector<std::pair<LayerCache, LayerCache>>
      cache_;  // Pair of key and value caches for all layers
  std::unique_ptr<BlockAllocator> block_allocator_;  // Allocator for blocks
  std::list<BlockTable> block_table_pool_;  // The pool of all block tables
  std::unordered_map<size_t, std::list<BlockTable>::iterator>
      block_tables_;  // Mapping of sequence_id to block_info
  std::unordered_map<size_t, size_t>
      copy_on_writes_;  // Mapping of block ids to copy
  std::unordered_map<size_t, std::vector<std::shared_ptr<Block>>>
      vllm_block_tables_;  // Mapping of block ids to sequence_id
};

}  // namespace Generators
