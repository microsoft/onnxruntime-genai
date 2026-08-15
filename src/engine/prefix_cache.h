// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "../span.h"
#include "block.h"

/**
 * @file prefix_cache.h
 * @brief Content-addressed index of filled paged-cache blocks, so a prompt that repeats a prefix
 *        another sequence already computed can adopt those blocks instead of recomputing them.
 *
 * Serving and agent traffic resends large identical prefixes every turn -- system prompts, tool
 * schemas, retrieved context, and the conversation so far. Each filled block gets an identity that
 * chains the identity of the block before it, so a block only matches when the whole preceding
 * token sequence matches too. A hash match is never trusted on its own: the stored tokens and the
 * parent identity are compared before a block is handed out, so a collision costs a missed hit
 * rather than a wrong answer.
 *
 * Only full blocks are indexed. A full block is never written again, which is what makes it safe
 * for several requests to point at the same physical block; a request's partially filled tail block
 * stays private to it. Blocks stay indexed after their request finishes (retention) so the next
 * turn of the same conversation can hit them, bounded by a block budget and reclaimable on demand
 * so retention can never starve a live request.
 */

namespace Generators {

struct PrefixCacheOptions {
  bool enabled{};
  // Upper bound on blocks the index may hold. Retention beyond this evicts the least recently used
  // unreferenced entry. Zero disables the cache regardless of `enabled`.
  size_t max_blocks{};
  // A match shorter than this is not worth adopting: the saved prefill has to outweigh the extra
  // block-table width and bookkeeping.
  size_t min_match_blocks{1};
  // Content hash used to address a block. Left null in production, where PrefixCache::ChainHash is
  // used. Overridable so the collision-verification path -- distinct contents landing on the same
  // identity -- can be exercised deterministically instead of hoping to find a real collision.
  uint64_t (*hash)(uint64_t parent_hash, std::span<const int32_t> tokens){nullptr};
};

// A run of already-resident blocks covering the leading `token_count` tokens of a prompt. The
// blocks are exactly the ones the adopting request will point at; `token_count` is always a whole
// multiple of the block size.
struct PrefixCacheMatch {
  size_t token_count{};
  std::vector<std::shared_ptr<Block>> blocks;

  bool Empty() const { return blocks.empty(); }
};

struct PrefixCacheMetrics {
  uint64_t lookups{};                 // Prompts offered to the index.
  uint64_t hits{};                    // Prompts that adopted at least one block.
  uint64_t queried_tokens{};          // Prompt tokens eligible for adoption across all lookups.
  uint64_t matched_tokens{};          // Prompt tokens actually adopted (prefill work skipped).
  uint64_t registered_blocks{};       // Blocks given a content identity.
  uint64_t duplicate_registrations{};  // Blocks whose content was already indexed elsewhere.
  uint64_t hash_collisions{};         // Distinct contents that hashed to an indexed identity.
  uint64_t evictions{};               // Indexed blocks dropped to make room.
  uint64_t retention_refusals{};      // Registrations skipped because nothing was evictable.
};

/**
 * @class PrefixCache
 * @brief Content-addressed index over a BlockPool's filled blocks with bounded LRU retention.
 *
 * The index holds one reference per indexed block, so an indexed block survives the request that
 * produced it. `Reclaim` gives that capacity straight back to the pool under memory pressure, which
 * is what keeps retention from ever starving a live request.
 */
class PrefixCache {
 public:
  PrefixCache(BlockPool& block_pool, PrefixCacheOptions options);
  PrefixCache(const PrefixCache&) = delete;
  PrefixCache& operator=(const PrefixCache&) = delete;
  ~PrefixCache();

  bool Enabled() const { return options_.enabled && options_.max_blocks != 0; }

  const PrefixCacheOptions& Options() const { return options_; }

  /**
   * @brief Longest run of block-aligned leading tokens of `tokens` that is already resident.
   * @param tokens The whole sequence the request wants computed.
   * @param max_adoptable_tokens Upper bound on tokens the caller may skip. A request must always
   *        compute at least its last token, so the caller passes one less than the sequence length.
   *
   * Marks every block it hands out as recently used, so adoption feeds the eviction order.
   */
  PrefixCacheMatch Match(std::span<const int32_t> tokens, size_t max_adoptable_tokens);

  /**
   * @brief Gives `block` a content identity and indexes it, taking a reference on it.
   * @param block A full block whose slots hold exactly `tokens`.
   * @param tokens The block-size-many tokens the block holds.
   * @param parent The exact identity of the block that precedes it, or null for the first block.
   * @return The identity the block after it chains from, or nothing when the chain has to stop.
   *
   * A block whose content is already indexed keeps no identity of its own and stays private: the
   * first physical copy serves every lookup, so a lookup never has to choose between duplicates.
   * The chain still continues through it, because the indexed copy holds the same tokens behind the
   * same parent.
   *
   * Nothing is returned when the identity is already taken by a different block (a collision) or
   * when the budget is full and nothing can be evicted. Neither this block nor anything after it is
   * reachable then, so the caller stops sealing.
   */
  std::shared_ptr<const BlockIdentity> Register(const std::shared_ptr<Block>& block,
                                                std::span<const int32_t> tokens,
                                                const std::shared_ptr<const BlockIdentity>& parent);

  /**
   * @brief Identity hash a chain starts from, before any block has contributed to it.
   */
  static uint64_t RootHash();
  static uint64_t ChainHash(uint64_t parent_hash, std::span<const int32_t> tokens);

  // The hash this cache addresses blocks with, which is ChainHash unless overridden.
  uint64_t Hash(uint64_t parent_hash, std::span<const int32_t> tokens) const {
    return options_.hash ? options_.hash(parent_hash, tokens) : ChainHash(parent_hash, tokens);
  }

  /**
   * @brief Returns indexed blocks that no request references back to the pool.
   * @param blocks_needed How many blocks the caller needs.
   * @return The number of blocks actually returned to the pool.
   *
   * Evicts in least-recently-used order and skips entries a request still holds, so reclaiming
   * never takes a block out from under a live sequence.
   */
  size_t Reclaim(size_t blocks_needed);

  // Indexed blocks no request currently references, which is the capacity Reclaim can hand back.
  size_t ReclaimableBlocks() const;

  size_t IndexedBlocks() const { return entries_.size(); }

  const PrefixCacheMetrics& Metrics() const { return metrics_; }

 private:
  struct Entry {
    std::shared_ptr<Block> block;
    std::shared_ptr<const BlockIdentity> identity;
    std::list<uint64_t>::iterator recency;
  };

  // Keeps `entry` ordered immediately before the entry it chains from, so a chain is always
  // evicted from its tail rather than its head.
  void Reorder(Entry& entry, const std::shared_ptr<const BlockIdentity>& parent);
  void Evict(std::unordered_map<uint64_t, Entry>::iterator it);

  BlockPool& block_pool_;
  PrefixCacheOptions options_;
  std::unordered_map<uint64_t, Entry> entries_;
  // Front is the least recently used identity, back the most recently used.
  std::list<uint64_t> recency_;
  PrefixCacheMetrics metrics_;
};

}  // namespace Generators
