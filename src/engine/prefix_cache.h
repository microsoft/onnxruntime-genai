// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
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

class FixedStatePrefixCheckpoint;
struct Dflash2PrefixCheckpoint;
using DraftPrefixBoundary =
    std::pair<std::shared_ptr<const BlockIdentity>, std::shared_ptr<const FixedStatePrefixCheckpoint>>;

struct PrefixCacheOptions {
  bool enabled{};
  // Upper bound on blocks the index may hold. Retention beyond this evicts the least recently used
  // unreferenced entry. Zero disables the cache regardless of `enabled`.
  size_t max_blocks{};
  // Hybrid target models require a fixed-state checkpoint at the same boundary as the paged
  // blocks. Paged-only models leave this false and continue matching every complete block.
  bool requires_checkpoint{};
  size_t max_checkpoints{};
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
  std::shared_ptr<const FixedStatePrefixCheckpoint> fixed_state_checkpoint;
  std::shared_ptr<const Dflash2PrefixCheckpoint> draft_checkpoint;

  bool Empty() const { return blocks.empty(); }
};

enum class PrefixCacheRegistrationStatus {
  Indexed,
  Duplicate,
  HashCollision,
  CapacityRefused,
};

struct PrefixCacheRegistration {
  PrefixCacheRegistrationStatus status{};
  std::shared_ptr<const BlockIdentity> identity;

  bool StopsSealing() const {
    return status == PrefixCacheRegistrationStatus::Duplicate ||
           status == PrefixCacheRegistrationStatus::HashCollision;
  }
};

struct PrefixCacheMetrics {
  uint64_t lookups{};                  // Prompts offered to the index.
  uint64_t matches{};                  // Lookups that found an adoptable prefix.
  uint64_t deferred_matches{};         // Matching candidates deferred before execution.
  uint64_t hits{};                     // Matches whose adoption committed.
  uint64_t queried_tokens{};           // Prompt tokens eligible for adoption across all lookups.
  uint64_t matched_tokens{};           // Prompt tokens actually adopted (prefill work skipped).
  uint64_t registered_blocks{};        // Blocks given a content identity.
  uint64_t duplicate_registrations{};  // Blocks whose content was already indexed elsewhere.
  uint64_t hash_collisions{};          // Distinct contents that hashed to an indexed identity.
  uint64_t evictions{};                // Indexed blocks dropped to make room.
  uint64_t retention_refusals{};       // Registrations skipped because nothing was evictable.
  uint64_t publication_refusals{};     // Boundaries skipped after metadata allocation failures.
};

/**
 * @class PrefixCache
 * @brief Content-addressed index over a BlockPool's filled blocks with bounded LRU retention.
 *
 * The index holds one reference per indexed block, so an indexed block survives the request that
 * produced it. `Reclaim` gives that capacity straight back to the pool under memory pressure, which
 * is what keeps retention from ever starving a live request.
 */
class PrefixCache final : private BlockReferenceObserver {
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
   * Matching does not affect eviction order. After adoption commits, the caller must pass the
   * adopted blocks to RecordAdoption so committed use refreshes recency and metrics.
   */
  PrefixCacheMatch Match(std::span<const int32_t> tokens, size_t max_adoptable_tokens);

  /**
   * @brief Gives `block` a content identity and indexes it, taking a reference on it.
   * @param block A full block whose slots hold exactly `tokens`.
   * @param tokens The block-size-many tokens the block holds.
   * @param parent The exact identity of the block that precedes it, or null for the first block.
   * @return The registration outcome and, when indexed, the identity the next block chains from.
   *
   * A block whose content is already indexed keeps no identity of its own and stays private: the
   * first physical copy serves every lookup, so a lookup never has to choose between duplicates.
   * Sealing stops at that duplicate because its request does not own the canonical physical block
   * and therefore cannot keep the canonical lineage alive for later blocks.
   *
   * Nothing is returned when the identity is already taken by a different block (a collision) or
   * when the budget is full and nothing can be evicted. Neither this block nor anything after it is
   * reachable then, so the caller stops sealing.
   */
  PrefixCacheRegistration Register(
      const std::shared_ptr<Block>& block,
      std::span<const int32_t> tokens,
      const std::shared_ptr<const BlockIdentity>& parent);

  // Publishes a hybrid suffix only with its checkpoint. Failed publication removes all identities
  // created by this call, leaving the physical blocks private and eligible for a later retry.
  PrefixCacheRegistrationStatus CheckCheckpointedPrefix(
      std::span<const std::shared_ptr<Block>> blocks,
      std::span<const int32_t> tokens,
      const std::shared_ptr<const BlockIdentity>& parent);
  PrefixCacheRegistration RegisterCheckpointedPrefix(
      std::span<const std::shared_ptr<Block>> blocks,
      std::span<const int32_t> tokens,
      const std::shared_ptr<const BlockIdentity>& parent,
      std::shared_ptr<const FixedStatePrefixCheckpoint> checkpoint);
  PrefixCacheRegistration RegisterCheckpointedPrefix(
      std::span<const std::shared_ptr<Block>> blocks,
      std::span<const int32_t> tokens,
      const std::shared_ptr<const BlockIdentity>& parent,
      const std::function<std::shared_ptr<const FixedStatePrefixCheckpoint>()>& capture_checkpoint);

  // Publishes and refreshes a match only after its adopting cache transaction commits.
  void RecordAdoption(
      std::span<const std::shared_ptr<Block>> blocks) noexcept;
  void RecordDeferredMatches(size_t count) noexcept {
    metrics_.deferred_matches += count;
  }
  void RecordPublicationRefusal() noexcept {
    ++metrics_.publication_refusals;
  }

  bool CanAttachCheckpoint(
      const std::shared_ptr<const BlockIdentity>& identity) const;
  bool AttachCheckpoint(
      const std::shared_ptr<const BlockIdentity>& identity,
      std::shared_ptr<const FixedStatePrefixCheckpoint> checkpoint);
  bool CanAttachDraftCheckpoint(const std::shared_ptr<const BlockIdentity>& identity,
                                size_t token_count) const;
  std::shared_ptr<const FixedStatePrefixCheckpoint> DraftBoundary(
      const std::shared_ptr<const BlockIdentity>& identity, size_t token_count) const;
  bool AttachDraftCheckpoint(const std::shared_ptr<const BlockIdentity>& identity,
                             const std::shared_ptr<const FixedStatePrefixCheckpoint>& fixed_checkpoint,
                             std::shared_ptr<const Dflash2PrefixCheckpoint> draft_checkpoint);
  void DropUnleasedDraftCheckpoints();
  bool ReclaimDraftCheckpoint();
  size_t ReclaimCheckpoints(size_t checkpoints_needed);
  bool ReclaimCheckpoint(const FixedStatePrefixCheckpoint* checkpoint);
  size_t ReclaimableCheckpoints() const;
  const FixedStatePrefixCheckpoint* ReclaimableCheckpoint(
      const std::shared_ptr<const BlockIdentity>& current_path = nullptr) const;
  size_t CheckpointCount() const { return checkpoint_entries_.size(); }

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
    std::shared_ptr<const FixedStatePrefixCheckpoint> checkpoint;
    std::shared_ptr<const Dflash2PrefixCheckpoint> draft_checkpoint;
    std::list<Entry*>::iterator recency;
    std::list<Entry*>::iterator reference_state;
    std::optional<size_t> parent_block_id;
    bool reclaimable{};
    bool promote_on_release{true};
  };

  struct CheckpointedPrefixPlan {
    PrefixCacheRegistrationStatus status{PrefixCacheRegistrationStatus::Indexed};
    std::vector<size_t> retiring_block_ids;
    std::vector<std::shared_ptr<const LogicalPrefixIdentity>> logical_identities;
  };
  CheckpointedPrefixPlan PlanCheckpointedPrefix(
      std::span<const std::shared_ptr<Block>> blocks,
      std::span<const int32_t> tokens,
      const std::shared_ptr<const BlockIdentity>& parent);
  PrefixCacheRegistration ReplaceCheckpointedPrefix(
      std::span<const std::shared_ptr<Block>> blocks,
      std::span<const int32_t> tokens,
      const std::shared_ptr<const BlockIdentity>& parent,
      const CheckpointedPrefixPlan& plan,
      const std::function<std::shared_ptr<const FixedStatePrefixCheckpoint>()>& capture_checkpoint);

  // Keeps `entry` ordered immediately before the entry it chains from, so a chain is always
  // evicted from its tail rather than its head.
  void Reorder(Entry& entry, const std::shared_ptr<const BlockIdentity>& parent);
  Entry* FindEntry(const std::shared_ptr<const BlockIdentity>& identity) const noexcept;
  Entry* FindPhysical(uint64_t hash, const std::shared_ptr<const BlockIdentity>& parent,
                      std::span<const int32_t> tokens) const;
  Entry* FindLogical(uint64_t hash,
                     const std::shared_ptr<const LogicalPrefixIdentity>& parent,
                     std::span<const int32_t> tokens) const;
  bool HasRetainedPhysicalPath(
      const Entry& endpoint, size_t block_count,
      std::span<const int32_t> tokens = {},
      std::vector<std::shared_ptr<Block>>* blocks = nullptr) const;
  bool IsProtectedByLeasedCheckpoint(const Entry& entry) const;
  static bool IsCheckpointUnleased(const Entry& entry);
  bool HasCheckpointedDescendant(const Entry& ancestor) const;
  void PromoteCheckpoint(Entry& entry) noexcept;
  void OnBlockBecameReferenced(Block& block, void* cookie) noexcept override;
  void OnBlockBecameReclaimable(Block& block, void* cookie) noexcept override;
  void Evict(std::unordered_map<size_t, Entry>::iterator it);

  BlockPool& block_pool_;
  PrefixCacheOptions options_;
  // Physical block ID owns entries. Hash lookup is a non-owning multimap because hybrid
  // checkpointed publication may retain several exact physical histories for one logical prefix.
  std::unordered_map<size_t, Entry> entries_;
  std::unordered_multimap<uint64_t, Entry*> entries_by_hash_;
  // Front is the least recently used identity, back the most recently used.
  std::list<Entry*> recency_;
  // Every entry always owns one preallocated node in exactly one of these lists. Reference-count
  // transitions splice that node without allocating, including in validated no-throw publication.
  std::list<Entry*> referenced_entries_;
  std::list<Entry*> reclaimable_entries_;
  std::vector<Entry*> entries_by_block_id_;
  // LRU checkpoint endpoints. Capacity is reserved at construction, so promotion and publication
  // are allocation-free after checkpoint capture starts.
  std::vector<Entry*> checkpoint_entries_;
  PrefixCacheMetrics metrics_;
};

}  // namespace Generators
