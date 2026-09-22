// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "prefix_cache.h"

#include <algorithm>
#include <exception>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fixed_state_pool.h"

namespace Generators {

namespace {

// FNV-1a. The identity only has to be well distributed and reproducible within one process; every
// hit is verified against the stored tokens, so the hash never decides correctness on its own.
constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

uint64_t HashBytes(uint64_t hash, const void* data, size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint64_t>(bytes[i]);
    hash *= kFnvPrime;
  }
  return hash;
}

}  // namespace

uint64_t PrefixCache::RootHash() {
  return kFnvOffsetBasis;
}

uint64_t PrefixCache::ChainHash(uint64_t parent_hash, std::span<const int32_t> tokens) {
  uint64_t hash = HashBytes(kFnvOffsetBasis, &parent_hash, sizeof(parent_hash));
  const uint64_t token_count = tokens.size();
  hash = HashBytes(hash, &token_count, sizeof(token_count));
  for (const int32_t token : tokens) {
    hash = HashBytes(hash, &token, sizeof(token));
  }
  return hash;
}

PrefixCache::PrefixCache(BlockPool& block_pool, PrefixCacheOptions options)
    : block_pool_{block_pool},
      options_{options},
      entries_by_block_id_(block_pool.Capacity()) {
  block_pool_.SetReferenceObserver(this);
}

PrefixCache::~PrefixCache() {
  block_pool_.SetReferenceObserver(nullptr);
  // Hand every retained block back so the pool's accounting is balanced when the cache outlives its
  // requests. Blocks a request still holds simply lose the cache's reference. The single-block
  // release allocates nothing, so teardown cannot fail on a failed allocation.
  for (auto& value : entries_) {
    auto& entry = value.second;
    block_pool_.ClearReferenceObserverCookie(entry.block);
    entry.block->ClearIdentity();
    block_pool_.Release(entry.block);
  }
  entries_.clear();
  recency_.clear();
  referenced_entries_.clear();
  reclaimable_entries_.clear();
}

PrefixCacheMatch PrefixCache::Match(std::span<const int32_t> tokens,
                                    size_t max_adoptable_tokens) {
  PrefixCacheMatch match;
  if (!Enabled()) {
    return match;
  }

  const size_t block_size = block_pool_.BlockSize();
  if (block_size == 0) {
    return match;
  }

  const size_t adoptable =
      std::min(max_adoptable_tokens, tokens.size()) / block_size * block_size;
  ++metrics_.lookups;
  metrics_.queried_tokens += adoptable;

  uint64_t parent_hash = RootHash();
  std::shared_ptr<const BlockIdentity> parent;
  std::vector<std::unordered_map<uint64_t, Entry>::iterator> hits;
  size_t safe_hit_count = 0;
  std::shared_ptr<const FixedStatePrefixCheckpoint> checkpoint;
  for (size_t offset = 0; offset + block_size <= adoptable; offset += block_size) {
    const auto chunk = tokens.subspan(offset, block_size);
    const uint64_t hash = Hash(parent_hash, chunk);
    const auto it = entries_.find(hash);
    if (it == entries_.end()) {
      break;
    }

    // A hash match is not a match. The block's exact tokens are compared, and its parent is
    // compared as the identity object it was computed behind rather than as a hash of it, so a
    // collision anywhere in the chain costs a missed hit and can never splice a block onto a prefix
    // it was not computed behind.
    const auto& identity = *it->second.identity;
    if (identity.parent != parent ||
        identity.tokens.size() != chunk.size() ||
        !std::equal(identity.tokens.begin(), identity.tokens.end(), chunk.begin())) {
      ++metrics_.hash_collisions;
      break;
    }
    if (!it->second.block->IsFull()) {
      break;
    }

    hits.push_back(it);
    if (it->second.checkpoint &&
        it->second.checkpoint->TokenCount() == offset + block_size) {
      safe_hit_count = hits.size();
      checkpoint = it->second.checkpoint;
    }
    parent = it->second.identity;
    parent_hash = hash;
  }

  if (options_.requires_checkpoint) {
    hits.resize(safe_hit_count);
  }
  if (hits.empty() || (options_.requires_checkpoint && !checkpoint)) {
    return match;
  }

  match.blocks.reserve(hits.size());
  for (const auto& it : hits) {
    match.blocks.push_back(it->second.block);
  }
  match.token_count = hits.size() * block_size;
  match.fixed_state_checkpoint = std::move(checkpoint);

  return match;
}

PrefixCacheRegistration PrefixCache::Register(
    const std::shared_ptr<Block>& block,
    std::span<const int32_t> tokens,
    const std::shared_ptr<const BlockIdentity>& parent) {
  if (!Enabled()) {
    return {PrefixCacheRegistrationStatus::CapacityRefused, nullptr};
  }
  if (!block) {
    throw std::runtime_error("Cannot index a null block in the prefix cache.");
  }
  if (!block->IsFull() || tokens.size() != block->Capacity()) {
    throw std::runtime_error("Only a full block whose tokens cover every slot can be indexed.");
  }
  if (block->HasIdentity()) {
    // Already indexed, which is the normal case for an adopted block being re-walked.
    return {PrefixCacheRegistrationStatus::Indexed, block->IdentityPtr()};
  }
  if (!block_pool_.Owns(block)) {
    throw std::runtime_error("Cannot index a block the pool does not own.");
  }

  const uint64_t hash = Hash(parent ? parent->hash : RootHash(), tokens);
  const auto existing = entries_.find(hash);
  if (existing != entries_.end()) {
    const auto& identity = *existing->second.identity;
    if (identity.parent != parent ||
        identity.tokens.size() != tokens.size() ||
        !std::equal(identity.tokens.begin(), identity.tokens.end(), tokens.begin())) {
      // A different block already holds this identity. Nothing after it can be reached either, so
      // the caller stops here rather than indexing entries no lookup can ever verify.
      ++metrics_.hash_collisions;
      return {PrefixCacheRegistrationStatus::HashCollision, nullptr};
    }
    // Two sequences computed the same prefix before either was indexed. The first physical copy
    // serves every lookup, so this one stays private. Stop this request's sealing here because it
    // does not hold a request reference on the canonical physical block.
    ++metrics_.duplicate_registrations;
    Reorder(existing->second, parent);
    return {PrefixCacheRegistrationStatus::Duplicate, nullptr};
  }

  if (entries_.size() >= options_.max_blocks && Reclaim(1) == 0) {
    // The budget is full and every indexed block is still in use. Leaving this block unindexed is
    // the safe outcome: it stays private and is freed with its request.
    ++metrics_.retention_refusals;
    return {PrefixCacheRegistrationStatus::CapacityRefused, nullptr};
  }

  auto identity = std::make_shared<BlockIdentity>();
  identity->hash = hash;
  identity->parent = parent;
  identity->tokens.assign(tokens.begin(), tokens.end());

  // Everything that can fail happens first. Order the entry just behind its parent, so a chain is
  // always evicted from its tail: the head is what every longer match starts from, and losing it
  // would orphan everything chained behind it.
  const auto parent_entry = parent ? entries_.find(parent->hash) : entries_.end();
  Entry* const parent_entry_ptr =
      parent_entry == entries_.end() ? nullptr : &parent_entry->second;
  auto [entry_it, inserted] = entries_.try_emplace(
      hash, Entry{block, identity, nullptr, {}, {}, !parent_entry_ptr ? std::optional<size_t>{} : std::optional<size_t>{parent_entry_ptr->block->Id()}});
  if (!inserted) {
    throw std::logic_error("Prefix cache identity became occupied during registration.");
  }
  auto& entry = entry_it->second;
  try {
    const auto recency_position =
        !parent_entry_ptr ? recency_.end() : parent_entry_ptr->recency;
    entry.recency = recency_.insert(recency_position, &entry);
  } catch (...) {
    entries_.erase(entry_it);
    throw;
  }
  try {
    entry.reference_state = referenced_entries_.insert(referenced_entries_.end(), &entry);
  } catch (...) {
    recency_.erase(entry.recency);
    entries_.erase(entry_it);
    throw;
  }

  // The index owns a reference of its own, which is what keeps the block alive once its request
  // releases it. Publish the observer cookie last so no reference transition can expose a partially
  // initialized entry.
  bool cache_reference_added = false;
  bool identity_set = false;
  try {
    block_pool_.AddRef(block);
    cache_reference_added = true;
    block->SetIdentity(identity);
    identity_set = true;
    entries_by_block_id_[block->Id()] = &entry;
    block_pool_.SetReferenceObserverCookie(block, &entry);
  } catch (...) {
    if (identity_set) {
      block->ClearIdentity();
    }
    if (cache_reference_added) {
      block_pool_.Release(block);
    }
    referenced_entries_.erase(entry.reference_state);
    recency_.erase(entry.recency);
    entries_.erase(entry_it);
    throw;
  }
  ++metrics_.registered_blocks;
  return {PrefixCacheRegistrationStatus::Indexed, std::move(identity)};
}

void PrefixCache::RecordAdoption(
    std::span<const std::shared_ptr<Block>> blocks) noexcept {
  if (blocks.empty()) {
    return;
  }
  // Root-to-deepest keeps each child immediately before its parent. The root ends as the most
  // recently used entry, while eviction still takes the deepest disposable suffix first.
  for (const auto& block : blocks) {
    if (!block || !block->HasIdentity()) {
      std::terminate();
    }
    const auto& identity = block->IdentityPtr();
    const auto entry = entries_.find(identity->hash);
    if (entry == entries_.end() ||
        entry->second.block != block ||
        entry->second.identity != identity) {
      std::terminate();
    }
    entry->second.promote_on_release = true;
    Reorder(entry->second, identity->parent);
  }
  ++metrics_.hits;
  metrics_.matched_tokens += blocks.size() * block_pool_.BlockSize();
}

bool PrefixCache::CanAttachCheckpoint(
    const std::shared_ptr<const BlockIdentity>& identity) const {
  if (!Enabled() || !identity || options_.max_checkpoints == 0) {
    return false;
  }
  const auto entry = entries_.find(identity->hash);
  return entry != entries_.end() &&
         entry->second.identity == identity &&
         !entry->second.checkpoint;
}

bool PrefixCache::AttachCheckpoint(
    const std::shared_ptr<const BlockIdentity>& identity,
    std::shared_ptr<const FixedStatePrefixCheckpoint> checkpoint) {
  if (!identity || !checkpoint) {
    throw std::invalid_argument(
        "A prefix checkpoint requires an indexed identity and fixed state.");
  }
  const auto entry = entries_.find(identity->hash);
  if (entry == entries_.end() || entry->second.identity != identity) {
    return false;
  }
  if (entry->second.checkpoint) {
    return true;
  }

  size_t block_count = 0;
  for (auto current = identity; current; current = current->parent) {
    ++block_count;
  }
  if (checkpoint->TokenCount() !=
      block_count * block_pool_.BlockSize()) {
    throw std::runtime_error(
        "Fixed state checkpoint does not match its paged prefix boundary.");
  }
  if (checkpoint_count_ >= options_.max_checkpoints &&
      ReclaimCheckpoints(1) == 0) {
    return false;
  }

  entry->second.checkpoint = std::move(checkpoint);
  ++checkpoint_count_;
  Reorder(entry->second, entry->second.identity->parent);
  return true;
}

size_t PrefixCache::ReclaimCheckpoints(size_t checkpoints_needed) {
  size_t reclaimed = 0;
  for (auto recency = recency_.begin();
       reclaimed < checkpoints_needed && recency != recency_.end();
       ++recency) {
    auto& entry = **recency;
    if (entry.checkpoint &&
        entry.checkpoint.use_count() == 1) {
      entry.checkpoint.reset();
      --checkpoint_count_;
      ++reclaimed;
    }
  }
  return reclaimed;
}

size_t PrefixCache::ReclaimableCheckpoints() const {
  return static_cast<size_t>(std::count_if(
      entries_.begin(), entries_.end(), [](const auto& value) {
        return value.second.checkpoint &&
               value.second.checkpoint.use_count() == 1;
      }));
}

size_t PrefixCache::Reclaim(size_t blocks_needed) {
  size_t reclaimed = 0;
  while (reclaimed < blocks_needed && !reclaimable_entries_.empty()) {
    Entry* entry = reclaimable_entries_.front();
    if (!entry || !entry->reclaimable || entry->block->RefCount() != 1) {
      throw std::logic_error("Prefix cache reclaimable order contains an invalid entry.");
    }
    const auto entry_it = entries_.find(entry->identity->hash);
    if (entry_it == entries_.end()) {
      throw std::logic_error("Prefix cache reclaimable order references an unknown identity.");
    }
    Evict(entry_it);
    ++reclaimed;
    ++metrics_.evictions;
  }
  return reclaimed;
}

size_t PrefixCache::ReclaimableBlocks() const {
  return reclaimable_entries_.size();
}

void PrefixCache::Reorder(Entry& entry, const std::shared_ptr<const BlockIdentity>& parent) {
  // Every entry sits immediately before the entry it chains from, so the head of a chain is always
  // the most recently used of its run and eviction takes the tail first.
  if (!parent) {
    recency_.splice(recency_.end(), recency_, entry.recency);
    if (entry.reclaimable) {
      reclaimable_entries_.splice(
          reclaimable_entries_.end(), reclaimable_entries_, entry.reference_state);
    }
    return;
  }
  const auto parent_entry = entries_.find(parent->hash);
  if (parent_entry == entries_.end()) {
    // No lookup can reach this entry any more, so it is the first thing worth reclaiming.
    recency_.splice(recency_.begin(), recency_, entry.recency);
    if (entry.reclaimable) {
      reclaimable_entries_.splice(
          reclaimable_entries_.begin(), reclaimable_entries_, entry.reference_state);
    }
    return;
  }
  recency_.splice(parent_entry->second.recency, recency_, entry.recency);
  if (entry.reclaimable) {
    const auto position = parent_entry->second.reclaimable
                              ? parent_entry->second.reference_state
                              : reclaimable_entries_.end();
    reclaimable_entries_.splice(position, reclaimable_entries_, entry.reference_state);
  }
}

void PrefixCache::OnBlockBecameReferenced(Block& block, void* cookie) noexcept {
  auto* entry = static_cast<Entry*>(cookie);
  if (!entry || entry->block.get() != &block || !entry->reclaimable) {
    std::terminate();
  }
  referenced_entries_.splice(
      referenced_entries_.end(), reclaimable_entries_, entry->reference_state);
  entry->reclaimable = false;
  entry->promote_on_release = false;
}

void PrefixCache::OnBlockBecameReclaimable(Block& block, void* cookie) noexcept {
  auto* entry = static_cast<Entry*>(cookie);
  if (!entry || entry->block.get() != &block || entry->reclaimable) {
    std::terminate();
  }

  auto position = reclaimable_entries_.end();
  if (entry->promote_on_release) {
    if (entry->parent_block_id) {
      Entry* parent = entries_by_block_id_[*entry->parent_block_id];
      if (parent && parent->identity != entry->identity->parent) {
        parent = nullptr;
      }
      if (!parent) {
        position = reclaimable_entries_.begin();
      } else if (parent->reclaimable) {
        position = parent->reference_state;
      }
    }
  } else {
    for (auto next = std::next(entry->recency); next != recency_.end(); ++next) {
      if ((*next)->reclaimable) {
        position = (*next)->reference_state;
        break;
      }
    }
  }

  reclaimable_entries_.splice(
      position, referenced_entries_, entry->reference_state);
  entry->reclaimable = true;
  entry->promote_on_release = true;
}

void PrefixCache::Evict(std::unordered_map<uint64_t, Entry>::iterator it) {
  auto block = it->second.block;
  if (it->second.checkpoint) {
    --checkpoint_count_;
  }
  if (it->second.reclaimable) {
    reclaimable_entries_.erase(it->second.reference_state);
  } else {
    referenced_entries_.erase(it->second.reference_state);
  }
  recency_.erase(it->second.recency);
  entries_by_block_id_[block->Id()] = nullptr;
  block_pool_.ClearReferenceObserverCookie(block);
  entries_.erase(it);
  block->ClearIdentity();
  block_pool_.Release(block);
}

}  // namespace Generators
