// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "prefix_cache.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

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
    : block_pool_{block_pool}, options_{options} {}

PrefixCache::~PrefixCache() {
  // Hand every retained block back so the pool's accounting is balanced when the cache outlives its
  // requests. Blocks a request still holds simply lose the cache's reference. The single-block
  // release allocates nothing, so teardown cannot fail on a failed allocation.
  for (auto& [hash, entry] : entries_) {
    entry.block->ClearIdentity();
    block_pool_.Release(entry.block);
  }
  entries_.clear();
  recency_.clear();
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

  const size_t adoptable = std::min(max_adoptable_tokens, tokens.size());
  ++metrics_.lookups;
  metrics_.queried_tokens += adoptable;

  uint64_t parent_hash = RootHash();
  std::shared_ptr<const BlockIdentity> parent;
  std::vector<std::unordered_map<uint64_t, Entry>::iterator> hits;
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
    parent = it->second.identity;
    parent_hash = hash;
  }

  if (hits.size() < options_.min_match_blocks) {
    return match;
  }

  match.blocks.reserve(hits.size());
  for (const auto& it : hits) {
    match.blocks.push_back(it->second.block);
  }
  // Refresh the run head first, so each block lands immediately before the one it chains from and
  // the head of the chain ends up the most recently used. Evicting the head would orphan every
  // block behind it -- their identities chain from it, so no lookup can reach them again -- while
  // evicting the tail just shortens the prefix that stays reusable.
  for (const auto& it : hits) {
    Reorder(it->second, it->second.identity->parent);
  }
  match.token_count = hits.size() * block_size;

  ++metrics_.hits;
  metrics_.matched_tokens += match.token_count;
  return match;
}

std::shared_ptr<const BlockIdentity> PrefixCache::Register(
    const std::shared_ptr<Block>& block,
    std::span<const int32_t> tokens,
    const std::shared_ptr<const BlockIdentity>& parent) {
  if (!Enabled()) {
    return nullptr;
  }
  if (!block) {
    throw std::runtime_error("Cannot index a null block in the prefix cache.");
  }
  if (!block->IsFull() || tokens.size() != block->Capacity()) {
    throw std::runtime_error("Only a full block whose tokens cover every slot can be indexed.");
  }
  if (block->HasIdentity()) {
    // Already indexed, which is the normal case for an adopted block being re-walked.
    return block->IdentityPtr();
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
      return nullptr;
    }
    // Two sequences computed the same prefix before either was indexed. The first physical copy
    // serves every lookup, so this one stays private; the chain still continues through it because
    // the indexed copy holds exactly the same tokens behind the same parent.
    ++metrics_.duplicate_registrations;
    Reorder(existing->second, parent);
    return existing->second.identity;
  }

  if (entries_.size() >= options_.max_blocks && Reclaim(1) == 0) {
    // The budget is full and every indexed block is still in use. Leaving this block unindexed is
    // the safe outcome: it stays private and is freed with its request.
    ++metrics_.retention_refusals;
    return nullptr;
  }

  auto identity = std::make_shared<BlockIdentity>();
  identity->hash = hash;
  identity->parent = parent;
  identity->tokens.assign(tokens.begin(), tokens.end());

  // Everything that can fail happens first. Order the entry just behind its parent, so a chain is
  // always evicted from its tail: the head is what every longer match starts from, and losing it
  // would orphan everything chained behind it.
  const auto parent_entry = parent ? entries_.find(parent->hash) : entries_.end();
  const auto position = parent_entry == entries_.end() ? recency_.end() : parent_entry->second.recency;
  const auto recency = recency_.insert(position, hash);
  try {
    entries_.emplace(hash, Entry{block, identity, recency});
  } catch (...) {
    recency_.erase(recency);
    throw;
  }

  // The index owns a reference of its own, which is what keeps the block alive once its request
  // releases it. Neither step allocates, so the entry above cannot end up without its reference.
  block_pool_.AddRef(block);
  block->SetIdentity(identity);
  ++metrics_.registered_blocks;
  return identity;
}

size_t PrefixCache::Reclaim(size_t blocks_needed) {
  size_t reclaimed = 0;
  auto recency_it = recency_.begin();
  while (reclaimed < blocks_needed && recency_it != recency_.end()) {
    const auto entry_it = entries_.find(*recency_it);
    if (entry_it == entries_.end()) {
      throw std::logic_error("Prefix cache recency order references an unknown identity.");
    }
    // A block a request still holds is not the cache's to give back.
    if (entry_it->second.block->RefCount() > 1) {
      ++recency_it;
      continue;
    }
    ++recency_it;
    Evict(entry_it);
    ++reclaimed;
    ++metrics_.evictions;
  }
  return reclaimed;
}

size_t PrefixCache::ReclaimableBlocks() const {
  size_t reclaimable = 0;
  for (const auto& [hash, entry] : entries_) {
    if (entry.block->RefCount() == 1) {
      ++reclaimable;
    }
  }
  return reclaimable;
}

void PrefixCache::Reorder(Entry& entry, const std::shared_ptr<const BlockIdentity>& parent) {
  // Every entry sits immediately before the entry it chains from, so the head of a chain is always
  // the most recently used of its run and eviction takes the tail first.
  if (!parent) {
    recency_.splice(recency_.end(), recency_, entry.recency);
    return;
  }
  const auto parent_entry = entries_.find(parent->hash);
  if (parent_entry == entries_.end()) {
    // No lookup can reach this entry any more, so it is the first thing worth reclaiming.
    recency_.splice(recency_.begin(), recency_, entry.recency);
    return;
  }
  recency_.splice(parent_entry->second.recency, recency_, entry.recency);
}

void PrefixCache::Evict(std::unordered_map<uint64_t, Entry>::iterator it) {
  auto block = it->second.block;
  recency_.erase(it->second.recency);
  entries_.erase(it);
  block->ClearIdentity();
  block_pool_.Release(block);
}

}  // namespace Generators
