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
#include "dflash2_drafter.h"

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

bool LogicalIdentityEquals(
    std::shared_ptr<const LogicalPrefixIdentity> left,
    std::shared_ptr<const LogicalPrefixIdentity> right) {
  while (left || right) {
    if (left == right) {
      return true;
    }
    if (!left || !right || left->hash != right->hash ||
        left->tokens.size() != right->tokens.size() ||
        !std::equal(left->tokens.begin(), left->tokens.end(), right->tokens.begin())) {
      return false;
    }
    left = left->parent;
    right = right->parent;
  }
  return true;
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
  entries_.reserve(options_.max_blocks);
  entries_by_hash_.reserve(options_.max_blocks);
  checkpoint_entries_.reserve(options_.max_checkpoints);
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
  entries_by_hash_.clear();
  recency_.clear();
  referenced_entries_.clear();
  reclaimable_entries_.clear();
  checkpoint_entries_.clear();
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

  if (options_.requires_checkpoint) {
    // A hybrid hit is an indivisible physical history plus its fixed-state snapshot. Scan only
    // bounded checkpoint endpoints, longest first and then most-recently-used for deterministic
    // ties. Rebuilding the exact parent path prevents logical variants from ever being spliced.
    for (auto it = checkpoint_entries_.rbegin(); it != checkpoint_entries_.rend(); ++it) {
      Entry* endpoint = *it;
      if (!endpoint || !endpoint->checkpoint) {
        std::terminate();
      }
      const size_t token_count = endpoint->checkpoint->TokenCount();
      if (token_count > adoptable || token_count <= match.token_count) {
        continue;
      }
      std::vector<std::shared_ptr<Block>> blocks;
      if (!HasRetainedPhysicalPath(
              *endpoint, token_count / block_size, tokens.first(token_count), &blocks)) {
        continue;
      }
      match.token_count = token_count;
      match.blocks = std::move(blocks);
      match.fixed_state_checkpoint = endpoint->checkpoint;
      match.draft_checkpoint = endpoint->draft_checkpoint;
    }
    if (!match.fixed_state_checkpoint) {
      return {};
    }
    ++metrics_.matches;
    return match;
  }

  uint64_t parent_hash = RootHash();
  std::shared_ptr<const BlockIdentity> parent;
  for (size_t offset = 0; offset + block_size <= adoptable; offset += block_size) {
    const auto chunk = tokens.subspan(offset, block_size);
    const uint64_t hash = Hash(parent_hash, chunk);
    Entry* entry = FindPhysical(hash, parent, chunk);
    if (!entry) {
      if (entries_by_hash_.find(hash) != entries_by_hash_.end()) {
        ++metrics_.hash_collisions;
      }
      break;
    }
    if (!entry->block->IsFull()) {
      break;
    }

    match.blocks.push_back(entry->block);
    parent = entry->identity;
    parent_hash = hash;
  }
  if (match.blocks.empty()) {
    return match;
  }
  match.token_count = match.blocks.size() * block_size;
  ++metrics_.matches;

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
  if (auto* existing = FindPhysical(hash, parent, tokens)) {
    // Two sequences computed the same prefix before either was indexed. The first physical copy
    // serves every paged-only lookup.
    ++metrics_.duplicate_registrations;
    Reorder(*existing, parent);
    return {PrefixCacheRegistrationStatus::Duplicate, nullptr};
  }
  if (entries_by_hash_.find(hash) != entries_by_hash_.end()) {
    // A different block already holds this identity. Nothing after it can be reached either, so
    // the caller stops here rather than indexing entries no lookup can ever verify.
    ++metrics_.hash_collisions;
    return {PrefixCacheRegistrationStatus::HashCollision, nullptr};
  }

  if (entries_.size() >= options_.max_blocks && Reclaim(1) == 0) {
    // The budget is full and every indexed block is still in use. Leaving this block unindexed is
    // the safe outcome: it stays private and is freed with its request.
    ++metrics_.retention_refusals;
    return {PrefixCacheRegistrationStatus::CapacityRefused, nullptr};
  }

  auto identity = std::make_shared<BlockIdentity>();
  identity->hash = hash;
  identity->block_id = block->Id();
  identity->parent = parent;
  if (options_.requires_checkpoint || options_.max_checkpoints != 0) {
    auto logical = std::make_shared<LogicalPrefixIdentity>();
    logical->hash = hash;
    logical->parent = parent ? parent->logical : nullptr;
    logical->tokens.assign(tokens.begin(), tokens.end());
    identity->logical = std::move(logical);
  } else {
    identity->tokens.assign(tokens.begin(), tokens.end());
  }

  // Everything that can fail happens first. Order the entry just behind its parent, so a chain is
  // always evicted from its tail: the head is what every longer match starts from, and losing it
  // would orphan everything chained behind it.
  Entry* const parent_entry_ptr = FindEntry(parent);
  const auto parent_block_id =
      parent_entry_ptr ? std::optional<size_t>{parent_entry_ptr->block->Id()} : std::nullopt;
  auto [entry_it, inserted] = entries_.try_emplace(
      block->Id(), Entry{block, identity, nullptr, nullptr, {}, {}, parent_block_id});
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
  try {
    entries_by_hash_.emplace(hash, &entry);
  } catch (...) {
    referenced_entries_.erase(entry.reference_state);
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
    const auto range = entries_by_hash_.equal_range(hash);
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second == &entry) {
        entries_by_hash_.erase(it);
        break;
      }
    }
    referenced_entries_.erase(entry.reference_state);
    recency_.erase(entry.recency);
    entries_.erase(entry_it);
    throw;
  }
  ++metrics_.registered_blocks;
  return {PrefixCacheRegistrationStatus::Indexed, std::move(identity)};
}

PrefixCacheRegistrationStatus PrefixCache::CheckCheckpointedPrefix(
    std::span<const std::shared_ptr<Block>> blocks,
    std::span<const int32_t> tokens,
    const std::shared_ptr<const BlockIdentity>& parent) {
  return PlanCheckpointedPrefix(blocks, tokens, parent).status;
}

PrefixCache::CheckpointedPrefixPlan PrefixCache::PlanCheckpointedPrefix(
    std::span<const std::shared_ptr<Block>> blocks,
    std::span<const int32_t> tokens,
    const std::shared_ptr<const BlockIdentity>& parent) {
  const size_t block_size = block_pool_.BlockSize();
  if (blocks.empty() || tokens.size() / block_size != blocks.size() ||
      tokens.size() % block_size != 0) {
    throw std::invalid_argument("A hybrid prefix publication requires complete blocks.");
  }
  CheckpointedPrefixPlan plan;
  if (options_.max_checkpoints == 0) {
    ++metrics_.retention_refusals;
    plan.status = PrefixCacheRegistrationStatus::CapacityRefused;
    return plan;
  }
  auto logical_parent = parent ? parent->logical : nullptr;
  Entry* duplicate_root = nullptr;
  size_t prefix_block_count = 0;
  for (auto identity = parent; identity; identity = identity->parent) {
    ++prefix_block_count;
  }
  for (size_t index = 0; index < blocks.size(); ++index) {
    const auto& block = blocks[index];
    if (!block || block->HasIdentity() || !block->IsFull() ||
        block->Capacity() != block_size || !block_pool_.Owns(block)) {
      throw std::logic_error("A hybrid prefix suffix must contain only private blocks.");
    }
    const auto chunk = tokens.subspan(index * block_size, block_size);
    const uint64_t hash = Hash(logical_parent ? logical_parent->hash : RootHash(), chunk);
    Entry* existing = FindLogical(hash, logical_parent, chunk);
    if (!existing && entries_by_hash_.find(hash) != entries_by_hash_.end()) {
      ++metrics_.hash_collisions;
      plan.status = PrefixCacheRegistrationStatus::HashCollision;
      return plan;
    }
    if (existing) {
      logical_parent = existing->identity->logical;
      if (!duplicate_root) {
        duplicate_root = existing;
      }
    } else {
      auto logical = std::make_shared<LogicalPrefixIdentity>();
      logical->hash = hash;
      logical->parent = logical_parent;
      logical->tokens.assign(chunk.begin(), chunk.end());
      logical_parent = std::move(logical);
    }
    plan.logical_identities.push_back(logical_parent);

    // Only a complete independently usable endpoint is a duplicate. An intermediate logical
    // block may be recomputed behind another physical history and must not stop hybrid sealing.
    if (index + 1 == blocks.size()) {
      const auto range = entries_by_hash_.equal_range(hash);
      for (auto candidate = range.first; candidate != range.second; ++candidate) {
        if (LogicalIdentityEquals(candidate->second->identity->logical, logical_parent) &&
            candidate->second->checkpoint &&
            candidate->second->checkpoint->TokenCount() ==
                (prefix_block_count + blocks.size()) * block_size &&
            HasRetainedPhysicalPath(
                *candidate->second, prefix_block_count + blocks.size())) {
          ++metrics_.duplicate_registrations;
          plan.status = PrefixCacheRegistrationStatus::Duplicate;
          return plan;
        }
      }
    }
  }

  // One checkpoint cannot retain independent histories. Preserve the established transactional
  // whole-suffix replacement policy in that configuration.
  if (options_.max_checkpoints <= 1 && duplicate_root) {
    for (const auto* entry : recency_) {
      auto identity = entry->identity;
      while (identity && identity != duplicate_root->identity) {
        identity = identity->parent;
      }
      if (!identity) {
        continue;
      }
      if (entry->block->RefCount() != 1 ||
          (entry->checkpoint && entry->checkpoint.use_count() != 1) ||
          (entry->draft_checkpoint && entry->draft_checkpoint.use_count() != 1)) {
        ++metrics_.retention_refusals;
        plan.status = PrefixCacheRegistrationStatus::CapacityRefused;
        return plan;
      }
      plan.retiring_block_ids.push_back(entry->block->Id());
    }
  }

  if (blocks.size() > options_.max_blocks) {
    ++metrics_.retention_refusals;
    plan.status = PrefixCacheRegistrationStatus::CapacityRefused;
    return plan;
  }
  size_t retained_after_retirement = entries_.size() - plan.retiring_block_ids.size();
  for (auto* entry : reclaimable_entries_) {
    if (retained_after_retirement + blocks.size() <= options_.max_blocks) {
      break;
    }
    if (!IsProtectedByLeasedCheckpoint(*entry) &&
        std::find(plan.retiring_block_ids.begin(), plan.retiring_block_ids.end(),
                  entry->block->Id()) == plan.retiring_block_ids.end()) {
      plan.retiring_block_ids.push_back(entry->block->Id());
      --retained_after_retirement;
    }
  }
  if (retained_after_retirement + blocks.size() > options_.max_blocks ||
      (CheckpointCount() >= options_.max_checkpoints && ReclaimableCheckpoints() == 0)) {
    ++metrics_.retention_refusals;
    plan.status = PrefixCacheRegistrationStatus::CapacityRefused;
  }
  return plan;
}

PrefixCacheRegistration PrefixCache::ReplaceCheckpointedPrefix(
    std::span<const std::shared_ptr<Block>> blocks,
    std::span<const int32_t> tokens,
    const std::shared_ptr<const BlockIdentity>& parent,
    const CheckpointedPrefixPlan& plan,
    const std::function<std::shared_ptr<const FixedStatePrefixCheckpoint>()>& capture_checkpoint) {
  std::unordered_map<size_t, Entry> staged;
  std::unordered_multimap<uint64_t, Entry*> staged_by_hash;
  std::list<Entry*> staged_recency;
  std::list<Entry*> staged_references;
  std::vector<size_t> staged_block_ids;
  staged_block_ids.reserve(blocks.size());
  entries_.reserve(entries_.size() + blocks.size());
  entries_by_hash_.reserve(entries_by_hash_.size() + blocks.size());
  auto identity = parent;
  size_t token_count = tokens.size();
  for (auto ancestor = parent; ancestor; ancestor = ancestor->parent) {
    token_count += block_pool_.BlockSize();
  }
  for (size_t index = 0; index < blocks.size(); ++index) {
    const auto chunk = tokens.subspan(index * block_pool_.BlockSize(), block_pool_.BlockSize());
    auto next = std::make_shared<BlockIdentity>();
    next->hash = Hash(identity ? identity->hash : RootHash(), chunk);
    if (next->hash != plan.logical_identities[index]->hash) {
      throw std::logic_error("A hybrid prefix hash changed between planning and publication.");
    }
    next->block_id = blocks[index]->Id();
    next->parent = identity;
    next->logical = plan.logical_identities[index];
    std::optional<size_t> parent_block_id;
    if (index != 0) {
      parent_block_id = blocks[index - 1]->Id();
    } else if (parent) {
      const auto* ancestor = FindEntry(parent);
      if (!ancestor) {
        throw std::logic_error("A hybrid replacement lost its adopted parent.");
      }
      parent_block_id = ancestor->block->Id();
    }
    auto [it, inserted] = staged.try_emplace(
        blocks[index]->Id(),
        Entry{blocks[index], next, nullptr, nullptr, {}, {}, parent_block_id});
    if (!inserted) {
      ++metrics_.hash_collisions;
      return {PrefixCacheRegistrationStatus::HashCollision, nullptr};
    }
    auto& entry = it->second;
    entry.recency = staged_recency.insert(staged_recency.end(), &entry);
    entry.reference_state = staged_references.insert(staged_references.end(), &entry);
    staged_by_hash.emplace(next->hash, &entry);
    staged_block_ids.push_back(blocks[index]->Id());
    identity = std::move(next);
  }
  auto checkpoint = capture_checkpoint();
  if (!checkpoint) {
    return {PrefixCacheRegistrationStatus::CapacityRefused, nullptr};
  }
  if (checkpoint->TokenCount() != token_count) {
    throw std::runtime_error("Fixed state checkpoint does not match its paged prefix boundary.");
  }

  // Node handles, list splices, and prevalidated block references publish without allocation.
  // Until this point the old suffix and its checkpoints remain available on any failure.
  for (const size_t block_id : plan.retiring_block_ids) {
    const auto entry = entries_.find(block_id);
    if (entry == entries_.end()) {
      std::terminate();
    }
    Evict(entry);
    ++metrics_.evictions;
  }
  if (CheckpointCount() >= options_.max_checkpoints && ReclaimCheckpoints(1) != 1) {
    std::terminate();
  }
  auto current = parent;
  for (size_t index = 0; index < blocks.size(); ++index) {
    const auto& block = blocks[index];
    const auto pending = staged.find(staged_block_ids[index]);
    if (pending == staged.end()) {
      std::terminate();
    }
    auto result = entries_.insert(staged.extract(pending));
    if (!result.inserted) {
      std::terminate();
    }
    auto& entry = result.position->second;
    Entry* ancestor = FindEntry(current);
    recency_.splice(
        ancestor ? ancestor->recency : recency_.end(),
        staged_recency, entry.recency);
    referenced_entries_.splice(
        referenced_entries_.end(), staged_references, entry.reference_state);
    const auto hash_range = staged_by_hash.equal_range(entry.identity->hash);
    auto hash_node = hash_range.first;
    while (hash_node != hash_range.second && hash_node->second != &entry) {
      ++hash_node;
    }
    if (hash_node == hash_range.second) {
      std::terminate();
    }
    entries_by_hash_.insert(staged_by_hash.extract(hash_node));
    block_pool_.AddRef(block);
    block->SetIdentity(entry.identity);
    entries_by_block_id_[block->Id()] = &entry;
    block_pool_.SetReferenceObserverCookie(block, &entry);
    current = entry.identity;
    ++metrics_.registered_blocks;
  }
  auto* endpoint = FindEntry(identity);
  if (!endpoint) {
    std::terminate();
  }
  endpoint->checkpoint = std::move(checkpoint);
  checkpoint_entries_.push_back(endpoint);
  return {PrefixCacheRegistrationStatus::Indexed, std::move(identity)};
}

PrefixCacheRegistration PrefixCache::RegisterCheckpointedPrefix(
    std::span<const std::shared_ptr<Block>> blocks,
    std::span<const int32_t> tokens,
    const std::shared_ptr<const BlockIdentity>& parent,
    std::shared_ptr<const FixedStatePrefixCheckpoint> checkpoint) {
  if (!checkpoint) {
    throw std::invalid_argument("A hybrid prefix publication requires a checkpoint.");
  }
  return RegisterCheckpointedPrefix(
      blocks, tokens, parent, [checkpoint = std::move(checkpoint)] { return checkpoint; });
}

PrefixCacheRegistration PrefixCache::RegisterCheckpointedPrefix(
    std::span<const std::shared_ptr<Block>> blocks,
    std::span<const int32_t> tokens,
    const std::shared_ptr<const BlockIdentity>& parent,
    const std::function<std::shared_ptr<const FixedStatePrefixCheckpoint>()>& capture_checkpoint) {
  if (!capture_checkpoint) {
    throw std::invalid_argument("A hybrid prefix publication requires a checkpoint capture.");
  }
  const auto plan = PlanCheckpointedPrefix(blocks, tokens, parent);
  if (plan.status != PrefixCacheRegistrationStatus::Indexed) {
    return {plan.status, nullptr};
  }
  return ReplaceCheckpointedPrefix(blocks, tokens, parent, plan, capture_checkpoint);
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
    auto* entry = FindEntry(identity);
    if (!entry || entry->block != block) {
      std::terminate();
    }
    entry->promote_on_release = true;
    Reorder(*entry, identity->parent);
    PromoteCheckpoint(*entry);
  }
  ++metrics_.hits;
  metrics_.matched_tokens += blocks.size() * block_pool_.BlockSize();
}

bool PrefixCache::CanAttachCheckpoint(
    const std::shared_ptr<const BlockIdentity>& identity) const {
  if (!Enabled() || !identity || options_.max_checkpoints == 0) {
    return false;
  }
  const auto* entry = FindEntry(identity);
  return entry && !entry->checkpoint;
}

bool PrefixCache::AttachCheckpoint(
    const std::shared_ptr<const BlockIdentity>& identity,
    std::shared_ptr<const FixedStatePrefixCheckpoint> checkpoint) {
  if (!identity || !checkpoint) {
    throw std::invalid_argument(
        "A prefix checkpoint requires an indexed identity and fixed state.");
  }
  auto* entry = FindEntry(identity);
  if (!entry) {
    return false;
  }
  if (entry->checkpoint) {
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
  if (CheckpointCount() >= options_.max_checkpoints &&
      ReclaimCheckpoints(1) == 0) {
    return false;
  }

  entry->checkpoint = std::move(checkpoint);
  checkpoint_entries_.push_back(entry);
  Reorder(*entry, entry->identity->parent);
  return true;
}

bool PrefixCache::CanAttachDraftCheckpoint(
    const std::shared_ptr<const BlockIdentity>& identity, size_t token_count) const {
  if (!Enabled() || !identity) {
    return false;
  }
  const auto* entry = FindEntry(identity);
  return entry && entry->checkpoint && entry->checkpoint->TokenCount() == token_count &&
         !entry->draft_checkpoint;
}

std::shared_ptr<const FixedStatePrefixCheckpoint> PrefixCache::DraftBoundary(
    const std::shared_ptr<const BlockIdentity>& identity, size_t token_count) const {
  const auto* entry = FindEntry(identity);
  return CanAttachDraftCheckpoint(identity, token_count) ? entry->checkpoint : nullptr;
}

bool PrefixCache::AttachDraftCheckpoint(
    const std::shared_ptr<const BlockIdentity>& identity,
    const std::shared_ptr<const FixedStatePrefixCheckpoint>& fixed_checkpoint,
    std::shared_ptr<const Dflash2PrefixCheckpoint> draft_checkpoint) {
  if (!draft_checkpoint || !fixed_checkpoint ||
      !CanAttachDraftCheckpoint(identity, draft_checkpoint->token_count)) {
    return false;
  }
  auto* entry = FindEntry(identity);
  if (!entry || entry->checkpoint != fixed_checkpoint) {
    return false;
  }
  entry->draft_checkpoint = std::move(draft_checkpoint);
  return true;
}

void PrefixCache::DropUnleasedDraftCheckpoints() {
  for (auto& [hash, entry] : entries_) {
    if (entry.draft_checkpoint && entry.draft_checkpoint.use_count() == 1) {
      entry.draft_checkpoint.reset();
    }
  }
}

bool PrefixCache::ReclaimDraftCheckpoint() {
  for (auto* entry : recency_) {
    if (entry->draft_checkpoint && entry->draft_checkpoint.use_count() == 1) {
      entry->draft_checkpoint.reset();
      return true;
    }
  }
  return false;
}

size_t PrefixCache::ReclaimCheckpoints(size_t checkpoints_needed) {
  size_t reclaimed = 0;
  while (reclaimed < checkpoints_needed) {
    const auto* selected = ReclaimableCheckpoint();
    if (!selected || !ReclaimCheckpoint(selected)) {
      break;
    }
    ++reclaimed;
  }
  return reclaimed;
}

bool PrefixCache::ReclaimCheckpoint(const FixedStatePrefixCheckpoint* checkpoint) {
  if (!checkpoint) {
    return false;
  }
  for (size_t index = 0; index < checkpoint_entries_.size(); ++index) {
    auto* entry = checkpoint_entries_[index];
    if (entry->checkpoint.get() == checkpoint && IsCheckpointUnleased(*entry)) {
      entry->checkpoint.reset();
      entry->draft_checkpoint.reset();
      checkpoint_entries_.erase(checkpoint_entries_.begin() + index);
      return true;
    }
  }
  return false;
}

const FixedStatePrefixCheckpoint* PrefixCache::ReclaimableCheckpoint(
    const std::shared_ptr<const BlockIdentity>& current_path) const {
  // Prefer the deepest checkpoint on the publishing request's current physical path. It is an
  // intermediate snapshot once that request advances and replacing it preserves independent
  // terminal branches.
  for (auto identity = current_path; identity; identity = identity->parent) {
    const auto* entry = FindEntry(identity);
    if (entry && IsCheckpointUnleased(*entry)) {
      return entry->checkpoint.get();
    }
  }
  // Next discard old intermediate endpoints that still have a retained checkpointed descendant.
  for (auto* entry : checkpoint_entries_) {
    if (IsCheckpointUnleased(*entry) &&
        HasCheckpointedDescendant(*entry)) {
      return entry->checkpoint.get();
    }
  }
  // Independent terminal endpoints are replaced only when capacity requires it.
  for (auto* entry : checkpoint_entries_) {
    if (IsCheckpointUnleased(*entry)) {
      return entry->checkpoint.get();
    }
  }
  return nullptr;
}

size_t PrefixCache::ReclaimableCheckpoints() const {
  return static_cast<size_t>(std::count_if(
      checkpoint_entries_.begin(), checkpoint_entries_.end(), [](const auto* entry) {
        return IsCheckpointUnleased(*entry);
      }));
}

size_t PrefixCache::Reclaim(size_t blocks_needed) {
  size_t reclaimed = 0;
  while (reclaimed < blocks_needed) {
    const auto candidate = std::find_if(
        reclaimable_entries_.begin(), reclaimable_entries_.end(),
        [this](const Entry* entry) {
          return entry && !IsProtectedByLeasedCheckpoint(*entry);
        });
    if (candidate == reclaimable_entries_.end()) {
      break;
    }
    Entry* entry = *candidate;
    if (!entry || !entry->reclaimable || entry->block->RefCount() != 1) {
      throw std::logic_error("Prefix cache reclaimable order contains an invalid entry.");
    }
    const auto entry_it = entries_.find(entry->block->Id());
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
  return static_cast<size_t>(std::count_if(
      reclaimable_entries_.begin(), reclaimable_entries_.end(),
      [this](const Entry* entry) {
        return entry && !IsProtectedByLeasedCheckpoint(*entry);
      }));
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
  auto* parent_entry = FindEntry(parent);
  if (!parent_entry) {
    // No lookup can reach this entry any more, so it is the first thing worth reclaiming.
    recency_.splice(recency_.begin(), recency_, entry.recency);
    if (entry.reclaimable) {
      reclaimable_entries_.splice(
          reclaimable_entries_.begin(), reclaimable_entries_, entry.reference_state);
    }
    return;
  }
  recency_.splice(parent_entry->recency, recency_, entry.recency);
  if (entry.reclaimable) {
    const auto position = parent_entry->reclaimable
                              ? parent_entry->reference_state
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

void PrefixCache::Evict(std::unordered_map<size_t, Entry>::iterator it) {
  auto block = it->second.block;
  if (it->second.checkpoint) {
    const auto checkpoint_entry =
        std::find(checkpoint_entries_.begin(), checkpoint_entries_.end(), &it->second);
    if (checkpoint_entry == checkpoint_entries_.end()) {
      std::terminate();
    }
    checkpoint_entries_.erase(checkpoint_entry);
  }
  if (it->second.reclaimable) {
    reclaimable_entries_.erase(it->second.reference_state);
  } else {
    referenced_entries_.erase(it->second.reference_state);
  }
  recency_.erase(it->second.recency);
  entries_by_block_id_[block->Id()] = nullptr;
  block_pool_.ClearReferenceObserverCookie(block);
  const auto range = entries_by_hash_.equal_range(it->second.identity->hash);
  for (auto hash_it = range.first; hash_it != range.second; ++hash_it) {
    if (hash_it->second == &it->second) {
      entries_by_hash_.erase(hash_it);
      break;
    }
  }
  entries_.erase(it);
  block->ClearIdentity();
  block_pool_.Release(block);
}

PrefixCache::Entry* PrefixCache::FindEntry(
    const std::shared_ptr<const BlockIdentity>& identity) const noexcept {
  if (!identity || identity->block_id >= entries_by_block_id_.size()) {
    return nullptr;
  }
  auto* entry = entries_by_block_id_[identity->block_id];
  return entry && entry->identity == identity ? entry : nullptr;
}

PrefixCache::Entry* PrefixCache::FindPhysical(
    uint64_t hash, const std::shared_ptr<const BlockIdentity>& parent,
    std::span<const int32_t> tokens) const {
  const auto range = entries_by_hash_.equal_range(hash);
  for (auto it = range.first; it != range.second; ++it) {
    auto* entry = it->second;
    const auto& identity = *entry->identity;
    const auto stored_tokens = identity.Tokens();
    if (identity.parent == parent && stored_tokens.size() == tokens.size() &&
        std::equal(stored_tokens.begin(), stored_tokens.end(), tokens.begin())) {
      return entry;
    }
  }
  return nullptr;
}

PrefixCache::Entry* PrefixCache::FindLogical(
    uint64_t hash, const std::shared_ptr<const LogicalPrefixIdentity>& parent,
    std::span<const int32_t> tokens) const {
  const auto range = entries_by_hash_.equal_range(hash);
  for (auto it = range.first; it != range.second; ++it) {
    auto* entry = it->second;
    const auto& logical = *entry->identity->logical;
    if (LogicalIdentityEquals(logical.parent, parent) &&
        logical.tokens.size() == tokens.size() &&
        std::equal(logical.tokens.begin(), logical.tokens.end(), tokens.begin())) {
      return entry;
    }
  }
  return nullptr;
}

bool PrefixCache::HasRetainedPhysicalPath(
    const Entry& endpoint, size_t block_count, std::span<const int32_t> tokens,
    std::vector<std::shared_ptr<Block>>* blocks) const {
  const size_t block_size = block_pool_.BlockSize();
  if (!tokens.empty() &&
      (tokens.size() % block_size != 0 || tokens.size() / block_size != block_count)) {
    return false;
  }
  if (blocks) {
    blocks->resize(block_count);
  }
  const auto fail = [&] {
    if (blocks) {
      blocks->clear();
    }
    return false;
  };
  auto identity = endpoint.identity;
  for (size_t remaining = block_count; remaining != 0; --remaining) {
    const auto* entry = FindEntry(identity);
    if (!entry || !entry->block->IsFull()) {
      return fail();
    }
    if (!tokens.empty()) {
      const auto chunk = tokens.subspan((remaining - 1) * block_size, block_size);
      const auto stored_tokens = identity->Tokens();
      if (stored_tokens.size() != chunk.size() ||
          !std::equal(stored_tokens.begin(), stored_tokens.end(), chunk.begin())) {
        return fail();
      }
    }
    if (blocks) {
      (*blocks)[remaining - 1] = entry->block;
    }
    identity = identity->parent;
  }
  if (identity) {
    return fail();
  }
  return true;
}

bool PrefixCache::IsProtectedByLeasedCheckpoint(const Entry& entry) const {
  for (const auto* endpoint : checkpoint_entries_) {
    if ((!endpoint->checkpoint || endpoint->checkpoint.use_count() == 1) &&
        (!endpoint->draft_checkpoint || endpoint->draft_checkpoint.use_count() == 1)) {
      continue;
    }
    for (auto identity = endpoint->identity; identity; identity = identity->parent) {
      if (identity == entry.identity) {
        return true;
      }
    }
  }
  return false;
}

bool PrefixCache::IsCheckpointUnleased(const Entry& entry) {
  return entry.checkpoint && entry.checkpoint.use_count() == 1 &&
         (!entry.draft_checkpoint || entry.draft_checkpoint.use_count() == 1);
}

bool PrefixCache::HasCheckpointedDescendant(const Entry& ancestor) const {
  for (const auto* candidate : checkpoint_entries_) {
    if (candidate == &ancestor) {
      continue;
    }
    for (auto identity = candidate->identity->parent; identity; identity = identity->parent) {
      if (identity == ancestor.identity) {
        return true;
      }
    }
  }
  return false;
}

void PrefixCache::PromoteCheckpoint(Entry& entry) noexcept {
  const auto found = std::find(checkpoint_entries_.begin(), checkpoint_entries_.end(), &entry);
  if (found != checkpoint_entries_.end() && std::next(found) != checkpoint_entries_.end()) {
    std::rotate(found, std::next(found), checkpoint_entries_.end());
  }
}

}  // namespace Generators
