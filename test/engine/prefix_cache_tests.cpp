// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for the content-addressed block index, the reference counting that makes block sharing
// safe, and the copy-on-write guard that keeps a shared block from being written. These exercise the
// production policy types directly and need no model, device, or engine.

#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engine/block.h"
#include "engine/paged_key_value_cache.h"
#include "engine/prefix_cache.h"

namespace Generators {
namespace test {
namespace {

constexpr size_t kBlockSize = 4;

PrefixCacheOptions MakeOptions(size_t max_blocks, size_t min_match_blocks = 1) {
  PrefixCacheOptions options;
  options.enabled = true;
  options.max_blocks = max_blocks;
  options.min_match_blocks = min_match_blocks;
  return options;
}

// Takes one full block from the pool, marks every slot used, and indexes it behind `parent`.
std::shared_ptr<Block> SealBlock(BlockPool& pool, PrefixCache& cache,
                                 std::span<const int32_t> tokens,
                                 std::shared_ptr<const BlockIdentity>& parent) {
  auto blocks = pool.AllocateBlocks(kBlockSize);
  EXPECT_EQ(blocks.size(), 1u);
  auto chained = cache.Register(blocks.front(), tokens, parent);
  if (chained) {
    parent = std::move(chained);
  }
  return blocks.front();
}

// A BlockCopier double that records the copies the divergence policy asks for.
struct RecordingBlockCopier final : BlockCopier {
  void CopyBlock(size_t source_block_id, size_t destination_block_id) override {
    copies.emplace_back(source_block_id, destination_block_id);
  }
  std::vector<std::pair<size_t, size_t>> copies;
};

// ---------------------------------------------------------------------------------------------
// Reference counting
// ---------------------------------------------------------------------------------------------

TEST(BlockReferenceCountingTest, BlockReturnsToThePoolOnlyWhenTheLastOwnerReleasesIt) {
  BlockPool pool{kBlockSize, 2};
  auto blocks = pool.AllocateBlocks(kBlockSize);
  ASSERT_EQ(blocks.size(), 1u);
  EXPECT_EQ(blocks.front()->RefCount(), 1u);
  EXPECT_EQ(pool.AvailableBlocks(), 1u);

  pool.AddRef(blocks);
  EXPECT_EQ(blocks.front()->RefCount(), 2u);
  EXPECT_TRUE(blocks.front()->IsShared());

  pool.Free(blocks);
  EXPECT_EQ(blocks.front()->RefCount(), 1u);
  EXPECT_EQ(pool.AvailableBlocks(), 1u);

  pool.Free(blocks);
  EXPECT_EQ(pool.AvailableBlocks(), 2u);
}

// One reservation admitting two requests that adopt the same prefix holds two references to the
// same block and releases both together.
TEST(BlockReferenceCountingTest, ABlockCanBeReleasedOncePerReferenceInOneCall) {
  BlockPool pool{kBlockSize, 1};
  auto blocks = pool.AllocateBlocks(kBlockSize);
  pool.AddRef(blocks);
  ASSERT_EQ(blocks.front()->RefCount(), 2u);

  const std::vector<std::shared_ptr<Block>> twice{blocks.front(), blocks.front()};
  pool.Free(twice);

  EXPECT_EQ(pool.AvailableBlocks(), 1u);
}

TEST(BlockReferenceCountingTest, ReleasingMoreReferencesThanAreHeldIsRejectedWithoutMutating) {
  BlockPool pool{kBlockSize, 1};
  auto blocks = pool.AllocateBlocks(kBlockSize);
  const std::vector<std::shared_ptr<Block>> twice{blocks.front(), blocks.front()};

  EXPECT_THROW(pool.Free(twice), std::runtime_error);
  EXPECT_EQ(blocks.front()->RefCount(), 1u);
  EXPECT_EQ(pool.AvailableBlocks(), 0u);
}

TEST(BlockReferenceCountingTest, AddingAReferenceToAForeignBlockIsRejected) {
  BlockPool pool{kBlockSize, 1};
  BlockPool other{kBlockSize, 1};
  auto foreign = other.AllocateBlocks(kBlockSize);

  EXPECT_THROW(pool.AddRef(foreign), std::runtime_error);
}

// ---------------------------------------------------------------------------------------------
// Content addressing
// ---------------------------------------------------------------------------------------------

TEST(PrefixCacheTest, ExactPrefixMatchAdoptsEveryFullBlock) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(8)};

  const std::array<int32_t, 9> prompt{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::shared_ptr<const BlockIdentity> parent;
  SealBlock(pool, cache, std::span<const int32_t>{prompt}.subspan(0, kBlockSize), parent);
  SealBlock(pool, cache, std::span<const int32_t>{prompt}.subspan(kBlockSize, kBlockSize), parent);

  const auto match = cache.Match(prompt, prompt.size() - 1);

  EXPECT_EQ(match.blocks.size(), 2u);
  EXPECT_EQ(match.token_count, 2 * kBlockSize);
  EXPECT_EQ(cache.Metrics().hits, 1u);
  EXPECT_EQ(cache.Metrics().matched_tokens, 2 * kBlockSize);
}

TEST(PrefixCacheTest, PartialPrefixMatchStopsAtTheFirstDivergingBlock) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(8)};

  const std::array<int32_t, 9> indexed{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::shared_ptr<const BlockIdentity> parent;
  SealBlock(pool, cache, std::span<const int32_t>{indexed}.subspan(0, kBlockSize), parent);
  SealBlock(pool, cache, std::span<const int32_t>{indexed}.subspan(kBlockSize, kBlockSize), parent);

  // Same first block, different second block.
  const std::array<int32_t, 9> probe{1, 2, 3, 4, 50, 60, 70, 80, 90};
  const auto match = cache.Match(probe, probe.size() - 1);

  EXPECT_EQ(match.blocks.size(), 1u);
  EXPECT_EQ(match.token_count, kBlockSize);
}

TEST(PrefixCacheTest, ADifferentPromptMatchesNothing) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(8)};

  const std::array<int32_t, 4> indexed{1, 2, 3, 4};
  std::shared_ptr<const BlockIdentity> parent;
  SealBlock(pool, cache, indexed, parent);

  const std::array<int32_t, 5> probe{9, 9, 9, 9, 9};
  EXPECT_TRUE(cache.Match(probe, probe.size() - 1).Empty());
  EXPECT_EQ(cache.Metrics().hits, 0u);
}

// The last token always has to go through the model, so a prompt that is exactly one indexed block
// long adopts nothing.
TEST(PrefixCacheTest, TheFinalTokenIsNeverAdopted) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(8)};

  const std::array<int32_t, 4> prompt{1, 2, 3, 4};
  std::shared_ptr<const BlockIdentity> parent;
  SealBlock(pool, cache, prompt, parent);

  EXPECT_TRUE(cache.Match(prompt, prompt.size() - 1).Empty());
}

// The identity chains the block before it, so identical tokens at a different offset do not match.
TEST(PrefixCacheTest, TheSameBlockContentAtADifferentOffsetDoesNotMatch) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(8)};

  const std::array<int32_t, 8> indexed{1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<const BlockIdentity> parent;
  SealBlock(pool, cache, std::span<const int32_t>{indexed}.subspan(0, kBlockSize), parent);
  SealBlock(pool, cache, std::span<const int32_t>{indexed}.subspan(kBlockSize, kBlockSize), parent);

  // Starts with the tokens of the second indexed block, which sat behind a different prefix.
  const std::array<int32_t, 5> probe{5, 6, 7, 8, 9};
  EXPECT_TRUE(cache.Match(probe, probe.size() - 1).Empty());
}

// A hash match is never treated as proof of a match. The hash is overridden so distinct blocks
// deterministically land on the same identity.
TEST(PrefixCacheTest, CollidingContentIsRejectedRatherThanServed) {
  BlockPool pool{kBlockSize, 8};
  auto options = MakeOptions(8);
  options.hash = [](uint64_t, std::span<const int32_t>) { return uint64_t{7}; };
  PrefixCache cache{pool, options};

  const std::array<int32_t, 4> indexed{1, 2, 3, 4};
  std::shared_ptr<const BlockIdentity> parent;
  SealBlock(pool, cache, indexed, parent);
  ASSERT_EQ(cache.IndexedBlocks(), 1u);

  // Different content, same identity: indexing it must not displace the first block.
  const std::array<int32_t, 4> colliding{5, 6, 7, 8};
  std::shared_ptr<const BlockIdentity> colliding_parent;
  auto collided = SealBlock(pool, cache, colliding, colliding_parent);
  EXPECT_FALSE(collided->HasIdentity());
  EXPECT_EQ(cache.IndexedBlocks(), 1u);
  EXPECT_EQ(cache.Metrics().hash_collisions, 1u);

  // And a lookup for the colliding content finds the identity but rejects the tokens.
  const std::array<int32_t, 5> probe{5, 6, 7, 8, 9};
  EXPECT_TRUE(cache.Match(probe, probe.size() - 1).Empty());
  EXPECT_EQ(cache.Metrics().hash_collisions, 2u);

  // The original content still matches, so the collision cost a miss and nothing else.
  const std::array<int32_t, 5> original{1, 2, 3, 4, 9};
  EXPECT_EQ(cache.Match(original, original.size() - 1).blocks.size(), 1u);
}

// The dangerous collision is not on a block's own tokens but on its ancestry: a block whose parent
// was evicted must never be spliced onto a different prefix that happens to hash the same. The
// block's parent is compared as the identity it was computed behind, not as a hash of it, so this
// cannot happen even when every hash collides.
TEST(PrefixCacheTest, ABlockIsNeverSplicedOntoAPrefixItWasNotComputedBehind) {
  BlockPool pool{kBlockSize, 8};
  auto options = MakeOptions(8);
  options.hash = [](uint64_t parent_hash, std::span<const int32_t> tokens) {
    // Every first block collides with every other first block, and every second block collides with
    // every other second block, so only the lineage check can tell them apart.
    return parent_hash == PrefixCache::RootHash() ? uint64_t{1} : uint64_t{2};
  };
  PrefixCache cache{pool, options};

  const std::array<int32_t, 8> original{1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<const BlockIdentity> parent;
  auto head = SealBlock(pool, cache, std::span<const int32_t>{original}.subspan(0, kBlockSize), parent);
  SealBlock(pool, cache, std::span<const int32_t>{original}.subspan(kBlockSize, kBlockSize), parent);
  ASSERT_EQ(cache.IndexedBlocks(), 2u);

  // The head is evicted, but the block behind it stays indexed under the colliding identity.
  pool.Free({head});
  ASSERT_EQ(cache.Reclaim(1), 1u);
  ASSERT_EQ(cache.IndexedBlocks(), 1u);

  // A different first block now takes the head's identity, and its follower is the original's.
  const std::array<int32_t, 8> impostor{9, 9, 9, 9, 5, 6, 7, 8};
  std::shared_ptr<const BlockIdentity> impostor_parent;
  SealBlock(pool, cache, std::span<const int32_t>{impostor}.subspan(0, kBlockSize), impostor_parent);

  const std::array<int32_t, 9> probe{9, 9, 9, 9, 5, 6, 7, 8, 1};
  const auto match = cache.Match(probe, probe.size() - 1);

  // The impostor's own block matches, but the block computed behind a different prefix does not.
  EXPECT_EQ(match.token_count, kBlockSize);
}

TEST(PrefixCacheTest, DuplicateContentKeepsTheFirstPhysicalBlock) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(8)};

  const std::array<int32_t, 4> tokens{1, 2, 3, 4};
  std::shared_ptr<const BlockIdentity> first_parent;
  auto first = SealBlock(pool, cache, tokens, first_parent);
  std::shared_ptr<const BlockIdentity> second_parent;
  auto second = SealBlock(pool, cache, tokens, second_parent);

  EXPECT_TRUE(first->HasIdentity());
  EXPECT_FALSE(second->HasIdentity());
  EXPECT_EQ(cache.IndexedBlocks(), 1u);
  EXPECT_EQ(cache.Metrics().duplicate_registrations, 1u);

  const std::array<int32_t, 5> probe{1, 2, 3, 4, 9};
  const auto match = cache.Match(probe, probe.size() - 1);
  ASSERT_EQ(match.blocks.size(), 1u);
  EXPECT_EQ(match.blocks.front()->Id(), first->Id());
}

TEST(PrefixCacheTest, AMatchShorterThanTheMinimumIsNotWorthAdopting) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(8, /*min_match_blocks=*/2)};

  const std::array<int32_t, 4> tokens{1, 2, 3, 4};
  std::shared_ptr<const BlockIdentity> parent;
  SealBlock(pool, cache, tokens, parent);

  const std::array<int32_t, 6> probe{1, 2, 3, 4, 9, 9};
  EXPECT_TRUE(cache.Match(probe, probe.size() - 1).Empty());
}

TEST(PrefixCacheTest, ADisabledCacheIndexesAndMatchesNothing) {
  BlockPool pool{kBlockSize, 8};
  PrefixCacheOptions options;
  options.enabled = false;
  options.max_blocks = 8;
  PrefixCache cache{pool, options};

  const std::array<int32_t, 4> tokens{1, 2, 3, 4};
  std::shared_ptr<const BlockIdentity> parent;
  auto block = SealBlock(pool, cache, tokens, parent);

  EXPECT_FALSE(block->HasIdentity());
  EXPECT_EQ(cache.IndexedBlocks(), 0u);
  const std::array<int32_t, 5> probe{1, 2, 3, 4, 9};
  EXPECT_TRUE(cache.Match(probe, probe.size() - 1).Empty());
}

// ---------------------------------------------------------------------------------------------
// Retention and eviction
// ---------------------------------------------------------------------------------------------

// A block its request has released survives on the index's own reference, which is what lets the
// next turn of the same conversation adopt it.
TEST(PrefixCacheTest, AnIndexedBlockOutlivesTheRequestThatProducedIt) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(8)};

  const std::array<int32_t, 4> tokens{1, 2, 3, 4};
  std::shared_ptr<const BlockIdentity> parent;
  auto block = SealBlock(pool, cache, tokens, parent);
  ASSERT_EQ(block->RefCount(), 2u);

  pool.Free({block});  // The request finished.

  EXPECT_EQ(block->RefCount(), 1u);
  EXPECT_EQ(pool.AvailableBlocks(), 7u);
  EXPECT_EQ(cache.ReclaimableBlocks(), 1u);
  const std::array<int32_t, 5> probe{1, 2, 3, 4, 9};
  EXPECT_EQ(cache.Match(probe, probe.size() - 1).blocks.size(), 1u);
}

TEST(PrefixCacheTest, ReclaimReturnsRetainedBlocksInLeastRecentlyUsedOrder) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(8)};

  const std::array<int32_t, 4> older{1, 2, 3, 4};
  const std::array<int32_t, 4> newer{5, 6, 7, 8};
  std::shared_ptr<const BlockIdentity> older_parent;
  auto older_block = SealBlock(pool, cache, older, older_parent);
  std::shared_ptr<const BlockIdentity> newer_parent;
  auto newer_block = SealBlock(pool, cache, newer, newer_parent);
  pool.Free({older_block});
  pool.Free({newer_block});
  ASSERT_EQ(cache.ReclaimableBlocks(), 2u);

  EXPECT_EQ(cache.Reclaim(1), 1u);

  // The older entry went first, so the newer one still matches.
  const std::array<int32_t, 5> newer_probe{5, 6, 7, 8, 9};
  EXPECT_EQ(cache.Match(newer_probe, newer_probe.size() - 1).blocks.size(), 1u);
  const std::array<int32_t, 5> older_probe{1, 2, 3, 4, 9};
  EXPECT_TRUE(cache.Match(older_probe, older_probe.size() - 1).Empty());
  EXPECT_EQ(cache.Metrics().evictions, 1u);
}

// Evicting the head of a chain would orphan everything behind it, because their identities chain
// from it and no lookup can reach them again. Eviction has to take the tail first.
TEST(PrefixCacheTest, EvictionTakesTheTailOfAChainBeforeItsHead) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(8)};

  const std::array<int32_t, 8> chain{1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<const BlockIdentity> parent;
  auto head = SealBlock(pool, cache, std::span<const int32_t>{chain}.subspan(0, kBlockSize), parent);
  auto tail = SealBlock(pool, cache, std::span<const int32_t>{chain}.subspan(kBlockSize, kBlockSize),
                        parent);
  pool.Free({head});
  pool.Free({tail});

  EXPECT_EQ(cache.Reclaim(1), 1u);

  // The head survives, so the shorter prefix is still reusable.
  const std::array<int32_t, 9> probe{1, 2, 3, 4, 5, 6, 7, 8, 9};
  const auto match = cache.Match(probe, probe.size() - 1);
  ASSERT_EQ(match.blocks.size(), 1u);
  EXPECT_EQ(match.blocks.front()->Id(), head->Id());
}

// A hit refreshes the whole run, and it has to keep the head newer than the tail so the next
// eviction still takes the tail.
TEST(PrefixCacheTest, AHitKeepsTheHeadOfTheChainNewerThanItsTail) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(8)};

  const std::array<int32_t, 8> chain{1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<const BlockIdentity> parent;
  auto head = SealBlock(pool, cache, std::span<const int32_t>{chain}.subspan(0, kBlockSize), parent);
  auto tail = SealBlock(pool, cache, std::span<const int32_t>{chain}.subspan(kBlockSize, kBlockSize),
                        parent);
  pool.Free({head});
  pool.Free({tail});

  const std::array<int32_t, 9> probe{1, 2, 3, 4, 5, 6, 7, 8, 9};
  ASSERT_EQ(cache.Match(probe, probe.size() - 1).blocks.size(), 2u);

  EXPECT_EQ(cache.Reclaim(1), 1u);
  const auto match = cache.Match(probe, probe.size() - 1);
  ASSERT_EQ(match.blocks.size(), 1u);
  EXPECT_EQ(match.blocks.front()->Id(), head->Id());
}

// Retention must never starve a live request: a block a request still holds is not the cache's to
// give back.
TEST(PrefixCacheTest, ReclaimSkipsBlocksARequestStillHolds) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(8)};

  const std::array<int32_t, 4> held{1, 2, 3, 4};
  const std::array<int32_t, 4> retained{5, 6, 7, 8};
  std::shared_ptr<const BlockIdentity> held_parent;
  auto held_block = SealBlock(pool, cache, held, held_parent);
  std::shared_ptr<const BlockIdentity> retained_parent;
  auto retained_block = SealBlock(pool, cache, retained, retained_parent);
  pool.Free({retained_block});

  EXPECT_EQ(cache.ReclaimableBlocks(), 1u);
  EXPECT_EQ(cache.Reclaim(2), 1u);
  EXPECT_EQ(held_block->RefCount(), 2u);
  EXPECT_TRUE(held_block->HasIdentity());
}

TEST(PrefixCacheTest, RetentionStaysWithinItsBlockBudget) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(/*max_blocks=*/2)};

  std::vector<std::shared_ptr<Block>> blocks;
  for (int32_t i = 0; i < 3; ++i) {
    const std::array<int32_t, 4> tokens{i, i, i, i};
    std::shared_ptr<const BlockIdentity> parent;
    auto block = SealBlock(pool, cache, tokens, parent);
    pool.Free({block});  // Each request finishes immediately.
    blocks.push_back(block);
  }

  EXPECT_EQ(cache.IndexedBlocks(), 2u);
  EXPECT_EQ(cache.Metrics().evictions, 1u);
}

TEST(PrefixCacheTest, AFullBudgetOfLiveBlocksRefusesRetentionRatherThanEvicting) {
  BlockPool pool{kBlockSize, 8};
  PrefixCache cache{pool, MakeOptions(/*max_blocks=*/1)};

  const std::array<int32_t, 4> live{1, 2, 3, 4};
  std::shared_ptr<const BlockIdentity> live_parent;
  auto live_block = SealBlock(pool, cache, live, live_parent);

  const std::array<int32_t, 4> refused{5, 6, 7, 8};
  std::shared_ptr<const BlockIdentity> refused_parent;
  auto refused_block = SealBlock(pool, cache, refused, refused_parent);

  EXPECT_TRUE(live_block->HasIdentity());
  EXPECT_FALSE(refused_block->HasIdentity());
  EXPECT_EQ(cache.Metrics().retention_refusals, 1u);
  EXPECT_EQ(cache.IndexedBlocks(), 1u);
}

// ---------------------------------------------------------------------------------------------
// Copy-on-write divergence
// ---------------------------------------------------------------------------------------------

// Two owners sharing a block that is still being written must not both write into it. The writer
// takes a private copy of the block's key-value data and diverges into that instead.
TEST(CopyOnWriteTest, AWriterDivergesFromASharedTailBlockByCopyingIt) {
  BlockPool pool{kBlockSize, 4};
  auto blocks = pool.AllocateBlocks(1);  // One slot used, so the block is still being written.
  ASSERT_EQ(blocks.size(), 1u);
  pool.AddRef(blocks);  // A second owner references the same partially filled block.

  PagedCacheBlockTable table;
  table.request_id = &table;
  table.committed_slots = 1;
  table.blocks = blocks;

  RecordingBlockCopier copier;
  const bool copied = MakeTailBlockExclusive(table, /*target_slots=*/2, pool, copier);

  ASSERT_TRUE(copied);
  ASSERT_EQ(copier.copies.size(), 1u);
  EXPECT_EQ(copier.copies.front().first, blocks.front()->Id());
  EXPECT_EQ(copier.copies.front().second, table.blocks.front()->Id());
  EXPECT_NE(table.blocks.front()->Id(), blocks.front()->Id());
  // The private copy carries the same tokens so far, and neither owner is shared any more.
  EXPECT_EQ(table.blocks.front()->Size(), 1u);
  EXPECT_FALSE(table.blocks.front()->IsShared());
  EXPECT_EQ(blocks.front()->RefCount(), 1u);
}

TEST(CopyOnWriteTest, AnExclusiveTailBlockIsWrittenInPlace) {
  BlockPool pool{kBlockSize, 4};
  auto blocks = pool.AllocateBlocks(1);

  PagedCacheBlockTable table;
  table.request_id = &table;
  table.committed_slots = 1;
  table.blocks = blocks;

  RecordingBlockCopier copier;
  EXPECT_FALSE(MakeTailBlockExclusive(table, /*target_slots=*/2, pool, copier));
  EXPECT_TRUE(copier.copies.empty());
  EXPECT_EQ(table.blocks.front()->Id(), blocks.front()->Id());
}

// A step that writes nothing has nothing to diverge from, so a shared block stays shared.
TEST(CopyOnWriteTest, AStepThatWritesNothingLeavesASharedBlockAlone) {
  BlockPool pool{kBlockSize, 4};
  auto blocks = pool.AllocateBlocks(kBlockSize);
  pool.AddRef(blocks);

  PagedCacheBlockTable table;
  table.request_id = &table;
  table.committed_slots = kBlockSize;
  table.blocks = blocks;

  RecordingBlockCopier copier;
  EXPECT_FALSE(MakeTailBlockExclusive(table, /*target_slots=*/kBlockSize, pool, copier));
  EXPECT_TRUE(blocks.front()->IsShared());
}

}  // namespace
}  // namespace test
}  // namespace Generators
