// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// End-to-end tests for prefix-aware paged key-value caching. These drive the production admission,
// planning, reservation, commit, and retention path -- the real PagedCacheManager, the real dynamic
// scheduler, and the real Engine transaction -- with only model execution replaced by a double, so
// they run deterministically without a GPU or a real model.

#include <array>
#include <memory>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "engine/engine_invariants.h"
#include "engine/paged_key_value_cache.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"

namespace Generators {
namespace test {
namespace {

constexpr size_t kBlockSize = 4;

class PrefixCachingEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadDummyDecoderModel();
    Config::Engine::DynamicBatching dynamic_batching;
    dynamic_batching.block_size = kBlockSize;
    dynamic_batching.num_blocks = 32;
    dynamic_batching.max_batch_size = 4;
    dynamic_batching.max_scheduled_tokens = 64;
    dynamic_batching.prefix_caching = true;
    dynamic_batching.prefix_cache_max_blocks = 16;
    model_->config_->engine.dynamic_batching = dynamic_batching;
  }

  Config::Engine::DynamicBatching& Batching() {
    return *model_->config_->engine.dynamic_batching;
  }

  // A prompt whose leading tokens repeat between turns, which is the shape agent traffic has: a
  // long shared system prompt and tool schema followed by a short varying suffix.
  static std::vector<int32_t> MakePrompt(size_t shared_tokens, int32_t suffix) {
    std::vector<int32_t> prompt(shared_tokens);
    std::iota(prompt.begin(), prompt.end(), 2);
    prompt.push_back(suffix);
    return prompt;
  }

  // Runs one request to completion, releases it the way an application would, and returns the
  // tokens the engine generated for it.
  std::vector<int32_t> RunToCompletion(PagedEngine& paged,
                                       const std::vector<int32_t>& prompt,
                                       int max_length) {
    auto params = MakeGreedyParams(*model_);
    params->search.max_length = max_length;
    auto request = std::make_shared<Request>(params);
    request->AddTokens(prompt);
    paged.engine->AddRequest(request);

    std::vector<int32_t> generated;
    while (paged.engine->HasPendingRequests()) {
      auto ready = paged.engine->Step();
      if (!ready) break;
      while (ready->HasUnseenTokens()) {
        generated.push_back(ready->UnseenToken());
      }
    }
    EXPECT_EQ(request->status_, RequestStatus::Completed);
    paged.engine->RemoveRequest(request);
    return generated;
  }

  std::shared_ptr<Model> model_;
};

// The decisive correctness property: a request served with a warm cache produces exactly the tokens
// it produces cold, while computing strictly less prefill.
TEST_F(PrefixCachingEngineTest, WarmCacheProducesTokenForTokenIdenticalOutput) {
  const auto prompt = MakePrompt(11, /*suffix=*/29);

  auto cold_engine = MakePagedEngine(model_);
  const auto cold = RunToCompletion(cold_engine, prompt, /*max_length=*/16);
  const size_t cold_tokens = cold_engine.executor->TotalDecodedTokens();

  auto warm_engine = MakePagedEngine(model_);
  RunToCompletion(warm_engine, prompt, /*max_length=*/16);  // Warms the index.
  const size_t warm_up_tokens = warm_engine.executor->TotalDecodedTokens();
  const auto warm = RunToCompletion(warm_engine, prompt, /*max_length=*/16);
  const size_t warm_tokens = warm_engine.executor->TotalDecodedTokens() - warm_up_tokens;

  EXPECT_FALSE(cold.empty());
  EXPECT_EQ(warm, cold);
  EXPECT_LT(warm_tokens, cold_tokens);

  const auto* metrics = warm_engine.engine->PrefixCacheStats();
  ASSERT_NE(metrics, nullptr);
  EXPECT_EQ(metrics->hits, 1u);
  // Prompt is 12 tokens and the last one always runs, so 11 are adoptable: two whole blocks.
  EXPECT_EQ(metrics->matched_tokens, 2 * kBlockSize);
  EXPECT_EQ(cold_tokens - warm_tokens, 2 * kBlockSize);
}

// The prefix a second request adopts is the one the first request is still holding, so sharing
// works between live requests and not only across turns.
TEST_F(PrefixCachingEngineTest, ConcurrentRequestsShareTheSamePrefixBlocks) {
  auto paged = MakePagedEngine(model_);
  const auto prompt = MakePrompt(11, /*suffix=*/29);

  auto params = MakeGreedyParams(*model_);
  params->search.max_length = 32;
  auto first = std::make_shared<Request>(params);
  first->AddTokens(prompt);
  paged.engine->AddRequest(first);
  paged.engine->Step();  // Prefill and seal the first request's blocks.

  auto second = std::make_shared<Request>(MakeGreedyParams(*model_));
  second->AddTokens(prompt);
  paged.engine->AddRequest(second);
  paged.engine->Step();

  // The whole uncached remainder of the prompt runs in one step, so the cursor lands on the end of
  // the prompt with the first eight tokens never recomputed.
  EXPECT_EQ(second->ProcessedSequenceLength(), static_cast<int64_t>(prompt.size()));
  EXPECT_EQ(second->AdoptedPrefixLength(), static_cast<int64_t>(2 * kBlockSize));

  const auto snapshot = paged.cache->Snapshot();
  EXPECT_TRUE(ValidateCacheInvariants(snapshot).empty());
  size_t shared_blocks = 0;
  for (const auto& block : snapshot.blocks) {
    if (block.ref_count > 2) ++shared_blocks;  // Both requests plus the index.
  }
  EXPECT_EQ(shared_blocks, 2u);
}

// A prompt that shares nothing pays exactly what it paid before the feature existed.
TEST_F(PrefixCachingEngineTest, AnUnrelatedPromptAdoptsNothing) {
  auto paged = MakePagedEngine(model_);

  RunToCompletion(paged, MakePrompt(11, /*suffix=*/29), /*max_length=*/16);
  const size_t after_first = paged.executor->TotalDecodedTokens();

  std::vector<int32_t> unrelated(12, 0);
  std::iota(unrelated.begin(), unrelated.end(), 17);
  RunToCompletion(paged, unrelated, /*max_length=*/16);

  const auto* metrics = paged.engine->PrefixCacheStats();
  ASSERT_NE(metrics, nullptr);
  EXPECT_EQ(metrics->hits, 0u);
  EXPECT_EQ(metrics->matched_tokens, 0u);
  EXPECT_EQ(paged.executor->TotalDecodedTokens() - after_first, after_first);
}

// A finished request's blocks stay indexed, so the next turn of the same conversation hits them
// even though nothing holds them any more.
TEST_F(PrefixCachingEngineTest, RetainedBlocksAreReusedByALaterRequest) {
  auto paged = MakePagedEngine(model_);
  const auto prompt = MakePrompt(11, /*suffix=*/29);

  RunToCompletion(paged, prompt, /*max_length=*/16);

  // Nothing holds the blocks now, but they are still indexed and still resident.
  const auto snapshot = paged.cache->Snapshot();
  EXPECT_TRUE(snapshot.requests.empty());
  size_t retained = 0;
  for (const auto& block : snapshot.blocks) {
    if (block.indexed && block.ref_count == 1) ++retained;
  }
  EXPECT_GE(retained, 2u);

  const size_t before = paged.executor->TotalDecodedTokens();
  RunToCompletion(paged, prompt, /*max_length=*/16);
  EXPECT_EQ(before - (paged.executor->TotalDecodedTokens() - before), 2 * kBlockSize);
}

// The processed cursor after an adoption has to land exactly on the adopted block boundary: one
// token too far and the request would skip a token whose keys and values were never computed.
TEST_F(PrefixCachingEngineTest, AdoptionLeavesTheProcessedCursorOnTheBlockBoundary) {
  auto paged = MakePagedEngine(model_);
  const auto prompt = MakePrompt(11, /*suffix=*/29);
  RunToCompletion(paged, prompt, /*max_length=*/16);

  auto warm = std::make_shared<Request>(MakeGreedyParams(*model_));
  warm->AddTokens(prompt);
  paged.engine->AddRequest(warm);
  EXPECT_EQ(warm->ProcessedSequenceLength(), 0);

  paged.engine->Step();

  // Adopted 8 tokens, computed the remaining 4, and sampled one more.
  EXPECT_EQ(warm->AdoptedPrefixLength(), static_cast<int64_t>(2 * kBlockSize));
  EXPECT_EQ(warm->ProcessedSequenceLength(), static_cast<int64_t>(prompt.size()));
  ASSERT_FALSE(paged.executor->decoded_token_counts.empty());
  EXPECT_EQ(paged.executor->decoded_token_counts.back(), prompt.size() - 2 * kBlockSize);
}

// A step that rolls back must leave the request exactly where it was, including the cursor the
// adoption moved and the references the reservation took on the shared blocks.
TEST_F(PrefixCachingEngineTest, RolledBackAdoptionRestoresTheCursorAndReleasesEveryReference) {
  auto paged = MakePagedEngine(model_);
  const auto prompt = MakePrompt(11, /*suffix=*/29);
  RunToCompletion(paged, prompt, /*max_length=*/16);

  const auto before = paged.cache->Snapshot();

  auto warm = std::make_shared<Request>(MakeGreedyParams(*model_));
  warm->AddTokens(prompt);
  paged.engine->AddRequest(warm);
  paged.executor->SetNextFailure(ScriptedExecutionFailure::RetryableBeforeExecution);

  EXPECT_THROW(paged.engine->Step(), EngineStepError);

  EXPECT_EQ(warm->ProcessedSequenceLength(), 0);
  EXPECT_EQ(warm->AdoptedPrefixLength(), 0);
  const auto after = paged.cache->Snapshot();
  EXPECT_TRUE(ValidateCacheInvariants(after).empty());
  EXPECT_EQ(after.free_blocks, before.free_blocks);
  ASSERT_EQ(after.blocks.size(), before.blocks.size());
  for (size_t i = 0; i < after.blocks.size(); ++i) {
    EXPECT_EQ(after.blocks[i].block_id, before.blocks[i].block_id);
    EXPECT_EQ(after.blocks[i].ref_count, before.blocks[i].ref_count);
  }

  // The retry adopts the prefix cleanly, so the rollback lost nothing but the step.
  paged.engine->Step();
  EXPECT_EQ(warm->AdoptedPrefixLength(), static_cast<int64_t>(2 * kBlockSize));
}

// Retention is bounded and always reclaimable: a live request under memory pressure takes the
// blocks back rather than being deferred forever.
TEST_F(PrefixCachingEngineTest, RetainedBlocksAreReclaimedForALiveRequestUnderPressure) {
  Batching().num_blocks = 6;
  Batching().prefix_cache_max_blocks = 6;
  auto paged = MakePagedEngine(model_);

  RunToCompletion(paged, MakePrompt(7, /*suffix=*/29), /*max_length=*/12);
  const auto retained = paged.cache->Snapshot();
  size_t retained_blocks = 0;
  for (const auto& block : retained.blocks) {
    if (block.indexed && block.ref_count == 1) ++retained_blocks;
  }
  ASSERT_GT(retained_blocks, 0u);

  // A prompt sharing nothing now needs the whole pool, so retention has to give way.
  std::vector<int32_t> unrelated(20, 0);
  std::iota(unrelated.begin(), unrelated.end(), 11);
  RunToCompletion(paged, unrelated, /*max_length=*/24);

  const auto* metrics = paged.engine->PrefixCacheStats();
  ASSERT_NE(metrics, nullptr);
  EXPECT_GT(metrics->evictions, 0u);
  EXPECT_TRUE(ValidateCacheInvariants(paged.cache->Snapshot()).empty());
}

// The engine's default configuration must behave exactly as it did before the feature existed.
TEST_F(PrefixCachingEngineTest, DefaultConfigurationDoesNotCacheAnyPrefix) {
  Batching().prefix_caching = Config::Engine::DynamicBatching{}.prefix_caching;
  ASSERT_FALSE(Batching().prefix_caching);
  auto paged = MakePagedEngine(model_);
  const auto prompt = MakePrompt(11, /*suffix=*/29);

  const auto first = RunToCompletion(paged, prompt, /*max_length=*/16);
  const size_t first_tokens = paged.executor->TotalDecodedTokens();
  const auto second = RunToCompletion(paged, prompt, /*max_length=*/16);
  const size_t second_tokens = paged.executor->TotalDecodedTokens() - first_tokens;

  EXPECT_EQ(second, first);
  EXPECT_EQ(second_tokens, first_tokens);
  const auto* metrics = paged.engine->PrefixCacheStats();
  ASSERT_NE(metrics, nullptr);
  EXPECT_EQ(metrics->lookups, 0u);
  EXPECT_EQ(metrics->registered_blocks, 0u);
}

// Static batching keeps a single contiguous cache with no content-addressed blocks, so it reports
// no prefix cache rather than pretending a hit occurred.
TEST_F(PrefixCachingEngineTest, StaticBatchingReportsNoPrefixCache) {
  model_->config_->engine.dynamic_batching.reset();
  model_->config_->engine.static_batching = Config::Engine::StaticBatching{};
  auto cache = CacheManager::Create(model_);

  EXPECT_FALSE(cache->SupportsDynamicBatching());
  EXPECT_EQ(cache->PrefixMetrics(), nullptr);
  auto request = MintRequest(*model_, std::array<int32_t, 2>{2, 3});
  EXPECT_EQ(cache->MatchPrefix(*request), nullptr);
}

// Beam search keeps several live sequences behind one request, which the engine does not model, so
// it is rejected up front rather than silently sharing a prefix between beams.
TEST_F(PrefixCachingEngineTest, BeamSearchIsRejectedRatherThanSharingAPrefix) {
  auto params = MakeGreedyParams(*model_);
  params->search.num_beams = 2;
  EXPECT_THROW(
      {
        auto rejected = std::make_shared<Request>(params);
        static_cast<void>(rejected);
      },
      std::runtime_error);
}

// The block table the model sees after an adoption has to name the adopted blocks first, and the
// step has to write at absolute positions that continue exactly where they end.
TEST_F(PrefixCachingEngineTest, AdoptedBlocksLeadTheBlockTableAndPositionsContinueFromThem) {
  PagedKeyValueCache cache{model_};
  auto engine = MakeDoublesEngine(model_, /*capacity=*/2, EosToken(*model_)).engine;

  const std::array<int32_t, 8> tokens{2, 3, 4, 5, 6, 7, 8, 9};
  auto first = MintAssignedRequest(engine, *model_, tokens);
  const std::array reserve_first{
      PagedCacheReservationRequest{first.get(), tokens.size(), true, tokens.size()},
  };
  auto first_reservation = cache.Reserve(reserve_first);
  first_reservation.Commit();
  cache.SealCommittedBlocks(first.get(), tokens);

  // A second prompt sharing the first seven tokens can adopt one whole block.
  const std::array<int32_t, 8> warm_tokens{2, 3, 4, 5, 6, 7, 8, 99};
  auto second = MintAssignedRequest(engine, *model_, warm_tokens);
  const auto match = cache.MatchPrefix(warm_tokens, warm_tokens.size() - 1);
  ASSERT_EQ(match.token_count, kBlockSize);
  ASSERT_EQ(match.blocks.size(), 1u);
  const int32_t adopted_block_id = static_cast<int32_t>(match.blocks.front()->Id());

  const std::array reserve_second{
      PagedCacheReservationRequest{second.get(), warm_tokens.size(), true, warm_tokens.size(), &match},
  };
  auto second_reservation = cache.Reserve(reserve_second);

  // The adopted block leads the row, so the step writes token j at absolute position 4 + j, which
  // lands in the block that follows it.
  const std::array<const void*, 1> ids{second.get()};
  std::vector<int32_t> row(2, -1);
  second_reservation.FillBlockTable(ids, 2, row);
  EXPECT_EQ(row[0], adopted_block_id);
  EXPECT_NE(row[1], adopted_block_id);
  EXPECT_NE(row[1], -1);

  EXPECT_TRUE(ValidateCacheInvariants(cache.Snapshot(second_reservation)).empty());
  second_reservation.Commit();

  const auto snapshot = cache.Snapshot();
  EXPECT_TRUE(ValidateCacheInvariants(snapshot).empty());
  ASSERT_EQ(snapshot.requests.size(), 2u);
  EXPECT_EQ(snapshot.requests[1].used_slots, warm_tokens.size());

  cache.Remove(first);
  cache.Remove(second);
}

// Adoption is capacity-neutral: the retained blocks a match claims stop being reclaimable, so they
// are charged to the step alongside the blocks it still has to take. This admission starts with an
// empty free pool and every block retained, and it has to reclaim exactly the shortfall.
TEST_F(PrefixCachingEngineTest, AdmissionWithNoFreeBlocksReclaimsExactlyWhatItNeeds) {
  Batching().num_blocks = 3;
  PagedKeyValueCache cache{model_};
  auto engine = MakeDoublesEngine(model_, /*capacity=*/2, EosToken(*model_)).engine;

  const std::array<int32_t, 12> tokens{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  auto first = MintAssignedRequest(engine, *model_, tokens);
  const std::array reserve_first{
      PagedCacheReservationRequest{first.get(), tokens.size(), true, tokens.size()},
  };
  auto first_reservation = cache.Reserve(reserve_first);
  first_reservation.Commit();
  cache.SealCommittedBlocks(first.get(), tokens);
  cache.Remove(first);

  const auto retained = cache.Snapshot();
  EXPECT_EQ(retained.free_blocks, 0u);
  EXPECT_EQ(retained.blocks.size(), 3u);

  auto second = MintAssignedRequest(engine, *model_, tokens);
  const auto match = cache.MatchPrefix(tokens, tokens.size() - 1);
  ASSERT_EQ(match.blocks.size(), 2u);

  StepPlan plan;
  RequestStepPlan entry;
  entry.request = second;
  entry.request_id = second.get();
  entry.target_cache_slots = match.token_count + 1;
  entry.whole_sequence_cache_slots = tokens.size();
  entry.newly_admitted = true;
  entry.prefix_match = std::make_shared<const PrefixCacheMatch>(match);
  plan.requests.push_back(std::move(entry));

  const auto result = cache.PlanStepResources(plan);
  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);

  const std::array reserve_second{
      PagedCacheReservationRequest{second.get(), match.token_count + 1, true, tokens.size(), &match},
  };
  auto second_reservation = cache.Reserve(reserve_second);
  EXPECT_EQ(second_reservation.ReservedBlockCount(), 1u);
  second_reservation.Commit();

  const auto snapshot = cache.Snapshot();
  EXPECT_TRUE(ValidateCacheInvariants(snapshot).empty());
  ASSERT_EQ(snapshot.requests.size(), 1u);
  EXPECT_EQ(snapshot.requests[0].block_ids.size(), 3u);

  cache.Remove(second);
}

// A cached prefix is only ever made of full blocks, because a partially filled block is still being
// written and sharing it would let one request's tokens land in another's key-value data.
TEST_F(PrefixCachingEngineTest, APartiallyFilledBlockIsNeverOfferedAsAPrefix) {
  PagedKeyValueCache cache{model_};
  auto engine = MakeDoublesEngine(model_, /*capacity=*/2, EosToken(*model_)).engine;

  // Six tokens fill one block and leave two slots of the next one used.
  const std::array<int32_t, 6> tokens{2, 3, 4, 5, 6, 7};
  auto request = MintAssignedRequest(engine, *model_, tokens);
  const std::array reserve{
      PagedCacheReservationRequest{request.get(), tokens.size(), true, tokens.size()},
  };
  auto reservation = cache.Reserve(reserve);
  reservation.Commit();
  cache.SealCommittedBlocks(request.get(), tokens);

  // Only the first four tokens are adoptable; the half-written second block is not offered.
  const std::array<int32_t, 7> probe{2, 3, 4, 5, 6, 7, 8};
  const auto match = cache.MatchPrefix(probe, probe.size() - 1);
  EXPECT_EQ(match.token_count, kBlockSize);

  cache.Remove(request);
}

}  // namespace
}  // namespace test
}  // namespace Generators
