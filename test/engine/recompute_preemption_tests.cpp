// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for recompute preemption on the continuous-batching dynamic path: the policy that picks a
// victim, the request transition that gives up cache residency, the scheduler that performs the
// suspension at a step boundary, and the end-to-end guarantee that a suspended request resumes with
// exactly the token stream it would have produced had it never been preempted.
//
// The scheduler and engine cases drive the production PagedCacheManager and DynamicBatchScheduler,
// with only model execution replaced, so capacity pressure here is real block pressure.

#include <array>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engine/engine.h"
#include "engine/engine_invariants.h"
#include "engine/recompute_preemption_policy.h"
#include "engine/scheduler.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"

namespace Generators {
namespace test {
namespace {

RecomputePreemptionCandidate Candidate(const void* id, size_t blocks,
                                       size_t preemptions, bool eligible = true,
                                       size_t decode_steps = 4) {
  return RecomputePreemptionCandidate{id, blocks, preemptions, decode_steps, eligible};
}

RecomputePreemptionSettings EnabledSettings(size_t max_victims_per_step = 1,
                                            size_t max_per_request = 0,
                                            size_t min_decode_steps = 1) {
  return RecomputePreemptionSettings{true, max_victims_per_step, max_per_request,
                                     min_decode_steps};
}

// Stand-in request identities for the policy tests, which never dereference them.
const char kA = 'a';
const char kB = 'b';
const char kC = 'c';

// A non-end-of-stream token, so a scripted step advances a request instead of completing it.
constexpr int32_t kFilledToken = 5;

TEST(RecomputePreemptionPolicyTest, OrdersFewestPreemptionsThenNewestResident) {
  const std::array candidates{
      Candidate(&kA, 4, 1),
      Candidate(&kB, 4, 0),
      Candidate(&kC, 4, 0),
  };

  const auto order = RecomputeVictimOrder(candidates);

  // Both never-preempted residents come before the one that has already paid, and between them the
  // later-admitted one is chosen so the longest-resident request keeps its accumulated work.
  ASSERT_EQ(order.size(), 3u);
  EXPECT_EQ(order[0], 2u);
  EXPECT_EQ(order[1], 1u);
  EXPECT_EQ(order[2], 0u);
}

TEST(RecomputePreemptionPolicyTest, SkipsIneligibleCandidates) {
  const std::array candidates{
      Candidate(&kA, 4, 0, /*eligible=*/false),
      Candidate(&kB, 4, 0),
  };

  const auto order = RecomputeVictimOrder(candidates);

  ASSERT_EQ(order.size(), 1u);
  EXPECT_EQ(order[0], 1u);
}

TEST(RecomputePreemptionPolicyTest, SelectsNothingWhenDisabled) {
  const std::array candidates{Candidate(&kA, 4, 0)};

  const auto decision = SelectRecomputeVictims(
      candidates, 1, RecomputePreemptionSettings{false, 1, 0, 1});

  EXPECT_TRUE(decision.Empty());
  EXPECT_EQ(decision.reclaimed_blocks, 0u);
}

TEST(RecomputePreemptionPolicyTest, DeclinesUntilAResidentHasEarnedItsAdmission) {
  // A resident that has not decoded since it was admitted has not been paid for the key-value
  // entries it rebuilt, so suspending it again would be pure churn.
  const std::array candidates{
      Candidate(&kA, 4, 0, /*eligible=*/true, /*decode_steps=*/0)};

  EXPECT_TRUE(SelectRecomputeVictims(candidates, 1, EnabledSettings(1, 0, 1)).Empty());

  const std::array served{
      Candidate(&kA, 4, 0, /*eligible=*/true, /*decode_steps=*/1)};
  EXPECT_EQ(SelectRecomputeVictims(served, 1, EnabledSettings(1, 0, 1)).victims.size(), 1u);
  // A larger quantum keeps the same resident protected for longer.
  EXPECT_TRUE(SelectRecomputeVictims(served, 1, EnabledSettings(1, 0, 4)).Empty());
}

TEST(RecomputePreemptionPolicyTest, SelectsNothingWhenNothingIsBlocked) {
  const std::array candidates{Candidate(&kA, 4, 0)};

  EXPECT_TRUE(SelectRecomputeVictims(candidates, 0, EnabledSettings()).Empty());
}

TEST(RecomputePreemptionPolicyTest, SelectsTheSmallestCoveringVictimSet) {
  const std::array candidates{
      Candidate(&kA, 2, 0),
      Candidate(&kB, 3, 0),
  };

  const auto decision = SelectRecomputeVictims(candidates, 3, EnabledSettings(2));

  // The newest resident is considered first and already covers the shortfall, so the older one is
  // left alone.
  ASSERT_EQ(decision.victims.size(), 1u);
  EXPECT_EQ(decision.victims[0], 1u);
  EXPECT_EQ(decision.reclaimed_blocks, 3u);
}

TEST(RecomputePreemptionPolicyTest, CombinesVictimsUpToTheStepBound) {
  const std::array candidates{
      Candidate(&kA, 2, 0),
      Candidate(&kB, 2, 0),
  };

  const auto decision = SelectRecomputeVictims(candidates, 4, EnabledSettings(2));

  ASSERT_EQ(decision.victims.size(), 2u);
  EXPECT_EQ(decision.reclaimed_blocks, 4u);
}

TEST(RecomputePreemptionPolicyTest, DeclinesWhenTheStepBoundCannotCoverTheShortfall) {
  const std::array candidates{
      Candidate(&kA, 2, 0),
      Candidate(&kB, 2, 0),
  };

  const auto decision = SelectRecomputeVictims(candidates, 4, EnabledSettings(1));

  EXPECT_TRUE(decision.Empty());
  EXPECT_EQ(decision.reclaimed_blocks, 0u);
}

TEST(RecomputePreemptionPolicyTest, DeclinesWhenNoEligibleVictimExists) {
  const std::array candidates{Candidate(&kA, 8, 0, /*eligible=*/false)};

  EXPECT_TRUE(SelectRecomputeVictims(candidates, 1, EnabledSettings()).Empty());
}

TEST(RecomputePreemptionPolicyTest, HonorsThePerRequestBound) {
  const std::array candidates{
      Candidate(&kA, 4, 2),
      Candidate(&kB, 4, 0),
  };

  const auto bounded = SelectRecomputeVictims(candidates, 8, EnabledSettings(2, 2));

  // Only one candidate is still under the per-request bound, so the pair cannot be formed.
  EXPECT_TRUE(bounded.Empty());

  const auto single = SelectRecomputeVictims(candidates, 4, EnabledSettings(2, 2));
  ASSERT_EQ(single.victims.size(), 1u);
  EXPECT_EQ(single.victims[0], 1u);
}

TEST(RecomputePreemptionPolicyTest, IgnoresCandidatesThatOwnNothing) {
  const std::array candidates{
      Candidate(&kA, 4, 0),
      Candidate(&kB, 0, 0),
  };

  const auto decision = SelectRecomputeVictims(candidates, 4, EnabledSettings(2));

  ASSERT_EQ(decision.victims.size(), 1u);
  EXPECT_EQ(decision.victims[0], 0u);
}

// Fabricates a decoder whose next token depends on every token a request has pushed through it
// since its key-value entries were last built, which is how a real decoder's output depends on its
// whole context. A resume that replayed the wrong tokens, dropped any, or replayed them at the
// wrong offset would produce a different stream, so comparing streams is a real continuity check.
class ContextDependentDecoder {
 public:
  ContextDependentDecoder(int32_t vocab_size, int32_t eos_token)
      : vocab_size_{vocab_size}, eos_token_{eos_token} {}

  int32_t operator()(Request& request) {
    auto& context = contexts_[&request];
    const auto processed = static_cast<size_t>(request.ProcessedSequenceLength());
    // Truncating to the committed boundary makes this idempotent: a rolled-back attempt's tokens
    // are dropped before the retry appends them again.
    if (processed < context.size())
      context.resize(processed);
    if (context.size() != processed)
      throw std::runtime_error("ContextDependentDecoder lost part of a request's context.");

    for (const int32_t token : request.UnprocessedTokensCpu())
      context.push_back(token);

    uint32_t hash = 2166136261u;
    for (const int32_t token : context) {
      hash ^= static_cast<uint32_t>(token);
      hash *= 16777619u;
    }
    // Stay clear of the end-of-stream token so generation length stays under the test's control.
    const int32_t candidate =
        static_cast<int32_t>(hash % static_cast<uint32_t>(vocab_size_));
    return candidate == eos_token_ ? (candidate + 1) % vocab_size_ : candidate;
  }

 private:
  int32_t vocab_size_;
  int32_t eos_token_;
  std::map<const Request*, std::vector<int32_t>> contexts_;
};

class RecomputePreemptionTest : public ::testing::Test {
 protected:
  // A small block pool with a large block size keeps the arithmetic obvious: one block holds one
  // request's short sequence, so "one more resident" costs exactly one block.
  std::shared_ptr<Model> LoadModel(size_t num_blocks, bool enable_preemption,
                                   size_t max_preemptions_per_step = 1,
                                   size_t max_preemptions_per_request = 0,
                                   size_t min_decode_steps = 1,
                                   size_t block_size = 32) {
    auto model = LoadDummyDecoderModel();
    Config::Engine::DynamicBatching dynamic_batching;
    dynamic_batching.block_size = block_size;
    dynamic_batching.num_blocks = num_blocks;
    dynamic_batching.max_batch_size = 8;
    dynamic_batching.max_scheduled_tokens = 64;
    dynamic_batching.enable_recompute_preemption = enable_preemption;
    dynamic_batching.max_preemptions_per_step = max_preemptions_per_step;
    dynamic_batching.max_preemptions_per_request = max_preemptions_per_request;
    dynamic_batching.min_decode_steps_before_preemption = min_decode_steps;
    model->config_->engine.dynamic_batching = dynamic_batching;
    return model;
  }

  static std::shared_ptr<Request> Mint(const Model& model,
                                       std::span<const int32_t> prompt,
                                       int max_length) {
    auto params = MakeGreedyParams(model);
    params->search.max_length = max_length;
    auto request = std::make_shared<Request>(params);
    request->AddTokens(prompt);
    return request;
  }

  static std::vector<RequestStateSnapshot> Snapshots(
      const std::vector<std::shared_ptr<Request>>& requests) {
    std::vector<RequestStateSnapshot> snapshots;
    snapshots.reserve(requests.size());
    for (const auto& request : requests)
      snapshots.push_back(request->Snapshot());
    return snapshots;
  }

  // Engine::Step() returns one ready request per call and drains a committed batch before running
  // the model again, so counting model invocations is how a test advances the engine by a known
  // number of committed steps.
  static void RunModelSteps(PagedDoublesEngine& paged, int model_steps) {
    const int target = paged.executor->decode_calls + model_steps;
    int guard = 0;
    while (paged.executor->decode_calls < target && guard++ < 256)
      paged.engine->Step();
    if (paged.executor->decode_calls < target)
      throw std::runtime_error("The engine stopped running the model before the step budget.");
  }

  static void ExpectConsistent(const PagedDoublesEngine& engine,
                               const std::vector<std::shared_ptr<Request>>& requests) {
    const auto violations =
        ValidateInvariants(engine.cache->Snapshot(), Snapshots(requests));
    for (const auto& violation : violations)
      ADD_FAILURE() << violation.message;
  }
};

TEST_F(RecomputePreemptionTest, SuspendRewindsTheCacheCursorAndKeepsTheSequence) {
  auto model = LoadModel(/*num_blocks=*/4, /*enable_preemption=*/true);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto request = Mint(*model, prompt, /*max_length=*/32);
  paged.engine->AddRequest(request);

  ASSERT_NE(paged.engine->Step(), nullptr);
  const auto resident = request->Snapshot();
  ASSERT_EQ(resident.status, RequestStatus::InProgress);
  ASSERT_EQ(resident.processed_sequence_length, 3);
  ASSERT_EQ(resident.current_sequence_length, 4);

  request->SuspendForRecompute();

  const auto suspended = request->Snapshot();
  EXPECT_EQ(suspended.status, RequestStatus::Suspended);
  EXPECT_EQ(suspended.processed_sequence_length, 0);
  // The whole logical sequence survives, including the token sampled by the last committed step.
  EXPECT_EQ(suspended.current_sequence_length, resident.current_sequence_length);
  EXPECT_EQ(suspended.seen_sequence_length, resident.seen_sequence_length);
  EXPECT_TRUE(suspended.is_prefill);
  EXPECT_EQ(suspended.preemption_count, 1u);
  EXPECT_EQ(suspended.recomputed_token_count, 3u);
}

TEST_F(RecomputePreemptionTest, SuspendRejectsRequestsThatAreNotResident) {
  auto model = LoadModel(/*num_blocks=*/4, /*enable_preemption=*/true);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto request = Mint(*model, prompt, /*max_length=*/32);

  EXPECT_THROW(request->SuspendForRecompute(), std::runtime_error);

  paged.engine->AddRequest(request);
  EXPECT_FALSE(request->IsPreemptible());
  EXPECT_THROW(request->SuspendForRecompute(), std::runtime_error);
}

TEST_F(RecomputePreemptionTest, SuspendedRequestRejectsNewTokens) {
  auto model = LoadModel(/*num_blocks=*/4, /*enable_preemption=*/true);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto request = Mint(*model, prompt, /*max_length=*/32);
  paged.engine->AddRequest(request);
  ASSERT_NE(paged.engine->Step(), nullptr);
  request->SuspendForRecompute();

  const std::array<int32_t, 1> extra{5};
  EXPECT_THROW(request->AddTokens(extra), std::runtime_error);
}

TEST_F(RecomputePreemptionTest, InvariantsRejectASuspendedRequestThatStillLooksResident) {
  RequestStateSnapshot snapshot;
  snapshot.request_id = &kA;
  snapshot.status = RequestStatus::Suspended;
  snapshot.current_sequence_length = 8;
  snapshot.processed_sequence_length = 8;
  snapshot.seen_sequence_length = 8;
  snapshot.is_prefill = false;
  snapshot.preemption_count = 0;

  const auto violations = ValidateRequestInvariants(snapshot);

  EXPECT_EQ(violations.size(), 3u);
}

TEST_F(RecomputePreemptionTest, DisabledByDefaultLeavesBlockedRequestsWaiting) {
  // One block holds one request's whole sequence, so the second request cannot be admitted.
  auto model = LoadModel(/*num_blocks=*/1, /*enable_preemption=*/false);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto resident = Mint(*model, prompt, /*max_length=*/32);
  auto waiting = Mint(*model, prompt, /*max_length=*/32);
  paged.engine->AddRequest(resident);
  RunModelSteps(paged, 2);
  paged.engine->AddRequest(waiting);

  RunModelSteps(paged, 3);

  EXPECT_EQ(resident->status_, RequestStatus::InProgress);
  EXPECT_EQ(waiting->status_, RequestStatus::Assigned);
  const auto& metrics = paged.scheduler->PreemptionMetrics();
  EXPECT_EQ(metrics.preemptions, 0u);
  EXPECT_EQ(metrics.block_starved_passes, 0u);
  ExpectConsistent(paged, {resident, waiting});
}

TEST_F(RecomputePreemptionTest, PreemptsAResidentToAdmitABlockedRequest) {
  auto model = LoadModel(/*num_blocks=*/1, /*enable_preemption=*/true);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto resident = Mint(*model, prompt, /*max_length=*/32);
  auto waiting = Mint(*model, prompt, /*max_length=*/32);
  paged.engine->AddRequest(resident);
  // One prefill step plus one decode step, so the resident has earned its admission.
  RunModelSteps(paged, 2);
  const auto before = resident->Snapshot();
  ASSERT_EQ(resident->DecodeStepsSinceAdmission(), 1u);
  paged.engine->AddRequest(waiting);

  RunModelSteps(paged, 1);

  const auto& metrics = paged.scheduler->PreemptionMetrics();
  EXPECT_EQ(metrics.block_starved_passes, 1u);
  EXPECT_EQ(metrics.preemptions, 1u);
  EXPECT_EQ(metrics.preemption_passes, 1u);
  EXPECT_EQ(metrics.reclaimed_blocks, 1u);
  EXPECT_EQ(metrics.recomputed_tokens,
            static_cast<uint64_t>(before.processed_sequence_length));
  EXPECT_EQ(metrics.declined_preemptions, 0u);

  // The resident gave up its residency and the waiting request took its place.
  EXPECT_EQ(resident->status_, RequestStatus::Suspended);
  EXPECT_EQ(resident->ProcessedSequenceLength(), 0);
  EXPECT_EQ(resident->PreemptionCount(), 1u);
  EXPECT_EQ(resident->DecodeStepsSinceAdmission(), 0u);
  EXPECT_EQ(waiting->status_, RequestStatus::InProgress);
  const auto snapshot = paged.cache->Snapshot();
  ASSERT_EQ(snapshot.requests.size(), 1u);
  EXPECT_EQ(snapshot.requests[0].request_id, waiting.get());
  ExpectConsistent(paged, {resident, waiting});
}

TEST_F(RecomputePreemptionTest, PreemptsToUnblockAResidentThatCannotGrow) {
  // One slot per block means every decode step needs a new block, so the pool runs dry with the
  // residents mid-sequence and no request able to run.
  auto model = LoadModel(/*num_blocks=*/6, /*enable_preemption=*/true,
                         /*max_preemptions_per_step=*/1,
                         /*max_preemptions_per_request=*/0,
                         /*min_decode_steps=*/1, /*block_size=*/1);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 2> prompt{2, 3};
  auto first = Mint(*model, prompt, /*max_length=*/16);
  auto second = Mint(*model, prompt, /*max_length=*/16);
  paged.engine->AddRequest(first);
  paged.engine->AddRequest(second);
  RunModelSteps(paged, 2);
  ASSERT_EQ(paged.cache->Snapshot().free_blocks, 0u);
  ASSERT_EQ(paged.scheduler->PreemptionMetrics().preemptions, 0u);

  RunModelSteps(paged, 1);

  // Rather than reporting that the engine is out of capacity, the newest resident is suspended so
  // the older one can keep decoding.
  const auto& metrics = paged.scheduler->PreemptionMetrics();
  EXPECT_EQ(metrics.preemptions, 1u);
  EXPECT_EQ(second->status_, RequestStatus::Suspended);
  EXPECT_EQ(first->status_, RequestStatus::InProgress);
  EXPECT_GT(first->ProcessedSequenceLength(), 3);
  ExpectConsistent(paged, {first, second});
}

TEST_F(RecomputePreemptionTest, SuspendedRequestIsRebuiltByChunkedPrefill) {
  auto model = LoadModel(/*num_blocks=*/1, /*enable_preemption=*/true);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto victim = Mint(*model, prompt, /*max_length=*/10);
  auto newcomer = Mint(*model, prompt, /*max_length=*/10);
  paged.engine->AddRequest(victim);
  RunModelSteps(paged, 2);
  paged.engine->AddRequest(newcomer);
  RunModelSteps(paged, 1);
  ASSERT_EQ(victim->status_, RequestStatus::Suspended);
  const int64_t sequence_length_when_suspended = victim->CurrentSequenceLength();
  const size_t recomputed_tokens = victim->RecomputedTokenCount();

  // Drive the engine until the victim's key-value entries have been rebuilt.
  int steps = 0;
  while (paged.engine->HasPendingRequests() && steps++ < 128) {
    paged.engine->Step();
    if (victim->status_ == RequestStatus::InProgress &&
        victim->ProcessedSequenceLength() >= sequence_length_when_suspended) {
      break;
    }
  }

  EXPECT_EQ(victim->status_, RequestStatus::InProgress);
  // Every token it held before the suspension is back in the cache, and no token was lost.
  EXPECT_GE(victim->ProcessedSequenceLength(), sequence_length_when_suspended);
  EXPECT_GE(victim->CurrentSequenceLength(), sequence_length_when_suspended);
  EXPECT_EQ(victim->PreemptionCount(), 1u);
  EXPECT_EQ(victim->RecomputedTokenCount(), recomputed_tokens);
  ExpectConsistent(paged, {victim, newcomer});
}

TEST_F(RecomputePreemptionTest, PrefillingResidentIsNotPreemptible) {
  // A resident whose key-value entries are only partly built owns blocks for a prefill it has not
  // finished; suspending it would discard work that is still in flight.
  auto model = LoadModel(/*num_blocks=*/4, /*enable_preemption=*/true);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 6> prompt{2, 3, 4, 5, 6, 7};
  auto request = Mint(*model, prompt, /*max_length=*/32);
  paged.engine->AddRequest(request);

  // Commit a partial prefill the way a chunked step would: cache slots advance, no token is
  // sampled, and the request becomes resident while still in prefill.
  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = request->CurrentSequenceLength();
  plan.target_cache_slots = 2;
  request->CommitStep(plan, RequestStepResult{});

  ASSERT_EQ(request->status_, RequestStatus::InProgress);
  ASSERT_TRUE(request->IsPrefill());
  EXPECT_FALSE(request->IsPreemptible());
  EXPECT_EQ(request->DecodeStepsSinceAdmission(), 0u);
}

TEST_F(RecomputePreemptionTest, DeclinesToPreemptUntilAResidentHasEarnedItsService) {
  // The resident has decoded once, but the configured quantum asks for more before its residency
  // can be taken away again.
  auto model = LoadModel(/*num_blocks=*/1, /*enable_preemption=*/true,
                         /*max_preemptions_per_step=*/1,
                         /*max_preemptions_per_request=*/0,
                         /*min_decode_steps=*/4);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto resident = Mint(*model, prompt, /*max_length=*/32);
  auto waiting = Mint(*model, prompt, /*max_length=*/32);
  paged.engine->AddRequest(resident);
  RunModelSteps(paged, 2);
  ASSERT_EQ(resident->DecodeStepsSinceAdmission(), 1u);
  paged.engine->AddRequest(waiting);

  RunModelSteps(paged, 1);

  const auto& metrics = paged.scheduler->PreemptionMetrics();
  EXPECT_GE(metrics.block_starved_passes, 1u);
  EXPECT_GE(metrics.declined_preemptions, 1u);
  EXPECT_EQ(metrics.preemptions, 0u);
  EXPECT_EQ(resident->status_, RequestStatus::InProgress);
  EXPECT_EQ(waiting->status_, RequestStatus::Assigned);
  ExpectConsistent(paged, {resident, waiting});

  // Once the quantum is paid, the same waiting request is admitted by suspending the resident.
  RunModelSteps(paged, 3);
  EXPECT_EQ(paged.scheduler->PreemptionMetrics().preemptions, 1u);
  EXPECT_EQ(resident->status_, RequestStatus::Suspended);
  ExpectConsistent(paged, {resident, waiting});
}

TEST_F(RecomputePreemptionTest, SpreadsPreemptionsAcrossResidentsInsteadOfRepeatingOne) {
  auto model = LoadModel(/*num_blocks=*/2, /*enable_preemption=*/true);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto first = Mint(*model, prompt, /*max_length=*/24);
  auto second = Mint(*model, prompt, /*max_length=*/24);
  paged.engine->AddRequest(first);
  paged.engine->AddRequest(second);
  RunModelSteps(paged, 2);
  ASSERT_EQ(first->status_, RequestStatus::InProgress);
  ASSERT_EQ(second->status_, RequestStatus::InProgress);

  auto third = Mint(*model, prompt, /*max_length=*/24);
  paged.engine->AddRequest(third);
  RunModelSteps(paged, 1);

  // The newest resident is suspended first, keeping the longest-resident request's work.
  EXPECT_EQ(second->status_, RequestStatus::Suspended);
  EXPECT_EQ(first->status_, RequestStatus::InProgress);

  auto fourth = Mint(*model, prompt, /*max_length=*/24);
  paged.engine->AddRequest(fourth);
  RunModelSteps(paged, 4);

  // No request is suspended twice while a resident that has never been suspended is still holding
  // capacity, so the cost is spread instead of falling on one victim.
  EXPECT_LE(second->PreemptionCount(), first->PreemptionCount() + 1u);
  EXPECT_GE(paged.scheduler->PreemptionMetrics().preemptions, 2u);
  ExpectConsistent(paged, {first, second, third, fourth});
}

TEST_F(RecomputePreemptionTest, SuspendedRequestIsAdmittedBeforeANeverAdmittedNewcomer) {
  auto model = LoadModel(/*num_blocks=*/1, /*enable_preemption=*/true);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto victim = Mint(*model, prompt, /*max_length=*/8);
  auto newcomer = Mint(*model, prompt, /*max_length=*/8);
  paged.engine->AddRequest(victim);
  RunModelSteps(paged, 2);
  paged.engine->AddRequest(newcomer);
  RunModelSteps(paged, 1);
  ASSERT_EQ(victim->status_, RequestStatus::Suspended);

  // A request that arrives after the suspension must not overtake the victim.
  auto latecomer = Mint(*model, prompt, /*max_length=*/8);
  paged.engine->AddRequest(latecomer);
  int steps = 0;
  while (paged.engine->HasPendingRequests() && steps++ < 128) {
    paged.engine->Step();
    if (victim->status_ == RequestStatus::InProgress)
      break;
  }

  EXPECT_EQ(victim->status_, RequestStatus::InProgress);
  EXPECT_EQ(latecomer->status_, RequestStatus::Assigned);
  ExpectConsistent(paged, {victim, newcomer, latecomer});
}

TEST_F(RecomputePreemptionTest, CapacityRetryDoesNotReadmitTheRequestTheStepJustSuspended) {
  auto model = LoadModel(/*num_blocks=*/1, /*enable_preemption=*/true);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto victim = Mint(*model, prompt, /*max_length=*/12);
  auto newcomer = Mint(*model, prompt, /*max_length=*/12);
  paged.engine->AddRequest(victim);
  RunModelSteps(paged, 2);
  paged.engine->AddRequest(newcomer);
  const size_t decodes_before = paged.executor->decoded_request_ids.size();

  // The step preempts the victim, then the newcomer's execution reports a capacity failure and the
  // engine replans a smaller batch for the same step.
  paged.executor->SetNextFailure(ScriptedExecutionFailure::CapacityExceeded);
  paged.engine->Step();

  ASSERT_EQ(victim->PreemptionCount(), 1u);
  ASSERT_GT(paged.executor->decoded_request_ids.size(), decodes_before + 1);
  // The retry must not hand the reclaimed block back to the request this step just suspended.
  const auto& retry_requests = paged.executor->decoded_request_ids[decodes_before + 1];
  ASSERT_EQ(retry_requests.size(), 1u);
  EXPECT_EQ(retry_requests[0], newcomer.get());
  ExpectConsistent(paged, {victim, newcomer});
}

TEST_F(RecomputePreemptionTest, PlannerHonorsThePerPlanAdmissionLimit) {
  auto model = LoadModel(/*num_blocks=*/4, /*enable_preemption=*/true);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto first = Mint(*model, prompt, /*max_length=*/16);
  auto second = Mint(*model, prompt, /*max_length=*/16);
  paged.engine->AddRequest(first);
  paged.engine->AddRequest(second);

  StepPlan plan;
  plan.transaction_id = 1;
  for (const auto& request : {first, second}) {
    RequestStepPlan entry;
    entry.request = request;
    entry.request_id = request.get();
    entry.sequence_length_before = request->CurrentSequenceLength();
    entry.unprocessed_token_count = 1;
    entry.target_cache_slots = 1;
    entry.whole_sequence_cache_slots =
        static_cast<size_t>(request->CurrentSequenceLength());
    entry.is_prefill = true;
    entry.newly_admitted = true;
    plan.requests.push_back(std::move(entry));
  }
  // Free blocks are plentiful, so only the per-plan admission cap can hold the second request back.
  plan.new_admission_limit = 1;

  const auto result = paged.cache->PlanStepResources(plan);

  ASSERT_TRUE(result.executable);
  ASSERT_EQ(plan.requests.size(), 1u);
  EXPECT_EQ(plan.requests[0].request, first);
  EXPECT_TRUE(result.capacity_deferred);
  // The cap is not a block shortage, so it must not be reported as one.
  EXPECT_FALSE(result.blocked_admission.Any());
}

TEST_F(RecomputePreemptionTest, SuspendedRequestCanBeRemoved) {
  auto model = LoadModel(/*num_blocks=*/1, /*enable_preemption=*/true);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto victim = Mint(*model, prompt, /*max_length=*/12);
  auto newcomer = Mint(*model, prompt, /*max_length=*/12);
  paged.engine->AddRequest(victim);
  RunModelSteps(paged, 2);
  paged.engine->AddRequest(newcomer);
  RunModelSteps(paged, 1);
  ASSERT_EQ(victim->status_, RequestStatus::Suspended);

  victim->Remove();

  EXPECT_EQ(victim->status_, RequestStatus::Unassigned);
  const auto snapshot = paged.cache->Snapshot();
  ASSERT_EQ(snapshot.requests.size(), 1u);
  EXPECT_EQ(snapshot.requests[0].request_id, newcomer.get());
  // The engine keeps serving the surviving request without the removed one reappearing.
  int steps = 0;
  while (paged.engine->HasPendingRequests() && steps++ < 128)
    paged.engine->Step();
  EXPECT_EQ(newcomer->status_, RequestStatus::Completed);
  ExpectConsistent(paged, {newcomer});
}

TEST_F(RecomputePreemptionTest, RejectsEnablingPreemptionOnACacheThatCannotReclaim) {
  auto model = LoadModel(/*num_blocks=*/4, /*enable_preemption=*/true);
  auto cache = std::make_shared<RecordingCacheManager>(model, /*capacity=*/4);
  cache->SetSupportsRecomputePreemption(false);

  EXPECT_THROW(DynamicBatchScheduler(model, cache), std::runtime_error);
}

TEST_F(RecomputePreemptionTest, RejectsEnablingPreemptionOnTheStaticPath) {
  auto model = LoadModel(/*num_blocks=*/4, /*enable_preemption=*/true);
  auto cache = std::make_shared<RecordingCacheManager>(
      model, /*capacity=*/4, /*trace=*/nullptr, /*supports_dynamic_batching=*/false);

  EXPECT_THROW(StaticBatchScheduler(model, cache), std::runtime_error);
}

TEST_F(RecomputePreemptionTest, RejectsAZeroPerStepPreemptionBound) {
  auto model = LoadModel(/*num_blocks=*/4, /*enable_preemption=*/true,
                         /*max_preemptions_per_step=*/0);
  auto cache = std::make_shared<RecordingCacheManager>(model, /*capacity=*/4);

  EXPECT_THROW(DynamicBatchScheduler(model, cache), std::invalid_argument);
}// Drives every request to completion and returns the exact token stream each one produced, which is
// the observable an application sees.
std::map<const Request*, std::vector<int32_t>> RunToCompletion(
    Engine& engine, const std::vector<std::shared_ptr<Request>>& requests,
    int max_steps) {
  std::map<const Request*, std::vector<int32_t>> streams;
  for (const auto& request : requests)
    streams[request.get()] = {};

  int steps = 0;
  while (engine.HasPendingRequests()) {
    if (steps++ > max_steps)
      throw std::runtime_error("The engine did not drain within the step budget.");
    auto ready = engine.Step();
    if (!ready)
      continue;
    while (ready->HasUnseenTokens())
      streams[ready.get()].push_back(ready->UnseenToken());
  }
  return streams;
}

TEST_F(RecomputePreemptionTest, PreemptionPreservesEachRequestsTokenStream) {
  // The fabricated decoder is a pure function of the context a request pushed through it, so this
  // proves the resume replays the identical logical context. It cannot prove floating-point
  // determinism on a real provider, where replaying a decode as part of a longer prefill query can
  // resolve a near-tied logit differently.
  const std::array<int32_t, 4> first_prompt{2, 3, 4, 5};
  const std::array<int32_t, 3> second_prompt{6, 7, 8};
  const std::array<int32_t, 5> third_prompt{9, 10, 11, 12, 13};
  // Every sequence stays inside one block, so residency is the only thing capacity decides.
  constexpr int kMaxLength = 16;

  std::vector<std::vector<int32_t>> baseline_streams;
  {
    // Enough blocks for every request to stay resident: nothing is ever preempted.
    auto model = LoadModel(/*num_blocks=*/8, /*enable_preemption=*/false);
    auto paged = MakePagedEngine(model, kFilledToken);
    ContextDependentDecoder decoder{
        static_cast<int32_t>(model->config_->model.vocab_size), EosToken(*model)};
    paged.executor->SetNextTokenSelector(decoder);

    std::vector<std::shared_ptr<Request>> requests{
        Mint(*model, first_prompt, kMaxLength),
        Mint(*model, second_prompt, kMaxLength),
        Mint(*model, third_prompt, kMaxLength),
    };
    for (const auto& request : requests)
      paged.engine->AddRequest(request);

    const auto streams = RunToCompletion(*paged.engine, requests, /*max_steps=*/2048);
    for (const auto& request : requests)
      baseline_streams.push_back(streams.at(request.get()));
    ASSERT_EQ(paged.scheduler->PreemptionMetrics().preemptions, 0u);
  }

  std::vector<std::vector<int32_t>> preempted_streams;
  uint64_t preemptions = 0;
  {
    // The same workload under a key-value budget that cannot hold all three requests at once.
    auto model = LoadModel(/*num_blocks=*/2, /*enable_preemption=*/true);
    auto paged = MakePagedEngine(model, kFilledToken);
    ContextDependentDecoder decoder{
        static_cast<int32_t>(model->config_->model.vocab_size), EosToken(*model)};
    paged.executor->SetNextTokenSelector(decoder);

    std::vector<std::shared_ptr<Request>> requests{
        Mint(*model, first_prompt, kMaxLength),
        Mint(*model, second_prompt, kMaxLength),
        Mint(*model, third_prompt, kMaxLength),
    };
    for (const auto& request : requests)
      paged.engine->AddRequest(request);

    const auto streams = RunToCompletion(*paged.engine, requests, /*max_steps=*/2048);
    for (const auto& request : requests)
      preempted_streams.push_back(streams.at(request.get()));
    preemptions = paged.scheduler->PreemptionMetrics().preemptions;
    ExpectConsistent(paged, requests);
  }

  EXPECT_GT(preemptions, 0u) << "the constrained run did not exercise preemption";
  ASSERT_EQ(preempted_streams.size(), baseline_streams.size());
  for (size_t i = 0; i < baseline_streams.size(); ++i) {
    EXPECT_FALSE(baseline_streams[i].empty());
    EXPECT_EQ(preempted_streams[i], baseline_streams[i])
        << "request " << i << " produced a different stream after preemption";
  }
}

TEST_F(RecomputePreemptionTest, RollbackAfterPreemptionLeavesTheVictimSuspended) {
  auto model = LoadModel(/*num_blocks=*/1, /*enable_preemption=*/true);
  auto paged = MakePagedEngine(model, kFilledToken);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  auto victim = Mint(*model, prompt, /*max_length=*/12);
  auto newcomer = Mint(*model, prompt, /*max_length=*/12);
  paged.engine->AddRequest(victim);
  RunModelSteps(paged, 2);
  paged.engine->AddRequest(newcomer);

  // The step that preempts the resident then fails during execution and is rolled back.
  paged.executor->SetNextFailure(ScriptedExecutionFailure::RetryableDuringExecution);
  EXPECT_THROW(paged.engine->Step(), EngineStepError);

  // Preemption committed at the step boundary before the transaction started, so the rollback does
  // not resurrect the victim's residency, and no request is left half-advanced.
  EXPECT_EQ(victim->status_, RequestStatus::Suspended);
  EXPECT_EQ(victim->ProcessedSequenceLength(), 0);
  EXPECT_EQ(newcomer->status_, RequestStatus::Assigned);
  EXPECT_EQ(newcomer->ProcessedSequenceLength(), 0);
  const auto snapshot = paged.cache->Snapshot();
  EXPECT_EQ(snapshot.requests.size(), 0u);
  EXPECT_EQ(snapshot.free_blocks, snapshot.total_blocks);
  ExpectConsistent(paged, {victim, newcomer});

  // The engine is still healthy and drains both requests afterwards.
  int steps = 0;
  while (paged.engine->HasPendingRequests() && steps++ < 512)
    paged.engine->Step();
  EXPECT_EQ(victim->status_, RequestStatus::Completed);
  EXPECT_EQ(newcomer->status_, RequestStatus::Completed);
}

}  // namespace
}  // namespace test
}  // namespace Generators
