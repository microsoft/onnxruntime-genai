// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engine/cache_manager.h"
#include "engine/engine.h"
#include "engine/engine_invariants.h"
#include "engine/model_executor.h"
#include "engine/scheduler.h"
#include "engine_test_helpers.h"

namespace Generators {
namespace test {
namespace {

constexpr int32_t kInvariantFailureToken = 31;
const std::vector<int32_t> kInitialPrompt{4, 6};
const std::vector<int32_t> kFirstTurnOutput{12};
const std::vector<int32_t> kContinuation{8, 3, 10, 5, 12, 7, 14, 9, 16};
const std::vector<int32_t> kSecondTurnOutput{28, 17};

std::shared_ptr<Model> LoadWindowedMultiwrapModel() {
  return CreateModel(
      GetOrtEnv(), MODEL_PATH "engine/synthetic-windowed-multiwrap");
}

std::vector<int32_t> RunTurn(const std::shared_ptr<Engine>& engine,
                             const std::shared_ptr<Request>& request) {
  std::vector<int32_t> output;
  size_t steps = 0;
  while (!request->IsTurnComplete()) {
    if (++steps >= 100) {
      throw std::runtime_error(
          "Synthetic windowed request did not complete.");
    }

    auto ready = engine->Step();
    if (!ready) {
      continue;
    }
    if (ready != request) {
      throw std::runtime_error(
          "Synthetic windowed Engine returned an unexpected request.");
    }
    while (ready->HasUnseenTokens()) {
      output.push_back(ready->UnseenToken());
    }
  }
  return output;
}

// Wraps the real paged cache only to make reservation lifetime observable. All
// planning, block-table construction, cache binding, and ownership changes
// still run through PagedCacheManager.
class ObservingPagedCacheManager final : public PagedCacheManager {
 public:
  explicit ObservingPagedCacheManager(std::shared_ptr<Model> model)
      : PagedCacheManager(std::move(model)) {}

  std::unique_ptr<CacheStepReservation> ReserveStep(
      const StepPlan& plan) override {
    class ObservedReservation final : public CacheStepReservation {
     public:
      ObservedReservation(
          ObservingPagedCacheManager& owner,
          std::unique_ptr<CacheStepReservation> inner)
          : owner_{owner}, inner_{std::move(inner)} {}

      ~ObservedReservation() override {
        if (active_) {
          active_ = false;
          owner_.OnAbandonedReservation();
        }
      }

      PagedCacheReservation* PagedReservation() override {
        return inner_->PagedReservation();
      }

      void Commit() override {
        inner_->Commit();
        if (active_) {
          active_ = false;
          owner_.OnCommittedReservation();
        }
      }

      void Release() override {
        inner_->Release();
        if (active_) {
          active_ = false;
          owner_.OnReleasedReservation();
        }
      }

     private:
      ObservingPagedCacheManager& owner_;
      std::unique_ptr<CacheStepReservation> inner_;
      bool active_{true};
    };

    auto reservation = PagedCacheManager::ReserveStep(plan);
    ++reservation_count_;
    ++active_reservations_;
    return std::make_unique<ObservedReservation>(
        *this, std::move(reservation));
  }

  size_t ReservationCount() const { return reservation_count_; }
  size_t ActiveReservations() const { return active_reservations_; }
  size_t CommitCount() const { return commit_count_; }
  size_t ReleaseCount() const { return release_count_; }
  size_t AbandonedReservationCount() const {
    return abandoned_reservation_count_;
  }

 private:
  void OnCommittedReservation() {
    --active_reservations_;
    ++commit_count_;
  }

  void OnReleasedReservation() {
    --active_reservations_;
    ++release_count_;
  }

  void OnAbandonedReservation() {
    --active_reservations_;
    ++abandoned_reservation_count_;
  }

  size_t reservation_count_{};
  size_t active_reservations_{};
  size_t commit_count_{};
  size_t release_count_{};
  size_t abandoned_reservation_count_{};
};

// Delegates to the production DecoderModelExecutor first. Returning from that
// delegate means the synchronous ORT run completed with the paged cache bound
// as both input and output. Only then does this wrapper observe the real logits
// and raise a retryable failure.
class RetryableAfterRealDecodeExecutor final : public ModelExecutor {
 public:
  enum class FailurePoint {
    None,
    BeforeSearchMutation,
    AfterSearchMutation,
  };

  explicit RetryableAfterRealDecodeExecutor(
      std::unique_ptr<ModelExecutor> inner,
      std::shared_ptr<ObservingPagedCacheManager> cache)
      : inner_{std::move(inner)}, cache_{std::move(cache)} {}

  void Decode(ScheduledRequests& scheduled_requests,
              ExecutionContext& context) override {
    inner_->Decode(scheduled_requests, context);
    ++completed_real_decodes_;

    auto failure_point = failure_point_;
    if (failure_point == FailurePoint::AfterSearchMutation &&
        context.plan && !context.plan->requests.empty() &&
        context.plan->requests.front().target_cache_slots !=
            failure_target_cache_slots_) {
      failure_point = FailurePoint::None;
    } else {
      failure_point_ = FailurePoint::None;
    }
    if (failure_point == FailurePoint::None) {
      return;
    }

    if (!context.plan || context.plan->requests.size() != 1) {
      throw std::logic_error(
          "Windowed rollback fault expected one planned request.");
    }
    const auto& entry = context.plan->requests.front();
    failed_unprocessed_token_count_ = entry.unprocessed_token_count;
    failed_target_cache_slots_ = entry.target_cache_slots;
    if (failure_point == FailurePoint::AfterSearchMutation) {
      request_before_failure_ = entry.request->Snapshot();
      cache_before_failure_ = cache_->Snapshot();
      std::vector<RequestStepResult> staged_results;
      scheduled_requests.GenerateNextTokensForTransaction(
          *context.plan, staged_results);
      observed_real_logits_ =
          staged_results.size() == scheduled_requests.size();
      staged_sequence_length_ =
          entry.request->CurrentSequenceLength();
      staged_search_mutation_ =
          staged_sequence_length_ > entry.sequence_length_before;
    } else {
      const auto logits = scheduled_requests.ProcessLogits();
      observed_real_logits_ =
          logits.size() == scheduled_requests.size();
    }
    injected_failure_ = true;
    throw ModelExecutionError{
        ExecutionFailureKind::RetryableAbort,
        "Injected retryable failure after real windowed model execution.",
    };
  }

  void FailNextDecode() {
    failure_point_ = FailurePoint::BeforeSearchMutation;
  }
  void FailNextDecodeAfterSearchMutationAtTarget(
      size_t target_cache_slots) {
    failure_point_ = FailurePoint::AfterSearchMutation;
    failure_target_cache_slots_ = target_cache_slots;
  }

  bool InjectedFailure() const { return injected_failure_; }
  bool ObservedRealLogits() const { return observed_real_logits_; }
  size_t CompletedRealDecodes() const { return completed_real_decodes_; }
  size_t FailedUnprocessedTokenCount() const {
    return failed_unprocessed_token_count_;
  }
  size_t FailedTargetCacheSlots() const {
    return failed_target_cache_slots_;
  }
  bool StagedSearchMutation() const { return staged_search_mutation_; }
  int64_t StagedSequenceLength() const {
    return staged_sequence_length_;
  }
  const RequestStateSnapshot& RequestBeforeFailure() const {
    return request_before_failure_;
  }
  const PagedCacheSnapshot& CacheBeforeFailure() const {
    return cache_before_failure_;
  }

 private:
  std::unique_ptr<ModelExecutor> inner_;
  std::shared_ptr<ObservingPagedCacheManager> cache_;
  FailurePoint failure_point_{FailurePoint::None};
  size_t failure_target_cache_slots_{};
  bool injected_failure_{};
  bool observed_real_logits_{};
  bool staged_search_mutation_{};
  size_t completed_real_decodes_{};
  size_t failed_unprocessed_token_count_{};
  size_t failed_target_cache_slots_{};
  int64_t staged_sequence_length_{};
  RequestStateSnapshot request_before_failure_;
  PagedCacheSnapshot cache_before_failure_;
};

struct FaultInjectingEngine {
  std::shared_ptr<Engine> engine;
  std::shared_ptr<ObservingPagedCacheManager> cache;
  RetryableAfterRealDecodeExecutor* executor{};
};

FaultInjectingEngine MakeFaultInjectingEngine(
    const std::shared_ptr<Model>& model) {
  auto cache =
      std::make_shared<ObservingPagedCacheManager>(model);
  auto scheduler = Scheduler::Create(model, cache);
  auto executor = std::make_unique<RetryableAfterRealDecodeExecutor>(
      ModelExecutor::Create(model, cache), cache);
  auto* executor_observer = executor.get();
  EngineDependencies dependencies{
      cache, std::move(scheduler), std::move(executor)};
  auto engine =
      std::make_shared<Engine>(model, std::move(dependencies));
  return FaultInjectingEngine{
      std::move(engine), std::move(cache), executor_observer};
}

void ExpectRequestBlocksEqual(const RequestBlockSnapshot& actual,
                              const RequestBlockSnapshot& expected) {
  EXPECT_EQ(actual.request_id, expected.request_id);
  EXPECT_EQ(actual.block_ids, expected.block_ids);
  EXPECT_EQ(actual.used_slots, expected.used_slots);
  EXPECT_EQ(actual.empty_slots, expected.empty_slots);
}

void ExpectCacheOwnershipRestored(const PagedCacheSnapshot& actual,
                                  const PagedCacheSnapshot& expected) {
  EXPECT_EQ(actual.block_size, expected.block_size);
  EXPECT_EQ(actual.total_blocks, expected.total_blocks);
  EXPECT_EQ(actual.free_blocks, expected.free_blocks);
  EXPECT_EQ(actual.AllocatedBlocks(), expected.AllocatedBlocks());
  EXPECT_TRUE(actual.transaction_reserved_block_ids.empty());
  EXPECT_TRUE(actual.reservations.empty());

  ASSERT_EQ(actual.requests.size(), expected.requests.size());
  for (size_t i = 0; i < actual.requests.size(); ++i) {
    ExpectRequestBlocksEqual(actual.requests[i], expected.requests[i]);
  }

  EXPECT_EQ(actual.window_blocks.total_blocks,
            expected.window_blocks.total_blocks);
  EXPECT_EQ(actual.window_blocks.free_blocks,
            expected.window_blocks.free_blocks);
  EXPECT_EQ(actual.window_blocks.blocks_per_request,
            expected.window_blocks.blocks_per_request);
  EXPECT_TRUE(
      actual.window_blocks.transaction_reserved_block_ids.empty());
  ASSERT_EQ(actual.window_blocks.requests.size(),
            expected.window_blocks.requests.size());
  for (size_t i = 0; i < actual.window_blocks.requests.size(); ++i) {
    ExpectRequestBlocksEqual(actual.window_blocks.requests[i],
                             expected.window_blocks.requests[i]);
  }
}

std::vector<int32_t> RunCleanReplay() {
  auto model = LoadWindowedMultiwrapModel();
  auto engine = std::make_shared<Engine>(model);
  std::vector<int32_t> replay_prompt = kInitialPrompt;
  replay_prompt.insert(replay_prompt.end(), kFirstTurnOutput.begin(),
                       kFirstTurnOutput.end());
  replay_prompt.insert(replay_prompt.end(), kContinuation.begin(),
                       kContinuation.end());
  auto request = MintRequest(*model, replay_prompt);
  engine->AddRequest(request);

  auto output = RunTurn(engine, request);
  engine->RemoveRequest(request);
  return output;
}

TEST(WindowedTransactionTest,
     ContinuedRingWritesRollbackAndRetryMatchCleanReplay) {
  const auto clean_replay_output = RunCleanReplay();
  ASSERT_EQ(clean_replay_output, kSecondTurnOutput);
  ASSERT_EQ(std::find(clean_replay_output.begin(),
                      clean_replay_output.end(),
                      kInvariantFailureToken),
            clean_replay_output.end());

  auto model = LoadWindowedMultiwrapModel();
  auto faulting = MakeFaultInjectingEngine(model);
  auto request = MintRequest(*model, kInitialPrompt);
  faulting.engine->AddRequest(request);
  ASSERT_EQ(RunTurn(faulting.engine, request), kFirstTurnOutput);
  ASSERT_EQ(request->Status(), RequestStatus::TurnComplete);
  ASSERT_FALSE(request->HasUnseenTokens());

  request->Continue(kContinuation);
  const auto request_before = request->Snapshot();
  const auto cache_before = faulting.cache->Snapshot();
  ASSERT_EQ(request_before.status, RequestStatus::Assigned);
  ASSERT_EQ(request_before.current_sequence_length, 12);
  ASSERT_EQ(request_before.processed_sequence_length, 3);
  ASSERT_EQ(cache_before.requests.size(), 1u);
  ASSERT_EQ(cache_before.window_blocks.requests.size(), 1u);
  ASSERT_EQ(cache_before.window_blocks.blocks_per_request, 2u);
  const size_t ring_period =
      cache_before.window_blocks.blocks_per_request *
      cache_before.block_size;
  ASSERT_EQ(ring_period, 4u);

  const size_t reservations_before =
      faulting.cache->ReservationCount();
  const size_t commits_before = faulting.cache->CommitCount();
  const size_t releases_before = faulting.cache->ReleaseCount();
  const size_t decodes_before =
      faulting.executor->CompletedRealDecodes();
  faulting.executor->FailNextDecode();

  try {
    static_cast<void>(faulting.engine->Step());
    FAIL() << "Expected retryable failure after real windowed decode.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind,
              StepOutcomeKind::RetryableBatchAbort);
  }

  EXPECT_TRUE(faulting.executor->InjectedFailure());
  EXPECT_TRUE(faulting.executor->ObservedRealLogits());
  EXPECT_EQ(faulting.executor->CompletedRealDecodes(),
            decodes_before + 1);
  EXPECT_EQ(faulting.executor->FailedUnprocessedTokenCount(), 2u);
  EXPECT_EQ(faulting.executor->FailedTargetCacheSlots(), 5u);
  // The failed run wrote absolute positions [3, 5), which map to ring
  // slots [3, 0]. Thus the injected fault occurred only after a real
  // continuation run crossed the four-slot ring boundary.
  EXPECT_EQ(
      static_cast<size_t>(
          request_before.processed_sequence_length) %
          ring_period,
      ring_period - 1);
  EXPECT_EQ(
      (faulting.executor->FailedTargetCacheSlots() - 1) %
          ring_period,
      0u);

  const auto request_after = request->Snapshot();
  EXPECT_EQ(request_after.status, RequestStatus::Assigned);
  EXPECT_EQ(request_after.current_sequence_length,
            request_before.current_sequence_length);
  EXPECT_EQ(request_after.processed_sequence_length,
            request_before.processed_sequence_length);
  EXPECT_FALSE(request->HasUnseenTokens());
  EXPECT_TRUE(faulting.engine->HasPendingRequests());

  const auto cache_after = faulting.cache->Snapshot();
  ExpectCacheOwnershipRestored(cache_after, cache_before);
  EXPECT_EQ(faulting.cache->ReservationCount(),
            reservations_before + 1);
  EXPECT_EQ(faulting.cache->CommitCount(), commits_before);
  EXPECT_EQ(faulting.cache->ReleaseCount(),
            releases_before + 1);
  EXPECT_EQ(faulting.cache->ActiveReservations(), 0u);
  EXPECT_EQ(faulting.cache->AbandonedReservationCount(), 0u);
  EXPECT_NO_THROW(ThrowIfInvariantsViolated(
      cache_after, std::vector<RequestStateSnapshot>{request_after}));

  const auto retry_output = RunTurn(faulting.engine, request);
  EXPECT_EQ(retry_output, clean_replay_output);
  EXPECT_EQ(retry_output, kSecondTurnOutput);
  EXPECT_EQ(std::find(retry_output.begin(), retry_output.end(),
                      kInvariantFailureToken),
            retry_output.end());
  EXPECT_EQ(request->Status(), RequestStatus::TurnComplete);
  EXPECT_EQ(request->CurrentSequenceLength(), 14);
  EXPECT_EQ(faulting.cache->ActiveReservations(), 0u);

  faulting.engine->RemoveRequest(request);
}

TEST(WindowedTransactionTest,
     ContinuedRingWritesAndStagedSearchRollbackTogetherBeforeRetry) {
  const auto clean_replay_output = RunCleanReplay();
  ASSERT_EQ(clean_replay_output, kSecondTurnOutput);

  auto model = LoadWindowedMultiwrapModel();
  auto faulting = MakeFaultInjectingEngine(model);
  auto request = MintRequest(*model, kInitialPrompt);
  faulting.engine->AddRequest(request);
  ASSERT_EQ(RunTurn(faulting.engine, request), kFirstTurnOutput);
  request->Continue(kContinuation);

  ASSERT_EQ(request->Status(), RequestStatus::Assigned);
  ASSERT_FALSE(request->HasUnseenTokens());

  const size_t reservations_before =
      faulting.cache->ReservationCount();
  const size_t commits_before = faulting.cache->CommitCount();
  const size_t releases_before = faulting.cache->ReleaseCount();
  // Step commits four two-token continuation chunks before its final one-token prefill. Inject only
  // at target 12: that transaction writes absolute position 11 into ring slot 3 after multiple
  // wraps, then stages token 28 in Search before raising the retryable failure.
  faulting.executor->FailNextDecodeAfterSearchMutationAtTarget(12);

  try {
    static_cast<void>(faulting.engine->Step());
    FAIL() << "Expected retryable failure after staged Search mutation.";
  } catch (const EngineStepError& error) {
    EXPECT_EQ(error.Outcome().kind,
              StepOutcomeKind::RetryableBatchAbort);
  }

  EXPECT_TRUE(faulting.executor->InjectedFailure());
  EXPECT_TRUE(faulting.executor->ObservedRealLogits());
  EXPECT_TRUE(faulting.executor->StagedSearchMutation());
  EXPECT_EQ(faulting.executor->StagedSequenceLength(), 13);
  EXPECT_EQ(faulting.executor->FailedUnprocessedTokenCount(), 1u);
  EXPECT_EQ(faulting.executor->FailedTargetCacheSlots(), 12u);

  const auto& request_before =
      faulting.executor->RequestBeforeFailure();
  const auto& cache_before =
      faulting.executor->CacheBeforeFailure();
  ASSERT_EQ(request_before.status, RequestStatus::Active);
  ASSERT_EQ(request_before.current_sequence_length, 12);
  ASSERT_EQ(request_before.processed_sequence_length, 11);

  const auto request_after = request->Snapshot();
  EXPECT_EQ(request_after.status, request_before.status);
  EXPECT_EQ(request_after.current_sequence_length,
            request_before.current_sequence_length);
  EXPECT_EQ(request_after.processed_sequence_length,
            request_before.processed_sequence_length);
  EXPECT_FALSE(request->HasUnseenTokens());
  EXPECT_TRUE(faulting.engine->HasPendingRequests());

  const auto cache_after = faulting.cache->Snapshot();
  ExpectCacheOwnershipRestored(cache_after, cache_before);
  EXPECT_EQ(faulting.cache->ReservationCount(),
            reservations_before + 5);
  EXPECT_EQ(faulting.cache->CommitCount(), commits_before + 4);
  EXPECT_EQ(faulting.cache->ReleaseCount(),
            releases_before + 1);
  EXPECT_EQ(faulting.cache->ActiveReservations(), 0u);
  EXPECT_NO_THROW(ThrowIfInvariantsViolated(
      cache_after, std::vector<RequestStateSnapshot>{request_after}));

  const auto retry_output = RunTurn(faulting.engine, request);
  EXPECT_EQ(retry_output, clean_replay_output);
  EXPECT_EQ(retry_output, kSecondTurnOutput);
  EXPECT_EQ(request->Status(), RequestStatus::TurnComplete);
  EXPECT_EQ(request->CurrentSequenceLength(), 14);

  faulting.engine->RemoveRequest(request);
}

}  // namespace
}  // namespace test
}  // namespace Generators
