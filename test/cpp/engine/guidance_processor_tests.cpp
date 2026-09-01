// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests Request's guidance bookkeeping - independent per-request state, transactional
// stage/rollback, and turn completion behavior using a lightweight fake
// ConstrainedLogitsProcessor instead of the real llguidance-backed one. That makes these tests
// model-free with respect to guidance itself (no grammar, no tokenizer.json, no USE_GUIDANCE
// build requirement): they only need the tiny checked-in dummy-decoder model that every other
// engine unit test already uses to mint a Request.
//
// See test/engine/request_lifecycle_tests.cpp for the equivalent tests against the real
// GuidanceLogitsProcessor (gated on USE_GUIDANCE, since that needs llguidance and a tokenizer).

#include <array>
#include <atomic>
#include <limits>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_helpers.h"
#include "engine_test_doubles.h"
#include "engine/request_status.h"
#include "engine/scheduled_requests.h"
#include "engine/step_plan.h"
#include "constrained_logits_processor.h"
#include "guidance_test_access.h"

namespace Generators {
namespace test {

namespace {

std::vector<int32_t> Prompt() { return {2, 3, 4}; }

DeviceSpan<float> LogitsForToken(Model& model, int32_t token) {
  auto logits = model.p_device_inputs_->Allocate<float>(
      static_cast<size_t>(model.config_->model.vocab_size));
  auto cpu_logits = logits.CpuSpan();
  std::fill(cpu_logits.begin(), cpu_logits.end(), 0.0f);
  cpu_logits[token] = 100.0f;
  logits.CopyCpuToDevice();
  return logits;
}

// A minimal stand-in for GuidanceLogitsProcessor: it records committed tokens, can force a single
// token through ProcessLogits (masking every other logit to -inf, exactly like a real grammar
// mask would once only one token is valid), and can be told to fail its next Reset() so tests can
// exercise failure paths without needing a grammar that actually fails to parse.
class FakeGuidanceProcessor final : public ConstrainedLogitsProcessor {
 public:
  // `reset_count` is shared across every clone descended from the same instance, so a test can
  // tell how many times Reset() ran anywhere in that lineage (e.g. across a transactional
  // checkpoint clone) while each instance still owns its own commit
  // history and forced-token configuration independently.
  explicit FakeGuidanceProcessor(std::shared_ptr<int> reset_count = std::make_shared<int>(0))
      : reset_count_(std::move(reset_count)) {}

  void CommitTokens(std::span<int32_t> tokens) override {
    committed_tokens_.insert(committed_tokens_.end(), tokens.begin(), tokens.end());
  }

  void ProcessLogits(DeviceSpan<float> logits) override {
    if (pending_failure_ &&
        !pending_failure_->consumed.exchange(true)) {
      pending_failure_->future.get();
    }
    if (!forced_token_) {
      return;
    }
    auto cpu_logits = logits.CpuSpan();
    for (size_t i = 0; i < cpu_logits.size(); ++i) {
      if (static_cast<int32_t>(i) != *forced_token_) {
        cpu_logits[i] = std::numeric_limits<float>::lowest();
      }
    }
  }

  void Reset() override {
    if (fail_reset_) {
      throw std::runtime_error("Injected guidance reset failure.");
    }
    ++*reset_count_;
    committed_tokens_.clear();
  }

  std::vector<int32_t> GetFFTokens(size_t /*index*/) override { return {}; }

  std::span<const uint32_t> GetReadyMask() override {
    return ready_mask_;
  }

  std::unique_ptr<ConstrainedLogitsProcessor> Clone() const override {
    auto clone = std::make_unique<FakeGuidanceProcessor>(reset_count_);
    clone->committed_tokens_ = committed_tokens_;
    clone->forced_token_ = forced_token_;
    clone->fail_reset_ = fail_reset_;
    clone->ready_mask_ = ready_mask_;
    clone->pending_failure_ = pending_failure_;
    return clone;
  }

  void ForceToken(std::optional<int32_t> token) { forced_token_ = token; }
  void SetReadyMask(std::vector<uint32_t> mask) {
    ready_mask_ = std::move(mask);
  }
  void FailPendingMaskOnce() {
    std::promise<void> promise;
    promise.set_exception(std::make_exception_ptr(
        std::runtime_error("Injected asynchronous guidance mask failure.")));
    pending_failure_ = std::make_shared<PendingFailure>(
        promise.get_future().share());
  }
  void SetFailReset(bool fail) { fail_reset_ = fail; }
  int ResetCount() const { return *reset_count_; }
  const std::vector<int32_t>& CommittedTokens() const { return committed_tokens_; }

 private:
  struct PendingFailure {
    explicit PendingFailure(std::shared_future<void> value)
        : future{std::move(value)} {}
    std::shared_future<void> future;
    std::atomic<bool> consumed{};
  };

  std::shared_ptr<int> reset_count_;
  std::vector<int32_t> committed_tokens_;
  std::optional<int32_t> forced_token_;
  bool fail_reset_{false};
  std::vector<uint32_t> ready_mask_;
  std::shared_ptr<PendingFailure> pending_failure_;
};

FakeGuidanceProcessor& InstallFake(Request& request) {
  RequestGuidanceTestAccess::Install(request, std::make_unique<FakeGuidanceProcessor>());
  return *static_cast<FakeGuidanceProcessor*>(RequestGuidanceTestAccess::Get(request));
}

class GuidanceProcessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadDummyDecoderModel();
    engine_ = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));
  }

  std::shared_ptr<Request> NewAssignedRequest(int32_t max_length_beyond_prompt) {
    auto params = MakeGreedyParams(*model_);
    const auto prompt = Prompt();
    params->search.max_length =
        static_cast<int32_t>(prompt.size()) + max_length_beyond_prompt;
    auto request = CreateEngineRequest(engine_.engine, *params);
    request->BeginTurn(prompt);
    return request;
  }

  std::shared_ptr<Model> model_;
  DoublesEngine engine_;
};

// Two requests must not share any guidance state: committing tokens on one must never appear on
// the other's grammar cursor. This is the basic per-request-instance guarantee that everything
// else in this file builds on.
TEST_F(GuidanceProcessorTest, TwoRequestsMaintainIndependentGrammarState) {
  auto request_a = NewAssignedRequest(/*max_length_beyond_prompt=*/4);
  auto request_b = NewAssignedRequest(/*max_length_beyond_prompt=*/4);
  auto& fake_a = InstallFake(*request_a);
  auto& fake_b = InstallFake(*request_b);

  constexpr int32_t token_a = 5;
  constexpr int32_t token_b = 7;
  request_a->GenerateNextTokens(LogitsForToken(*model_, token_a));
  request_a->CompleteGeneration();
  request_b->GenerateNextTokens(LogitsForToken(*model_, token_b));
  request_b->CompleteGeneration();

  EXPECT_EQ(fake_a.CommittedTokens(), (std::vector<int32_t>{token_a}));
  EXPECT_EQ(fake_b.CommittedTokens(), (std::vector<int32_t>{token_b}));

  // A second round on A must still only ever see A's tokens.
  request_a->GenerateNextTokens(LogitsForToken(*model_, token_b));
  request_a->CompleteGeneration();
  EXPECT_EQ(fake_a.CommittedTokens(), (std::vector<int32_t>{token_a, token_b}));
  EXPECT_EQ(fake_b.CommittedTokens(), (std::vector<int32_t>{token_b}));
}

// The dynamic (transactional) path clones the live processor into a checkpoint before staging a
// step. A rolled-back attempt must leave the request with the pre-stage grammar state - not the
// mutated staged one - and a retried attempt afterwards must start from that same clean state
// rather than from whatever the rolled-back attempt left behind.
TEST_F(GuidanceProcessorTest, RollbackIsolatesRetriesFromDiscardedAttempts) {
  auto request = NewAssignedRequest(/*max_length_beyond_prompt=*/4);
  InstallFake(*request);
  const auto before = request->Snapshot();

  RequestStepPlan plan;
  plan.request = request;
  plan.request_id = request.get();
  plan.sequence_length_before = before.current_sequence_length;
  plan.target_cache_slots = static_cast<size_t>(before.current_sequence_length);
  PrepareRequestStep(model_, plan);

  // First attempt: stage a token, then roll it back without committing.
  constexpr int32_t discarded_token = 9;
  request->SaveStateForTransaction();
  const auto staged = request->ApplyLogitsForTransaction(LogitsForToken(*model_, discarded_token));
  ASSERT_TRUE(staged.token_appended);
  ASSERT_EQ(staged.token, discarded_token);
  auto* staged_fake = static_cast<FakeGuidanceProcessor*>(RequestGuidanceTestAccess::Get(*request));
  EXPECT_EQ(staged_fake->CommittedTokens(), (std::vector<int32_t>{discarded_token}));
  request->RestoreStateForTransaction();

  // The restored processor must be the pre-stage checkpoint, not the mutated staged instance, and
  // it must show no trace of the discarded attempt's committed token.
  auto* restored_fake = static_cast<FakeGuidanceProcessor*>(RequestGuidanceTestAccess::Get(*request));
  EXPECT_NE(restored_fake, staged_fake);
  EXPECT_TRUE(restored_fake->CommittedTokens().empty());

  // Retry with a different token and commit for real this time.
  constexpr int32_t committed_token = 11;
  request->SaveStateForTransaction();
  const auto retried = request->ApplyLogitsForTransaction(LogitsForToken(*model_, committed_token));
  ASSERT_EQ(retried.token, committed_token);
  request->CommitStateForTransaction();
  request->CommitStep(plan, retried);

  auto* committed_fake = static_cast<FakeGuidanceProcessor*>(RequestGuidanceTestAccess::Get(*request));
  // Only the committed retry's token should ever have reached the live processor.
  EXPECT_EQ(committed_fake->CommittedTokens(), (std::vector<int32_t>{committed_token}));
}

TEST_F(GuidanceProcessorTest, BatchedMasksPreserveUnguidedRows) {
  auto guided = NewAssignedRequest(/*max_length_beyond_prompt=*/4);
  auto unguided = NewAssignedRequest(/*max_length_beyond_prompt=*/4);
  auto& processor = InstallFake(*guided);
  processor.SetReadyMask({0x5u, 0xau});
  guided->BindScheduledTokenCount(Prompt().size());
  unguided->BindScheduledTokenCount(Prompt().size());
  const std::array requests{guided, unguided};
  std::vector<uint32_t> masks;

  EXPECT_EQ(CollectBatchedGuidanceMasks(requests, 2, masks),
            BatchedGuidanceMaskStatus::Ready);
  EXPECT_EQ(masks, (std::vector<uint32_t>{
                       0x5u, 0xau,
                       std::numeric_limits<uint32_t>::max(),
                       std::numeric_limits<uint32_t>::max()}));
}

TEST_F(GuidanceProcessorTest, BatchedMasksSkipPartialPrefillRows) {
  auto partial_prefill = NewAssignedRequest(/*max_length_beyond_prompt=*/4);
  auto guided_decode = NewAssignedRequest(/*max_length_beyond_prompt=*/4);
  auto& partial_processor = InstallFake(*partial_prefill);
  auto& decode_processor = InstallFake(*guided_decode);
  partial_processor.SetReadyMask({0x1u});
  decode_processor.SetReadyMask({0x2u});
  partial_prefill->BindScheduledTokenCount(Prompt().size() - 1);
  guided_decode->BindScheduledTokenCount(Prompt().size());
  const std::array requests{partial_prefill, guided_decode};
  std::vector<uint32_t> masks;

  EXPECT_EQ(CollectBatchedGuidanceMasks(requests, 1, masks),
            BatchedGuidanceMaskStatus::Ready);
  EXPECT_EQ(masks, (std::vector<uint32_t>{
                       std::numeric_limits<uint32_t>::max(), 0x2u}));
}

TEST_F(GuidanceProcessorTest, BatchedMasksFallBackOnInvalidGuidedRow) {
  auto guided = NewAssignedRequest(/*max_length_beyond_prompt=*/4);
  auto& processor = InstallFake(*guided);
  processor.SetReadyMask({0x1u});
  guided->BindScheduledTokenCount(Prompt().size());
  const std::array requests{guided};
  std::vector<uint32_t> masks;

  EXPECT_EQ(CollectBatchedGuidanceMasks(requests, 2, masks),
            BatchedGuidanceMaskStatus::FallbackRequired);
}

TEST_F(GuidanceProcessorTest, AsyncMaskFailureRollsBackAndCanRetry) {
  auto request = NewAssignedRequest(/*max_length_beyond_prompt=*/4);
  auto& processor = InstallFake(*request);
  processor.FailPendingMaskOnce();
  engine_.executor->SetForcedToken(EosToken(*model_));
  const auto before = request->Snapshot();

  const auto retryable = RunOne(*engine_.engine);
  EXPECT_EQ(retryable.flags, EngineEventFlagRetryable);
  EXPECT_EQ(retryable.error_code, EngineErrorCode::RetryableExecution);

  const auto rolled_back = request->Snapshot();
  EXPECT_EQ(rolled_back.status, before.status);
  EXPECT_EQ(rolled_back.current_sequence_length,
            before.current_sequence_length);
  EXPECT_EQ(rolled_back.processed_sequence_length,
            before.processed_sequence_length);

  const auto event = RunOne(*engine_.engine);
  EXPECT_EQ(event.request, request);
  EXPECT_NE(event.flags & EngineEventFlagTurnFinished, 0u);
}

// Turn completion resets the installed cursor so a subsequent BeginTurn starts from the grammar's
// initial state. An EOS stop signal is not appended, so the cursor sees no CommitTokens call.
TEST_F(GuidanceProcessorTest, TurnCompletionResetsGrammarCursor) {
  auto request = NewAssignedRequest(/*max_length_beyond_prompt=*/4);
  auto& fake = InstallFake(*request);

  request->GenerateNextTokens(LogitsForToken(*model_, EosToken(*model_)));
  request->CompleteGeneration();

  ASSERT_TRUE(request->IsTurnComplete());
  EXPECT_EQ(RequestGuidanceTestAccess::Get(*request), &fake);
  EXPECT_EQ(fake.ResetCount(), 1);
  EXPECT_TRUE(fake.CommittedTokens().empty());
}

}  // namespace
}  // namespace test
}  // namespace Generators
