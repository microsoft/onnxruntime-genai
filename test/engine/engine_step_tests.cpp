// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Component-contract tests for Engine::Step. They wire an Engine with recording cache-manager
// and model-executor doubles over a tiny CPU fixture model. The doubles never run the model: the
// executor fabricates end-of-stream logits so each scheduled request's real greedy search completes
// deterministically. This lets the tests assert how Step orchestrates its collaborators -- that it
// allocates a batch before decoding it, decodes once per internal cycle, drains ready requests
// without redundant model runs, and forms a fresh batch across steps under capacity backpressure.

#include <algorithm>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "engine_test_helpers.h"
#include "engine_test_doubles.h"

namespace Generators {
namespace test {
namespace {

std::vector<int32_t> Prompt(int32_t seed) {
  const int32_t base = 2 + (seed % 8);
  return {base, base + 1, base + 2};
}

// Index of the first occurrence of `entry` in the trace, or -1 if absent.
int IndexOf(const CallTrace& trace, const std::string& entry) {
  auto it = std::find(trace.entries.begin(), trace.entries.end(), entry);
  return it == trace.entries.end() ? -1 : static_cast<int>(it - trace.entries.begin());
}

class EngineStepTest : public ::testing::Test {
 protected:
  void SetUp() override { model_ = LoadDummyDecoderModel(); }

  std::shared_ptr<Model> model_;
};

// One request: Step schedules it, decodes the batch exactly once, and returns the request. The trace
// shows the batch is allocated before it is decoded.
TEST_F(EngineStepTest, SingleRequestSchedulesThenDecodesThenReturns) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));

  auto prompt = Prompt(10);
  auto request = MintRequest(*model_, prompt);
  engine.engine->AddRequest(request);
  ASSERT_TRUE(engine.engine->HasPendingRequests());

  auto ready = engine.engine->Step();

  ASSERT_NE(ready, nullptr);
  EXPECT_EQ(ready, request);
  EXPECT_TRUE(request->IsDone());
  EXPECT_EQ(engine.executor->decode_calls, 1);
  ASSERT_EQ(engine.executor->decoded_batch_sizes.size(), 1u);
  EXPECT_EQ(engine.executor->decoded_batch_sizes[0], 1u);

  const int allocate_at = IndexOf(*engine.trace, "Allocate");
  const int decode_at = IndexOf(*engine.trace, "Decode");
  ASSERT_GE(allocate_at, 0);
  ASSERT_GE(decode_at, 0);
  EXPECT_LT(allocate_at, decode_at);
}

// Several requests that all fit are decoded together in a single batch, and the remaining ready
// requests are drained without any further model execution.
TEST_F(EngineStepTest, FittingRequestsShareOneDecodeAndDrainWithoutReexecuting) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));

  std::vector<std::shared_ptr<Request>> requests;
  for (int32_t seed : {10, 20, 30}) {
    auto prompt = Prompt(seed);
    auto request = MintRequest(*model_, prompt);
    engine.engine->AddRequest(request);
    requests.push_back(request);
  }

  std::vector<std::shared_ptr<Request>> returned;
  while (auto ready = engine.engine->Step()) {
    returned.push_back(ready);
  }

  EXPECT_EQ(returned.size(), 3u);
  auto sorted_requests = requests;
  std::sort(sorted_requests.begin(), sorted_requests.end());
  std::sort(returned.begin(), returned.end());
  EXPECT_EQ(returned, sorted_requests);
  for (const auto& request : requests) {
    EXPECT_TRUE(request->IsDone());
  }
  EXPECT_EQ(engine.executor->decode_calls, 1);
  ASSERT_EQ(engine.executor->decoded_batch_sizes.size(), 1u);
  EXPECT_EQ(engine.executor->decoded_batch_sizes[0], 3u);
  EXPECT_FALSE(engine.engine->HasPendingRequests());
}

// Under capacity backpressure Step decodes only the requests that fit, then forms a fresh batch for
// the deferred request on a later step -- one decode per internal cycle, never an over-capacity run.
TEST_F(EngineStepTest, BackpressureFormsAFreshBatchAcrossSteps) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/2, EosToken(*model_));

  std::vector<std::shared_ptr<Request>> requests;
  for (int32_t seed : {10, 20, 30}) {
    auto prompt = Prompt(seed);
    auto request = MintRequest(*model_, prompt);
    engine.engine->AddRequest(request);
    requests.push_back(request);
  }

  std::vector<std::shared_ptr<Request>> returned;
  while (auto ready = engine.engine->Step()) {
    returned.push_back(ready);
  }

  EXPECT_EQ(returned.size(), 3u);
  auto sorted_requests = requests;
  std::sort(sorted_requests.begin(), sorted_requests.end());
  std::sort(returned.begin(), returned.end());
  EXPECT_EQ(returned, sorted_requests);
  for (const auto& request : requests) {
    EXPECT_TRUE(request->IsDone());
  }
  ASSERT_EQ(engine.executor->decoded_batch_sizes.size(), 2u);
  EXPECT_EQ(engine.executor->decoded_batch_sizes[0], 2u);
  EXPECT_EQ(engine.executor->decoded_batch_sizes[1], 1u);
}

// With nothing queued, Step reports no work and returns cleanly.
TEST_F(EngineStepTest, StepWithNoRequestsReturnsNull) {
  auto engine = MakeDoublesEngine(model_, /*capacity=*/8, EosToken(*model_));

  EXPECT_FALSE(engine.engine->HasPendingRequests());
  EXPECT_EQ(engine.engine->Step(), nullptr);
  EXPECT_EQ(engine.executor->decode_calls, 0);
}

}  // namespace
}  // namespace test
}  // namespace Generators
