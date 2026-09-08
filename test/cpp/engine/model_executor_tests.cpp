// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "engine/model_executor.h"
#include "engine/scheduler.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"
#include "models/utils.h"

namespace Generators {
namespace test {
namespace {

struct ThrowingDecoder : Decoder {
  explicit ThrowingDecoder(std::string message)
      : message_{std::move(message)} {}

  void Decode(ScheduledRequests&, ExecutionContext&) override {
    Ort::ThrowOnError(
        Ort::api->CreateStatus(ORT_FAIL, message_.c_str()));
  }

 private:
  std::string message_;
};

class ModelExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadDummyDecoderModel();
    cache_ = std::make_shared<RecordingCacheManager>(model_, 1);
  }

  DecoderModelExecutor CreateExecutor(std::string message) {
    return DecoderModelExecutor{
        model_, cache_, std::make_unique<ThrowingDecoder>(std::move(message))};
  }

  ScheduledRequests EmptyBatch() {
    return ScheduledRequests{
        std::vector<std::shared_ptr<Request>>{}, model_, nullptr, nullptr};
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<RecordingCacheManager> cache_;
};

TEST_F(ModelExecutorTest, TranslatesOrtArenaAllocationFailure) {
  auto executor = CreateExecutor(
      "BFCArena::AllocateRawInternal Failed to allocate memory for requested "
      "buffer of size 2443129088");
  auto scheduled_requests = EmptyBatch();
  ExecutionContext context;

  try {
    executor.Decode(scheduled_requests, context);
    FAIL() << "Expected an execution capacity failure.";
  } catch (const ModelExecutionError& error) {
    EXPECT_EQ(error.FailureKind(), ExecutionFailureKind::CapacityExceeded);
    EXPECT_NE(std::string{error.what()}.find("2443129088"),
              std::string::npos);
  }
}

TEST_F(ModelExecutorTest, PropagatesUnrelatedOrtFailure) {
  auto executor =
      CreateExecutor("Non-zero status code returned while running a kernel.");
  auto scheduled_requests = EmptyBatch();
  ExecutionContext context;

  EXPECT_THROW(executor.Decode(scheduled_requests, context), Ort::Exception);
}

void ExpectPackedHiddenStates(const std::shared_ptr<Model>& model) {
  auto cache = std::shared_ptr<CacheManager>{CacheManager::Create(model)};
  auto scheduler = Scheduler::Create(model, cache);
  auto assign_target = MakeDoublesEngine(model, /*capacity=*/8, EosToken(*model)).engine;
  const std::vector<int32_t> first_prompt{5, 9, 13};
  const std::vector<int32_t> second_prompt{7, 2, 20, 4};
  auto first = CreateRequestWithPrompt(assign_target, first_prompt);
  auto second = CreateRequestWithPrompt(assign_target, second_prompt);
  scheduler->AddRequest(first);
  scheduler->AddRequest(second);

  StepPlan plan;
  const auto planning_result = scheduler->PlanStep(plan);
  ASSERT_TRUE(planning_result.executable);
  auto reservation = cache->ReserveStep(plan);
  auto scheduled_requests = scheduler->CreateScheduledRequests(plan);
  ExecutionContext context{&plan};
  context.cache_reservation = reservation->PagedReservation();
  context.fixed_state_slots = reservation->FixedStateSlots();
  context.fixed_state_bindings = reservation->FixedStateBindings();
  context.fixed_state_staging_bytes = reservation->FixedStateStagingBytes();
  scheduled_requests.BeginTransaction();

  auto executor = ModelExecutor::Create(model, cache);
  executor->Decode(scheduled_requests, context);

  auto* hidden_states = scheduled_requests.HiddenStates();
  ASSERT_NE(hidden_states, nullptr);
  EXPECT_EQ(hidden_states->GetType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  EXPECT_EQ(hidden_states->GetShape(), (std::vector<int64_t>{7, 1}));
  const auto* values = hidden_states->GetData<Ort::Float16_t>();
  const std::vector<int32_t> expected{5, 9, 13, 7, 2, 20, 4};
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(ToFloat32(values[index]), static_cast<float>(expected[index]));
  }
  reservation->Release();
}

TEST(ModelExecutorHiddenStatesTest, PreservesPackedRowsWithPrunedLogits) {
  auto model = LoadSyntheticPagedMtpModel();
  auto cache = std::shared_ptr<CacheManager>{CacheManager::Create(model)};
  EXPECT_FALSE(ModelExecutor::Create(model, cache)->SupportsDraftVerification());
  ExpectPackedHiddenStates(model);
}

TEST(ModelExecutorHiddenStatesTest, ForwardsPackedRowsThroughHybridDecoder) {
  ExpectPackedHiddenStates(LoadSyntheticCompositeMtpModel());
}

TEST(ModelExecutorHiddenStatesTest, DoesNotBindOutputWithoutMtpDemand) {
  auto model = LoadSyntheticPagedModel();
  auto cache = std::shared_ptr<CacheManager>{CacheManager::Create(model)};
  auto scheduler = Scheduler::Create(model, cache);
  auto assign_target = MakeDoublesEngine(model, /*capacity=*/8, EosToken(*model)).engine;
  auto request = CreateRequestWithPrompt(assign_target, std::array<int32_t, 1>{5});
  scheduler->AddRequest(request);
  StepPlan plan;
  ASSERT_TRUE(scheduler->PlanStep(plan).executable);
  auto reservation = cache->ReserveStep(plan);
  auto scheduled_requests = scheduler->CreateScheduledRequests(plan);
  ExecutionContext context{&plan};
  context.cache_reservation = reservation->PagedReservation();
  auto executor = ModelExecutor::Create(model, cache);
  executor->Decode(scheduled_requests, context);

  EXPECT_EQ(scheduled_requests.HiddenStates(), nullptr);
  reservation->Release();
}

}  // namespace
}  // namespace test
}  // namespace Generators
