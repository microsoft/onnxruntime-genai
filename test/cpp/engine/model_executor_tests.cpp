// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "engine/model_executor.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"

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

}  // namespace
}  // namespace test
}  // namespace Generators
