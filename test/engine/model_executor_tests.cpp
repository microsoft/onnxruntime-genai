// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "engine/model_executor.h"

namespace Generators {
namespace test {
namespace {

TEST(ModelExecutorTest, ClassifiesOrtArenaAllocationFailureAsCapacityExceeded) {
  EXPECT_EQ(
      ClassifyOrtExecutionFailure(
          "BFCArena::AllocateRawInternal Failed to allocate memory for "
          "requested buffer of size 2443129088"),
      ExecutionFailureKind::CapacityExceeded);
}

TEST(ModelExecutorTest, LeavesUnrelatedOrtFailureUnclassified) {
  EXPECT_EQ(
      ClassifyOrtExecutionFailure(
          "Non-zero status code returned while running a CUDA kernel."),
      ExecutionFailureKind::Unknown);
}

}  // namespace
}  // namespace test
}  // namespace Generators
