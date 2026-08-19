// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdexcept>
#include <type_traits>

#include <gtest/gtest.h>

#include "cuda/sampler_state_index_pool.h"

namespace Generators {
namespace {

TEST(SamplerStateIndexPoolTest, NewIndexPreparationFailureIsAtomic) {
  SamplerStateIndexPool pool;

  EXPECT_THROW(
      pool.Acquire([](int index, int required_size) {
        EXPECT_EQ(index, 0);
        EXPECT_EQ(required_size, 1);
        throw std::runtime_error("Injected sampler state growth failure.");
      }),
      std::runtime_error);
  EXPECT_EQ(pool.Size(), 0);
  EXPECT_EQ(pool.ActiveCount(), 0u);
  EXPECT_EQ(pool.FreeCount(), 0u);

  const int index =
      pool.Acquire([](int, int) {});
  EXPECT_EQ(index, 0);
  EXPECT_EQ(pool.Size(), 1);
  EXPECT_EQ(pool.ActiveCount(), 1u);
}

TEST(SamplerStateIndexPoolTest, ReusedIndexPreparationFailureKeepsItFree) {
  SamplerStateIndexPool pool;
  const int original = pool.Acquire([](int, int) {});
  pool.Release(original);

  EXPECT_THROW(
      pool.Acquire([](int, int) {
        throw std::runtime_error("Injected sampler state initialization failure.");
      }),
      std::runtime_error);
  EXPECT_EQ(pool.Size(), 1);
  EXPECT_EQ(pool.ActiveCount(), 0u);
  EXPECT_EQ(pool.FreeCount(), 1u);

  const int retried = pool.Acquire([](int, int) {});
  EXPECT_EQ(retried, original);
  EXPECT_EQ(pool.ActiveCount(), 1u);
}

TEST(SamplerStateIndexPoolTest, ReleaseIsNoexceptAndAllocationFree) {
  SamplerStateIndexPool pool;
  const int first = pool.Acquire([](int, int) {});
  const int second = pool.Acquire([](int, int) {});

  static_assert(noexcept(pool.Release(first)));
  EXPECT_NO_THROW(pool.Release(first));
  EXPECT_NO_THROW(pool.Release(second));
  EXPECT_EQ(pool.ActiveCount(), 0u);
  EXPECT_EQ(pool.FreeCount(), 2u);
}

}  // namespace
}  // namespace Generators
