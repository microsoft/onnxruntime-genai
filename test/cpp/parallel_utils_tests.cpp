// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "models/parallel_utils.h"

namespace Generators {
namespace {

TEST(ParallelCopyTests, HandlesEmptySmallLargeNullAndWorkerCounts) {
  for (size_t workers : {0U, 1U, 3U}) {
    ThreadPool pool{workers};
    for (size_t size : {0U, 1U, 16U, 200000U}) {
      std::vector<int32_t> source(size);
      std::iota(source.begin(), source.end(), 1);
      std::vector<int32_t> destination(size);
      ParallelCopy(&pool, std::span<const int32_t>{source},
                   std::span<int32_t>{destination});
      EXPECT_EQ(destination, source);
    }
  }
  std::vector<double> source(10000, 3.5), destination(source.size());
  ParallelCopy(nullptr, std::span<const double>{source},
               std::span<double>{destination});
  EXPECT_EQ(destination, source);
}

TEST(ParallelCopyTests, RejectsSizeMismatch) {
  std::vector<int> source(3), destination(2);
  EXPECT_THROW(ParallelCopy(nullptr, std::span<const int>{source},
                            std::span<int>{destination}),
               std::invalid_argument);
}

TEST(ParallelTransformTests, ConvertsTypesWithImmutableCapture) {
  std::vector<int64_t> source(100000);
  std::iota(source.begin(), source.end(), -50000);
  std::vector<float> destination(source.size());
  ThreadPool pool{3};
  constexpr float scale = 0.25f;
  ParallelTransform(&pool, std::span<const int64_t>{source},
                    std::span<float>{destination}, 2.0,
                    [scale](int64_t value) {
                      return static_cast<float>(value) * scale;
                    });
  for (size_t i = 0; i < source.size(); ++i)
    EXPECT_EQ(destination[i], static_cast<float>(source[i]) * scale);
}

TEST(ParallelTransformTests, HandlesEmptyMismatchAndExceptions) {
  std::vector<int> empty;
  EXPECT_NO_THROW(ParallelTransform(nullptr, std::span<const int>{empty},
                                    std::span<int>{empty}, 1.0,
                                    [](int value) { return value; }));
  std::vector<int> source(10000, 1), destination(9999);
  EXPECT_THROW(ParallelTransform(nullptr, std::span<const int>{source},
                                 std::span<int>{destination}, 1.0,
                                 [](int value) { return value; }),
               std::invalid_argument);
  destination.resize(source.size());
  source[0] = -1;
  ThreadPool pool{3};
  EXPECT_THROW(
      ParallelTransform(&pool, std::span<const int>{source},
                        std::span<int>{destination}, 100.0, [](int value) {
                          if (value < 0)
                            throw std::runtime_error("transform");
                          return value;
                        }),
      std::runtime_error);
}

TEST(ParallelTransformInPlaceTests, MatchesScalarAndPropagatesExceptions) {
  std::vector<int> values(100000);
  std::iota(values.begin(), values.end(), 0);
  auto expected = values;
  std::transform(expected.begin(), expected.end(), expected.begin(),
                 [](int value) { return value * 3 + 1; });
  ThreadPool pool{3};
  ParallelTransformInPlace(&pool, std::span<int>{values}, 2.0,
                           [](int value) { return value * 3 + 1; });
  EXPECT_EQ(values, expected);

  values.assign(10000, 1);
  values[0] = -1;
  EXPECT_THROW(ParallelTransformInPlace(
                   &pool, std::span<int>{values}, 100.0, [](int value) {
                     if (value < 0)
                       throw std::runtime_error("in-place");
                     return value;
                   }),
               std::runtime_error);
}

TEST(ParallelFillTests, HandlesEmptySmallLargeAndWorkerCounts) {
  for (size_t workers : {0U, 1U, 3U}) {
    ThreadPool pool{workers};
    for (size_t size : {0U, 8U, 100000U}) {
      std::vector<uint64_t> values(size);
      ParallelFill(&pool, std::span<uint64_t>{values}, uint64_t{42});
      EXPECT_EQ(values, std::vector<uint64_t>(size, 42));
    }
  }
}

}  // namespace
}  // namespace Generators
