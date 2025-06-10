// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "narrow.h"

#include <complex>
#include <limits>

#include <gtest/gtest.h>

// These tests were adapted from:
// https://github.com/microsoft/GSL/blob/a3534567187d2edc428efd3f13466ff75fe5805c/tests/utils_tests.cpp#L127-L152

namespace Generators::test {

#define ASSERT_NARROW_FAILURE(expr) \
  ASSERT_THROW((expr), narrowing_error)

TEST(NarrowTest, Basic) {
  constexpr int n = 120;
  constexpr char c = narrow<char>(n);
  EXPECT_EQ(c, 120);

  EXPECT_EQ(narrow<uint32_t>(int32_t(0)), uint32_t{0});
  EXPECT_EQ(narrow<uint32_t>(int32_t(1)), uint32_t{1});
  constexpr auto int32_max = std::numeric_limits<int32_t>::max();
  EXPECT_EQ(narrow<uint32_t>(int32_max), static_cast<uint32_t>(int32_max));

  EXPECT_EQ(narrow<std::complex<float>>(std::complex<double>(4, 2)), std::complex<float>(4, 2));
}

TEST(NarrowTest, CharOutOfRange) {
  constexpr int n = 300;
  ASSERT_NARROW_FAILURE(narrow<char>(n));
}

TEST(NarrowTest, MinusOneToUint32OutOfRange) {
  ASSERT_NARROW_FAILURE(narrow<uint32_t>(int32_t(-1)));
}

TEST(NarrowTest, Int32MinToUint32OutOfRange) {
  constexpr auto int32_min = std::numeric_limits<int32_t>::min();
  ASSERT_NARROW_FAILURE(narrow<uint32_t>(int32_min));
}

TEST(NarrowTest, UnsignedOutOfRange) {
  constexpr int n = -42;
  ASSERT_NARROW_FAILURE(narrow<unsigned>(n));
}

namespace {
constexpr double kDoubleWithLossyRoundTripFloatConversion = 4.2;
static_assert(static_cast<double>(static_cast<float>(kDoubleWithLossyRoundTripFloatConversion)) !=
              kDoubleWithLossyRoundTripFloatConversion);
}  // namespace

TEST(NarrowTest, FloatLossyRoundTripConversion) {
  ASSERT_NARROW_FAILURE(narrow<float>(kDoubleWithLossyRoundTripFloatConversion));
}

TEST(NarrowTest, ComplexFloatLossyRoundTripConversion) {
  ASSERT_NARROW_FAILURE(narrow<std::complex<float>>(std::complex<double>(kDoubleWithLossyRoundTripFloatConversion)));
}

}  // namespace Generators::test
