// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include <gtest/gtest.h>

#include "config.h"

namespace Generators::test {
namespace {

Config LoadDynamicConfig(std::string_view dynamic_batching) {
  const std::string overlay =
      R"({ "engine": { "dynamic_batching": )" +
      std::string{dynamic_batching} + " } }";
  return Config{
      fs::path{std::string{MODEL_PATH "engine/dummy-decoder"}},
      overlay};
}

TEST(DynamicBatchingConfigTest, ScheduledTokenBudgetDefaultsTo2048) {
  const auto config = LoadDynamicConfig(R"({ "max_batch_size": 4 })");

  ASSERT_TRUE(config.engine.dynamic_batching.has_value());
  EXPECT_EQ(config.engine.dynamic_batching->max_scheduled_tokens, 2048u);
}

TEST(DynamicBatchingConfigTest, ScheduledTokenBudgetAcceptsOverride) {
  const auto config =
      LoadDynamicConfig(R"({ "max_scheduled_tokens": 321 })");

  ASSERT_TRUE(config.engine.dynamic_batching.has_value());
  EXPECT_EQ(config.engine.dynamic_batching->max_scheduled_tokens, 321u);
}

class InvalidScheduledTokenBudgetTest
    : public ::testing::TestWithParam<const char*> {};

TEST_P(InvalidScheduledTokenBudgetTest, RejectsNonPositiveOrNonIntegralValue) {
  EXPECT_THROW(
      LoadDynamicConfig(
          std::string{R"({ "max_scheduled_tokens": )"} + GetParam() + " }"),
      std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidValues,
    InvalidScheduledTokenBudgetTest,
    ::testing::Values("0", "-1", "1.5", "4000000000"));

}  // namespace
}  // namespace Generators::test
