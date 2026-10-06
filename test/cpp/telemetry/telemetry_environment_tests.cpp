// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scoped-environment-variable.h"
#include "telemetry/telemetry_environment.h"

#include <deque>
#include <utility>

namespace {

using Generators::test::ScopedEnvironmentVariable;
using namespace Generators::TelemetryInternal;

class TelemetryEnvironmentTests : public ::testing::Test {
 protected:
  void SetUp() override {
    for (const char* name : kTelemetrySuppressionVariables) {
      variables_.emplace_back(name, std::nullopt);
    }
    variables_.emplace_back("ORT_DISABLE_TELEMETRY", std::nullopt);
  }

 private:
  std::deque<ScopedEnvironmentVariable> variables_;
};

TEST_F(TelemetryEnvironmentTests, ExplicitOptOutRequiresAffirmativeValue) {
  ScopedEnvironmentVariable opt_out{"ORT_DISABLE_TELEMETRY"};
  const std::pair<const char*, bool> cases[] = {
      {"1", true},
      {"true", true},
      {"TRUE", true},
      {" yes ", true},
      {"on", true},
      {"Y", true},
      {"", false},
      {" ", false},
      {"0", false},
      {"false", false},
      {"no", false},
      {"off", false},
      {"random", false},
  };
  for (const auto& [value, expected] : cases) {
    opt_out.Set(value);
    EXPECT_EQ(ShouldSuppressTelemetryFromEnvironment(), expected) << value;
  }
  opt_out.Set(std::string(Generators::kMaxTelemetryInputBytes + 1, ' '));
  EXPECT_TRUE(ShouldSuppressTelemetryFromEnvironment());
  opt_out.Set(std::nullopt);
  EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment());
}

TEST_F(TelemetryEnvironmentTests, RecognizesEveryCiAndTestFlagAndFailsClosed) {
  EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment());
  const std::string oversized(Generators::kMaxTelemetryInputBytes + 1, ' ');
  for (const char* name : kTelemetrySuppressionVariables) {
    ScopedEnvironmentVariable variable{name, "random"};
    EXPECT_TRUE(ShouldSuppressTelemetryFromEnvironment()) << name;
    variable.Set("0");
    EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment()) << name;
    variable.Set(oversized);
    EXPECT_TRUE(ShouldSuppressTelemetryFromEnvironment()) << name;
    variable.Set(std::nullopt);
    EXPECT_FALSE(ShouldSuppressTelemetryFromEnvironment()) << name;
  }
}

}  // namespace
